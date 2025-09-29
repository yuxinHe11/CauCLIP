import torch.utils.data as data
from torch.utils.data import DataLoader
from .data_utils import *
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import random
from PIL import Image
import torch
from torchvision.transforms import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class AugSurgVisdom(data.Dataset):
    def __init__(self, img_file, labels_file,
                 num_segments, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=0, kfold=False,
                 alpha=1):  

        self.img_file = img_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.kfold = kfold
        self.alpha = alpha  
        self.backgrounds = []
        self.background_groups = {0: [], 1: []}  
        self._parse_list()
        self.initialized = False
    
    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def _parse_list(self):
        if not self.kfold:
            lines = [x.strip().split(' ') for x in open(self.list_file)]
        else:
            lines = self.list_file

        self.video_list = []
        for idx, line in enumerate(lines):
            record = VideoRecord(line[:3])  # path, num_frames, label
            background = int(line[3])       # background:0, 1
            self.video_list.append(record)
            self.backgrounds.append(background)
            self.background_groups[background].append(idx)

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                start_offset = randint(1, record.num_frames) 
                return np.mod(np.arange(self.total_length) + start_offset, record.num_frames) + self.index_bias
            else:
                offsets = np.concatenate((
                    np.arange(1, record.num_frames),  
                    randint(1, record.num_frames, size=self.total_length - record.num_frames + 1)  
                ))
                return np.sort(offsets)[:self.total_length] + self.index_bias  
        else:
            offsets = list()
            ticks = [i * record.num_frames // self.num_segments for i in range(self.num_segments + 1)]

            for i in range(self.num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    if tick == 0:
                        if tick_len >= self.seg_length:
                            tick += randint(0, tick_len - self.seg_length)
                    else:
                        tick += randint(0, tick_len - self.seg_length)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=np.int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=int) + self.index_bias


    def _sample_from_other_background(self, current_background):
        other_background = 1 - current_background
        return random.choice(self.background_groups[other_background])
    
    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record, segment_indices)
    
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def get(self, record, indices):
        images = []
        for seg_ind in indices:
            p = max(1, int(seg_ind))
            try:
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)  
            except OSError:
                print(f'ERROR: Could not read image "{record.path}"')
                raise

     
        img_idx = self.video_list.index(record)
        img_background = self.backgrounds[img_idx]

        other_idx = self._sample_from_other_background(img_background)
        other_record = self.video_list[other_idx]
        other_indices = self._sample_indices(other_record) if self.random_shift else self._get_val_indices(other_record)
        other_images = []
        for seg_ind in other_indices:
            p = max(1, int(seg_ind))
            try:
                seg_imgs = self._load_image(other_record.path, p)
                other_images.extend(seg_imgs)
            except OSError:
                print(f'ERROR: Could not read image "{other_record.path}"')
                raise

        img_origs = []  
        img_augs = []  
        
        for img_orig, img_other in zip(images, other_images):
            img1 = np.array(img_orig)
            img2 = np.array(img_other)

            img_aug, _ = spectrum_colorful_mix(img1, img2, alpha=self.alpha)
        
            if self.transform is not None:
                orig_tensor = self.transform(img_orig)  
                aug_tensor = self.transform(Image.fromarray(img_aug))  
            else:
                orig_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
                aug_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0

            img_origs.append(orig_tensor)
            img_augs.append(aug_tensor)

        img_origs = torch.stack(img_origs)  
        img_augs = torch.stack(img_augs)  

        return (img_origs, img_augs), record.label
    
    def __len__(self):
        return len(self.video_list)

class SurgVisDom(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=0, kfold=False):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.kfold = kfold

        if self.index_bias is None:
             if self.image_tmpl == "frame{:d}.jpg":
                 self.index_bias = 0
             else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        if not self.kfold:
            self.video_list = [VideoRecord(x.strip().split(' '))
                               for x in open(self.list_file)]
        else:
            self.video_list = [VideoRecord(x)
                               for x in self.list_file]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                start_offset = randint(1, record.num_frames)  
                return np.mod(np.arange(self.total_length) + start_offset, record.num_frames) + self.index_bias
            else:
                offsets = np.concatenate((
                    np.arange(1, record.num_frames),
                    randint(1, record.num_frames, size=self.total_length - record.num_frames + 1)  
                ))
                return np.sort(offsets)[:self.total_length] + self.index_bias  
        else:
            offsets = list()
            ticks = [i * record.num_frames // self.num_segments for i in range(self.num_segments + 1)]

            for i in range(self.num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    if tick == 0:
                        tick += randint(1, tick_len - self.seg_length + 1)
                    else:
                        tick += randint(0, tick_len - self.seg_length)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=np.int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


    def get(self, record, indices):
        images = list()
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            if p == 0:
                p = 1  
            try:
                seg_imgs = self._load_image(record.path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
