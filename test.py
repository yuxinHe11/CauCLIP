import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import clip
import torch.nn as nn
from datasets import SurgVisDom
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *
import pdb
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import logging



class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)
      

def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, logger):
    model.eval()
    if fusion_model is not None:
        fusion_model.eval()
    num = 0
    corr_id = [0, 0, 0]
    f1_dict={
        0:[0, 0, 0],
        1:[0, 0, 0],
        2:[0, 0, 0]
    }
    num_id = [0, 0, 0]
    encode_tensors = []
    encode_tensors_2 = []
    channels = 3 if not config.network.a_cha else 4

    with torch.no_grad():
        text_inputs = classes.to(device)                
        text_features = model.encode_text(text_inputs) 
        p = (torch.ones(3) / 3).to(device) 
        m = 0.9   
        lamda = 1   

        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, channels) + image.size()[-2:]) # [b, 16, c, h, w]
            b, t, c, h, w = image.size()
            assert c == channels
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input)

            if fusion_model is not None:
                image_features = image_features.view(b, t, -1)
                image_features = fusion_model(image_features) 

            image_features /= image_features.norm(dim=-1, keepdim=True)                      
            assert image_features.shape[0] == b and image_features.shape[1] == 512
            
            text_features /= text_features.norm(dim=-1, keepdim=True)                        
            similarity = (100.0 * image_features @ text_features.T)                          
            similarity = similarity.view(b, num_text_aug, -1)                                 

            similarity = similarity.softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)                                
            values_1, indices_1 = similarity.topk(1, dim=-1)                                 
            num += b

            batch_p = torch.zeros(3).to(device) 
            
            for i in range(b):
                batch_p[indices_1[i]] += 1
                encode_tensors.append((image_features[i], indices_1[i]))
                encode_tensors_2.append((image_features[i], class_id[i]))
                num_id[class_id[i]] += 1
                label = int(class_id[i])
                predict = int(indices_1[i])
                f1_dict[label][predict] += 1
                if indices_1[i] == class_id[i]:
                    corr_id[class_id[i]] += 1
            
            batch_p = batch_p / b
            p = m*p + (1-m)*batch_p
            
        acc_0 = corr_id[0] / num_id[0] if num_id[0] != 0 else 0
        acc_1 = corr_id[1] / num_id[1] if num_id[1] != 0 else 0
        acc_2 = corr_id[2] / num_id[2] if num_id[2] != 0 else 0
        bacc = (acc_0 + acc_1 + acc_2) / 3.0

        confusion = f1_dict  # alias
    precision = []
    recall = []
    f1 = []
    total_samples = sum(num_id)
    weighted_f1_sum = 0

    TP_total = 0
    FP_total = 0
    FN_total = 0

    for i in range(3):
        TP = confusion[i][i]
        FP = sum(confusion[j][i] for j in range(3) if j != i)
        FN = sum(confusion[i][j] for j in range(3) if j != i)

        TP_total += TP
        FP_total += FP
        FN_total += FN

        prec = TP / (TP + FP) if (TP + FP) != 0 else 0
        rec = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_i = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_i)

        weighted_f1_sum += num_id[i] * f1_i

    macro_f1 = sum(f1) / 3
    weighted_f1 = weighted_f1_sum / total_samples

    micro_prec = TP_total / (TP_total + FP_total) if (TP_total + FP_total) != 0 else 0
    micro_rec = TP_total / (TP_total + FN_total) if (TP_total + FN_total) != 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) != 0 else 0

    logger.info('Epoch:{} bacc:{:.3f} Weighted F1:{:.3f} Macro F1:{:.3f} Micro F1:{:.3f}'.format(
    epoch, bacc, weighted_f1, macro_f1, micro_f1
    ))


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/SurgVisDom.yaml')
    parser.add_argument('--log_time', default='test')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)

    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)
    print(transform_val)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(fusion_model)

    test_data = SurgVisDom(config.data.test_list, config.data.label_list, num_segments=config.data.num_segments,
                          image_tmpl=config.data.image_tmpl,
                          transform=transform_val, random_shift=config.random_shift, test_mode= True)
    test_loader = DataLoader(test_data, batch_size=config.data.batch_size, num_workers=1, shuffle=False,
                            pin_memory=True, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'], strict=False)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(test_data)

    def getLogger():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                      datefmt="%a %b %d %H:%M:%S %Y")

        sHandler = logging.StreamHandler()
        sHandler.setFormatter(formatter)

        logger.addHandler(sHandler)

        pth = os.path.join(working_dir, working_dir.split('/')[-1]+'.txt')

        fHandler = logging.FileHandler(pth, mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)
        return logger

    logger = getLogger()

    validate(start_epoch, test_loader, classes, device, model, fusion_model, config, num_text_aug, logger)

if __name__ == '__main__':
    main()
