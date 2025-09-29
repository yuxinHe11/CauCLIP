import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch.nn as nn
from datasets import SurgVisDom
from datasets import AugSurgVisDom
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from utils.KLLoss import KLLoss
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *
from utils.FactorizationLoss import factorization_loss
import logging

S = 'cfg'
prompt_type = 1
print('prompt type: ', prompt_type)

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)
    
def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-%s' % S, default='./configs/SurgVisDom.yaml')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], time_now)

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
    shutil.copy('train.py', working_dir)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, clip_state_dict = clip.load(config.network.arch,device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,
                                       pretrain=config.network.init, joint = config.network.joint)  # Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    fusion_model = visual_prompt(config.network.sim_header,
                                 clip_state_dict,config.data.num_segments, 6)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    if device == "cpu":
        model_text.float()
        model_image.float()
    else :
        clip.model.convert_weights(model_text) 
        clip.model.convert_weights(model_image)

    loss_img = KLLoss()
    loss_txt = KLLoss()

    start_epoch = config.solver.start_epoch
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    train_data = AugSurgVisDom(config.data.train_list, config.data.label_list,
                                 num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                                 random_shift=config.data.random_shift, transform=transform_train, alpha=config.solver.alpha)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size,
                              num_workers=0, shuffle=True, pin_memory=False, drop_last=True)
    
    classes, num_text_aug, text_dict = text_prompt(train_data, prompt=prompt_type)
    optimizer = _optimizer(config, model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)
 
    best_prec1 = 0.0
    for epoch in range(start_epoch, config.solver.epochs):
        epoch_loss = 0.0
        model_image.train()
        model_text.train()
        fusion_model.train()
        for kkk, ((images,images_aug), list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])

            b,t,c,h,w = images.size()
            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            images= images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(device)

            images_aug = images_aug.view((-1,config.data.num_segments,3)+images.size()[-2:])
            images_aug = images_aug.to(device).view(-1,c,h,w)

            image_embedding = model_image(images)  # [512, 512]
            image_embedding = image_embedding.view(b,t,-1)  # [bs, num_seg, 512]
            image_embedding = fusion_model(image_embedding)  # [bs, 512]

            image_aug_embedding = model_image(images_aug)  # [512, 512]
            image_aug_embedding = image_aug_embedding.view(b,t,-1)  # [bs, num_seg, 512]
            image_aug_embedding = fusion_model(image_aug_embedding)  # [bs, 512]
            loss_fac = factorization_loss(image_embedding, image_aug_embedding)

            text_embedding = model_text(texts)  # [bs, 512]

            if config.network.fix_text:
                text_embedding.detach_()
            if config.network.fix_img:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)
            logits_per_image_aug, logits_per_text_aug = create_logits(image_aug_embedding, text_embedding, logit_scale)
            # [bs,bs]

            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)

            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            loss_imgs_aug = loss_img(logits_per_image_aug, ground_truth)
            loss_texts_aug = loss_txt(logits_per_text_aug, ground_truth)

            loss_clip = (loss_imgs + loss_texts)/2.0 
            loss_clip_aug = (loss_imgs_aug + loss_texts_aug)/2.0 
            total_loss = loss_clip + config.solver.fac_weight * loss_fac + config.solver.aug_weight * loss_clip_aug

            total_loss.backward()
            epoch_loss += total_loss.cpu().data

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        print('epoch loss: ', epoch_loss/(kkk+1))
        logger.info(f'Epoch:{epoch}, loss:{epoch_loss/(kkk+1)}')
        filename = "{}/last_model.pt".format(working_dir)
        epoch_saving(epoch, model, fusion_model, optimizer, filename)

        
if __name__=='__main__':
    main()