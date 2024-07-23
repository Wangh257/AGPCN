import cv2
import os
import torch
import logging
import wandb
import argparse

from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, Discrimate_GAN
from loss import FocalLoss, SSIM, MseDirectionLoss
from tqdm import tqdm

from model_with_attention import ReconstructiveSubNetwork_CBAMattention, ReconstructiveSubNetwork_Spattention
from model_with_attention import ReconstructiveSubNetwork_Chattention, ReconstructiveSubNetwork_Spattention_only_encoder
from ResUNet import ResUNet
from unet import UNet, UNet_attention
from res_net_18_50_101 import Resnet_Unet_18_50_101, Resnet_Unet_18_50_101_attention
# import unet.unet_model.UNet as UNet

from evaluate import evaluate
from utils.data_loading import BasicDataset

from glob import glob

# prepare data 

dir_img = '/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu/images_GT'
dir_mask = '/home/wangh20/data/structure/metal_dataset/Network_use_small_1_fenzi_fenmu'

dataset = BasicDataset(dir_img, dir_mask, 1.0)

test_loader = DataLoader(dataset, shuffle=False, batch_size=1, drop_last=True, num_workers=16)

device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

# load model and checkpoints 
model = model = UNet_attention(4, 8, False)
checkpoint_path = './tools/checkpoints/fenzi_fenmu/add_plat_qiu/UNet_attention/bs8_lr0.001_1000'
checkpoint_path = glob(os.path.join(checkpoint_path, "*.pth"))[0]
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device=device)
# save 
save_dir = '/home/wangh20/projects/DSAR_light/Generation_fenzi_fenmu/tools/results/attention_map'
os.makedirs(save_dir, exist_ok=True)
save_attention_map_in_path =  '/home/wangh20/projects/DSAR_light/Generation_fenzi_fenmu/tools/results/attention_map/attention_map_in.bmp'
save_attention_map_out_path = '/home/wangh20/projects/DSAR_light/Generation_fenzi_fenmu/tools/results/attention_map/attention_map_out.bmp'
save_attention_map_in_2_path = '/home/wangh20/projects/DSAR_light/Generation_fenzi_fenmu/tools/results/attention_map/attention_map_in_2.bmp'
save_attention_map_out_2_path = '/home/wangh20/projects/DSAR_light/Generation_fenzi_fenmu/tools/results/attention_map/attention_map_out_2.bmp'


x1 = torch.randn(1, 64, 544, 640)
x1_mean = x1.mean(dim=1)
x1_mean = x1_mean.squeeze().detach().cpu().numpy() 
x1_mean = x1_mean * 255
x1_mean = x1_mean.astype('uint8')
import cv2
cv2.imwrite(save_attention_map_in_path, x1_mean)


w_x1 = torch.randn(1, 64, 544, 640)
w_x1_mean = w_x1.mean(dim=1)
w_x1_mean = w_x1_mean.squeeze().detach().cpu().numpy() 
w_x1_mean = w_x1_mean * 255
w_x1_mean = w_x1_mean.astype('uint8')
cv2.imwrite(save_attention_map_out_path, w_x1_mean)

# x2_mean = x2.mean(dim=1)
# x2_mean = x2_mean.squeeze().detach().cpu().numpy() 
# x2_mean = x2_mean * 255
# x2_mean = x2_mean.astype('uint8')
# import cv2
# cv2.imwrite(save_attention_map_in_2_path, x2_mean)



# w_x2_mean = w_x2.mean(dim=1)
# w_x2_mean = w_x2_mean.squeeze().detach().cpu().numpy() 
# w_x2_mean = w_x2_mean * 255
# w_x2_mean = w_x2_mean.astype('uint8')
# cv2.imwrite(save_attention_map_out_2_path, w_x2_mean)



# test process
for index, batch in tqdm(enumerate(test_loader), desc="A Processing Bar Test: "):
    images, true_masks = batch['image'], batch['mask']
    name = batch['name'][0]
    if name != '587':
        continue
    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    true_masks = true_masks.to(device=device, dtype=torch.float32)
    masks_pred, _ = model(images)
