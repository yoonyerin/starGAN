import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import matplotlib

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch

import random


import sys

import argparse
from torch.backends import cudnn

sys.path.append("/Users/yerinyoon/Documents/cubig/anonymousNet/data/celeba/")

from model import *
from data_extract import *

print(sys.version)
import wandb

parser = argparse.ArgumentParser()
def str2bool(v):
    return v.lower() in ('true')

def main(config):
    #os.environ("CUDA_VISIBLE_DEVICES")=0
    wandb.init(project="starGAN")
    wandb.config.update(config)


    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader=get_loader(config, config.celeba_image_dir, config.attr_path,
                                   config.all_attrs, config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
       # service_loader=service_get_loader(config, config.celeba_image_dir, config.attr_path, config.selected, config.all_attrs, config.celeba_crop_size, config.image_size, 1, 'CelebA', config.mode, config.num_workers)
 #        gan_loader, full_label=get_loader(config.celeba_image_dir, config.attr_path,
 #                config.selected_attrs, config.all_attrs, config.celeba_crop_size,
 #                config.image_size, 300000, 'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)
    

    # Solver for training and testing StarGAN.
    if config.mode=='service':
        solver = Solver(celeba_loader, rafd_loader,  config, wandb)

    #############################################################
    #       calculate c_dim for the changing selected_attrs     #
    ##############################################################
    
    c_dim=len(config.selected[config.group_index])

    solver = Solver(celeba_loader, rafd_loader, config, wandb, c_dim)
 
    if config.mode == 'train':

        if config.dataset in ['CelebA', 'RaFD']:
            solver.train(config)
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()
    elif config.mode == 'service':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.create_img_data()
        elif config.dataset in ['Both']:
            solver.test_multi() 

parser.add_argument("--target_attr", type=list, default=["Wearing_Lipstick","Attractive", "High_Cheekbones", "High_Cheekbones", "Mouth_Slightly_Open", "Smiling"])
parser.add_argument("--selected", type=list, default=[["Narrow_Eyes", "Straight_Hair", "Pale_Skin", "Receding_Hairline", "Bald", "Oval_Face", 
            "Double_Chin", "Bushy_Eyebrows", "5_o_Clock_Shadow", "Wavy_Hair", "Wearing_Earrings", "No_Beard", "Arched_Eyebrows", "Attractive", "Male",
            "Heavy_Makeup"], [ "Mouth_Slightly_Open", "Straight_Hair", "Bushy_Eyebrows", "Bangs", "5_o_Clock_Shadow", "Big_Lips", "Narrow_Eyes", 
             "Double_Chin", "Wavy_Hair", "Pointy_Nose", "Chubby", "Arched_Eyebrows", "Big_Nose", "Young", "Male", "Heavy_Makeup", "Wearing_Lipstick"], ["Young", "Straight_Hair", "Receding_Hairline", "Chubby", "Big_Lips", "5_o_Clock_Shadow", "No_Beard", "Oval_Face", "Rosy_Cheeks", "Male",
            "Heavy_Makeup", "Wearing_Lipstick", "Mouth_Slightly_Open"], ["Narrow_Eyes", "Straight_Hair", "Pale_Skin", "Mouth_Slightly_Open", "Receding_Hairline", "Oval_Face", "Goatee", "Big_Nose",
            "Attractive", "Arched_Eyebrows", "5_o_Clock_Shadow", "No_Beard", "Heavy_Makeup", "Wearing_Lipstick"],
            ["Bald", "Pointy_Nose", "Young", "Straight_Hair", "Attractive", "Chubby", "Oval_Face", "Male", "Narrow_Eyes", "Rosy_Cheeks",
            "High_Cheekbones"],[ "Straight_Hair", "Big_Lips","Young",
            "Bags_Under_Eyes", "No_Beard", "Male", "Attractive", "Oval_Face", "Rosy_Cheeks", "Mouth_Slightly_Open", "High_Cheekbones"]])

parser.add_argument('--reversed_attrs', type=list, nargs='+', help='selected attributes for the CelebA dataset',
                    default=[
                        #1
                       ["Narrow_Eyes", "Straight_Hair", "Pale_Skin",  "Receding_Hairline", "Bald", "Oval_Face",
            "Double_Chin", "Bushy_Eyebrows"],
                       #2
            [ "Mouth_Slightly_Open", "Straight_Hair", "Bushy_Eyebrows", "Bangs", "5_o_Clock_Shadow", "Big_Lips", "Narrow_Eyes"],
            #3
            ["Young", "Straight_Hair", "Receding_Hairline", "Chubby", "Big_Lips"  ],
            #4
            ["Narrow_Eyes", "Straight_Hair", "Pale_Skin", "Mouth_Slightly_Open", "Receding_Hairline", "Oval_Face"],
            #5
            ["Bald", "Pointy_Nose", "hair_color", "Young", "Straight_Hair", "Attractive", "Chubby"],
            #6
            [ "Straight_Hair", "Big_Lips","Young"]])
parser.add_argument('--fixed_attrs', type=list, nargs='+', help='selected attributes for the CelebA dataset',
                    default=[
                        #1
                        ["5_o_Clock_Shadow", "Wavy_Hair", "Wearing_Earrings", "No_Beard", "Arched_Eyebrows", "Attractive", "Male", "Heavy_Makeup"],
                        #2
                        ["Double_Chin", "Wavy_Hair", "Pointy_Nose", "Chubby", "Arched_Eyebrows", "Big_Nose", "Young", "Male", "Heavy_Makeup", "Wearing_Lipstick"], 
                        #3
                        ["5_o_Clock_Shadow", "No_Beard", "Oval_Face", "Rosy_Cheeks", "Male",
            "Heavy_Makeup", "Wearing_Lipstick", "Mouth_Slightly_Open"],
                        #4
                        ["Goatee", "Big_Nose", "Attractive", "Arched_Eyebrows", "5_o_Clock_Shadow", "No_Beard", "Heavy_Makeup", "Wearing_Lipstick"], 
                        #5
                        ["Oval_Face", "Male", "Narrow_Eyes", "Rosy_Cheeks", "High_Cheekbones"],
                        #6
                        ["Bags_Under_Eyes", "No_Beard", "Male", "Attractive", "Oval_Face", "Rosy_Cheeks", "Mouth_Slightly_Open", "High_Cheekbones"]])


           

# Model configuration.

#parser.add_argument('--c_dim', type=int, default=13, help='dimension of domain labels (1st dataset)')
parser.add_argument('--c2_dim', type=int, default=6, help='dimension of domain labels (2nd dataset)')
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
parser.add_argument('--image_size', type=int, default=128, help='image resolution')
parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

# Training configuration.
parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')


#parser.add_argument('--num_iters', type=int, default=300000, help='number of total iterations for training D')
parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')

parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
#<<<<<<< HEAD
parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
parser.add_argument('--all_attrs',type=list, nargs='+', default=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
                       'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 
                       'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                       'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 
                       'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                       'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])
 #parser.add_argument('--selected_attrs',type=list, nargs='+', help='selected attributes for the CelebA dataset',
 #        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline',  'Narrow_Eyes', 'Pointy_Nose',  'Bushy_Eyebrows','Arched_Eyebrows', 'Big_Nose', 'Male', 'High_Cheekbones', "Pale_Skin"])
 #'''parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
 #                    default=['Arched_Eyebrows', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Chubby', 'Double_Chin',
 #'High_Cheekbones', 'Narrow_Eyes', 'Oval_Face', 'Pointy_Nose', 'Sideburns'])
 #'''
 #parser.add_argument('--reversed_attrs', type=list, nargs='+', help='selected attributes for the CelebA dataset',
 #                    default=["Narrow_Eyes", "Bushy_Eyebrows","Arched_EyeBrows", "Big_Nose"])
 #parser.add_argument('--fixed_attrs', type=list, nargs='+', help='selected attributes for the CelebA dataset',
 #                    default=['Pointy_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline', "Male", "High_CheekBones", "Pale_Skin"])
 #
 #
#=======
 #parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
 #parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
 #                    default=['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Big_Lips',
 #                            'Big_Nose','Black_Hair','Bushy_Eyebrows','Chubby','Double_Chin',
 #                            'High_Cheekbones','Mustache','Narrow_Eyes','Oval_Face','Pale_Skin',
 #                            'Pointy_Nose','Sideburns','Straight_Hair'])
 #                    
                    # Bangs, Black_Hair, BushEyebrows
'''parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                    default=['Arched_Eyebrows', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Chubby', 'Double_Chin',
'High_Cheekbones', 'Narrow_Eyes', 'Oval_Face', 'Pointy_Nose', 'Sideburns'])
'''
parser.add_argument('--service_attrs', type=list, nargs='+', help='selected attributes for the CelebA dataset',
                    default=["Narrow_Eyes", "Pointy_Nose", "Gray_Hair", "Straight_Hair", "Bushy_Eyebrows", "PaleSkin"])
#>>>>>>> 9b620f8b6ff58ae098d16c62779eb6dbd4a59e6c
# Test configuration.
parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

# Miscellaneous.
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'service'])

parser.add_argument('--group_index', type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
 #
 #
 #
 #parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
 #parser.add_argument('--use_tensorboard', type=str2bool, default=True)
 #
# Directories.
parser.add_argument('--celeba_image_dir', type=str, default="/data/datasets/CelebA/img_align_celeba")
parser.add_argument('--attr_path', type=str, default="/data/datasets/CelebA/list_attr_celeba.txt")
# parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
parser.add_argument('--log_dir', type=str, default='../')
parser.add_argument('--model_save_dir', type=str, default='/data/checkpoints/yerin/starGAN/1/celeba128/experiment/onlyReversed6_stargan_lr0.0001_bs16_recon10_lambda1.0/')
parser.add_argument('--sample_dir', type=str, default='/data/checkpoints/yerin/starGAN/1/celeba128/experiment/onlyReversed6_stargan_lr0.0001_bs16_recon10_lambda1.0/iter230000')
parser.add_argument('--result_dir', type=str, default='/data/checkpoints/yerin/starGAN/1/celeba128/experiment/onlyReversed6_stargan_lr0.0001_bs16_recon10_lambda1.0/iter230000')

parser.add_argument('--service_model_save_dir', type=str, default='/home/yerinyoon/code/anonymousNet/service_model_save_point')
parser.add_argument('--real_img_dir', type=str, default="/home/yerinyoon/code/anonymousNet/real_dir/real_img_dir")
parser.add_argument('--real_label_file', type=str, default="/home/yerinyoon/code/anonymousNet/real_dir/real_label_file.csv")
parser.add_argument('--fake_img_dir', type=str, default="/home/yerinyoon/code/anonymousNet/fake_dir/fake_img_dir")
parser.add_argument('--fake_label_file', type=str, default="/home/yerinyoon/code/anonymousNet/fake_dir/fake_label_file.csv")
 #
 #parser.add_argument('--log_dir', type=str, default='./log')
 #parser.add_argument('--model_save_dir', type=str, default='/Users/yerinyoon/Documents/cubig/anonymousNet/service_model_save_point')
 #parser.add_argument('--sample_dir', type=str, default='/Users/yerinyoon/Documents/cubig/anonymousNet/service_model_save_point')
 #parser.add_argument('--result_dir', type=str, default='/Users/yerinyoon/Documents/cubig/anonymousNet/service_model_save_point')
 #
 #parser.add_argument('--service_model_save_dir', type=str, default='/Users/yerinyoon/Documents/cubig/anonymousNet/service_model_save_point')
 #
# Step size.
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=10000)
parser.add_argument('--lr_update_step', type=int, default=1000)

#model save
parser.add_argument("--output-prefix", default="model")

parser.add_argument('--resume-from', default="/data/checkpoints/yerin/starGAN/1/celeba128/experiment/onlyReversed6_stargan_lr0.0001_bs16_recon10_lambda1.0/iter200000")

#parser.add_argument('--resume-from', default="/Users/yerinyoon/Documents/cubig/anonymousNet/service_model_save_point")
parser.add_argument('--input-dim', type=int)
parser.add_argument("--gpu_id", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("--group_folder",type=str, default="/home/yerinyoon/code/anonymousNet/mobile_select_attribute/group_data/")



config = parser.parse_args()
# print(config)

main(config)
