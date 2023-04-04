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

sys.path.append("/home/yerinyoon/code/anonymousNet/data/celeba/")
from data_extract import *

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)



class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)#fake/real 여부
        out_cls = self.conv2(h)#도메인 개수가 5개  
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))




class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader,service_loader,config, wandb):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
        self.service_loader=service_loader
        
        self.wandb=wandb

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.reversed_attrs=config.reversed_attrs
        self.fixed_attrs=config.fixed_attrs
        # Test configurations.
        self.test_iters = config.test_iters


        self.real_img_dir=config.real_img_dir
        self.fake_img_dir=config.fake_img_dir
        self.fake_label_file=config.fake_label_file
        self.real_label_file=config.real_label_file
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

        torch.cuda.set_device(config.gpu_id)
        print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name(torch.cuda.current_device())}")

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        self.service_model_save_dir = config.service_model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        
        self.selected_attrs=config.selected[config.group_index]
        self.fixed_attrs=config.fixed_attrs[config.group_index]
        self.reversed_attrs=config.reversed_attrs[config.group_index]

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        self.mode=config.mode        



    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
#         self.print_network(self.G, 'G')
#         self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
    
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
#         from logger import Logger
#         self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
#train

    def create_labels(self, c_org, c_dim=6, dataset='CelebA'):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if self.mode=="train":
            if dataset == 'CelebA':
                hair_color_indices = [] #hair color는 하나의 column에서 제공하고자 한다
                for i, attr_name in enumerate(self.selected_attrs):
                    if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                        hair_color_indices.append(i)
    
    
            c_trg_list = []
            c_trg=c_org.clone()
            for i in range(c_dim):
      
                if dataset == 'CelebA':
                    #c_trg = c_org.clone()
                    
                    if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                        c_trg[:, i] = 1
                        for j in hair_color_indices:
                            if j != i:
                                c_trg[:, j] = 0
                    else:
                       # print(i)
                     
                        c_trg[:, i] = (c_trg[:, i] == 0)  # Revoerse attribute value.
                elif dataset == 'RaFD':
                    c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
    
                c_trg_list.append(c_trg.to(self.device))
            return c_trg_list
       #        # Get hair color indices.
        else:
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                hair_color_indices = []#hair color는 하나의 column에서 제공하고자 한다
                for i, attr_name in enumerate(selected_attrs): 
                    if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                        hair_color_indices.append(i)     
  
                c_trg_list = []
                for i, attr_name in enumerate(selected_attrs): 
                    if i in hair_color_indices:
                        c_trg[:, i] = 1
                        for j in hair_color_indices:
                            if j != i:
                                c_trg[:, j] = 0

                for i, attr_name in enumerate(selected_attrs):
                    c_trg[:, i]=(c_trg[:, i]==0)
                    if attr_name in self.fixed_attrs:
                        c_trg[:, i] = (c_trg[:, i] == 0)
                    #c_trg[:, i] = (c_trg[:, i] == 0) 
    #                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
    #                    hair_color_indices.append(i)     
    
          #  c_trg_list = []
                #for i, attr_name in enumerate(selected_attrs): 
    #                if i in hair_color_indices:
    #                    c_trg[:, i] = 1
    #                    for j in hair_color_indices:
    #                        if j != i:
    #                            c_trg[:, j] = 0
    #                else:
    #                    if attr_name in service_attrs:
                    #c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
    
            return c_trg.to(self.device)
    
#
#    def create_labels(self, c_org, c_dim=6, dataset='CelebA',selected_attrs=None, reversed_attrs=None, fixed_attrs=None):
#        """Generate target domain labels for debugging and testing."""
#        # Get hair color indices.
#        if dataset == 'CelebA':
#            c_trg = c_org.clone()
#            hair_color_indices = []#hair color는 하나의 column에서 제공하고자 한다
#            for i, attr_name in enumerate(selected_attrs):
#                
#                if attr_name in fixed_attrs:
#                    c_trg[:, i] = (c_trg[:, i] == 0)
#                #c_trg[:, i] = (c_trg[:, i] == 0) 
##                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
##                    hair_color_indices.append(i)     
#
#      #  c_trg_list = []
#            #for i, attr_name in enumerate(selected_attrs): 
##                if i in hair_color_indices:
##                    c_trg[:, i] = 1
##                    for j in hair_color_indices:
##                        if j != i:
##                            c_trg[:, j] = 0
##                else:
##                    if attr_name in service_attrs:
#                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
#        elif dataset == 'RaFD':
#            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
#
#        return c_trg.to(self.device)
#

 #
 #   
 #
 #
 #
    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self, config):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        seleted_pics = [17689, 12133, 150707, 112343, 23272, 11278, 6056, 34538, 36297]
        
        
        transform = []
        transform.append(T.RandomHorizontalFlip())
        transform.append(T.CenterCrop(178))
        transform.append(T.Resize(128))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        
        
        tmp_dataset = CelebA(config.celeba_image_dir, config.attr_path, self.selected_attrs, transform, mode='train')
        x_fixed = [tmp_dataset[i-2000][0] for i in seleted_pics]
        c_org = [tmp_dataset[i-2000][1] for i in seleted_pics]# 
        
        x_fixed = torch.stack(x_fixed)
        c_org = torch.stack(c_org)
        
#         data_iter = iter(data_loader)
#         x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

 
        loss={}
        loss['D/loss_real'] =[]
        loss['D/loss_fake'] = []
        loss['D/loss_cls'] = []
        loss['D/loss_gp'] = []
        loss['G/loss_fake'] = []
        loss['G/loss_rec'] = []
        loss['G/loss_cls'] = []
        
        self.wandb.watch(self.G)
        self.wandb.watch(self.D)
 #
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in tqdm(range(start_iters, self.num_iters)):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
            print(label_org)

            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
 #            
 #            group_data=pd.read_csv(f"{config.group_folder}/{config.target_attr}_group_number.csv")
 #            target_label=group_data.iloc[:, :-1]
 #
 #            import random
 #
 #            # Generate target domain labels randomly.
 #            rand_idx =random.choices(range(1,group_data.shape[0]), weights=group_data.iloc[:, -1]) #weight=count 
 #            label_trg = target_label[rand_idx]   ##TODO: check.
 #

            

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real) # real/face and domain 
            ###real:큰 값 fake: 작은 값
            d_loss_real = - torch.mean(out_src) # log를 안써도 되나?? 
            #실제 수식
            #E_xc'(-log(D_cls(C'|x))): log가 실제로 안중요할 것 같긴하지만...
            #out_src가 큰값이 등장하면 진짜 
             
            
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)
            ###도메인 구별 여부 D_cls

            # Compute loss with fake images.
            
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)
            
            
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            loss["D/loss_real"].append(d_loss_real.item())
            loss["D/loss_fake"].append(d_loss_fake.item())
            loss["D/loss_cls"].append(d_loss_cls.item())
            loss["D/loss_gp"].append(d_loss_gp.item())
           # Logging.
#            
#            self.wandb.log({'D/loss_real': d_loss_real.item(), 
#                'D/loss_fake':d_loss_fake.item(),
#                'D/loss_cls': d_loss_cls.item(),
#                'D/loss_gp':d_loss_gp.item()}, step=i)
#            
#        


            # =================================================================================== #
            #                               3.glg Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

               # Logging.
                loss['G/loss_fake'].append(g_loss_fake.item())
                loss['G/loss_rec'].append(g_loss_rec.item())
                loss['G/loss_cls'].append(g_loss_cls.item())
               
                self.wandb.log({'D/loss_real': d_loss_real.item(), 
                    'D/loss_fake':d_loss_fake.item(),
                     'D/loss_cls': d_loss_cls.item(),
                 'D/loss_gp':d_loss_gp.item(), "G/loss_fake":g_loss_fake.item(), "G/loss_rec":g_loss_rec.item(),
                 "G/loss_cls":g_loss_cls.item()}, step=i)
             

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                   
                    log += ", {}: {:.4f}".format(tag, value[-1])
                print(log)
               
#                 if self.use_tensorboard:
#                     for tag, value in loss.items():
#                         self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            self.sample_step = 2000
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1, i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))
                    
                    import numpy as np
                    import matplotlib.pyplot as plt
    
                    def show(img):
                        npimg = img.numpy()
                        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
                        plt.rcParams['figure.figsize'] = (100.0,50.0)
                        plt.show()

                    from torchvision.utils import make_grid
                    #show(make_grid(self.denorm(x_concat.data.cpu()), nrow = 1, padding=0))
                    save_image(self.denorm(x_concat.data.cpu()),
                               'sample_{}.jpg'.format(i),
                               nrow=1, padding=0)
                    
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir,"iter{}", '{}-G.ckpt'.format(i+1, i+1))
                D_path = os.path.join(self.model_save_dir,"iter{}", '{}-D.ckpt'.format(i+1, i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
       # self.wandb.log(loss)
#
#  def test(self):
#        """Translate images using StarGAN trained on a single dataset."""
#        # Load the trained generator.
#        
#        # Set data loader.
#        if self.dataset == 'CelebA':
#            data_loader = self.celeba_loader
#        elif self.dataset == 'RaFD':
#            data_loader = self.rafd_loader
#        
#        with torch.no_grad():
#            for i, (x_real, c_org) in enumerate(data_loader):
#
#                # Prepare input images and target domain labels.
#                
#                x=x_real.clone()
#                x_real=x_real.to(self.device)
#                x=x.to(self.device)
#                c_trg_list = self.create_labels(c_org, 3, self.dataset, self.service_attrs)
#               # print(len(c_trg_list))
#                # Translate images.
#               # for i in c_trg_list:
#                   # print(i.shape())
#                x_fake_list = [x_real]
#                for c_trg in c_trg_list:
#                    x=self.G(x, c_trg)
#                    x_fake_list.append(x)
#                    
#
#                # Save the translated images.
#                x_concat = torch.cat(x_fake_list, dim=3)
#                result_path = os.path.join(self.service_model_save_dir, '{}-images.jpg'.format(i+1))
#               # print(result_path)
#                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
#                print('Saved real and fake images into {}...'.format(result_path))
#
#





   
#real test
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg= self.create_labels(c_org, self.c_dim, self.dataset,self.selected_attrs)

                # Translate 
                x_fake_list = [x_real]
                x_fake_list.append(self.G(x_real, c_trg))
                  #

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-skin-fixed-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
                
    def create_img_data(self):
        print(self.full_label)
        print(self.full_label.shape)
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.service_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        # Label for new image
        new_labels=[]
        labels=[]

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                #x_fake=self.G(x_real, c_trg)
                c_trg= self.create_labels(c_org, self.c_dim, self.dataset,self.selected_attrs)
               # print(c_trg)
                #print(c_trg.data.cpu())
               
               # for j in self.seleted_attrs:
                #    self.full_label[j]

               
             
               
                new_labels.append(c_trg.data.cpu().tolist())
                #labels.append(c_org.data.cpu().tolist())

                x_fake=self.G(x_real, c_trg)
                real_result_path = os.path.join(self.real_img_dir, '{}-original-images.jpg'.format(i+1))
                save_image(self.denorm(x_real.data.cpu()), real_result_path, nrow=1, padding=0)
                fake_result_path = os.path.join(self.fake_img_dir, '{}-deidentify-images.jpg'.format(i+1))
                save_image(self.denorm(x_fake.data.cpu()), fake_result_path, nrow=1, padding=0)

                print('Saved real and fake images into {}...'.format(i))
            import pandas as pd
            
            new_labels=np.array(new_labels)
            new_labels=np.squeeze(new_labels, 1)
            #labels=np.array(labels)
            #labels=np.squeeze(labels, 1)

            print(new_labels.shape)
            new_labels=pd.DataFrame(new_labels, columns=self.selected_attrs)
            service_labels=self.full_label.copy()
            service_labels[self.selected_attrs]=new_labels

            service_labels.to_csv(self.fake_label_file)
            self.full_label.to_csv(self.real_label_file)
            
            #labels=pd.DataFrame(labels, columns=self.selected_attrs)
            #labels.to_csv(self.real_label_file, index=False)
        
