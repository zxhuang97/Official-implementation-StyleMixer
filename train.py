import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import model
from utils import *
import net


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

parser = argparse.ArgumentParser()

#parameters
parser.add_argument('--num',type=int,default=1)
parser.add_argument('--mode',type=str,default='add')
parser.add_argument('--bandwidth',type=int,default=1)
parser.add_argument('--p_size',type=int,default=3)

parser.add_argument('--reshuffle', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=6)

parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=3.0)
parser.add_argument('--identity_weight', type=float, default=1.0)
parser.add_argument('--tv_weight',type=float,default=0.0)
parser.add_argument('--cx_weight', type=float, default=3.0)

parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=40000)
parser.add_argument('--save_img_interval',default=300)

#Basic options
parser.add_argument('--content_dir', type=str, required=False, default='./coco', \
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=False, default='./wikiart', \
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./checkpoint/vgg_normalised.pth')

# training options
parser.add_argument('--sample_dir',default='./sample/',
                    help='Directory to save transfer sample')
parser.add_argument('--save_dir', default='./checkpoint/',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs/',
                    help='Directory to save the log')
args = parser.parse_args()
args.use_cx = True if args.cx_weight > 0 else False
args.use_iden = True if args.identity_weight > 0 else False
config = vars(args)
exp_name, config = preparation(config)
print(exp_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=config['log_dir'])

print(torch.cuda.get_device_name(0))

network = net.Net(**config)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(config['content_dir'], content_tf, reshuffle=config['reshuffle'])
style_dataset = FlatFolderDataset(config['style_dir'], style_tf, reshuffle=config['reshuffle'])

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=config['batch_size'],
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=config['n_threads']))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=config['batch_size'],
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=config['n_threads']))

# def adjust_learning_rate(optimizer, iteration_count):
#     """Imitating the original implementation"""
#     lr = config['lr'] / (1.0 + config['lr_decay'] * iteration_count)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr
optimizer = torch.optim.Adam(network.parameters(), lr=config['lr'], weight_decay=config['lr_decay'])
start_iter=0

for name, param in network.named_parameters():
    if param.requires_grad:
        print(name)
for i in range(config['max_iter']):
    optimizer.zero_grad()
    # crt_lr=adjust_learning_rate(optimizer, iteration_count=(i+start_iter))
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    
    res, loss_c, loss_s, loss_i, loss_cx, loss_tv = network(content_images, style_images)

    if i%100==0:
        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)  
        writer.add_scalar('loss_identity', loss_i.item(), i + 1)
        if config['use_cx']: writer.add_scalar('loss_cx', loss_cx.item(), i + 1)
        writer.add_scalar('loss_tv', loss_tv.item(), i+1)
        writer.add_scalar('lr',crt_lr,i+1)
      
   
    loss_i =config['identity_weight'] * loss_i
    loss_c = config['content_weight'] * loss_c
    loss_s = config['style_weight'] * loss_s
    loss_cx = config['cx_weight'] * loss_cx
    loss_tv = config['tv_weight'] * loss_tv
    loss = loss_i + loss_c + loss_s +loss_cx + loss_tv
    if i%100==0:  
    	writer.add_scalar('total_loss',loss.item(),i+1)
    	print("iter %d "%i,loss.item())

    loss.backward()
    optimizer.step()

    if(i%config['save_img_interval']==0):
        d = res.data[0].to(torch.device('cpu'))
        img = content_images.data[0].to(torch.device('cpu'))
        save_image(img, './content.jpg')
        save_image(d, './see.jpg')
        
    if (i + 1) % config['save_model_interval'] == 0 or (i + 1) == config['max_iter']:
        state_dict = network.state_dict()
        torch.save(state_dict,
                   '{:s}/iter_{:d}.pth.tar'.format(config['save_dir'],
                                                           i + 1))
writer.close()
