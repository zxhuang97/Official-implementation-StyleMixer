import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import model
import net
from function import adaptive_instance_normalization
from function import coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(network, content, style, alpha=1.0, num_cluster=10, loc_weight=0.0):
    return network.multi_transfer(content, style, alpha=alpha, num_cluster=num_cluster, loc_weight=loc_weight)

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--net',type=str,default='nonlocal')
parser.add_argument('--use_ada', action='store_true', default=False)
parser.add_argument('--bandwidth',type=int,default=1)
parser.add_argument('--alpha', type=float, default=0.8,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument('--p', type=int, default=5,
                    help='patch_size')
parser.add_argument('--c', type=int, default=10,
                    help='num of clusters')
parser.add_argument('--loc_weight', type=float, default=0.0)
parser.add_argument('--max_layer', type=int, default=4)
parser.add_argument('--name', type=str)
parser.add_argument('--iter',type=int,default=8)
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default='input/content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, 
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,default='input/style',
                    help='Directory path to a batch of style images')
parser.add_argument('--style_dir2', type=str,default='input/style',
                    help='Directory path to a batch of style images')
parser.add_argument('--mw',type=str, default='./checkpoint/')

parser.add_argument('--vgg', type=str, default='./checkpoint/vgg_normalised.pth')
# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=448,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', default=False,  #add --crop, then config[crop=True  
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='./sample/',
                    help='Directory to save the output image(s)')

# Advanced options

config  = vars(parser.parse_args())

setting = config['name'].split('_')
if setting[2][:2]=='bw':
    config['bandwidth'] = setting[1][-1]
config['mw'] += "%s/iter_%d0000.pth.tar" % (config['name'], config['iter'])
config['output'] += config['name']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


if config['content']:
    content_paths = [config['content']]
else:
    content_paths = [os.path.join(config['content_dir'], f) for f in
                     os.listdir(config['content_dir'])]
if config['style']:
    style_paths = config['style'].split(',')
else:
    style_paths = [os.path.join(config['style_dir'], f) for f in
                   os.listdir(config['style_dir'])]

if not os.path.exists(config['output']):
    os.mkdir(config['output'])


network = net.Net(**config)
network.load_state_dict(torch.load(config['mw']))
network.eval()
network.to(device)

content_tf = test_transform(config['content_size'], config['crop'])
style_tf = test_transform(config['style_size'], config['crop'])


styles = [style_tf(Image.open(p).convert('RGB')).to(device) for p in style_paths]
styles = [style.unsqueeze(0) for style in styles]

out =config['output']

for content_path in content_paths:
    # process one content and one style
    
    content = content_tf(Image.open(content_path).convert('RGB'))
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(network, content, styles, alpha = config['alpha'],
            num_cluster = config['c'], loc_weight = config['loc_weight'])
    output = output.cpu()
    output_name = '{:s}/{:s}{:s}'.format(
        out, splitext(basename(content_path))[0],
        config['save_ext']
    )
    print(output_name)
    save_image(output, output_name)
