import numpy as np
import os
from PIL import Image
from PIL import ImageFile
from torch.utils import data
from torchvision import transforms
import glob
import torch
import random
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

def preparation(config):
    name = '_'.join([
        item for item in [
        'styleMixer',
        'bw%d' % config['bandwidth'],
        'p_%d'
        'reshuf' if config['reshuffle'] else None,        
        'style%2.2f' % config['style_weight'],
        'cont%2.2f' % config['content_weight'],
        'iden%2.2f' % config['identity_weight'] if config['use_iden'] else None,
        'cx%2.2f' % config['cx_weight'] if config['use_cx'] else  None,
        '%d' % config['num']
        ] if item is not None])

    config['save_dir'] += name
    config['log_dir'] += name
    config['sample_dir'] += name

    if not os.path.exists(config['save_dir']):
        os.mkdir(config['save_dir'])
    if not os.path.exists(config['log_dir']):
        os.mkdir(config['log_dir'])
    if not os.path.exists(config['sample_dir']):
        os.mkdir(config['sample_dir'])

    return name, config

def train_transform():
    transform_list = [
        transforms.Resize(size=512),
        transforms.RandomCrop(256),
    ]
    return transforms.Compose(transform_list)

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

def MyFlip(image):
    out=image
    if random.random()<0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random()<0.5:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    return out

def MyReshuffle(image,block_num=4):
    new_image=Image.new("RGB",(256,256),0)
    block_size = 256 // block_num
    l=block_num*block_num
    random_list=batch = np.random.permutation(l)
    for i in range(l):
        col=random_list[i]%block_num
        row=int(random_list[i]/block_num)
        box=(col*block_size,row*block_size,col*block_size+block_size,row*block_size+block_size)
        region= MyFlip(image.crop(box))

        col1=i%block_num
        row1=int(i/block_num)
        box1=(col1*block_size,row1*block_size,col1*block_size+block_size,row1*block_size+block_size)
        new_image.paste(region,box1)
    return new_image

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform,reshuffle=False):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        if(self.root=='/public/zixuhuang3/wikiart'):
            self.paths = glob.glob('/public/zixuhuang3/wikiart/*/*')
        else:
            self.paths = glob.glob(self.root+'/*')
        self.num=len(self.paths)
        print(self.num)
        self.transform = transform
        self.reshuffle=reshuffle

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        if(self.reshuffle):
            new_img= MyFlip(img)
            img_tensor = transforms.ToTensor()(img)
            new_img_tensor = transforms.ToTensor()(new_img)# H,W,C -> C,H,W [0,1]
            return torch.stack([img_tensor,new_img_tensor])
        else:
            img_tensor = transforms.ToTensor()(img)
            return img_tensor

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

