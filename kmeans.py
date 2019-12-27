import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from function import calc_mean_std
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
import random

def extract_PatchesAndNorm(feats, ksize=3, padding=1, stride=1):
    '''
    feats = (B,C,H,W), with B being 1
    core function: torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
    return: patches = (num_patch, C, patch_size, patch_size) , kernel_norm = (num_patch, 1, 1, 1)
    '''
    B,C,H,W = feats.size()
    padding = ksize//2

    # => (patch_size * patch_size * C, num_patch)
    unfold = nn.Unfold(kernel_size=(ksize,ksize), padding=0, stride=stride)
    content_pad = nn.ReflectionPad2d(padding)
    feats = content_pad(feats)
    raw_patches = unfold(feats)
    raw_patches = torch.squeeze(raw_patches,0)

    kernels_norm = torch.norm(raw_patches, 2, 0, keepdim=True)
    kernels_norm = kernels_norm.view(-1,1,1,1)

    # => (num_patch, C, patch_size, patch_size)
    patches = raw_patches.view(C,ksize, ksize,-1)
    patches = patches.permute(3,0,1,2)                                            

    return patches, kernels_norm


def styleSwap(cont_feats, style_feats, patch_size=5, stride=1):
    '''
    Both con_feats and style_feats are normalized feature in shape of (B,C,H,W)
    Return: swapped feature in the same shape of cont_feats
    '''
    C, H, W = cont_feats.size()[1:] 
    style_kernels, kernels_norm = extract_PatchesAndNorm(style_feats, ksize=patch_size)
    _, cont_norm = extract_PatchesAndNorm(cont_feats, ksize=patch_size, stride=stride)

    ######################################
    # calculate the normalization factor #
    ######################################
    mask = torch.cuda.FloatTensor(H,W).fill_(1)
    fullmask = torch.cuda.FloatTensor(H + patch_size -1, W + patch_size -1).fill_(0)
    for x in range(patch_size):
        for y in range(patch_size):
            padding = (x, patch_size - x - 1, y, patch_size - y - 1)
            mask_pad = nn.ConstantPad2d(padding,0)
            padded_mask = mask_pad(mask)
            fullmask += padded_mask
    pad_width = int((patch_size-1) / 2)
    deconv_norm = fullmask[ pad_width:pad_width+H , pad_width:pad_width+W ]
    deconv_norm = deconv_norm.view(1,1,H,W)
    
    ########################
    # starting convolution #
    ########################
    pad_total = patch_size-1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padding = (pad_beg,pad_end,pad_beg,pad_end)

    content_pad = nn.ReflectionPad2d(padding)
    net = content_pad(cont_feats)

    net = F.conv2d(net, torch.div(style_kernels, kernels_norm + 1e-7),stride=stride)
    # normalized for fair comparison between location
    # net = net.div_(cont_norm.view(1, 1, H, W))

    # Binarize the result by retrieve the max #
    best_match_ids = torch.argmax(net, dim=1, keepdim=True).cuda()
    best_match_map = torch.cuda.FloatTensor(1, *net.size()[1:] ).fill_(0)
    best_match_map.scatter_(1,best_match_ids,1.0)

    best_match_factor =  torch.max(net, dim=1, keepdim=False)[0].cuda()
    best_match_factor = best_match_factor.view(net.size()[0],-1)#->to vector

    # Find the patch and warping the output #
    unnormalized_output = F.conv_transpose2d(
        best_match_map,
        style_kernels,
        stride=stride)

    unnormalized_output = unnormalized_output[:,:,pad_beg : H + pad_beg, pad_beg : W+pad_beg]
    normalized_output = torch.div(unnormalized_output, deconv_norm)

    return normalized_output, best_match_factor
    #B,C,H,W, B,H*W

def forgy(X, Y, n_clusters):
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state , Y[indices]

def pairwise_distance(feat, loc, feat_state, loc_state, loc_weight=1.0, device=-1):
    '''
    args:
    feat:[H*W, C] loc:[X*W, 2] feat_state:[num, C]
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    A = feat.unsqueeze(dim=1)
    B = feat_state.unsqueeze(dim=0)
    feat_dis = (A-B)**2.0
    # A_norm = torch.norm(A, 2, 2, keepdim=True)
    # B_norm = torch.norm(B, 2, 2, keepdim=True)
    # feat_dis = (A * B)/(A_norm * B_norm + 1e-7)

    C = loc.unsqueeze(dim=1)
    D = loc_state.unsqueeze(dim=0)

    loc_dis = (C-D)**2.0

    feat_dis = feat_dis.sum(dim=-1)
    loc_dis = loc_dis.sum(dim=-1)**(1/2)

    # print(feat_dis.min(dim=1)[0].view(-1).mean())
    dis = feat_dis + loc_weight * loc_dis
    return dis.squeeze()

def kmeans(X, locMap, n_clusters=5, device=0, tol=1e-4, loc_weight= 1.0):
    # X:[C,H*W], locMap[2,H*W]
    C, num = X.size()
    X = X.permute(1, 0)
    locMap = locMap.permute(1, 0)
    initial_feat, initial_loc= forgy(X, locMap, n_clusters)

    for cnt in range(200):
        dis = pairwise_distance(X, locMap, initial_feat, initial_loc, loc_weight=loc_weight)
        choice_cluster = torch.argmin(dis, dim=1)
        # initial_state_pre = initial_state.clone()
 
        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster==index).squeeze()
            selected_feat = torch.index_select(X, 0, selected)
            selected_loc = torch.index_select(locMap, 0, selected)
            initial_feat[index] = selected_feat.mean(dim=0)
            initial_loc[index] = selected_loc.mean(dim=0)

        # center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
 
        # if center_shift ** 2 < tol:
        #     break
    # print(choice_cluster[0])

    # H*W
    return choice_cluster

def adaptive_instance_normalization(content_feat):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return content_mean,content_std,normalized_feat

def adaptive_instance_colorization(projected_features, feature_kernels, mean_features):
    return projected_features * feature_kernels.expand(projected_features.size()) + mean_features.expand(projected_features.size())

#https://github.com/xxradon/PytorchWCT/blob/master/util.py
def zca_normalization(features,ty='C',th=0.00001): 
    size = features.size()#B=1,C,H,W
    features=features.squeeze().view(size[1],-1)

    # get unbiased_features
    mean_features = torch.mean(features,dim=1,keepdim=True) #c x (h x w)
    unbiased_features = features - mean_features
    # get covariance matrix
    if(ty=='C'):
        gram = torch.matmul(unbiased_features,unbiased_features.t()).div(features.size()[-1]-1)+\
            torch.eye(size[1]).type(torch.cuda.FloatTensor)#(C,C)
    elif (ty=='S'):
        gram = torch.matmul(unbiased_features,unbiased_features.t()).div(features.size()[-1]-1)
    # svd and demension reduction
    u,s,v = torch.svd(gram,some=False)
    k = size[1]
    for i in range(size[1]):
        if s[i] < th:
            k = i
            break 

    sqrt_s_effective=s[:k].pow(0.5)
    sqrt_inv_s_effective=s[:k].pow(-0.5)

    # normalized features

    step1 = torch.matmul(v[:,:k],torch.diag(sqrt_inv_s_effective))
    step2 = torch.matmul(step1,v[:,:k].t())
    whiten_features = torch.matmul(step2,unbiased_features)
    whiten_features=whiten_features.view(size[1],size[2],size[3]).unsqueeze(dim=0)
    # colorization kernel

    colorization_kernel=torch.matmul(torch.matmul(v[:,:k],torch.diag(sqrt_s_effective)),(v[:,:k].t()))

    return mean_features, colorization_kernel, whiten_features

def zca_colorization(whiten_features, colorization_kernel, mean_features):
    size=whiten_features.size()#B=1,C,H,W
    whiten_features=whiten_features.squeeze().view(size[1],-1)
    colorized_features = torch.matmul(colorization_kernel,whiten_features) + mean_features
    colorized_features = colorized_features.view(size[1],size[2],size[3])
    return colorized_features.unsqueeze(dim=0)


def getLocMap(H,W):
    index = [i for i in range(H*W)]
    x = torch.cuda.FloatTensor(index).view(1,H*W)%W
    y = torch.cuda.FloatTensor(index).view(1,H*W)//W
    loc = torch.cat((x,y),dim=0)
    return loc

def multi_style_warp(content, styles, alpha, coding='ada', patch_size=5, num_cluster = 5, loc_weight=0.0):
    #content,style: B,C,H,W

    if coding=='ada':
        normalization = adaptive_instance_normalization
        colorization = adaptive_instance_colorization
    else:
        normalization = zca_normalization
        colorization = zca_colorization

    B, C, H, W=content.size()
    cont_mean, cont_std, normalized_cont = normalization(content)
    choice_maps=[]

    for i in range(B):
        tmp_content = content[i].view(C, H*W)
        locMap = getLocMap(H,W)
        choice_map = kmeans(tmp_content, locMap, num_cluster, loc_weight)
        choice_maps.append(choice_map.unsqueeze(dim=0))

    style_num=len(styles)
    style_warps=[]
    style_maps=[]
    for i in range(style_num):
        style_mean, style_kernel, normalized_style = normalization(styles[i])
        style_warp, style_map = styleSwap(normalized_cont, normalized_style, patch_size=patch_size)
        style_warp = style_warp * alpha + normalized_cont * (1-alpha)
        style_warp = colorization(style_warp, style_kernel, style_mean)

        style_warps.append(style_warp.view(C, H*W))
        style_maps.append(style_map)

    #Here we have content B==1
    mult_swap_feature_map = torch.zeros(C, H*W).type(torch.cuda.FloatTensor)
    style_alloc_map = torch.zeros(B, H*W).type(torch.cuda.FloatTensor)
    count=0
    for i in range(num_cluster):
        choice_cluster = (choice_maps[0] == i).type(torch.cuda.FloatTensor)
        score_list = np.zeros((style_num))
        for j in range(style_num):
            score = choice_cluster*(style_maps[j])
            total_score = torch.sum(score)
            score_list[j] = (total_score)
        style_id = np.argmax(score_list)
        count += style_id
        # if random.random() < 0.5:
        #     style_id=0
        # else:
        #     style_id=1
        mult_swap_feature_map = mult_swap_feature_map + style_warps[style_id].mul(choice_cluster)
        style_alloc_map += style_id * choice_cluster

    print(count/num_cluster)

    return mult_swap_feature_map.view(1,C,H,W), style_alloc_map.view(1,1,H,W)

def multi_style_warp(content, feats_map, alpha,  num_cluster = 5, loc_weight=0.0):
    #content B,C,H,W
    #style_feats style_num * [1,C,H*,W*]
    #conf_maps [B,1,H,W]
    style_feats = [x[0] for x in feats_map]
    conf_maps = [x[1] for x in feats_map]

    B, C, H, W=content.size()
    choice_maps=[]

    for i in range(B):
        tmp_content = content[i].view(C, H*W)
        locMap = getLocMap(H,W)
        choice_map = kmeans(tmp_content, locMap, num_cluster, loc_weight)
        choice_maps.append(choice_map.unsqueeze(dim=0))

    style_num = len(style_feats)
    style_warps = []
    style_maps = []
    for i in range(style_num):
        style_warp, style_map = style_feats[i], conf_maps[i].view(1,-1)
        # style_warp = style_warp * alpha + normalized_cont * (1-alpha)
        style_warps.append(style_warp.view(C, H*W))
        style_maps.append(style_map)

    #Here we have content B==1
    mult_swap_feature_map = torch.zeros(C, H*W).type(torch.cuda.FloatTensor)
    style_alloc_map = torch.zeros(B, H*W).type(torch.cuda.FloatTensor)
    count=0

    for i in range(num_cluster):
        choice_cluster = (choice_maps[0] == i).type(torch.cuda.FloatTensor)
        score_list = np.zeros((style_num))
        for j in range(style_num):
            score = choice_cluster*(style_maps[j])
            score = score.squeeze()
            score = score[torch.nonzero(score)]
            sorted_score,__ = torch.sort(score,descending=True)
            leng= sorted_score.size(0)
            total_score = torch.sum(sorted_score[:int(leng*0.95)])
            total_score = torch.sum(score)
            score_list[j] = (total_score)
        style_id = np.argmax(score_list)
        count += style_id
        # if random.random() < 0.5:
        #     style_id=0
        # else:
        #     style_id=1
        mult_swap_feature_map = mult_swap_feature_map + style_warps[style_id].mul(choice_cluster)
        style_alloc_map += style_maps[style_id] * choice_cluster

    print(count/num_cluster)

    return mult_swap_feature_map.view(1,C,H,W), style_alloc_map.view(1,1,H,W)




