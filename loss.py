import torch.nn as nn
import torch

from ContextualLoss import ContextualLoss
from function import adaptive_instance_normalization as adain
from function import calc_mean_std

mse_loss = nn.MSELoss()
l1 = nn.L1Loss()
cx = ContextualLoss()

def calc_style_loss( input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)

def normalized_mse_loss( input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    size = target.size()
    eps = 1e-6
    cs_mean, cs_std = calc_mean_std(input)
    c_mean,c_std = calc_mean_std(target)
    cs_normalized_feat = (input - cs_mean.expand(size)) / (cs_std.expand(size)+eps)
    c_normalized_feat = (target - c_mean.expand(size)) / (c_std.expand(size)+eps)
    return mse_loss(cs_normalized_feat, c_normalized_feat)

def perceptual_loss(cs_feats, cont_feats):
    loss_p = 0.0
    for cs, cont in zip(cs_feats,cont_feats):
        loss_p += normalized_mse_loss(cs, cont)
    return loss_p

def adain_style_loss(cs_feats, style_feats):
    loss_s = 0.0
    for i in range(0, 5):
        loss_s += calc_style_loss(cs_feats[i], style_feats[i])
    return loss_s

def identity_loss( cc, cc_feats, content, cont_feats, ss, ss_feats, style, style_feats, weight):
    loss_i = mse_loss(cc, content) + mse_loss(ss, style)
    for cc_feat, c_feat in zip(cc_feats, cont_feats):
        loss_i += weight * mse_loss(cc_feat, c_feat)
    for ss_feat, s_feat in zip(ss_feats, style_feats):
        loss_i += weight * mse_loss(ss_feat, s_feat)
    return loss_i

def contextual_loss(cs_feats, style_feats):
    loss_cx = 0.0
    for i in range(2, 4):
        loss_cx += cx(cs_feats[i], style_feats[i])
    return loss_cx

def total_variation(x):
    loss_y = l1(x[:,:,:,1:],x[:,:,:,:-1])
    loss_x = l1(x[:,:,1:,:],x[:,:,:-1,:])
    return loss_x + loss_y