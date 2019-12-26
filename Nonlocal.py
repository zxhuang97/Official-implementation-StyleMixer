import torch.nn as nn
import torch
from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from torch.nn import functional as F

class Fusion(nn.Module):
    def __init__(self,in_channels,k_size=1):
        super(Fusion,self).__init__()

    def forward(self, content, style, state=None):
        output = content + style
        return output

class Nonlocal(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2,bandwidth=1.0):
        super(Nonlocal, self).__init__()
        assert dimension in [1, 2, 3]
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dimension = dimension
        self.bandwidth = bandwidth
        
        #set inter_channel
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
        #dimension == 2 
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d

        #Parameter initialization
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias,0)
        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.W.weight)
        nn.init.constant_(self.W.bias, 0)

        self.operation_function = self.op

      
    def extract_PatchesAndNorm(self,feats, ksize=3, padding=1, stride=1):
        '''
        feats = (B,C,H,W), with B being 1
        core function: torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
        return: patches = (num_patch, C, patch_size, patch_size) , kernel_norm = (num_patch, 1, 1, 1)
        '''
        B,C,H,W = feats.size()
        padding = ksize//2

        # => (B, patch_size * patch_size * C, num_patch)
        unfold = nn.Unfold(kernel_size=(ksize,ksize), padding=0, stride=stride)
        content_pad = nn.ReflectionPad2d(padding)
        feats = content_pad(feats)
        raw_patches = unfold(feats)
        
        # => （B, H * W, patch_size * patch_size * C）
        raw_patches = raw_patches.permute(0,2,1)
      
        # => ksize = patch_size, kernel (C * patch_size * patch_size)
        # kernel=torch.tensor([[1,2,1],[2,4,2],[1,2,1]]).type(torch.cuda.FloatTensor)*(1/16)    #assign Gaussion weight
        kernel=torch.tensor([[1,1,1],[1,1,1],[1,1,1]]).type(torch.cuda.FloatTensor)             #assign equal weight
        kernel=kernel.expand([C,ksize,ksize])
        kernel=kernel.contiguous().view(C*ksize*ksize)
        raw_patches=raw_patches*kernel

        return raw_patches
    
    def forward(self,content,style,fusion_style,isTraining=True,normal=True):#whether normalize inputs to calculate correspondence
        output = self.operation_function(content,style,fusion_style,isTraining,normal)
        return output

    def op(self, x, s, fusion_style, isTraining, normal, eps=1e-5):
        batch_size = x.size(0)

        #recalibrate features 
        g_s = self.g(fusion_style).view(batch_size, self.inter_channels, -1)
        g_s = g_s.permute(0, 2, 1)
        if(normal==True):
            x_size = x.size()
            s_size = s.size()
            x_mean, x_std = calc_mean_std(x)
            s_mean, s_std=calc_mean_std(s)
            x_normalized_feat = (x - x_mean.expand(x_size)) / x_std.expand(x_size)
            s_normalized_feat = (s - s_mean.expand(s_size)) / s_std.expand(s_size)
        theta_x = self.theta(x_normalized_feat)
        phi_s = self.phi(s_normalized_feat)
        theta_x=self.extract_PatchesAndNorm(theta_x)
        phi_s=self.extract_PatchesAndNorm(phi_s).permute(0,2,1)

        # calculate attention map, f_div_C->normalized feature map
        f = torch.matmul(theta_x, phi_s) * self.bandwidth
        f_div_C = F.softmax(f, dim=-1)
        if(isTraining==False):
            conf_map=f_div_C*f
            conf_map=conf_map.sum(-1).view(batch_size, 1, *x.size()[2:])
        else:
            conf_map=0

        # swap style features
        y = torch.matmul(f_div_C, g_s)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        
        return W_y,conf_map