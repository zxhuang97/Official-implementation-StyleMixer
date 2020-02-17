import torch.nn as nn
import torch
import model
from torch.nn import functional as F
from Nonlocal import Nonlocal, Fusion
from kmeans import multi_style_warp
import loss

class ScaleLayer(nn.Module):
   def __init__(self, init_value=2.0):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))
   def forward(self, input):
       return input * self.scale

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Feature_pyramid(nn.Module):
    def __init__(self):
        super(Feature_pyramid, self).__init__()
        self.latlayer1 = nn.Conv2d(256, 256, (1,1))
        self.latlayer2 = nn.Conv2d(512, 256, (1,1))
        self.latlayer3 = nn.Conv2d(512, 256, (1,1))

        self.channel_attention=SELayer(3 * 256)
        self.reflectPad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.squeeze = nn.Conv2d(3 * 256, 512, (3, 3),padding=(0,0))


    def forward(self,feats):
        top,mid,btm = feats[0],feats[1],feats[2]
        top_sample = self.latlayer1(top)
        top_sample = F.interpolate(top_sample, size=mid.size()[2:], mode='bilinear',align_corners=True)
        btm_sample = self.latlayer3(btm)
        btm_sample = F.interpolate(btm_sample, size=mid.size()[2:], mode='bilinear',align_corners=True)
        mid_sample =self.latlayer2(mid)

        result = torch.cat((top_sample,mid_sample,btm_sample),1)
        #channel wise attention
        result=self.channel_attention(result)
        result = self.reflectPad(result)
        result = self.squeeze(result)
        return result

arch = [0, 4, 11, 18, 31, 44]

class Net(nn.Module):
    def __init__(self, bandwidth=1, p_size=3, train=True, vgg='', use_iden=True, use_cx=True, **kwargs):

        super(Net, self).__init__()
        encoder, decoder = model.get_vgg(vgg)

        self.is_train = train
        self.use_iden = use_iden
        self.use_cx = use_cx

        enc_layers = list(encoder.children())
        self.encoder = []
        for i in range(len(arch)-1):
            self.encoder += [nn.Sequential(*enc_layers[arch[i]:arch[i+1]])]
        self.encoder = nn.ModuleList(self.encoder)

        self.non_local = Nonlocal(512, bandwidth=bandwidth, p_size=p_size)
        self.reflectPad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.feature_pyramid = Feature_pyramid()
        self.amplifier = ScaleLayer()
        self.fusion = Fusion(512)
        self.decoder = decoder

        # fix the encoder
        for module in self.encoder:
            for param in module.parameters():
                param.requires_grad = False
   
    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for block in self.encoder:
            results.append(block(results[-1]))
        return results[1:]

    def pair_inference(self, cont_feats, style_feats, hidden_cont_feats, hidden_style_feats, iden=False):
        cs_nonlocal, cs_map = self.non_local(cont_feats[-2], style_feats[-2], hidden_style_feats)
        cs_fused = self.amplifier(cs_nonlocal) if iden else self.fusion(hidden_cont_feats, cs_nonlocal)
        cs =   self.decoder(cs_fused)
        cs_feats = self.encode_with_intermediate(cs)
        return cs, cs_feats

    def forward(self, content, style, alpha=1.0):
     
        style_feats = self.encode_with_intermediate(style)
        cont_feats = self.encode_with_intermediate(content)
        
        hidden_cont_feats = self.feature_pyramid(cont_feats[-3:])
        hidden_style_feats = self.feature_pyramid(style_feats[-3:])

        cs, cs_feats = self.pair_inference(cont_feats, style_feats, hidden_cont_feats, hidden_style_feats)
        if not self.training:
            return cs

        # perceptual
        loss_c = loss.perceptual_loss(cs_feats[-3:], cont_feats[-3:])         

        # Style Loss
        loss_s = loss.adain_style_loss(cs_feats, style_feats)

        result = (cs, loss_c, loss_s)

        if self.use_iden:
            cc, cc_feats = self.pair_inference(cont_feats, cont_feats, hidden_cont_feats, hidden_cont_feats, True)
            ss, ss_feats = self.pair_inference(style_feats, style_feats, hidden_style_feats, hidden_style_feats, True)
            loss_i = loss.identity_loss(cc, cc_feats, content, cont_feats, 
                                         ss, ss_feats, style, style_feats, 50)
            result += (loss_i,)
        else:
            result += (0,)
        if self.use_cx:
            loss_cx = loss.contextual_loss(cs_feats, style_feats)
            result +=(loss_cx,)
        else:
            result += (0,)
        result += (loss.total_variation(cs),)
        return result

    def multi_transfer(self,content, styles, alpha=0.5, num_cluster=8, loc_weight=1.0):
        print(self.amplifier.scale)
        cont_feats = self.encode_with_intermediate(content)
        styles_feats = [self.encode_with_intermediate(style) for style in styles]

        hidden_cont_feats = self.feature_pyramid(cont_feats[-3:])
        hidden_styles_feats = [self.feature_pyramid(s[-3:]) for s in styles_feats]

        cs_maps = [self.non_local(cont_feats[-2], style_feats[-2], hidden_style_feats, isTraining=False)
                                 for style_feats,hidden_style_feats in zip(styles_feats, hidden_styles_feats)]
        cc_map = self.non_local(cont_feats[-2], cont_feats[-2], hidden_cont_feats, isTraining=True)[0]

        cs, conf_maps = multi_style_warp(cont_feats[-2], cs_maps, alpha=alpha, num_cluster=num_cluster, loc_weight=loc_weight)
        cs = (1-alpha)*cc_map + alpha*cs
        print(alpha)
        cs_fused = self.fusion(hidden_cont_feats, cs)
        result = self.decoder(cs_fused)
        return result


    def getHidden(self, content, mode):
        cont_feats = self.encode_with_intermediate(content)
        hidden_cont_feats = self.feature_pyramid(cont_feats[-3:])
        
        if mode==1:
            return cont_feats[-2]
        elif mode==2:
            return cont_feats[-1]
        return hidden_cont_feats

