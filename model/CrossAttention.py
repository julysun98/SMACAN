import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torchvision import models

# class CrossAttention(nn.Module):
#     def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.k_dim = k_dim
#         self.v_dim = v_dim
        
#         self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
#         self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
#         self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
#         self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        
#     def forward(self, x1, x2, mask=None):
#         batch_size, seq_len1, in_dim1 = x1.size()
#         seq_len2 = x2.size(1)
        
#         q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
#         k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
#         v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        
#         attn = torch.matmul(q1, k2) / self.k_dim**0.5
        
#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)
        
#         attn = F.softmax(attn, dim=-1)
#         output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
#         output = self.proj_o(output)
        
#         return output

class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, att_dropout=0.0, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, Flow, RGB, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, c, h, w = Flow.shape
        bb, cc, hh, ww = RGB.shape

        Flow = self.proj_in(Flow)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        Flow = rearrange(Flow, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        RGB = self.proj_in(RGB)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        RGB = rearrange(RGB, 'bb cc hh ww -> bb (hh ww) cc')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(Flow)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(RGB)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(RGB)

        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        out = self.proj_out(out)   # [batch_size, c, h, w]

        # print(out.shape)

        return out #, att_weights


class SMACAN(nn.Module):
    def __init__(self):
        super(MotionGuide, self).__init__()
        self.inch_flow = 4*2
        self.inch_RGB = 4*3
        ################################resnet101 Flow#######################################
        # feats_Flow = models.resnet101(pretrained=True)
        self.conv0_Flow =  nn.Sequential(
            nn.Conv2d(in_channels=self.inch_flow, out_channels=64, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv1_Flow = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=2,padding=3,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2_Flow = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=2,padding=3,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3_Flow = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=2,padding=3,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # self.crosscov_Flow = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=4,bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv4_Flow = feats_Flow.layer4
        # self.cross_attention = CrossAttention(512, 512, att_dropout=0.0, aropout=0.0)
         # Cross-Attention layers
        self.cross_attention0 = CrossAttention(64, 64, att_dropout=0.0, dropout=0.0)
        self.cross_attention1 = CrossAttention(128, 128, att_dropout=0.0, dropout=0.0)
        self.cross_attention2 = CrossAttention(256, 256, att_dropout=0.0, dropout=0.0)
        self.cross_attention3 = CrossAttention(512, 512, att_dropout=0.0, dropout=0.0)

        
        ################################resnet101 RGB#######################################
        # feats_RGB = models.resnet101(pretrained=True)
        self.conv0_RGB = nn.Sequential(
            nn.Conv2d(in_channels=self.inch_RGB, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv1_RGB = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding=3,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2_RGB = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding=3,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3_RGB = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, padding=3,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # self.crosscov_RGB = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=4,bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )
        # self.sig = nn.Sigmoid()
        #================decoder RGB ========================
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv0 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, bias=False)

 #================decoder Flow ========================

        self.deconv3_ = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0)
        )
        self.conv3_ = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv2_ = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0)
        )
        self.conv2_ = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv1_ = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0)
        )
        self.conv1_ = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv00_ = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, bias=False)


        self.conv1x1_0 = nn.Conv2d(64, 1, 1, bias=False)
        self.conv1x1_1 = nn.Conv2d(128, 1, 1, bias=False)
        self.conv1x1_2 = nn.Conv2d(256, 1, 1, bias=False)
        self.conv1x1_3 = nn.Conv2d(512, 1, 1, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, input, flow):
        c0_Flow = self.conv0_Flow(flow)  # N,64,32,32(64,16,16)
        # c0_Flow_ = self.conv1x1_0(c0_Flow)
        c1_Flow = self.conv1_Flow(c0_Flow)  # N,128,16,16
        # c1_Flow_ = self.conv1x1_1(c1_Flow)
        c2_Flow = self.conv2_Flow(c1_Flow)  # N,256,8,8
        # c2_Flow_ = self.conv1x1_2(c2_Flow)
        c3_Flow = self.conv3_Flow(c2_Flow)  # N,512,4,4
        # c4_Flow = c3_Flow.view(c3_Flow.size(0), c3_Flow.size(1), -1)
        # c3_Flow_ = self.conv1x1_3(c3_Flow) #N,1,4,4 
        # c4_Flow = self.conv4_Flow(c3_Flow)  # N,2048,12,12

        c0_RGB = self.conv0_RGB(input)  # N,64,32,32
        c0_RGB_ = self.cross_attention0(c0_Flow,c0_RGB)+c0_RGB

        c1_RGB = self.conv1_RGB(c0_RGB) # N,128,16,16
        c1_RGB_ = self.cross_attention1(c1_Flow,c1_RGB)+c1_RGB

        c2_RGB = self.conv2_RGB(c1_RGB) # N,256,8,8
        c2_RGB_ = self.cross_attention2(c2_Flow,c2_RGB)+c2_RGB

        c3_RGB = self.conv3_RGB(c2_RGB) # N,512,4,4
        # c4_RGB = c3_RGB.view(c3_RGB.size(0), c3_RGB.size(1), -1)
        c3_RGB_ = self.cross_attention3(c3_Flow,c3_RGB)+c3_RGB

        
        # d3 = nn.Sigmoid()(c3_RGB)*c3_Flow + c3_RGB
        # d3 = nn.Sigmoid()(c3_Flow)*c3_RGB + c3_RGB
        
        d3 = c3_RGB_ #N,512,4,4
        # print("d3.shape",d3.shape)
        dd3 = c3_Flow #N,512,4,4
        # print("dd3.shape",dd3.shape)
        d3_feature =  self.global_avg_pool(d3).view(d3.size(0), -1)
        # print(self.global_avg_pool(d3).shape)
        dd3_feature =  self.global_avg_pool(dd3).view(dd3.size(0), -1)
        
        d2 = self.deconv3(d3)
        d2 = torch.cat((d2, c2_RGB), dim=1)
        d2 = self.conv3(d2)

        d1 = self.deconv2(d2)
        d1 = torch.cat((d1,c1_RGB),dim=1)
        d1 = self.conv2(d1)

        d0 = self.deconv1(d1)
        d0 = torch.cat((d0,c0_RGB),dim=1)  #v1
        d0 = self.conv1(d0)

        dd2 = self.deconv3_(dd3)
        dd2 = torch.cat((dd2, c2_Flow),dim=1)
        dd2 = self.conv3_(dd2)

        dd1 = self.deconv2_(dd2)
        dd1 = torch.cat((dd1,c1_Flow),dim=1)
        dd1 = self.conv2_(dd1)

        dd0 = self.deconv1_(dd1)
        dd0 = torch.cat((dd0,c0_Flow),dim=1)  #v1
        dd0 = self.conv1_(dd0)

        out = self.conv0(d0)
        out_Flow = self.conv00_(dd0)


        return out,out_Flow,d3_feature,dd3_feature


if __name__ == "__main__":
    model = SMACAN()
    input = torch.autograd.Variable(torch.randn(256, 12, 32, 32))

    flow = torch.autograd.Variable(torch.randn(256, 8, 32, 32))
    output,output_Flow,_,__ = model(input, flow)
    print(output.shape,output_Flow.shape,_.shape,__.shape)
