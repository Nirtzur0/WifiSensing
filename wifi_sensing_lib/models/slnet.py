# The implementation of {SLNet}: A Spectrogram Learning Neural Network for Deep Wireless Sensing
# is based on : https://github.com/aiot-lab/RFBoost/blob/main/source/model/slnet.py
# and the official implementation: https://github.com/SLNetRelease/SLNetCode/blob/main/Case_Gesture/SLNet_ges.py


import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

import os, sys, math
import numpy as np
import scipy.io as scio
import torch, torchvision
import torch.nn as nn
from torch import sigmoid
from torch.fft import fft, ifft
from torch.nn.functional import relu
from torch import sigmoid

class m_Linear(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        # Creation
        self.weights_real = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.weights_imag = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(2, size_out, dtype=torch.float32))

        # Initialization
        nn.init.xavier_uniform_(self.weights_real, gain=1)
        nn.init.xavier_uniform_(self.weights_imag, gain=1)
        nn.init.zeros_(self.bias)
    
    def swap_real_imag(self, x):
        # [@,*,2,Hout]
        # [real, imag] => [-1*imag, real]
        h = x                   # [@,*,2,Hout]
        h = h.flip(dims=[-2])   # [@,*,2,Hout]  [real, imag]=>[imag, real]
        h = h.transpose(-2,-1)  # [@,*,Hout,2]
        # Removed .cuda(), moved to device
        h = h * torch.tensor([-1,1]).to(h.device)     # [@,*,Hout,2] [imag, real]=>[-1*imag, real]
        h = h.transpose(-2,-1)  # [@,*,2,Hout]
        
        return h

    def forward(self, x):
        # x shape: [..., 2, size_in]
        # x_real: [..., 0, size_in]
        # x_imag: [..., 1, size_in]
        
        xr = x[..., 0, :]
        xi = x[..., 1, :]
        
        yr = torch.matmul(xr, self.weights_real) - torch.matmul(xi, self.weights_imag) + self.bias[0]
        yi = torch.matmul(xr, self.weights_imag) + torch.matmul(xi, self.weights_real) + self.bias[1]
        
        return torch.stack([yr, yi], dim=-2)

class m_pconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_front_pconv_layer=False):
        super().__init__()
        self.is_front_pconv_layer = is_front_pconv_layer
        self.in_channels = in_channels # Store for potential use in forward
        
        # Simple conv3d implementation to match usage
        # kernel_size and stride are lists [D, F, T]
        # Multiplied out_channels by 2 to support the split into Real/Imag (or 2 channels) in forward reshape
        self.conv3d = nn.Conv3d(in_channels * 2, out_channels * 2, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding=(0,0,0)) # padding?
        
    def _polarize(self, h, F):
        # Placeholder for polarization logic if needed, or simplifed
        # If input is [B, 2, C, D, F, T]
        # For now just return as is or handle dimensions
        return h

    def forward(self, x):
        h = x # Assuming [B, 2, C, D, F, T] 
        
        # If x has 6 dims, we need to handle it.
        if len(x.shape) == 6:
            # [B, 2, C, D, F, T] -> [B, 2*C, D, F, T]
            b, cplx, c, d, f, t = x.shape
            h = x.reshape(b, cplx*c, d, f, t)
            
        # Polarize for the front pconv layer (if needed, based on original logic)
        
        h = self.conv3d(h) # [B, 2*C_out, ...]
        
        # Reshape back to [B, 2, C_out, ...]
        # Output shape: [B, 2*C_out, D_out, F_out, T_out]
        # We want [B, 2, C_out, D_out, F_out, T_out]
        b, c2, d, f, t = h.shape
        h = h.reshape(b, 2, c2//2, d, f, t)
        
        return h

class SLNet(nn.Module):
    # Need customization
    def __init__(self, input_shape, class_num):
        super(SLNet, self).__init__()
        self.input_shape = input_shape  # [2, sensor_channel, sensor_num, freq_bins, time_steps]
        self.IN_CHANNEL = input_shape[1]
        self.NUM_RX = input_shape[2]
        self.T_MAX = input_shape[4]
        self.class_num = class_num

        # pconv+FC
        self.complex_fc_1 = m_Linear(32*7, 128)
        self.complex_fc_2 = m_Linear(128, 64)
        self.fc_1 = nn.Linear(32*7, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_4 = nn.Linear(self.NUM_RX*(self.T_MAX-8)*32, 256)
        self.fc_5 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, self.class_num)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.dropout_4 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.2)
        self.pconv3d_1 = m_pconv3d(in_channels=self.IN_CHANNEL,out_channels=16,kernel_size=[1,5,5],stride=[1,1,1],is_front_pconv_layer=True)
        self.pconv3d_2 = m_pconv3d(in_channels=16,out_channels=32,kernel_size=[1,5,5],stride=[1,1,1],is_front_pconv_layer=False)
        self.mpooling3d_1 = nn.MaxPool3d(kernel_size=[1,3,1],stride=[1,3,1])
        self.mpooling3d_2 = nn.MaxPool3d(kernel_size=[1,5,1],stride=[1,5,1])

    def forward(self, x):
        h = x   # [@,2,R,C,F,T] ~ (@,2,1,6,121,T_MAX) or (@,2,3,6,121,T_MAX)

        # pconv
        h = self.pconv3d_1(h)                       # (@,2,R,6,121,T_MAX)=>(@,2,16,6,117,T_MAX-4)
        h = h.reshape((-1,16,self.NUM_RX,117,self.T_MAX-4))   # (@,2,16,6,117,T_MAX-4)=>(@*2,16,6,117,T_MAX-4)
        h = self.mpooling3d_1(h)                    # (@*2,16,6,117,T_MAX-4)=>(@*2,16,6,39,T_MAX-4)
        h = h.reshape((-1,2,16,self.NUM_RX,39,self.T_MAX-4))  # (@*2,16,6,39,T_MAX-4)=>(@,2,16,6,39,T_MAX-4)

        h = self.pconv3d_2(h)                       # (@,2,16,6,39,T_MAX-4)=>(@,2,32,6,35,T_MAX-8)
        h = h.reshape((-1,32,self.NUM_RX,35,self.T_MAX-8))    # (@,2,32,6,35,T_MAX-8)=>(@*2,32,6,35,T_MAX-8)
        h = self.mpooling3d_2(h)                    # (@*2,32,6,35,T_MAX-8)=>(@*2,32,6,7,T_MAX-8)
        h = h.reshape((-1,2,32,self.NUM_RX,7,self.T_MAX-8))   # (@*2,32,6,7,T_MAX-8)=>(@,2,32,6,7,T_MAX-8)

        # Complex FC
        h = h.permute(0,3,5,1,2,4)                  # (@,2,32,6,7,T_MAX-8)=>(@,6,T_MAX-8,2,32,7)
        h = h.reshape((-1,self.NUM_RX,self.T_MAX-8,2,32*7))   # (@,6,T_MAX-8,2,32,7)=>(@,6,T_MAX-8,2,32*7)
        h = self.dropout_1(h)
        h = self.complex_fc_1(h)                    # (@,6,T_MAX-8,2,32*7)=>(@,6,T_MAX-8,2,128)
        h = self.dropout_2(h)
        h = self.complex_fc_2(h)                    # (@,6,T_MAX-8,2,128)=>(@,6,T_MAX-8,2,64)

        # FC
        h = torch.linalg.norm(h,dim=3)              # (@,6,T_MAX-8,2,64)=>(@,6,T_MAX-8,64)
        h = relu(self.fc_3(h))                      # (@,6,T_MAX-8,64)=>(@,6,T_MAX-8,32)
        h = h.reshape((-1,self.NUM_RX*(self.T_MAX-8)*32))     # (@,6,T_MAX-8,32)=>(@,6*(T_MAX-8)*32)
        h = self.dropout_3(h)
        h = relu(self.fc_4(h))                      # (@,6*(T_MAX-8)*32)=>(@,256)
        h = self.dropout_4(h)
        h = relu(self.fc_5(h))                      # (@,256)=>(@,128)
        h = self.dropout_5(h)
        output = self.fc_out(h)           # (@,128)=>(@,n_class)  (No need for activation when using CrossEntropyLoss)

        return output
 
 
def slnet_standard(sensor_num, sensor_in_channels, freq_bins, time_steps, num_classes):
    return SLNet([2, sensor_in_channels, sensor_num, freq_bins, time_steps], num_classes)
    