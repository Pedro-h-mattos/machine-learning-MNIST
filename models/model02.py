#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 15:15:07 2025

@author: pedro

Pytorch model code to build a convolutional neural network
"""
import torch
from torch import nn, optim

class CNN(nn.Module):
    def init__(self):
        super(CNN, self).__init__()
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.layer_2 = nn.sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )

        self.out = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output