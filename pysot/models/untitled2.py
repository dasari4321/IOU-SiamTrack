#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:50:23 2019

@author: ee18d001
"""

import torch

class Mybridge(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx,locs,idx):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        locs2=locs.permute(1,2,3,0).contiguous().view(4,-1)
        locs2[:,idx].fill_(1)
        ctx.save_for_backward(locs2)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        locs2, = ctx.saved_tensors
        grad_input = grad_output.sum()*locs2
        grad_input.reshape(20,25,25,1).permute(3,0,1,2)
        return grad_input
