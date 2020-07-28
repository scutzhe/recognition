#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : topZhex.py
# @time    : 7/28/20 1:54 PM
# @desc    : 
'''

import torch
def topkzhex(tensor):
    """
    @param tensor:
    @return:
    """
    # k 指明是得到前k个数据以及其index
    # dim： 指定在哪个维度上排序， 默认是最后一个维度
    # largest：如果为True，按照大到小排序； 如果为False，按照小到大排序
    # sorted：返回的结果按照顺序返回
    values, indices = tensor.topk(k=2,dim=1,largest=True,sorted=True)
    indexValue, indexMax = tensor.max(dim=1,keepdim=True)
    return values, indices, indexValue, indexMax

if __name__ == "__main__":
    pred = torch.randn((4, 5))
    print("pred=",pred)
    values, indices, indexValue, indexMax = topkzhex(pred)
    print("values=",values)
    print("indices=",indices)
    print("indexValue=",indexValue)
    print("indexMax=",indexMax)
