#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : dealNum.py
# @time    : 7/8/20 2:15 PM
# @desc    : 
'''
import math
def is_number(s):
    """
    :param s:
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def integer(x):
    """
    :param x:
    :return:
    """
    assert is_number(x), "{} is not a number !!!".format(x)
    if type(x) == int:
        return x
    else:
        intPart = math.floor(x)
        if x - intPart >= 0.5:
            return math.ceil(x) # > 0.5 向上取整
        else:
            return math.floor(x) # < 0.5 向下取整