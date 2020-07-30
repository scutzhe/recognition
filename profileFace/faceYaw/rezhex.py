#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : rezhex.py
# @time    : 7/27/20 3:58 PM
# @desc    : 
'''
import re

content = 'Hello 123456789 Word_This is just a test 666 Test'
result = re.search("\d", content).start()

print(result)