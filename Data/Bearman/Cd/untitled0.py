#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 10:25:56 2020

@author: bradleydavy
"""

import glob

files = glob.glob('*.txt')
print(files[11])
f = open(files[11],'r')
t = f.read().split('\n')
t.reverse()
j = ''
for lines in t:
    j = j +','+ lines.split(',')[1]
print(j)