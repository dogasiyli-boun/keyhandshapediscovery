#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:14:48 2019

@author: batu
"""

from dtw import dtw
import numpy as np

a = np.asarray([1, 1, 1, 1, 1, 2, 3],dtype='float64')
b = np.asarray([1, 2, 2,2,3],dtype='float64')

d,e,f,g = dtw(a,b)