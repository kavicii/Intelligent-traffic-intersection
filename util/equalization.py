# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:13:01 2020

@author: kavic
"""

def list_equalization(_list,x,_len):
    if len(_list) <= _len:
        _list.append(x)
    else:
        _list.pop(0)
        _list.append(x)
    list_mean = sum(_list)/len(_list)
    return(list_mean)
