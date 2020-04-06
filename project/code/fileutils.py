#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:06:16 2020

@author: henric
"""

def dict_to_str(params, key_sep='-', value_sep=':'):
    if len(params)==0:
        return ''
    return key_sep.join(sorted(['{}{}{}'.format(k,value_sep, v) for k,v in params.items()]))

def str_to_dict(params_str, key_sep='-', value_sep=':'):
    if not params_str:
        return {}
    splitted = params_str.split('-')
    splitted = [dict_formatting(s, value_sep) for s in splitted]
    return eval('{{{}}}'.format(','.join(splitted)))

def dict_formatting(s, value_sep=':'):
    k,v = s.split(value_sep)
    try:
        eval(v)
    except:
        v = '"{}"'.format(v)
    return value_sep.join(['"{}"'.format(k), v])

def give_filename(filename_rad, prefix='', params={}, extension='ply', sep='_'):
    filename = sep.join(filter(None, [prefix, filename_rad, dict_to_str(params)]))
    return '{}.{}'.format(filename, extension)

def parse_filename(filename, sep='_', key_sep='-', value_sep=':'):
    filename = '.'.join(filename.split('.')[:-1])
    parts = filename.split(sep)
    
    prefix, filename_rad, params_str = '', '', ''
    
    if len(parts)==1:
        filename_rad = parts[0]
    elif len(parts)==3:
        prefix, filename_rad, params_str = parts
    elif len(parts)==2:
        if value_sep in parts[-1]:
            params_str = parts[-1]
            filename_rad = parts[0]
        else:
            prefix, filename_rad = parts
    else:
        raise ValueError('wrong format')
        
    params = str_to_dict(params_str, key_sep, value_sep)
    return prefix, filename_rad, params
        
    
    