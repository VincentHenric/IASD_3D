#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:03:04 2020

@author: henric
"""

import PIL
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt

# PIL.PngImagePlugin.PngImageFile

class Material:
    def __init__(self, albedo, kd, ks, S):
        self.albedo = albedo
        self.kd = kd
        self.ks = ks
        self.S = S
        
    def lambert(self):
        return self.kd * self.albedo/math.pi
    
    def blinn_phong(self, n, wi, wo):
        wh = (wi + wo) / np.linalg.norm(wi+wo, axis=1)
        return self.ks * (n @ wh.T)**self.S

class LightSource:
    def __init__(self, positions, colors, intensity):
        self.positions = positions
        self.colors = colors
        self.intensity = intensity
        

def shade(img, lightsources, material, kind='lambert'):
    # sum_i Li f n.wi
    if kind == 'lambert':
        render = np.clip((material.lambert()[None,None,None,:] * (img @ lightsources.positions.T)[:,:,:,None] * lightsources.colors[None,None,:,:]).sum(axis=2),0,1)
    elif kind == 'phong':
        render = np.clip(((material.lambert()[None,None,None,:] + material.blinn_phong(img, lightsources.positions, np.array([0,0,0]))[:,:,:,None]) * (img @ lightsources.positions.T)[:,:,:,None] * lightsources.colors[None,None,:,:]).sum(axis=2),0,1)
    else:
        raise ValueError()
    return render
 

if __name__ == '__main__':
    material = Material(np.array([0.75, 0.9, 0.6]), kd=1, ks=1, S=1)

    if False:
        lightsources = LightSource(np.array([[0,3,-1]]), np.array([[1,1,1]]), np.ones(1))
        
        normalImage = Image.open("normal.png")
        img = np.array(normalImage)[:,:,:-1]/255
        print(normalImage)
        render = shade(img, lightsources, material)
        plt.imshow(render)
        
        render_img = Image.fromarray((render * 255).astype(np.uint8))
        render_img.save("render_lambert_1.png")
        
    if False:
        lightsources = LightSource(np.array([[0,3,-1],[3,-1,0],[-1,0,3]]),
                                   np.array([[1,1,1],[1,1,1], [1,1,1]]),
                                   np.ones(3))
        
        normalImage = Image.open("normal.png")
        img = np.array(normalImage)[:,:,:-1]/255
        print(normalImage)
        render = shade(img, lightsources, material)
        plt.imshow(render)
        
        render_img = Image.fromarray((render * 255).astype(np.uint8))
        render_img.save("render_lambert_3.png")
        
    if False:
        lightsources = LightSource(np.array([[0,3,-1]]), np.array([[1,1,1]]), np.ones(1))
        
        normalImage = Image.open("normal.png")
        img = np.array(normalImage)[:,:,:-1]/255
        print(normalImage)
        render = shade(img, lightsources, material, kind='phong')
        plt.imshow(render)
        
        render_img = Image.fromarray((render * 255).astype(np.uint8))
        render_img.save("render_phong_1.png")
        