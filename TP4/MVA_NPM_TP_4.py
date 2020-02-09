#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:03:04 2020

@author: henric
@author: ducos

"""

from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt


# PIL.PngImagePlugin.PngImageFile

class Material:
    def __init__(self, albedo, kd, ks, S, alpha, beta, specular_color):
        # Lambert BRDF
        self.albedo = albedo
        self.kd = kd
        
        # Blinn-Phong BRDF
        self.ks = ks
        self.S = S
        
        # Cook-Torrance BRDF
        self.alpha = alpha # roughness
        self.beta = beta
        self.specular_color = specular_color

    def lambert(self):
        return self.kd * self.albedo / math.pi

    def blinn_phong(self, n, wi, wo):
        wh = (wi + wo) / np.linalg.norm(wi + wo, axis=-1, keepdims=True)
        dot_prod_h = (n[:,:,np.newaxis,:] * wh).sum(axis=-1)
        dot_prod_h = np.clip(dot_prod_h, 0, 1)
        
        return self.ks * (dot_prod_h) ** self.S

    def microfacet(self, n, wi, wo):
        eps = 1e-16
        wh = (wi + wo) / np.linalg.norm(wi + wo, axis=-1, keepdims=True)
        
        # calculations of dot products. We project negative values to 0
        dot_prod_h = (n[:,:,np.newaxis,:] * wh).sum(axis=-1)
        dot_prod_h = np.clip(dot_prod_h, 0, 1)
        dot_prod_i = (n[:,:,np.newaxis,:] * wi).sum(axis=-1)
        dot_prod_i = np.clip(dot_prod_i, 0, 1)
        dot_prod_o = (n[:,:,np.newaxis,:] * wo).sum(axis=-1)
        dot_prod_o = np.clip(dot_prod_o, 0, 1)
        dot_prod_i_h = (wo * wi).sum(axis=-1)
        dot_prod_i_h = np.clip(dot_prod_i_h, 0, 1)

        # GGX distribution
        D = self.alpha ** 2 / (math.pi * ((dot_prod_h) ** 2 * (self.alpha ** 2 - 1) + 1) ** 2)

        # Geometrical term; Schlick approximation of the Smith model
        k = (math.sqrt(self.alpha) + 1) ** 2 / 8
        G_i = (dot_prod_i) / ((dot_prod_i) * (1 - k) + k)
        G_o = (dot_prod_o) / ((dot_prod_o) * (1 - k) + k)
        G = G_i * G_o

        # Smith 
        #F = self.specular_color + (1 - self.specular_color) * pow(2, (-5.55473 * wo @ wh.T - 6.98316) * wo @ wh.T)
        
        # spheical gaussian variant of the Schlick Fresnel approximation
        F = self.specular_color + (1 - self.specular_color) * (1 - np.maximum(np.zeros((dot_prod_i_h[:,:,:,np.newaxis].shape)), dot_prod_i_h[:,:,:,np.newaxis]))**5
        
        f = (D[:,:,:,None] * F * G[:,:,:,None]) / (4 * (dot_prod_i[:,:,:,None]) * (dot_prod_o[:,:,:,None]) + eps)

        return f


class LightSource:
    def __init__(self, positions, colors, intensity):
        self.positions = positions
        self.colors = colors
        self.intensity = intensity


def coor_matrix(img, z=0):
    nrow, ncol, _ = img.shape
    mesh_X, mesh_Y = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing='ij')
    coor = np.stack([mesh_X, mesh_Y], axis=-1)
    coor = np.append(coor, z * np.ones((nrow, ncol, 1)), axis=-1)
    return coor

def shade(img, lightsources, material, xo, kind='lambert'):
    # sum_i Li f n.wi
    positions = coor_matrix(img)
    n = img
    xi = lightsources.positions
    
    wi = xi[np.newaxis, np.newaxis,:,:] - positions[:,:,np.newaxis,:]
    wi = wi / np.linalg.norm(wi, axis=-1, keepdims=True)
    wo = xo[np.newaxis, np.newaxis,:,:] - positions[:,:,np.newaxis,:]
    wo = wo / np.linalg.norm(wo, axis=-1, keepdims=True)
    
    dot_prod_i = (n[:,:,np.newaxis,:] * wi).sum(axis=-1)
    dot_prod_i = np.clip(dot_prod_i, 0, 1)
    
    if kind == 'lambert':
        render = np.clip(
                (material.lambert()[None, None, None, :] *
                 (dot_prod_i)[:, :, :,None] *
                 lightsources.colors[None, None, :, :]
                 ).sum(axis=2),
                0, 1)
    elif kind == 'phong':
        render = np.clip((
                (material.lambert()[None, None, None, :] +
                 material.blinn_phong(n=n, wi=wi, wo=wo)[:, :, :,None]
                 ) * (dot_prod_i)[:, :, :,None] * lightsources.colors[None, None, :,:]
                ).sum(axis=2)
            , 0, 1)
    elif kind == 'microfacet':
        render = np.clip((
                (material.lambert()[None, None, None, :] +
                 material.microfacet(n=n, wi=wi, wo=wo)
                 ) * (dot_prod_i)[:, :,:, None] * lightsources.colors[None,None, :, :]
                ).sum(axis=2),
                0, 1)
    else:
        raise ValueError()
    return render

def import_and_clean(filename="normal.png"):
    normalImage = Image.open(filename)

    img = np.array(normalImage)[:, :, :-1].astype(np.float)
    
    
    norms = np.linalg.norm(img, axis=-1, keepdims=True)
    mask = (norms <= 10).squeeze()
    img = (img - 127)/127
    mask = mask & (img[:,:,-1]<0)
    
    img[:,:,:2] = (img[:,:,:2] - 127)/127
    img[:,:,-1] = img[:,:,-1] / 255
    
    img[mask] = 1
    img = img / np.linalg.norm(img, axis=-1, keepdims=True)
    img[mask] = 0
    plt.imshow(img)
    print(normalImage)
    return img

def save(render, filename):
    render_img = Image.fromarray((render * 255).astype(np.uint8))
    render_img.save(filename)
    
def hist(img):
    x1,x2,_ = img.shape
    img = img.reshape((x1*x2,3))
    plt.hist(img)
    plt.show()

if __name__ == '__main__':
    material = Material(np.array([0.75, 0.9, 0.6]), kd=1, ks=1, S=1, alpha=1, beta=1,
                        specular_color=np.array([1, 0, 0]))
    
    xo = np.array([[100, 350, 1000]]) # location of camera along Z axis
    xo = np.array([[0,1,1]])

    if False:
        material = Material(np.array([0.75, 0.9, 0.6]), kd=math.pi, ks=1, S=1, alpha=0.5, beta=1,
                        specular_color=np.array([0.05,0.54,0.05]))
        
        lightsources = LightSource(np.array([[0,1,1000]]), np.array([[1, 1, 1]]), np.ones(1))

        img = import_and_clean("normal.png")
        
        render = shade(img, lightsources, material, xo)
        plt.imshow(render)

        #save(render, "render_lambert_1.png")

    if False:
        material = Material(np.array([0.75, 0.9, 0.6]), kd=math.pi, ks=1, S=1, alpha=0.5, beta=1,
                        specular_color=np.array([0.05,0.54,0.05]))
        
        lightsources = LightSource(np.array([[1000, -100, 10], [200, -100, 1000]]),
                                   np.array([[1, 0, 0], [0, 1, 1]]),
                                   np.ones(2))
        
        lightsources = LightSource(np.array([[1000, -100, 10]]), np.array([[1,0,0]]), np.ones(1))
        lightsources = LightSource(np.array([[800, 1000, -10]]), np.array([[0,0,1]]), np.ones(1))
        lightsources = LightSource(np.array([[0,1000,-1]]), np.array([[0,0,1]]), np.ones(1))

        img = import_and_clean("normal.png")
        
        render = shade(img, lightsources, material, xo)
        plt.imshow(render)

        # save(render, "render_lambert_3.png")
        

    if False:
        material = Material(np.array([0.75, 0.9, 0.6]), kd=math.pi, ks=1, S=1, alpha=0.5, beta=1,
                        specular_color=np.array([0.05,0.54,0.05]))
        lightsources = LightSource(np.array([[0,1,1000]]), np.array([[1,1,1]]), np.ones(1))

        img = import_and_clean("normal.png")
        
        render = shade(img, lightsources, material, xo, kind='phong')
        plt.imshow(render)
        
        #save(render, "render_phong_1.png")


    if False:
        material = Material(np.array([0.75, 0.9, 0.6]), kd=math.pi, ks=1, S=1, alpha=0.5, beta=1,
                        specular_color=np.array([1,1,1]))
        
        lightsources = LightSource(np.array([[0,1,1000]]), np.array([[1,1,1]]), np.ones(1))
        
        img = import_and_clean(filename="normal.png")

        render = shade(img, lightsources, material, xo, kind='microfacet')
        plt.imshow(render)
        
        #save(render, "render_microfacet_3.png")
        
    if False:
        material = Material(np.array([0.75, 0.9, 0.6]), kd=1, ks=1, S=1, alpha=1, beta=1,
                        specular_color=np.array([0.05,0.05,0.05]))
        
        lightsources = LightSource(np.array([[0, 3, -1], [3, -1, 0], [-1, 0, 3]]),
                                   np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                                   np.ones(3))
        
        img = import_and_clean(filename="normal.png")
        
        alphas = [1,5,10,50,100]
        for alpha in alphas:
            material = Material(np.array([0.75, 0.9, 0.6]), kd=1, ks=1, S=1, alpha=alpha, beta=1,
                        specular_color=np.array([0.05,0.05,0.05]))
            render = shade(img, lightsources, material, xo, kind='microfacet')
            plt.clf()
            plt.imshow(render)
            
            save(render, "render_microfacet_3_{}.png".format(alpha))

