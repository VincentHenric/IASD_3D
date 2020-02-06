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
        self.albedo = albedo
        self.kd = kd
        self.ks = ks
        self.S = S
        self.alpha = alpha
        self.beta = beta
        self.specular_color = specular_color

    def lambert(self):
        return self.kd * self.albedo / math.pi

    def blinn_phong(self, n, wi, wo):
        wh = (wi + wo) / np.linalg.norm(wi + wo, axis=1)
        return self.ks * (n @ wh.T) ** self.S

    def microfacet(self, n, wi, wo):
        wh = (wi + wo) / np.linalg.norm(wi + wo, axis=1)

        D = self.alpha ** 2 / (math.pi * ((n @ wh.T) ** 2 * (self.alpha ** 2 - 1) + 1) ** 2)

        k = (math.sqrt(self.alpha) + 1) ** 2 / 8
        G_i = (n @ wi.T) / ((n @ wi.T) * (1 - k) + k)
        G_o = (n @ wo.T) / ((n @ wo.T) * (1 - k) + k)
        G = G_i * G_o

        F = self.specular_color + (1 - self.specular_color) * pow(2, (-5.55473 - 6.98316) * wo @ wh.T)

        return (D * F * G) / (4 * (n @ wi.T) * (n @ wo.T))


class LightSource:
    def __init__(self, positions, colors, intensity):
        self.positions = positions
        self.colors = colors
        self.intensity = intensity


def shade(img, lightsources, material, kind='lambert'):
    # sum_i Li f n.wi
    n = img
    wi = lightsources.positions
    wo = np.array([[1, 1, 1]])
    if kind == 'lambert':
        render = np.clip((material.lambert()[None, None, None, :] * (img @ lightsources.positions.T)[:, :, :,
                                                                    None] * lightsources.colors[None, None, :, :]).sum(
            axis=2), 0, 1)
    elif kind == 'phong':
        render = np.clip(((material.lambert()[None, None, None, :] + material.blinn_phong(n=n, wi=wi, wo=wo)[:, :, :,
                                                                     None]) * (img @ lightsources.positions.T)[:, :, :,
                                                                              None] * lightsources.colors[None, None, :,
                                                                                      :]).sum(axis=2), 0, 1)
    elif kind == 'microfacet':
        render = np.clip((material.microfacet(n=n, wi=wi, wo=wo)[:, :, None, :] * (img @ lightsources.positions.T)[:, :,
                                                                                  :, None] * lightsources.colors[None,
                                                                                             None, :, :]).sum(axis=2),
                         0, 1)
    else:
        raise ValueError()
    return render


if __name__ == '__main__':
    material = Material(np.array([0.75, 0.9, 0.6]), kd=1, ks=1, S=1, alpha=1, beta=1,
                        specular_color=np.array([1, 0, 0]))

    if True:
        lightsources = LightSource(np.array([[0, 3, -1]]), np.array([[1, 1, 1]]), np.ones(1))

        normalImage = Image.open("normal.png")
        img = np.array(normalImage)[:, :, :-1] / 255
        print(normalImage)
        render = shade(img, lightsources, material)
        plt.imshow(render)

        render_img = Image.fromarray((render * 255).astype(np.uint8))
        render_img.save("render_lambert_1.png")

    if True:
        lightsources = LightSource(np.array([[0, 3, -1], [3, -1, 0], [-1, 0, 3]]),
                                   np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                                   np.ones(3))

        normalImage = Image.open("normal.png")
        img = np.array(normalImage)[:, :, :-1] / 255
        print(normalImage)
        render = shade(img, lightsources, material)
        plt.imshow(render)

        render_img = Image.fromarray((render * 255).astype(np.uint8))
        render_img.save("render_lambert_3.png")

    if True:
        lightsources = LightSource(np.array([[0, 3, -1]]), np.array([[1, 1, 1]]), np.ones(1))

        normalImage = Image.open("normal.png")
        img = np.array(normalImage)[:, :, :-1] / 255
        print(normalImage)
        render = shade(img, lightsources, material, kind='phong')
        plt.imshow(render)

        render_img = Image.fromarray((render * 255).astype(np.uint8))
        render_img.save("render_phong_1.png")

    if True:
        lightsources = LightSource(np.array([[0, 3, -1], [3, -1, 0], [-1, 0, 3]]),
                                   np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                                   np.ones(3))
        normalImage = Image.open("normal.png")
        img = np.array(normalImage)[:, :, :-1] / 255

        render = shade(img, lightsources, material, kind='microfacet')
        plt.imshow(render)

        render_img = Image.fromarray((render * 255).astype(np.uint8))
        render_img.save("render_microfacet_3.png")
