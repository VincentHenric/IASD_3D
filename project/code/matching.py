#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:28:05 2020

@author: henric
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from scipy.stats import entropy
import scipy
from utils.ply import write_ply, read_ply
from fileutils import dict_to_str, give_filename, parse_filename
import time
import os
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix, tril

def generalized_eigenvalues(C1, C2):
    return scipy.linalg.eigh(C1, b=C2, lower=False, eigvals_only=True)
    #return np.linalg.svd(cov1[0], hermitian=True, compute_uv=False)[1]

def eigen_dist(C1, C2):
    """
    calculates the log-eigenvalue measure
    """
    eigenvalues = np.abs(generalized_eigenvalues(C1, C2))
    return np.linalg.norm(np.log(eigenvalues))
#eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
def flatten_eigen_dist(dim):
    def func(C1, C2):
        return eigen_dist(C1.reshape(dim,dim), C2.reshape(dim,dim))
    return func
    
def get_distance_matrix(C1, C2):
    """
    calculates the distance matrix between keypoints
    """
    dim = C1.shape[-1]
    distances = scipy.spatial.distance.cdist(C1.reshape(-1,dim**2), C2.reshape(-1,dim**2), metric=flatten_eigen_dist(dim))
    return distances

def is_valid_match(d1, d2, ratio):
    return (d1 <= ratio * d2.min(axis=0)) & (d1 <= ratio * d2.min(axis=0))

def get_correspondences(distances, ratio):
    """
    compute the candidates that pass the inclusive ratio strategy
    """
    dist = distances.copy()
    mask = (dist<=ratio*dist.min(axis=1, keepdims=True)) & (dist<=ratio*dist.min(axis=0, keepdims=True))
    dist[~mask] = 0
    dist = coo_matrix(dist)   
    matches = np.concatenate((dist.row[:,None], dist.col[:,None], dist.data[:,None]), axis=1)
    return matches
        
        
#def delta(gamma=1):
#    def func(dist1, dist2):
#        return min(dist1, dist2)/max(dist1, dist2)*np.exp(-np.abs(dist1-dist2)/gamma)
#    return func

def delta(gamma=0.01):
    """
    computes delta function for one match x one match
    """
    def func(match1, match2):
        if match1[0]==match2[0] or match1[1]==match2[1]:
            return 0
        d1, d2 = match1[-1], match2[-1]
        d = min(d1, d2)/max(d1, d2)*np.exp(-np.abs(d1-d2)/gamma)
        if d<0.1:
            return 0
        return d
    return func

def delta_arr(gamma=0.01):
    """
    computes delta function for one match x one match array (half vectorized, so faster than point to point)
    """
    def func(match1, match_array):
        distances = np.zeros((len(match_array)))
        indices = np.where((match_array[:,0]==match1[0]) | (match_array[:,1]==match1[1]))
        distances = np.minimum(match_array[:,-1], match1[-1])/np.maximum(match_array[:,-1], match1[-1])*np.exp(-np.abs(match_array[:,-1]-match1[-1])/gamma)
        distances[distances<0.1] = 0
        distances[indices] = 0
        return distances
    return func

#def delta_arr_arr(gamma=0.01):
#    def func(match_array_1, match_array_2):
#        distances = np.zeros((len(match_array)))
#        indices = np.where((match_array[:,0]==match1[0]) | (match_array[:,1]==match1[1]))
#        distances = np.minimum(match_array[:,-1], match1[-1])/np.maximum(match_array[:,-1], match1[-1])*np.exp(-np.abs(match_array[:,-1]-match1[-1])/gamma)
#        distances[distances<0.1] = 0
#        distances[indices] = 0
#        return distances
#    return func

def get_payoff_matrix3(matches, gamma=0.01):
    """
    implementation of the payoff matrix computation
    """
    func = delta(gamma)
    row = []
    col = []
    data = []
    for i, ri in enumerate(matches[:]):
        for j, rj in enumerate(matches[:i]):
            d = func(ri, rj)
            if d>0.1:
                row.append(i)
                col.append(j)
                data.append(d)
                
    return coo_matrix((data, (row,col)), shape=(len(matches), len(matches)))

def get_payoff_matrix(matches, gamma=0.01):
    """
    optimized implementation of the payoff matrix computation
    """
    func = delta_arr(gamma)
    row = []
    col = []
    data = []
    for i, ri in enumerate(matches[:]):
        payoff = func(ri, matches[:i])
        indices = np.where(payoff!=0)[0]
        row += [i]*len(indices)
        col += list(indices)
        data += list(payoff[indices])
                
    mat = coo_matrix((data, (row,col)), shape=(len(matches), len(matches)))
    return mat+mat.T-scipy.sparse.diags(mat.diagonal(),dtype=int)

def get_payoff_matrix2(matches, gamma=0.01):
    """
    implementation of the payoff matrix computation
    """
    distances = scipy.spatial.distance.cdist(matches, matches, metric = delta(gamma))
    distances[distances<0.1] = 0
    return coo_matrix(distances)


def iterate(payoff, kind='conv', conv=0, nb_iter=0):
    """
    compute iterative steps of the non cooperative game
    """
    n = payoff.shape[0]
    x = 1/n * np.ones(n)
    finished = False
    i = 0
    while not finished:
        x_new = x * (payoff @ x)/(x.T @ payoff @ x)
        if np.linalg.norm(x_new-x)/np.linalg.norm(x) < conv and kind=='conv':
            finished=True
        elif i>nb_iter and kind=='iter':
            finished=True
        x = x_new
        i += 1
    return x


if __name__ == '__main__':
    data_path = '../data'
    saved_data_path = '../saved_data'

    # Find a good radius for neighbors
    # ********************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        filename1 = 'keypoints_bunny-original_factor:0.5-geom:0-q:0.95-radius:0.026-subs:decimation-t0:0.021.ply'
        prefix1, name1, params1 = parse_filename(filename1)
        cov_filename1 = give_filename(name1, prefix='covariance', params=params1, extension='npy')
        
        filename2 = 'keypoints_bunny-perturbed_factor:0.5-geom:0-q:0.95-radius:0.026-subs:decimation-t0:0.021.ply'
        prefix2, name2, params2 = parse_filename(filename2)
        cov_filename2 = give_filename(name2, prefix='covariance', params=params2, extension='npy')
        
        cloud_ply1 = read_ply(os.path.join(saved_data_path, filename1))
        keypoints1 = cloud_ply1['keypoints']
        
        cloud_ply2 = read_ply(os.path.join(saved_data_path, filename2))
        keypoints2 = cloud_ply2['keypoints']
        
        
        cov1 = np.load(os.path.join(saved_data_path, cov_filename1))
        interesting_points_indices1 = np.where(keypoints1==1)[0]
        #query_points_indices1 = np.load(os.path.join(data_path, 'query_indices_{}.npy'.format(name1)))
        
        cov2 = np.load(os.path.join(saved_data_path, cov_filename2))
        interesting_points_indices2 = np.where(keypoints2==1)[0]
        #query_points_indices2 = np.load(os.path.join(data_path, 'query_indices_{}.npy'.format(name2)))

        
#        indices1 = features.select_top_indices(cov1, q=0.7)
#        indices2 = features.select_top_indices(cov2, q=0.7)
#        interesting_points_indices1 = interesting_points_indices1[indices1]
#        interesting_points_indices2 = interesting_points_indices2[indices2]
#        cov1 = cov1[indices1]
#        cov2 = cov2[indices2]
        
        
#        distances = get_distance_matrix(cov1 + np.tile((1e-5*np.identity(10))[None,:,:], [len(cov1),1,1]),
#                                        cov2 + np.tile((1e-5*np.identity(10))[None,:,:], [len(cov2),1,1]))
        
        distances = get_distance_matrix(cov1, cov2)

        ratio = 1.3
        matches = get_correspondences(distances, ratio)
        
        payoff_matrix = get_payoff_matrix(matches)
        
        x = iterate(payoff_matrix, kind='conv', conv=0.001, nb_iter=10)
        
        matching_indices = np.where(x>1e-4)[0]
        correspondences = matches[matching_indices,:2]
        
        correspondences_indices = np.zeros((len(correspondences),2))
        correspondences_indices[:,0] = interesting_points_indices1[correspondences[:,0].astype('int')]
        correspondences_indices[:,1] = interesting_points_indices2[correspondences[:,1].astype('int')]
        
        params = {k+'1':v for k,v in params1.items()}
        params.update({k+'2':v for k,v in params2.items()})
        correspondences_filename = give_filename(name1+'-'+name2, prefix='corresp', params=params, extension='npy')
        np.save(os.path.join(saved_data_path, correspondences_filename), correspondences_indices)
        
