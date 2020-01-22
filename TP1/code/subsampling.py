#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.preprocessing import label_binarize

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time
from collections import defaultdict

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[0:-1:factor,:]
    decimated_colors = colors[0:-1:factor,:]
    decimated_labels = labels[0:-1:factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    # YOUR CODE
    # we take voxels starting from the point (0,0,0)
    voxel_indices = np.floor(points/voxel_size).astype('int')
    
    stats = defaultdict(lambda: (np.zeros(3),0))
    for i, voxel_index in enumerate(voxel_indices):
        s,c = stats[tuple(voxel_index)]
        stats[tuple(voxel_index)] = (s+points[i], c+1)

    subsampled_points = np.array([v[0]/v[1] for v in stats.values()])

    return subsampled_points


def grid_subsampling_colors(points, colors, voxel_size):

    # YOUR CODE
    # we take voxels starting from the point (0,0,0) (data is centered or reasonably centered)
    voxel_indices = np.floor(points/voxel_size).astype('int')
    
    stats = defaultdict(lambda: ((np.zeros(3),np.zeros(3),0)))

    for i, voxel_index in enumerate(voxel_indices):
        sum_p, sum_col, c = stats[tuple(voxel_index)]
        stats[tuple(voxel_index)] = (sum_p+points[i], sum_col+colors[i], c+1)


    subsampled_points = np.array([v[0]/v[-1] for v in stats.values()])
    subsampled_colors = np.array([np.clip((v[1]/v[-1]).astype('uint8'),0,255) for v in stats.values()])

    return subsampled_points, subsampled_colors

def grid_subsampling_categorical_labels(points, colors, labels, voxel_size):
    
    # YOUR CODE
    # we take voxels starting from the point (0,0,0) (data is centered or reasonably centered)
    voxel_indices = np.floor(points/voxel_size).astype('int')
    
    unique_labels = np.unique(labels)
    #unique_labels_binarized = label_binarize(labels, classes=unique_labels)
    #labels_dict = {unique_labels_binarized[i]:unique_labels [i] for i in range(len(unique_labels))}
    
    binarized_labels = label_binarize(labels, classes=unique_labels)
    
    stats = defaultdict(lambda: ((np.zeros(3),np.zeros(3),np.zeros(binarized_labels.shape[1]),0)))

    for i, voxel_index in enumerate(voxel_indices):
        sum_p, sum_col, sum_lab, c = stats[tuple(voxel_index)]
        stats[tuple(voxel_index)] = (sum_p+points[i], sum_col+colors[i], sum_lab+binarized_labels[i], c+1)


    subsampled_points = np.array([v[0]/v[-1] for v in stats.values()])
    subsampled_colors = np.array([np.clip((v[1]/v[-1]).astype('uint8'),0,255) for v in stats.values()])
    subsampled_labels = np.array([unique_labels[v[2].argmax()] for v in stats.values()])

    return subsampled_points, subsampled_colors, subsampled_labels

def grid_subsampling_numeric_labels(points, colors, labels, voxel_size):
    
    # YOUR CODE
    # we take voxels starting from the point (0,0,0) (data is centered or reasonably centered)
    voxel_indices = np.floor(points/voxel_size).astype('int')
    
    stats = defaultdict(lambda: ((np.zeros(3),np.zeros(3),np.zeros(labels.shape[1]),0)))

    for i, voxel_index in enumerate(voxel_indices):
        sum_p, sum_col, sum_lab, c = stats[tuple(voxel_index)]
        stats[tuple(voxel_index)] = (sum_p+points[i], sum_col+colors[i], sum_lab+labels[i], c+1)


    subsampled_points = np.array([v[0]/v[-1] for v in stats.values()])
    subsampled_colors = np.array([np.clip((v[1]/v[-1]).astype('uint8'),0,255) for v in stats.values()])
    subsampled_labels = np.array([v[2]/v[-1] for v in stats.values()])

    return subsampled_points, subsampled_colors, subsampled_labels


def grid_subsampling_labels(points, colors, labels, voxel_size, kind='categorical'):
    if kind == 'categorical':
        print('categorical')
        return grid_subsampling_categorical_labels(points, colors, labels, voxel_size)
    print('numeric')
    return grid_subsampling_numeric_labels(points, colors, labels, voxel_size)

# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']

    # choose what computations to do
    flag_concatenation = False
    flag_simple_subsample = False
    flag_color_subsample = True
    flag_label_subsample = False    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    if flag_concatenation:
        t0 = time.time()
        decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
        t1 = time.time()
        print('decimation done in {:.3f} seconds'.format(t1 - t0))
    
        # Save
        write_ply('../data/decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

        print('Concatenation done')
        
    # Subsample the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2
    
    

    # Subsample
    if flag_simple_subsample:
        t0 = time.time()
        subsampled_points = grid_subsampling(points, voxel_size)
        t1 = time.time()
        print('Subsampling done in {:.3f} seconds'.format(t1 - t0))
        
        # Save
        write_ply('../data/grid_subsampled.ply', [subsampled_points], ['x', 'y', 'z'])
        
        print('Subsample done')
    
    # Color subsample
    if flag_color_subsample:
        t0 = time.time()
        subsampled_points, subsampled_colors = grid_subsampling_colors(points, colors, voxel_size)
        t1 = time.time()
        print('Subsampling done in {:.3f} seconds'.format(t1 - t0))
        
        # Save
        write_ply('../data/grid_subsampled_colors.ply', [subsampled_points, subsampled_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Subsample color done')
    
    # Label subsample
    if flag_label_subsample:
        t0 = time.time()
        subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling_labels(points, colors, labels, voxel_size)
        t1 = time.time()
        print('Subsampling done in {:.3f} seconds'.format(t1 - t0))
        
        if len(subsampled_labels.shape)==1:
            subsampled_labels = subsampled_labels[:,None]
    
        # Save
        write_ply('../data/grid_subsampled_labels.ply', [subsampled_points, subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

        print('Subsample label done')

    print('Done')