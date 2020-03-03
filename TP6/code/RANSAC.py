#
#
#      0===================0
#      |    6 Modelling    |
#      0===================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):

    
#    point = np.zeros((3,1))
#    normal = np.zeros((3,1))
    
    # TODO
#    normal, d = np.linalg.solve(np.concatenate((points, np.ones((len(points),1))), axis=1),0)
#    normal = normal/np.linalg.norm(normal)
    
    point = points[0]
    
    #np.linalg.solve(np.stack(((points[1]-points[0]),(points[2]-points[0])), axis=0), np.ones(3))
    
    #normal = (points[1]-points[0])*(points[2]-points[0])
    normal = np.cross(points[1]-points[0],
                         points[2]-points[0]
                         )
    normal = normal / np.linalg.norm(normal)
    
    return point[:,None], normal[:,None]

def compute_planes(points):
    ref_points = points[:,0,:]
    normals = np.cross(points[:,1,:]-points[:,0,:],
                       points[:,2,:]-points[:,0,:])
    normals = normals / np.linalg.norm(normals, axis=-1)[:,None]
    
    return ref_points, normals

def in_plane(points, ref_pt, normal, threshold_in=0.1):
    
    #indices = np.zeros(len(points), dtype=bool)
    
    # TODO: return a boolean mask of points in range
    indices = np.abs((points-ref_pt.T)@normal)<threshold_in
        
    return indices

def in_planes(points, ref_pts, normals, threshold_in=0.1):
    indices = np.abs(((points[None,:,:]-ref_pts[:,None,:])*normals[:,None,:]).sum(axis=-1))<threshold_in
    return indices

def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    
    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    best_vote = 0
    
    # TODO:
    for k in range(NB_RANDOM_DRAWS):
        candidate_pts = points[np.random.randint(0, len(points), size=3)]
        ref_pt, normal = compute_plane(candidate_pts)
        mask = in_plane(points, ref_pt, normal, threshold_in)
        nb_vote = sum(mask)
        if nb_vote > best_vote:
            best_ref_pt = ref_pt
            best_normal = normal
            best_vote = nb_vote
            
    print(nb_draws(best_vote, len(points), 0.99, q=3))
    
    return best_ref_pt, best_normal

def RANSAC_vectorized(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    
    candidate_pts = points[np.random.randint(0, len(points), size=3*NB_RANDOM_DRAWS)].reshape(NB_RANDOM_DRAWS,3,3)
    ref_pts, normals = compute_planes(candidate_pts)
    masks = in_planes(points, ref_pts, normals, threshold_in)
    nb_votes = np.sum(masks, axis=-1)
    
    best_vote_index = np.argmax(nb_votes)
    
    print(nb_draws(nb_votes[best_vote_index], len(points), 0.99, q=3))
    
    return ref_pts[best_vote_index], normals[best_vote_index]

def multi_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2, vectorized = True):
    
    # TODO:
    remaining_points = points
    remaining_indices = np.arange(len(points))
    
    plane_labels = -1 * np.ones(len(points), dtype=np.int64)
    
    for k in range(NB_PLANES):
        if vectorized:
            best_ref_pt, best_normal = RANSAC_vectorized(remaining_points, NB_RANDOM_DRAWS, threshold_in)
        else:
            best_ref_pt, best_normal = RANSAC(remaining_points, NB_RANDOM_DRAWS, threshold_in)
        
        points_in_plane = in_plane(remaining_points, best_ref_pt, best_normal, threshold_in)
        
        plane_inds = remaining_indices[points_in_plane]
        remaining_indices = remaining_indices[~points_in_plane]
        plane_labels[plane_inds] = k
        
        remaining_points = remaining_points[~points_in_plane]
        
    plane_indices = np.where(plane_labels>=0)
    
    return plane_indices, remaining_indices, plane_labels[plane_indices]

def nb_draws(nb_vote, nb_points, proba, q=3):
    return np.log(1/(1-proba))*(nb_points/nb_vote)**q

# ----------------------------------------------------------------------------------------------------------------------
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
    N = len(points)

    # Computes the plane passing through 3 randomly chosen points
    # ***********************************************************
    #

    if False:

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]

        # Save the 3 points and their corresponding plane for verification
        pts_clr = np.zeros_like(pts)
        pts_clr[:, 0] = 1.0
        write_ply('../triplet.ply',
                  [pts, pts_clr],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../triplet_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #

    if False:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 100
        #threshold_in = 0.05
        #threshold_in = 0.01
        threshold_in = 0.01

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save the best extracted plane and remaining points
        write_ply('../best_plane_RANSAC.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points_RANSAC.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        
    # the same, but with a fully vectorized function, which increases the performances (400 draws in 3 seconds)
    if False:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 400
        #threshold_in = 0.05
        #threshold_in = 0.01
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC_vectorized(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save the best extracted plane and remaining points
        write_ply('../best_plane_RANSAC.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points_RANSAC.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Find multiple planes in the cloud
    # *********************************
    #

    if True:

        # Define parameters of multi_RANSAC
        NB_RANDOM_DRAWS = 400
        threshold_in = 0.05
        NB_PLANES = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('\nmulti RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes_multi_RANSAC.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels[:,None].astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_multi_RANSAC.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
