#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : math_utils.py
#   Author       : Yufeng Liu
#   Date         : 2021-07-19
#   Description  : 
#
#================================================================

import math
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree


def calc_included_angles_from_vectors(vecs1, vecs2, return_rad=False, epsilon=1e-7, spacing=None, return_cos=False):
    if vecs1.ndim == 1:
        vecs1 = vecs1.reshape((1,-1))
    if vecs2.ndim == 1:
        vecs2 = vecs2.reshape((1,-1))

    if spacing is not None:
        spacing_reshape = np.array(spacing).reshape(1,-1)
        # rescale vectors according to spacing
        vecs1 = vecs1 * spacing_reshape
        vecs2 = vecs2 * spacing_reshape

    inner = (vecs1 * vecs2).sum(axis=1)
    norms = np.linalg.norm(vecs1, axis=1) * np.linalg.norm(vecs2, axis=1)
    cos_ang = inner / (norms + epsilon)

    if return_cos:
        return_val = cos_ang
    else:
        rads = np.arccos(np.clip(cos_ang, -1, 1))
        if return_rad:
            return_val = rads
        else:
            return_val = np.rad2deg(rads)
    return return_val    


def calc_included_angles_from_coords(anchor_coords, coords1, coords2, return_rad=False, epsilon=1e-7, spacing=None, return_cos=False):
    anchor_coords = np.array(anchor_coords)
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    v1 = coords1 - anchor_coords
    v2 = coords2 - anchor_coords
    angs = calc_included_angles_from_vectors(
                v1, v2, return_rad=return_rad, 
                epsilon=epsilon, spacing=spacing,
                return_cos=return_cos)
    return angs


def memory_safe_min_distances(voxels1, voxels2, num_thresh=50000, return_index=False):
    # verified
    nv1 = len(voxels1)
    nv2 = len(voxels2)
    if (nv1 > num_thresh) or (nv2 > num_thresh):
        # use block wise calculation
        vq1 = [voxels1[i*num_thresh:(i+1)*num_thresh] for i in range(int(math.ceil(nv1/num_thresh)))]
        vq2 = [voxels2[i*num_thresh:(i+1)*num_thresh] for i in range(int(math.ceil(nv2/num_thresh)))]

        dists1 = np.ones(nv1) * 1000000.
        dists2 = np.ones(nv2) * 1000000.
        if return_index:
            min_indices1 = np.ones(nv1) * -1
            min_indices2 = np.ones(nv2) * -1
        for i,v1 in enumerate(vq1):
            idx00 = i * num_thresh
            idx01 = i * num_thresh + len(v1)
            for j,v2 in enumerate(vq2):
                idx10 = j * num_thresh
                idx11 = j * num_thresh + len(v2)

                d = distance_matrix(v1, v2)
                dmin1 = d.min(axis=1)
                dmin0 = d.min(axis=0)
                dists1[idx00:idx01] = np.minimum(dmin1, dists1[idx00:idx01])
                dists2[idx10:idx11] = np.minimum(dmin0, dists2[idx10:idx11])
                if return_index:
                    dargmin1 = np.argmin(d, axis=1)
                    dargmin0 = np.argmin(d, axis=0)
                    mask1 = np.nonzero(dmin1 < dists1[idx00:idx01])
                    min_indices1[idx00:idx01][mask1[0]] = dargmin1[mask1[0]] + idx00
                    mask0 = np.nonzero(dmin0 < dists2[idx10:idx11])
                    min_indices2[idx10:idx11][mask0[0]] = dargmin0[mask0[0]] + idx10
    else:
        pdist = distance_matrix(voxels1, voxels2)
        dists1 = pdist.min(axis=1)
        dists2 = pdist.min(axis=0)
        if return_index:
            min_indices1 = pdist.argmin(axis=1)
            min_indices2 = pdist.argmin(axis=0)

    if return_index:
        return dists1, dists2, min_indices1, min_indices2
    else:
        return dists1, dists2


def min_distances_between_two_sets(voxels1, voxels2, topk=1, reciprocal=True, return_index=False):
    """
    We should use kd-tree instead of brute-force method for large-scale data inputs. Arguments are:
    @params voxels1: coordinates of points, np.ndarray in shape[N, 3]
    @params voxels2: coordinates of points, np.ndarray in shape[M, 3]
    @params topk: the number of top-ranking match
    @params reciprocal: whether to calculate 2->1, except for 1->2
    @params return_index: whehter to return the indices of points with minimal distances
    """
    tree2 = KDTree(voxels2, leaf_size=2)
    dmin1, imin1 = tree2.query(voxels1, k=topk)
    if reciprocal:
        tree1 = KDTree(voxels1, leaf_size=2)
        dmin2, imin2 = tree1.query(voxels2, k=topk)
        if return_index:
            return dmin1, dmin2, imin1, imin2
        else:
            return dmin1, dmin2
    else:
        if return_index:
            return dmin1, imin1
        else:
            return dmin1



