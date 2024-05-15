from ecut import swc_handler
from ecut.annealing import MorphAnneal
from ecut.graph_cut import ECut, Adjacency
from ecut.soma_detection import DetectTracingMask
from sklearn.neighbors import KDTree
from ecut.error_prune import ErrorPruning
from ecut.morphology import Morphology
from tempfile import TemporaryDirectory
import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # tree = swc_handler.parse_swc('../test/data/gcut_input.swc_sorted.swc')
    tree = swc_handler.parse_swc(r'D:\rectify\my_app2\17302_14358_42117_2799.swc')
    # tree = swc_handler.parse_swc(r"C:\Users\zzh\Downloads\2965_9868_7229.swc")
    # tree = swc_handler.parse_swc(r'D:\rectify\my_app2\15257_16445_16836_4489.swc')
    tree = [t for t in tree if not (t[1] == t[2] == t[3] == 0)]
    res = .2
    maxr = max([t[5] for t in tree]) * res
    res = [res, res, 1]
    rad = max(maxr * .5, 3.)
    centers = DetectTracingMask(rad, 20.).predict(tree, res)
    # anneal
    a = MorphAnneal(tree, res=res)
    tree = a.run()
    swc_handler.write_swc(tree, f'../test/data/ann.swc')
    # tree = swc_handler.parse_swc(f'../test/data/ann.swc')
    # graph cut
    if len(centers) < 1:
        centers = [[512, 512, 128]]
    kd = KDTree([t[2:5] for t in tree])
    inds = kd.query(centers, return_distance=False)
    inds = [tree[i[0]][0] for i in inds]
    print(inds, centers)
    centers = np.array(centers)
    # fig, ax = plt.subplots()
    # plt.scatter(centers[:, 0], centers[:, 1], c=centers[:, 2])
    # ax.invert_yaxis()
    # plt.show()

    e = ECut(tree, inds, Adjacency(tree))
    e.run()
    trees = e.export_swc()

    # trees = list(trees.values())
    # dist = []
    # for v in trees:
    #     soma = [t[2:5] for t in v if t[6] == -1][0]
    #     dist.append(np.linalg.norm(np.array(soma) - [512, 512, 128]))
    # tree = trees[np.argmin(dist)]
    # v = swc_handler.sort_swc(tree)
    # p = ErrorPruning([res,res,1], anchor_dist=20., soma_radius=10.)
    # morph = Morphology(v)
    # a = p.branch_prune(morph, 60, 1.5)
    # b = p.crossover_prune(morph, 2, 60, 90, short_tips_thr=10., no_multi=False)
    # c = p.crossover_prune(morph, 2, 60, 90, check_bif=True, short_tips_thr=10.)
    # v = swc_handler.prune(v, a | b | c)
    # swc_handler.write_swc(v, f'../test/data/main.swc')

    # pruning
    for k, v in trees.items():
        v = swc_handler.sort_swc(v)
        a = MorphAnneal(v, res=res)
        v = a.run()
        # p = ErrorPruning([res,res,1], anchor_dist=20., soma_radius=10.)
        # morph = Morphology(v)
        # a = p.branch_prune(morph, 60, 1.5)
        # b = p.crossover_prune(morph, 2, 60, 90, short_tips_thr=10., no_multi=False)
        # c = p.crossover_prune(morph, 2, 60, 90, check_bif=True, short_tips_thr=10.)
        # v = swc_handler.prune(v, a | b | c)
        swc_handler.write_swc(v, f'../test/data/nulti_{k}.swc')
