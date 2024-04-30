from ecut import swc_handler
from ecut.annealing import MorphAnneal
from ecut.graph_cut import ECut
from ecut.soma_detection import DetectTracingMask
from sklearn.neighbors import KDTree
from ecut.error_prune import ErrorPruning
from ecut.morphology import Morphology


if __name__ == '__main__':
    # tree = swc_handler.parse_swc('../test/data/gcut_input.swc_sorted.swc')
    tree = swc_handler.parse_swc(r'D:\rectify\my_app2\17302_14358_42117_2799.swc')
    tree = [t for t in tree if not (t[1] == t[2] == t[3] == 0)]
    # tree = swc_handler.parse_swc(r'D:\rectify\my_app2\15257_16445_16836_4489.swc')

    maxr = max([t[5] for t in tree]) * .3
    rad = max(maxr * .5, 5.)
    centers = DetectTracingMask(rad, 20.).predict(tree, [.3, .3, 1])

    # anneal
    a = MorphAnneal(tree)
    tree = a.run()

    # graph cut
    if len(centers) < 1:
        centers = [[512, 512, 128]]
    kd = KDTree([t[2:5] for t in tree])
    inds = kd.query(centers, return_distance=False)
    inds = [tree[i[0]][0] for i in inds]
    print(inds)
    e = ECut(tree, inds)
    e.run()
    trees = e.export_swc()

    # pruning
    for k, v in trees.items():
        v = swc_handler.sort_swc(v)
        p = ErrorPruning([.3,.3,1], anchor_dist=20., soma_radius=10.)
        morph = Morphology(v)
        a = p.branch_prune(morph, 60, 1.5)
        b = p.crossover_prune(morph, 2, 60, 90, short_tips_thr=10., no_multi=False)
        c = p.crossover_prune(morph, 2, 60, 90, check_bif=True, short_tips_thr=10.)
        v = swc_handler.prune(v, a | b | c)
        swc_handler.write_swc(v, f'../test/data/multi_{k}.swc')
        