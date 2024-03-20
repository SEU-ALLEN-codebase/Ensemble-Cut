from ecut import swc_handler
from ecut.annealing import MorphAnneal
from ecut.graph_cut import ECut
from ecut.soma_detection import DetectTracingMask
from sklearn.neighbors import KDTree
from ecut.error_prune import ErrorPruning
from ecut.morphology import Morphology


if __name__ == '__main__':
    # tree = swc_handler.parse_swc('../test/data/gcut_input.swc_sorted.swc')
    tree = swc_handler.parse_swc(r'D:\rectify\my_app2\15257_16445_16836_4489.swc')

    # detect soma
    maxr = max([t[5] for t in tree]) * .3
    soma = DetectTracingMask(min(maxr, 5), 100).predict(tree, [.3, .3, 1.])
    print(soma)

    # anneal
    a = MorphAnneal(tree)
    tree = a.run()

    # map soma
    kd = KDTree([t[2:5] for t in tree])
    inds = kd.query(soma, return_distance=False)
    inds = [tree[i[0]][0] for i in inds]
    print(inds)

    # graph cut
    e = ECut(tree, inds)
    e.run()
    trees = e.export_swc()
    # swc_handler.write_swc(trees, f'../test/data/whole.swc')
    # exit()

    # pruning
    for k, v in trees.items():
        v = swc_handler.sort_swc(v)
        p = ErrorPruning([1., 1., 1], anchor_dist=20, soma_radius=20.)
        morph = Morphology(v)
        a = p.branch_prune(morph, 60, 1.5)
        b = p.crossover_prune(morph, 5, 60, 90)
        c = p.crossover_prune(morph, 5, 90, 120, check_bif=True)
        v = swc_handler.prune(v, a | b | c)
        swc_handler.write_swc(v, f'../test/data/whole_{k}.swc')
        