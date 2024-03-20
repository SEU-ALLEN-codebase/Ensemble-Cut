from ecut import swc_handler
from ecut.annealing import MorphAnneal
from ecut.graph_cut import ECut
from ecut.soma_detection import DetectTracingMask
from sklearn.neighbors import KDTree
from ecut.error_prune import ErrorPruning
from ecut.morphology import Morphology


if __name__ == '__main__':
    # tree = swc_handler.parse_swc('../test/data/gcut_input.swc_sorted.swc')
    tree = swc_handler.parse_swc(r'D:\rectify\my_app2\17545_17012_13613_3775.swc')

    # detect soma
    d = DetectTracingMask(5, 100)
    soma = d.predict(tree, [.3, .3, 1.])

    # anneal
    a = MorphAnneal(tree, radius_gap=.5)
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

    # pruning
    for k, v in trees.items():
        p = ErrorPruning([.3, .3, 1], anchor_reach=(5., 20.))
        morph = Morphology(v)
        a = p.branch_prune(morph, 45, 10)
        b = p.crossover_prune(morph, 5, 90)
        # c = p.crossover_prune(morph, check_bif=True)
        t = swc_handler.prune(v, a )
        swc_handler.write_swc(t, f'../test/data/whole_{k}.swc')