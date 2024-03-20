from ecut import swc_handler
from ecut.annealing import MorphAnneal
from ecut.graph_cut import ECut
from ecut.soma_detection import DetectTracingMask
from sklearn.neighbors import KDTree
from ecut.error_prune import ErrorPruning
from ecut.morphology import Morphology
from traceback import print_exc


def main(args):
    in_path, out_path = args
    try:
        tree = [t for t in swc_handler.parse_swc(in_path) if not (t[1] == t[2] == t[3] == 0)]

        # detect soma
        d = DetectTracingMask(5)
        soma = d.predict(tree, [.3, .3, 1])

        # anneal
        a = MorphAnneal(tree)
        tree = a.run()

        # map soma
        kd = KDTree([t[2:5] for t in tree])
        inds = kd.query(soma, return_distance=False)
        inds = [tree[i[0]][0] for i in inds]

        # graph cut
        if len(inds) > 1:
            e = ECut(tree, inds)
            e.run()
            trees = e.export_swc()
        else:
            trees = {0: tree}

        # pruning
        for k, v in trees.items():
            p = ErrorPruning([.3, .3, 1], anchor_reach=(5., 20.))
            morph = Morphology(v)
            a = p.branch_prune(morph, 45, 2)
            b = p.crossover_prune(morph, 5, 90)
            # c = p.crossover_prune(morph, check_bif=True)
            t = swc_handler.prune(v, a | b)
            swc_handler.write_swc(t, str(out_path) + f'_{k}.swc')
    except:
        print_exc()
        print(in_path)


if __name__ == '__main__':
    from pathlib import Path
    from tqdm import tqdm
    from multiprocessing import Pool
    indir = Path('D:/rectify/my_app2')
    outdir = Path('D:/rectify/pruned')
    outdir.mkdir(exist_ok=True)
    files = sorted(indir.glob('*.swc'))
    outfiles = [outdir / f.name for f in files]
    arglist = [*zip(files, outfiles)]
    with Pool(16) as p:
        for i in tqdm(p.imap(main, arglist), total=len(arglist)):
            pass