from ecut import swc_handler
from ecut.annealing import MorphAnneal
from ecut.graph_cut import ECut
from ecut.soma_detection import *
from sklearn.neighbors import KDTree
from ecut.error_prune import ErrorPruning
from ecut.morphology import Morphology
from traceback import print_exc
from pathlib import Path
from v3dpy.loaders import PBD

img_dir = Path(r"D:\rectify\crop_8bit")


def main(args):
    in_path, out_path = args
    try:
        tree = [t for t in swc_handler.parse_swc(in_path) if not (t[1] == t[2] == t[3] == 0)]
        res = [.3, .3, 1.]

        # detect soma
        img = PBD().load(r"D:\rectify\crop_8bit\18453_9442_3817_6561.v3dpbd")[0]
        centers_list = []
        centers = DetectImage().predict(img, res)
        centers_list.append(centers)
        centers = DetectTiledImage([300, 300, 200]).predict(img, res)
        centers_list.append(centers)
        centers = DetectTracingMask().predict(tree, res)
        centers_list.append(centers)
        centers = DetectDistanceTransform().predict(img, res)
        centers_list.append(centers)
        centers = DetectTiledImage(base_detector=DetectDistanceTransform()).predict(img, res)
        centers_list.append(centers)
        centers = soma_consensus(*centers_list, res=res)

        # anneal
        a = MorphAnneal(tree)
        tree = a.run()

        # graph cut
        if len(centers) > 1:
            kd = KDTree([t[2:5] for t in tree])
            inds = kd.query(centers, return_distance=False)
            inds = [tree[i[0]][0] for i in inds]
            e = ECut(tree, inds)
            e.run()
            trees = e.export_swc()
        else:
            trees = {0: tree}

        # pruning
        for k, v in trees.items():
            p = ErrorPruning(res, anchor_reach=(5., 20.))
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
    from tqdm import tqdm
    from multiprocessing import Pool
    indir = Path('D:/rectify/my_app2')
    outdir = Path('D:/rectify/pruned')
    outdir.mkdir(exist_ok=True)
    files = sorted(indir.glob('*.swc'))
    outfiles = [outdir / f.name for f in files]
    arglist = [*zip(files, outfiles)]
    with Pool(8) as p:
        for i in tqdm(p.imap(main, arglist), total=len(arglist)):
            pass