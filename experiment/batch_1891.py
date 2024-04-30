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
        # img = PBD().load(r"D:\rectify\crop_8bit\18453_9442_3817_6561.v3dpbd")[0]
        # centers_list = []
        # centers = DetectImage().predict(img, res)
        # centers_list.append(centers)
        # centers = DetectTiledImage([300, 300, 200]).predict(img, res)
        # centers_list.append(centers)
        # maxr = max([t[5] for t in tree]) * res[0]
        # centers = DetectTracingMask(maxr * .75, maxr * 3).predict(tree, res)
        # centers_list.append(centers)
        # centers = DetectDistanceTransform().predict(img, res)
        # centers_list.append(centers)
        # centers = DetectTiledImage(base_detector=DetectDistanceTransform()).predict(img, res)
        # centers_list.append(centers)
        # centers = soma_consensus(*centers_list, res=res)

        maxr = max([t[5] for t in tree]) * res[0]
        rad = max(maxr * .5, 5.)
        centers = DetectTracingMask(rad, 20.).predict(tree, res)

        # anneal
        a = MorphAnneal(tree)
        tree = a.run()

        # graph cut
        if len(centers) < 1:
            centers = [[512, 512, 128]]
        kd = KDTree([t[2:5] for t in tree])
        inds = kd.query(centers, return_distance=False)
        inds = [tree[i[0]][0] for i in inds]
        e = ECut(tree, inds)
        e.run()
        trees = e.export_swc()

        # pruning
        for k, v in trees.items():
            v = swc_handler.sort_swc(v)
            # p = ErrorPruning(res, anchor_dist=20., soma_radius=10.)
            # morph = Morphology(v)
            # a = p.branch_prune(morph, 60, 1.5)
            # b = p.crossover_prune(morph, 2, 60, 90, short_tips_thr=10., no_multi=False)
            # c = p.crossover_prune(morph, 2, 60, 90, check_bif=True, short_tips_thr=10.)
            # v = swc_handler.prune(v, a | b | c)
            swc_handler.write_swc(v, str(out_path) + f'_{k}.swc')
    except:
        print_exc()
        print(in_path)


if __name__ == '__main__':
    from tqdm import tqdm
    from multiprocessing import Pool

    indir = Path('D:/rectify/my_app2')
    outdir = Path('D:/rectify/pruned_3')
    outdir.mkdir(exist_ok=True)
    files = sorted(indir.glob('*.swc'))
    outfiles = [outdir / f.name for f in files]
    arglist = [*zip(files, outfiles)]
    with Pool(12) as p:
        for i in tqdm(p.imap(main, arglist), total=len(arglist)):
            pass