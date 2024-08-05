from ecut import swc_handler
from ecut.soma_detection import *
import pickle
from pathlib import Path
from v3dpy.loaders import PBD

img_dir = Path(r"D:\rectify\my")
swc_dir = Path(r"D:\rectify\my_app2")
marker_dir = Path(r"D:\rectify\multi_soma_test\gt")
out_dir = Path('D:/rectify/multi_soma_test/detected')


def main(marker_file):
    img_file = marker_file.name.split(".")[0]
    brain = img_file.split("_")[0]
    swc_file = swc_dir / (img_file + '.swc')
    img_file = img_dir / (img_file + '.v3dpbd')
    tree = [t for t in swc_handler.parse_swc(swc_file) if not (t[1] == t[2] == t[3] == 0)]
    metadata = pd.read_csv('D:/rectify/supplement.csv', index_col=1, header=0).iloc[:, 3]
    res = metadata.loc[int(brain)]
    if np.isnan(res):
        res = .25
    res = [res, res, 1.]

    # detect soma
    img = PBD().load(img_file)[0]
    centers_list = []

    centers1 = DetectTiledImage(base_detector=DetectImage()).predict(img, res)
    centers1 = [c[::-1] for c in centers1]
    centers_list.append(centers1)

    centers2 = DetectTiledImage(base_detector=DetectDistanceTransform()).predict(img, res)
    centers2 = [c[::-1] for c in centers2]
    centers_list.append(centers2)

    centers3 = DetectTracingMask().predict(tree, res)

    centers4 = soma_consensus(*centers_list, res=res)

    centers_list.append(centers3)
    centers5 = soma_consensus(*centers_list, res=res)

    with open(out_dir / f'{img_file.stem}.pkl', 'wb') as f:
        out = {'def': centers1, 'dt': centers2, 'swc': centers3, 'con_img': centers4, 'con_all': centers5}
        pickle.dump(out, f)


if __name__ == '__main__':
    from tqdm import tqdm
    from multiprocessing import Pool
    out_dir.mkdir(exist_ok=True)
    # main(Path(r"D:\rectify\multi_soma_test\gt\15257_15413_13333_3271.v3dpbd.marker"))
    # exit()
    marker_files = sorted(marker_dir.glob('*.marker'))
    with Pool(10) as p:
        for i in tqdm(p.imap(main, marker_files), total=len(marker_files)):
            pass