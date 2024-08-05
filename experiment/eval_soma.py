# evaluate different reconstruction against gold standard

import pandas as pd
from pathlib import Path
from sklearn.neighbors import KDTree
import numpy as np

if __name__ == '__main__':
    import pickle
    from tqdm import tqdm

    marker_files = sorted(Path(r"D:\rectify\multi_soma_test\gt").glob('*.marker'))
    markers = []
    hit = {
        'def': 0,
        'dt': 0,
        'swc': 0,
        'con_img': 0,
        'con_all': 0
    }
    miss = {
        'def': 0,
        'dt': 0,
        'swc': 0,
        'con_img': 0,
        'con_all': 0
    }
    for m in tqdm(marker_files):
        r = Path(r"D:\rectify\multi_soma_test\detected") / (m.name.split('.')[0] + '.pkl')
        metadata = pd.read_csv('D:/rectify/supplement.csv', index_col=1, header=0).iloc[:, 3]
        reso = metadata.loc[int(m.name.split('_')[0])]
        if np.isnan(reso):
            reso = .25
        with open(m, 'r') as f:
            f.readline()
            gt = []
            for i in f.readlines():
                x, y, z = i.split(',')[:3]
                x = float(x) * reso
                y = float(y) * reso
                z = float(z)
                gt.append((x, y, z))
        with open(r, 'rb') as f:
            res = pickle.load(f)

        gt_tree = KDTree(gt)
        for k, v in res.items():
            if not v:
                continue
            v = np.array(v) * [reso, reso, 1]
            d, i = gt_tree.query(v, 1)
            for s in d:
                if s < 10:
                    hit[k] += 1
                else:
                    miss[k] += 1
    with open(r"D:\rectify\multi_soma_test\eval.pkl", "wb") as f:
        pickle.dump((hit, miss), f)
