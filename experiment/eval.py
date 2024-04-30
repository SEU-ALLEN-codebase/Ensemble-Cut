"""
Comparing APP2 reconstruction before and after pruning

categorized as sparse (863) and dense image blocks, using the tracing result of NIEND enhanced images.

Here are the pairs:

sparse against GS
sparse prune against GS
dense against GS
dense prune against GS

"""

# evaluate different reconstruction against gold standard

from utils.metrics import DistanceEvaluation
import pandas as pd
from pathlib import Path
import sys
import os


wkdir = Path(r"D:\rectify")
pruned_path = wkdir / 'pruned_3'



class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main(name):
    man = wkdir / 'manual' / name
    before = wkdir / 'my_app2' / name
    afters = pruned_path.glob(f'{before.stem}*')
    de = DistanceEvaluation(15)
    with HidePrint():
        before = de.run(before, man) if before.exists() else None
    ret = {
            'before_recall': 1 - before[2, 1] if before is not None else 0,
            'before_precision': 1 - before[2, 0] if before is not None else 1,
        }
    for after in afters:
        after = de.run(after, man) if after.exists() else None
        with HidePrint():
            recall = 1 - after[2, 1] if after is not None else 0
            precision = 1 - after[2, 0] if after is not None else 1
        if 'after_recall' not in ret or ret['after_recall'] < recall:
            ret['after_recall'] = recall
            ret['after_precision'] = precision
    return ret


if __name__ == '__main__':
    from multiprocessing import Pool
    from tqdm import tqdm

    # get the sparse and dense labels
    files = [i.name for i in (wkdir / 'manual').glob('*.swc')]
    tab = pd.read_csv(wkdir / 'filter.csv', index_col=0)

    # main('18457_14455_13499_5478.swc')
    with Pool(14) as p:
        # sparse
        sparse = [*filter(lambda f: tab.at[f, 'sparse'] == 1, files)]
        res = []
        for r in tqdm(p.imap(main, sparse), total=len(sparse)):
            res.append(r)
        pd.DataFrame.from_records(res, index=sparse).to_csv('../results/eval_sparse.csv')

        # dense
        dense = [*filter(lambda f: tab.at[f, 'sparse'] == 0, files)]
        res = []
        for r in tqdm(p.imap(main, dense), total=len(dense)):
            res.append(r)
        pd.DataFrame.from_records(res, index=dense).to_csv('../results/eval_dense.csv')


