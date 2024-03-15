import unittest
from pathlib import Path
from ecut.swc_handler import parse_swc, write_swc
from ecut.graph_cut import ECut


dat_dir = Path('data')


class TestCut(unittest.TestCase):
    def test_real_swc(self):
        tree = [list(t) for t in parse_swc(dat_dir / 'gcut_input.swc')]
        for t in tree:
            t[0] -= 1
            if t[6] != -1:
                t[6] -= 1
        tree = [tuple(t) for t in tree]
        g = ECut(tree, [2437, 1397], adjacency=3)
        g.run()
        trees = g.export_swc()
        for i, t in trees.items():
            write_swc(t, dat_dir / f'gcut_output_{i}.swc')

    def test_pseudo_swc(self):
        tree = [list(t) for t in parse_swc(dat_dir / 'gcut_pseudo.swc')]
        for t in tree:
            t[0] -= 1
            if t[6] != -1:
                t[6] -= 1
        tree = [tuple(t) for t in tree]
        g = ECut(tree, [0, 10], adjacency=3)
        g.run()
        trees = g.export_swc()
        # write_swc(trees, dat_dir / f'gcut_pseudo_output.swc')
        for i, t in trees.items():
            write_swc(t, dat_dir / f'gcut_pseudo_output_{i}.swc')


if __name__ == '__main__':
    unittest.main()
