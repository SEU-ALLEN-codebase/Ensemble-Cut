import unittest
from ecut.error_prune import *
from ecut.swc_handler import parse_swc, write_swc, prune
from ecut.morphology import Morphology


class TestPrune(unittest.TestCase):
    def setUp(self):
        self.swc = parse_swc('data/anneal_output.swc')

    def test_prune(self):
        pruner = ErrorPruning([.25, .25, 1], anchor_reach=(5., 20.))
        morph = Morphology(self.swc)
        a = pruner.branch_prune(morph, 45)
        b = pruner.crossover_prune(morph, 5)
        c = pruner.crossover_prune(morph, 5, check_bif=True)
        tree = prune(self.swc, a|b|c)
        write_swc(tree, 'data/pruned_tree.swc')


if __name__ == '__main__':
    unittest.main()
