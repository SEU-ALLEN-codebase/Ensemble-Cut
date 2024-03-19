import unittest
from ecut.annealing import *
from ecut.swc_handler import parse_swc, write_swc


class TestAnneal(unittest.TestCase):

    def setUp(self):
        # self.swc = parse_swc('data/gcut_input.swc_sorted.swc')
        self.swc = parse_swc(r'D:\rectify\my_app2\17302_14358_42117_2799.swc')
        self.swc = [t for t in self.swc if not (t[1] == t[2] == t[3] == 0)]

    def test1(self):
        a = MorphAnneal(self.swc)
        tree = a.run()
        write_swc(tree, 'data/anneal_output.swc')


if __name__ == '__main__':
    unittest.main()
