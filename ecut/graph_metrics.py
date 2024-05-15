import numpy as np
from .base_types import BaseNode, BaseMetric, BaseFragment
from .gcut_utils.distribution import Distribution
from ._utils import get_angle, get_gof


class EnsembleFragment(BaseFragment):

    def __init__(self):
        super().__init__()
        self.path_len = 0.


class EnsembleFragmentNode(BaseNode):
    """
    The dist will be used for global cost calculation.
    """

    def __init__(self, id):
        super().__init__(id)
        self.gof_cost = 0.  # the gof of this fragment w.r.t. a soma
        self.gap_cost = 0.
        self.a_cost = 0.
        self.r_cost = 0.
        self.path_dist = 0.  # the path distance to soma
        self.tot_cost = 0.
        self.weights = np.array([0., 0., 0.])

    def update(self, node):
        self.path_dist = node.path_dist
        self.gof_cost = node.gof_cost
        self.gap_cost = node.gap_cost
        self.a_cost = node.a_cost
        self.r_cost = node.r_cost
        self.tot_cost = node.tot_cost
        self.dij_cost = node.dij_cost
        self.parent = node.parent
        self.reverse = node.reverse
        self.weights = node.weights


class EnsembleMetric(BaseMetric):
    def __init__(self, gof_weight=1., angle_weight=1., radius_weight=1., gap_weight=3., anchor_dist=5.,
                 branch_len_norm=10., distribution=Distribution(), soma_radius=10., epsilon=1e-7):
        """
        :param gof_weight: the weight of the global gof metric
        :param angle_weight: the weight of the local angle metric
        :param radius_weight: the weight of the local radius metric
        :param anchor_dist: the distance to calculate fragment angle
        :param branch_len_norm: the (expected) average branch length, used to scale the probability of gof
        :param distribution: GOF distribution class, should load the data before use
        :param epsilon: for preventing zero division
        """
        self._gof_dist = distribution
        self._weights = np.array((angle_weight, radius_weight, gof_weight, gap_weight))
        self._anchor_dist = anchor_dist
        self._norm_bl = branch_len_norm
        self._soma_radius = soma_radius
        self._eps = epsilon
        distribution.load_distribution()

    def init_fragment(self, cut, frag: EnsembleFragment):
        # calculate path length
        pts_list = np.array([cut.swc[i][2:5] for i in frag.nodes])
        frag.path_len = np.linalg.norm((pts_list[1:] - pts_list[:-1]) * cut.res, axis=1).sum()

    def __call__(self, cut, soma, parent, child, reverse):
        """
        Calculating cost on a fragment when it's attached to a parent.

        :param cut: E-Cut object.
        :param soma: the source soma ID.
        :param parent: parent fragment ID.
        :param child: child fragment ID.
        :param reverse: if reverse to the original direction of this fragment.
        :return: costs as a dict.
        """
        test = EnsembleFragmentNode(-1)
        test.reverse = reverse
        test.parent = parent

        pts_par, radius_par, par_len = self._path_upstream(cut, soma, parent)
        pts_ch, radius_ch, ch_len = self._path_within(cut, child, not reverse)
        par_node: EnsembleFragmentNode = cut.fragment_trees[soma][parent]
        frag_ch: EnsembleFragment = cut.fragment[child]
        test.path_dist = par_node.path_dist + frag_ch.path_len

        gap = np.linalg.norm((pts_par[0] - pts_ch[0]) * cut.res)

        a1, a2 = get_angle(pts_par, pts_ch, cut.res)
        test.a_cost = a1 / np.pi
        test.gap_cost = a2 / np.pi
        test.r_cost = max(max(radius_ch) - np.median(radius_par), 0) / np.mean(radius_ch)

        gof_prob = self._gof_dist.probability(get_gof(pts_ch, cut.swc[soma][2:5], cut.res, self._eps))
        # probability decay by distance
        test.gof_cost = 1 - gof_prob * min(1, np.log(1 + 1 / (par_node.path_dist / self._norm_bl + self._eps)))

        # short fragments are less confident for radius and angle calculation
        # square, as the par_len usually equals the anchor dist, but ch_len can be short
        conf1 = par_len * ch_len / self._anchor_dist ** 2
        # weighted by the ch_len, but normalized by the path distance to soma
        # gof is less effective in shorter and farther fragments
        conf2 = frag_ch.path_len / test.path_dist
        # gap cost is angles alleviated by the gap dist
        conf3 = max(gap, 0) / max(radius_par)
        # near soma, the cost should be low
        test.weights = self._weights * (conf1, conf1, conf2, conf3) * min(1., par_node.path_dist / self._soma_radius)
        test.tot_cost = test.weights.dot((test.a_cost, test.r_cost, test.gof_cost, test.gap_cost)) * frag_ch.path_len
        test.dij_cost = par_node.dij_cost + test.tot_cost
        return test

    def _path_upstream(self, cut, soma: int, frag_id: int):
        """
        Construct a list of nodes based on an existing fragment tree, starting from current fragment
        towards the soma of the fragment tree.
        """
        path_dist = 0
        fragment_tree = cut.fragment_trees[soma]
        frag_node = fragment_tree[frag_id]
        pts_list = []
        radius_list = []
        while path_dist < self._anchor_dist:  # stop when exceeding the anchor dist
            # nodes in a fragment start from child to parent in the original tree
            # the path needs to start from a far end whenever
            a, b, c = self._path_within(cut, frag_node.id, frag_node.reverse, path_dist)
            pts_list.extend(a)
            radius_list.extend(b)
            path_dist += c
            # finish one fragment and get its parent
            p = frag_node.parent
            if p == -1:
                break
            frag_node = fragment_tree[p]
        return pts_list, radius_list, path_dist

    def _path_within(self, cut, frag_id: int, reverse: bool, path_dist=0.):
        """
        Construct a list of nodes departing from the soma within current fragment.
        The direction is explicitly given.
        """
        pts_list = []
        radius_list = []
        nodes = cut.fragment[frag_id].nodes
        if reverse:
            nodes = reversed(nodes)
        for i in nodes:
            pts_list.append(np.array(cut.swc[i][2:5]))
            radius_list.append(cut.swc[i][5])
            if len(pts_list) > 1:
                path_dist += np.linalg.norm((pts_list[-2] - pts_list[-1]) * cut.res)
            if path_dist > self._anchor_dist:
                break  # stop when exceeding the anchor dist
        return pts_list, radius_list, path_dist

