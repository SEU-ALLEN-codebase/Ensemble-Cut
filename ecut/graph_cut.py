import numpy as np
from .morphology import Morphology
from .swc_handler import get_child_dict
from sklearn.neighbors import KDTree
from ._queue import PriorityQueue
from .base_types import BaseCut, ListNeuron
from .graph_metrics import EnsembleMetric, EnsembleFragmentNode, EnsembleFragment


# class Fragmentation:
#     """
#     Turn swc into a fragment graph. In future this function would be seperated from the graph cut
#     """
#     def __init__(self, tree, res=(1., 1., 1.)):
#         self._res = np.array(res)
#         self._morph = Morphology(tree)
#
#     def get_adjacency(self, search_dist=30., up_dist=10., gap_ratio=5., path_euc_thr=3.):
#         all_trails = {}
#         n2trail = {}
#         for t in self._morph.tips:
#             trail = [t]
#             n2trail[t] = [(t, 0)]
#             pn = self._morph.pos_dict[t][6]
#             while pn != self._morph.p_soma:
#                 trail.append(pn)
#                 if pn not in n2trail:
#                     n2trail[pn] = []
#                 n2trail[pn].append((trail[0], len(n2trail[pn])))
#                 if pn not in self._morph.unifurcation:
#                     all_trails[trail[0]] = trail
#                     if pn in all_trails or pn == self._morph.idx_soma:
#                         break
#                     trail = [pn]
#                     n2trail[pn].append((pn, len(n2trail[pn])))
#                 pn = self._morph.pos_dict[pn][6]
#         kd = KDTree([t[2:5] * self._res for t in self._morph.tree])
#         pos = [self._morph.pos_dict[p][2:5] * self._res for p in self._morph.tips]
#
#         # search through all the branches
#         new_conn = {} # adjacency
#         for inds, dists, tip in zip(*kd.query_radius(pos, search_dist, return_distance=True), self._morph.tips):
#             tr_this = n2trail[tip]      # tip trail
#             p = self._morph.pos_dict[k][6]
#             r1 = self._morph.pos_dict[k][5]
#             while p != -1 and np.linalg.norm(
#                     (np.array(self._morph.pos_dict[k][2:5]) - self._morph.pos_dict[p][2:5]) * self._res) < up_dist:
#                 r1 = max(r1, self._morph.pos_dict[p][5])
#                 p = self._morph.pos_dict[p][6]
#             r1 *= v
#
#             for i, d in zip(inds, dists):
#                 t = self._morph.pos_dict[i]
#                 if t[0] == tip:     # same node
#                     continue
#                 tr = n2trail[t]
#                 if len(tr) > 1:     # branch node
#
#                     if t not in new_conn:
#                         new_conn[t] = []
#                     new_conn[t].append(tip)
#                     continue
#                 tr = tr[0]      # unifurcation
#                 if tr[0] == tr_this[0]:     # belong to the same trail
#                     continue


class Adjacency:
    def __init__(self, tree: ListNeuron, search_dist=30., up_dist=10., gap_ratio=5., path_euc_thr=3., res=(1., 1., 1.)):
        """
        Generate an adjacency map from the current tree. Parent and children are excluded.

        :return: the adjacency dictionary
        """
        res = np.array(res)
        morph = Morphology(tree)

        def get_path(n1, n2):
            t1 = [n1]
            t2 = [n2]
            while morph.pos_dict[t1[-1]][6] != -1:
                t1.append(morph.pos_dict[t1[-1]][6])
                if t1[-1] == n2:
                    return t1
            while morph.pos_dict[t2[-1]][6] != -1:
                t2.append(morph.pos_dict[t2[-1]][6])
                if t2[-1] == n1:
                    return t2
            while t1[-1] == t2[-1]:
                t1.pop(-1)
                t2.pop(-1)
            t2.reverse()
            return t1 + t2

        def get_path_distance(n1, n2):
            path = get_path(n1, n2)
            pts = [morph.pos_dict[t][2:5] for t in path]
            return np.linalg.norm((np.array(pts[1:]) - pts[:-1]) * res)

        adjacency = {}
        keys = list(morph.pos_dict.keys())
        for k in keys:
            adjacency[k] = set()
        kd = KDTree([morph.pos_dict[k][2:5] * res for k in keys])
        v = res.dot((.5, .5, 0))
        for k in morph.tips:
            p = morph.pos_dict[k][6]
            r1 = morph.pos_dict[k][5]
            while p != -1 and np.linalg.norm((np.array(morph.pos_dict[k][2:5]) - morph.pos_dict[p][2:5]) * res) < up_dist:
                r1 = max(r1, morph.pos_dict[p][5])
                p = morph.pos_dict[p][6]
            r1 *= v
            candid = []
            inds, dists = kd.query_radius([morph.pos_dict[k][2:5]] * res, search_dist, return_distance=True)
            for i, d in zip(inds[0], dists[0]):
                i = keys[i]
                r2 = morph.pos_dict[i][5] * v
                if k != i and get_path_distance(k, i) > d * path_euc_thr and d < gap_ratio * (r1 + r2):
                    candid.append(i)
            if k not in adjacency:
                adjacency[k] = set(candid)
            else:
                adjacency[k] |= set(candid)
            for i in candid:
                if i not in adjacency:
                    adjacency[i] = set()
                adjacency[i].add(k)
        self._adj = adjacency

    def __getitem__(self, item):
        return self._adj[item]


class ECut(BaseCut):
    """
    ECut differs from G-Cut in the following aspects:

    1. We allow more mechanisms to be considered, such as angle.

    2. We use fragment instead of branch as the basis for linear programming. It allows us to more naturally process
    somata and breakups.
    """

    def __init__(self, swc: ListNeuron, soma: list[int], adjacency: Adjacency, metric=EnsembleMetric(),
                 res=(.3, .3, 1.), *args, **kwargs):
        """

        :param swc: swc tree, whose id should match the line number
        :param soma: list of soma nodes, must match the tree node id
        :param children: children dict of the swc tree,
        :param adjacency: close non-connecting neighbours
        :param metric: the metric to compute the cost on each fragment
        """
        super().__init__(swc, soma, res, *args, **kwargs)
        self._metric = metric
        self._children = self._get_children()
        self._adjacency = adjacency
        self._end2frag: dict[int, set[int]] | None = None

    def _get_children(self) -> dict[int, set]:
        """
        Generate a children dict from the current tree. Elements are sets, tips are empty.
        """
        children = get_child_dict(list(self._swc.values()))
        for t in self._swc.values():
            if t[0] in children:
                children[t[0]] = set(children[t[0]])
            else:
                children[t[0]] = set()
        return children

    def run(self):
        if self._verbose:
            print("Starting G-Cut.")
        self._extract_fragment()
        self._resolve_fragment_tree()
        self._linear_programming()
        if self._verbose:
            print("Finished G-Cut.")

    def _extract_fragment(self):
        """
        Extract fragment from the swc tree.

        The connection between the fragments is defined by both swc and adjacency.
        """
        self._fragment = {}       # fragments are indexed by their last node
        self._end2frag = {}     # how each end node is mapped to the fragment
        for k, v in self._children.items():
            if len(v) > 0:     # tip nodes, start fragment recording
                continue
            new_frag = self._fragment[k] = EnsembleFragment()
            new_frag.nodes.append(k)
            self._end2frag[k] = {k}     # map the ends to their fragment, non-unique
            p = self._swc[k][6]
            while p != -1:
                new_frag.nodes.append(p)       # we add the fragment node to the fragment as well
                if len(self._children[p]) > 1 or len(self._adjacency[p]) > 0 or p in self._soma:
                    # start a new recording when it meets:
                    # a real branch node, close but non-connected nodes or soma
                    if p in self._end2frag:     # meeting the end of an already searched fragment, add and leave
                        self._end2frag[p].add(k)
                        break
                    # proceed to construct the next fragment
                    self._end2frag[p] = {k, p}       # also add the next fragment
                    k = p
                    new_frag = self._fragment[k] = EnsembleFragment()
                    new_frag.nodes.append(k)
                p = self._swc[p][6]
            else:
                # a special case for a tree root
                if len(new_frag.nodes) < 2:
                    del self._fragment[k]
                    self._end2frag[k].remove(k)
                else:
                    p = new_frag.nodes[-1]
                    if p not in self._end2frag:
                        self._end2frag[p] = {k}
                    else:
                        self._end2frag[p].add(k)
        # extract connectivity among fragments
        # a fragment can be connected at two ends
        # and non-connecting but close fragment ends will also be considered
        for k, v in self._fragment.items():
            end1 = v.nodes[0]
            v.end1_adj = self._end2frag[end1] - {k}        # omit self
            for i in self._adjacency[end1]:     # find adjacent nodes
                v.end1_adj |= self._end2frag[i]     # add any frag related
            end2 = v.nodes[-1]
            v.end2_adj = self._end2frag[end2] - {k}
            for i in self._adjacency[end2]:
                v.end2_adj |= self._end2frag[i]
            self._metric.init_fragment(self, v)
        if self._verbose:
            print("Finished fragment extraction.")

    def _resolve_fragment_tree(self):
        """
        Build MST for each soma based on the fragments.

        Our simplification:
        In the original G-Cut, some branches can be determined directly from the topology, we do similar things here
        after retrieving the fragment tree from each soma, the fragments are marked with traversed sources
        fragments with only one soma are omitted in linear programming
        """
        self._fragment_trees = {}   # corresponding to the soma list
        for i in self._soma:
            self._prim(i)
        if self._verbose:
            print("Finished fragment MST construction for all soma.")

    def _prim(self, soma: int):
        """
        MST to build a fragment tree. this tree share the fragments with others, but with different soma, so the MST
        will be different because the fragment orientation and angle calculation change.

        In the meantime, since it traverses the graph from one soma, it can document the traversed path until meeting
        another soma--the original G-Cut requires that branches leaving other soma should be unique to that one, so they
        can be excluded.

        Note that our fragments are highly interconnected--we can't just leave the fragments as inf or
        there would be a roundabout. Instead, after meeting other soma, we should color the downstream structures.

        :param soma: the node index of the soma, this should be one end of the fragments
        :return: a fragment tree for the soma
        """

        # priority queue (can update the content dynamically)
        queue = PriorityQueue()

        # initialize the fragment tree
        fragment_tree = self._fragment_trees[soma] = {}
        other_soma = set(self._soma) - {soma}
        for i in self._fragment:
            fragment_tree[i] = EnsembleFragmentNode(i)

        # initialize the starting fragments (connected to the designated soma)
        for i in self._end2frag[soma]:
            cur_node = fragment_tree[i]
            queue.add_task(i, 0)
            # when soma is the far end of this fragment, invert
            cur_node.set(dij_cost=0, parent=-1, reverse=(soma == self._fragment[i].nodes[0]))

        # BFS
        while not queue.empty():
            cheapest = queue.pop_task()
            if cheapest is None:        # the queue is empty (when the tasks are all None)
                break
            cur_node = fragment_tree[cheapest]
            cur_node.visited = True       # popped by the priority queue are visited
            cur_frag = self._fragment[cheapest]
            if not cur_node.passing_other_soma:
                # only a controversial _fragment is considered for linear programming
                # it has more than 1 traversed soma without passing other soma
                cur_frag.traversed.add(soma)
            # update the cost of the rest of the fragment nodes, which is reflected in the priority queue
            for i in cur_frag.end2_adj if cur_node.reverse else cur_frag.end1_adj:
                # reversed fragment the connection is on end2
                # search far ends
                next_node = fragment_tree[i]
                next_frag = self._fragment[i]
                if next_node.visited:
                    continue
                # here a fragment can be connected on both ends, a rare but possible case
                # if the connection is on end1, means the next fragment is reversed
                if cheapest in next_frag.end1_adj:
                    test = self._metric(self, soma, cheapest, i, reverse=True)
                    if test.dij_cost < next_node.dij_cost:
                        # if the cost is lower, update the fragment node
                        # it's on end1, far end, so invert
                        next_node.update(test)
                        queue.add_task(i, test.dij_cost)
                        # if this fragment is leaving another soma
                        if next_frag.nodes[0] in other_soma:
                            next_node.passing_other_soma = True
                        else:
                            next_node.passing_other_soma = cur_node.passing_other_soma
                if cheapest in next_frag.end2_adj:
                    test = self._metric(self, soma, cheapest, i, reverse=False)
                    if test.dij_cost < next_node.dij_cost:
                        # it's on end2, near end, keep original
                        next_node.update(test)
                        queue.add_task(i, test.dij_cost)
                        if next_frag.nodes[-1] in other_soma:
                            next_node.passing_other_soma = True
                        else:
                            next_node.passing_other_soma = cur_node.passing_other_soma
            # update by extending the adjacency
            par_cheapest = cur_node.parent
            for i in cur_frag.end1_adj if cur_node.reverse else cur_frag.end2_adj:
                # reversed fragment the connection is on end2
                # search far ends
                next_node = fragment_tree[i]
                next_frag = self._fragment[i]
                if next_node.visited:
                    continue
                if cheapest in next_frag.end1_adj:
                    if par_cheapest == -1:
                        next_node.set(dij_cost=0, parent=-1, reverse=True)
                        queue.add_task(i, 0)
                    else:
                        test = self._metric(self, soma, par_cheapest, i, reverse=True)
                        if test.dij_cost < next_node.dij_cost:
                            # if the cost is lower, update the fragment node
                            # it's on end1, far end, so invert
                            next_node.update(test)
                            queue.add_task(i, test.dij_cost)
                            # if this fragment is leaving another soma
                            if next_frag.nodes[0] in other_soma:
                                next_node.passing_other_soma = True
                            else:
                                next_node.passing_other_soma = cur_node.passing_other_soma
                if cheapest in next_frag.end2_adj:
                    if par_cheapest == -1:
                        next_node.set(dij_cost=0, parent=-1, reverse=False)
                        queue.add_task(i, 0)
                    else:
                        test = self._metric(self, soma, par_cheapest, i, reverse=False)
                        if test.dij_cost < next_node.dij_cost:
                            # it's on end2, near end, keep original
                            next_node.update(test)
                            queue.add_task(i, test.dij_cost)
                            if next_frag.nodes[-1] in other_soma:
                                next_node.passing_other_soma = True
                            else:
                                next_node.passing_other_soma = cur_node.passing_other_soma

        if self._verbose:
            print(f"Finished fragment MST construction for soma {soma}.")
