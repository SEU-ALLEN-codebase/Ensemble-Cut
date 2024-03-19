import numpy as np

from .swc_handler import get_child_dict
from sklearn.neighbors import KDTree
from ._queue import PriorityQueue
from .base_types import BaseCut
from .graph_metrics import EnsembleMetric, EnsembleNode, EnsembleFragment


class ECut(BaseCut):
    """
    ECut differs from G-Cut in the following aspects:

    1. We allow more mechanisms to be considered, such as angle.

    2. We use fragment instead of branch as the basis for linear programming. It allows us to more naturally process
    somata and breakups.

    """

    def __init__(self, swc: list[tuple], soma: list[int], children: dict[set] = None,
                 adjacency: dict[int, set] | float = 5., metric=EnsembleMetric(), *args, **kwargs):
        """

        :param swc: swc tree, whose id should match the line number
        :param soma: list of soma nodes, must match the tree node id
        :param children: children dict of the swc tree,
        :param adjacency: close non-connecting neighbours
        :param metric: the metric to compute the cost on each fragment
        """
        super().__init__(swc, soma, *args, **kwargs)
        self._metric = metric
        self._children = self._get_children() if children is None else children
        if isinstance(adjacency, dict):
            self._adjacency = adjacency
        else:
            self._adjacency = self._get_adjacency(adjacency)
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

    def _get_adjacency(self, dist: float) -> dict[int, set]:
        """
        Generate an adjacency map from the current tree. Parent and children are excluded.

        :param dist: the distance threshold to consider connection between 2 nodes.
        :return: the adjacency dictionary
        """
        kd = KDTree([t[2:5] for t in self._swc.values()])
        keys = np.array(list(self._swc.keys()))
        inds, dists = kd.query_radius([t[2:5] for t in self._swc.values()], dist, return_distance=True)
        adjacency = {}
        for k, i, d in zip(self._swc.values(), inds, dists):
            adjacency[k[0]] = set(keys[i[d < dist]]) - {k[0], k[6]} - self._children[k[0]]
        # ensure it's undirected graph
        for k, v in adjacency.items():
            for i in v:
                adjacency[i].add(k)
        if self._verbose:
            print("Adjacency computed.")
        return adjacency

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
            v.end1_adj =  self._end2frag[end1] - {k}        # omit self
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
            fragment_tree[i] = EnsembleNode(i)

        # initialize the starting fragments (connected to the designated soma)
        for i in self._end2frag[soma]:
            cur_node = fragment_tree[i]
            queue.add_task(i, 0)
            # when soma is the far end of this fragment, invert
            cur_node.update(cost=0, parent=-1, reverse=(soma == self._fragment[i].nodes[0]))

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
                    metrics = self._metric(self, soma, cheapest, i, reverse=True)
                    if metrics['cost'] < next_node.cost:
                        # if the cost is lower, update the fragment node
                        # it's on end1, far end, so invert
                        next_node.update(**metrics, parent=cheapest, reverse=True)
                        queue.add_task(i, metrics['cost'])
                        # if this fragment is leaving another soma
                        if next_frag.nodes[0] in other_soma:
                            next_node.passing_other_soma = True
                        else:
                            next_node.passing_other_soma = cur_node.passing_other_soma
                if cheapest in next_frag.end2_adj:
                    metrics = self._metric(self, soma, cheapest, i, reverse=False)
                    if metrics['cost'] < next_node.cost:
                        # it's on end2, near end, keep original
                        next_node.update(**metrics, parent=cheapest, reverse=False)
                        queue.add_task(i, metrics['cost'])
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
                        next_node.update(cost=0, parent=-1, reverse=True)
                        queue.add_task(i, 0)
                    else:
                        metrics = self._metric(self, soma, par_cheapest, i, reverse=True)
                        if metrics['cost'] < next_node.cost:
                            # if the cost is lower, update the fragment node
                            # it's on end1, far end, so invert
                            next_node.update(**metrics, parent=par_cheapest, reverse=True)
                            queue.add_task(i, metrics['cost'])
                            # if this fragment is leaving another soma
                            if next_frag.nodes[0] in other_soma:
                                next_node.passing_other_soma = True
                            else:
                                next_node.passing_other_soma = cur_node.passing_other_soma
                if cheapest in next_frag.end2_adj:
                    if par_cheapest == -1:
                        next_node.update(cost=0, parent=-1, reverse=False)
                        queue.add_task(i, 0)
                    else:
                        metrics = self._metric(self, soma, par_cheapest, i, reverse=False)
                        if metrics['cost'] < next_node.cost:
                            # it's on end2, near end, keep original
                            next_node.update(**metrics, parent=par_cheapest, reverse=False)
                            queue.add_task(i, metrics['cost'])
                            if next_frag.nodes[-1] in other_soma:
                                next_node.passing_other_soma = True
                            else:
                                next_node.passing_other_soma = cur_node.passing_other_soma

        if self._verbose:
            print(f"Finished fragment MST construction for soma {soma}.")
