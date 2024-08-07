import numpy as np
import pulp


ListNeuron = list[tuple[int, int, float, float, float, float, int]]


class BaseFragment:
    """
    A fragment is a series of connected non-branching nodes. We don't use branch in our computation for many reasons,
    one of which is that there can be breakup within a true branch, but mostly, fragments are branches.

    The 2 ends of a fragment can connect to multiple other nodes.

    It also records whether it is can be posses by multiple sources.
    """

    def __init__(self):
        self.nodes = []  # from end to start in the initial tree
        self.end1_adj = set()  # multiple other nodes, connected or close
        self.end2_adj = set()
        self.traversed = set()  # for the simplification of the problem
        self.source = {}   # the neuron it belongs to, source id -> likelihood


class BaseNode:
    """
    A fragment node is a node representation of a fragment.
    This tree is a new tree regarding fragments in the original tree as its nodes.

    Therefore, this node has a direction. By default, it uses the direction in the original tree.

    The cost will be used for optimization. It also checks whether it has passed another soma.
    """

    def __init__(self, id: int):
        self.id = id
        self.parent = None
        self.reverse = False
        self.dij_cost = np.inf
        self.visited = False
        self.passing_other_soma = False

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class BaseCut:
    def __init__(self, swc: ListNeuron, soma: list[int], res, likelihood_thr=None, verbose=False):
        """

        :param swc: swc tree
        :param soma: list of soma id
        :param res: resolution in x, y, z
        :param likelihood_thr: the minimum likelihood allowed for a fragment to be attached to a neuron, left as None
        to attach it to just the biggest. When multiple sources share a common or big enough likelihood, all of them will be considered.
        :param verbose:
        """
        self._verbose = verbose
        self._swc = dict([(t[0], t) for t in swc])
        self.res = np.array(res)
        self._soma = soma
        self._fragment: dict[int, BaseFragment] = {}
        self._fragment_trees: dict[int, dict[int, BaseNode]] = {}
        self._problem: pulp.LpProblem | None = None
        self._likelihood_thr: float = likelihood_thr

    @property
    def swc(self):
        return self._swc

    @property
    def fragment(self):
        return self._fragment

    @property
    def fragment_trees(self):
        return self._fragment_trees

    def export_swc(self, partition=True):
        """
        Export the result swc.

        :param partition: whether to bipartite the solution tree to multiple swc
        :return: an swc or a dict of swc
        """
        if not partition:
            tree = dict([(t[0], list(t)) for t in self._swc.values()])
            tag = dict(zip(self._soma, range(len(self._soma))))
            for frag in self._fragment.values():
                for i in frag.nodes:
                    a = list(frag.source.values())
                    b = list(frag.source.keys())
                    a = np.argmax(a)
                    tree[i][1] = tag[b[a]]
            tree = [tuple(t) for t in tree.values()]
            return tree

        trees = dict([(i, {(-1, 1): (1, *self._swc[i][1:6], -1)}) for i in self._soma])
        for frag_id, frag in self._fragment.items():
            candid = []
            a = list(frag.source.values())
            b = list(frag.source.keys())
            if self._likelihood_thr is None:    # max only mode
                m = None
                for i in np.argsort(a)[::-1]:
                    if m is not None and m > a[i]:
                        break
                    candid.append(b[i])
                    m = a[i]
            else:                               # thresholding mode, bigger than this will all be considered
                for i in np.argsort(a)[::-1]:
                    if a[i] < self._likelihood_thr:
                        break
                    candid.append(b[i])

            # for each candid source, append the frag nodes
            for src in candid:
                frag_node = self._fragment_trees[src][frag_id]
                nodes = self._fragment[frag_id].nodes
                if not frag_node.reverse:
                    nodes = nodes[::-1]
                par_frag_id = frag_node.parent
                if par_frag_id == -1:
                    last_id = -1, 1
                else:
                    par_frag_node = self._fragment_trees[src][par_frag_id]
                    par_nodes = self._fragment[par_frag_id].nodes
                    if par_frag_node.reverse:
                        last_id = par_frag_id, par_nodes[-1]
                    else:
                        last_id = par_frag_id, par_nodes[0]

                tree = trees[src]
                for i in nodes:
                    n = list(self._swc[i])
                    n[6] = last_id
                    n[0] = len(tree) + 1
                    tree[(frag_id, i)] = tuple(n)
                    last_id = frag_id, i

        for s, t in trees.items():
            for k, v in t.items():
                n = list(v)
                if n[6] != -1:
                    n[6] = t[n[6]][0]
                t[k] = tuple(n)
            trees[s] = list(t.values())
        return trees

    def _linear_programming(self):
        """
        Using linear programming to retrieve the weights of each node
        """
        self._problem = pulp.LpProblem('ECut', pulp.LpMinimize)

        # finding variables for fragment/soma pairs that require solving
        scores = {}      # var_i_s, i: fragment id, s: soma id
        for i, frag in self._fragment.items():
            if len(frag.traversed) > 1:  # mixed sources
                scores[i] = {}
                for s in frag.traversed:
                    scores[i][s] = pulp.LpVariable(f'Score_{i}_{s}', 0)        # non-negative
            elif len(frag.traversed) == 1:
                scores[i] = {}
                for s in frag.traversed:
                    scores[i][s] = pulp.LpVariable(f'Score_{i}_{s}', 1, 1)        # const
            else:
                pass

        # objective func: cost * score
        self._problem += pulp.lpSum(
            pulp.lpSum(
                self._fragment_trees[s][i].dij_cost * score for s, score in frag_vars.items()
            ) for i, frag_vars in scores.items()
        ), "Global Penalty"

        for i, frag_vars in scores.items():
            for s, score in frag_vars.items():
                if self._fragment_trees[s][i].dij_cost == np.inf:
                    print(self._fragment_trees[s][i].dij_cost == np.inf)
                    print(s, i, self.fragment[i].traversed, self._fragment_trees[s][i].parent, self._fragment_trees[s][i].dij_cost)

        # constraints
        for i, frag_vars in scores.items():
            self._problem += (pulp.lpSum(score for score in frag_vars.values()) == 1,
                              f"Membership Normalization for Fragment {i}")
            for s, score in frag_vars.items():
                p = self._fragment_trees[s][i].parent
                if p in scores:
                    self._problem += score <= scores[p][s], \
                        f"Tree Topology Enforcement for Score_{i}_{s}"

        self._problem.solve(pulp.PULP_CBC_CMD(msg=0))

        for frag in self._fragment.values():
            frag.source = dict.fromkeys(frag.traversed, 1)

        for variable in self._problem.variables():
            if 'dummy' in variable.name:
                continue
            frag_id, src = variable.name.split('_')[1:]
            frag_id, src = int(frag_id), int(src)
            frag = self._fragment[frag_id]
            assert src in frag.source
            frag.source[src] = variable.varValue

        if self._verbose:
            print("Finished linear programming.")


class BaseMetric:

    def init_fragment(self, cut: BaseCut, frag: BaseFragment):
        """
        Initialize some of the properties of the fragment, e.g. path length

        :param cut: the class to perform graph cut
        :param frag: the fragment object to mutate
        """
        pass

    def __call__(self, cut: BaseCut, soma: int, frag_par: int, frag_ch: int, reverse: bool):
        """
        Get the cost of the current fragment in a fragment tree.

        :param cut: the class to perform graph cut
        :param soma: the soma id of the current fragment tree
        :param frag_par: the parent fragment
        :param frag_ch: the child fragment
        :param reverse: whether to reverse the fragment
        :return: the computed metrics
        """
        pass
