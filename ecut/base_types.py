import numpy as np


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


class BaseNode:
    """
    A fragment node is a node representation of a fragment.
    This tree is a new tree regarding fragments in the original tree as its nodes.

    Therefore, this node has a direction. By default, it uses the direction in the original tree.

    The cost will be used for optimization. It also checks whether it has passed another soma.
    """

    def __init__(self, id):
        self.id = id
        self.parent = None
        self.reverse = False
        self.cost = np.inf
        self.visited = False
        self.passing_other_soma = False

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class BaseCut:
    def __init__(self, swc: list[tuple], soma: list[int]):
        self._swc = swc
        self._soma = soma
        self._fragment: dict[int, BaseFragment] = {}
        self._fragment_trees: dict[int, dict[int, BaseNode]] = {}

    @property
    def swc(self):
        return self._swc

    @property
    def fragment(self):
        return self._fragment

    @property
    def fragment_trees(self):
        return self._fragment_trees


class BaseMetric:

    def init_fragment(self, cut: BaseCut, frag: BaseFragment):
        """
        Initialize some of the properties of the fragment, e.g. path length

        :param cut: the class to perform graph cut
        :param frag: the fragment object to mutate
        """
        pass

    def __call__(self, cut: BaseCut, soma: int, frag_par: int, frag_ch: int, reverse: bool) -> dict:
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
