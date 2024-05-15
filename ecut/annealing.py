"""
Anneal refers to a heat treatment process that alters the properties of a material to increase its ductility
(the ability to deform without breaking) and reduce its hardness.

Neuron morphology in graph formats can suffer from strange branching patterns that cause inaccurate angle and branch
length estimation. In this case, this module is designed to restore the correct morphology by proofing these structures.

This process is also similar to all-path-pruning, except that the latter can control segments that can never be pruned
correctly in extreme cases.
"""
from .base_types import ListNeuron
from .morphology import Morphology
from ._queue import PriorityQueue
from .swc_handler import get_child_dict
import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree


class MorphAnneal:
    def __init__(self, swc: ListNeuron, radius_gap_factor=1., min_step=.2, step_size=.1,
                 epsilon=1e-7, res=(.3, .3, 1), drop_len=5.):
        self.morph = Morphology(swc)
        self._eps = epsilon
        self._step_size = step_size
        self._radius_gap_factor = radius_gap_factor
        self._min_step = min_step
        self._res = np.array(res)
        self._drop_len = drop_len
        self.new_swc = self.seg_tree = None

    def _dist(self, p1, p2=(0, 0, 0), axis=None):
        return np.linalg.norm((np.array(p1) - np.array(p2)) * self._res, axis=axis)

    def _get_seg_len(self, nodes: list[int]):
        pts1 = [self.new_swc[i][2:5] for i in nodes]
        p = self.new_swc[nodes[0]][6]
        if p == -1:
            pts2 = pts1[:-1]
            pts1 = pts1[1:]
            if len(pts1) == 0:
                return 0
        else:
            pts2 = [self.new_swc[p][2:5], *pts1[:-1]]
        tot_len = self._dist(pts2, pts1, axis=-1).sum()
        return tot_len

    def _get_interp(self, nodes: list[int], root=None):
        if root is not None:
            nodes = [root, *nodes]
        else:
            nodes = [self.new_swc[nodes[0]][6], *nodes]
        rad = np.array([self.new_swc[i][5] for i in nodes])
        pos = np.array([self.new_swc[i][2:5] for i in nodes])
        lengths = self._dist(pos[1:], pos[:-1], axis=-1)
        pick = [True, *(lengths > self._eps)]
        if sum(pick) < 2:
            return None, None
        rad = rad[pick]
        pos = pos[pick]
        cum_len = [0, *np.cumsum(lengths[pick[1:]])]
        if len(cum_len) > 2:
            pos_interp = interp1d(cum_len, pos, 'quadratic', 0, fill_value='extrapolate')
            rad_interp = interp1d(cum_len, rad, 'quadratic', 0, fill_value='extrapolate')
        else:
            pos_interp = interp1d(cum_len, pos, 'linear', 0, fill_value='extrapolate')
            rad_interp = interp1d(cum_len, rad, 'linear', 0, fill_value='extrapolate')
        return pos_interp, rad_interp, np.cumsum(lengths)

    def _try_merge(self, pos_interp1, rad_interp1, cum_len1, pos_interp2, rad_interp2, cum_len2):
        step = min(cum_len1[-1], cum_len2[-1])
        while step > 0:
            p1 = pos_interp1(step)
            p2 = pos_interp2(step)
            r1 = rad_interp1(step) * self._res.dot((.5,.5,0))
            r2 = rad_interp2(step) * self._res.dot((.5,.5,0))
            gap = self._dist(p1, p2) - r1 - r2
            if gap < self._radius_gap_factor * (r1 + r2):
                break
            step -= self._step_size
        i = j = -1
        for c in cum_len1:
            if c > step:
                break
            i += 1
        for c in cum_len2:
            if c > step:
                break
            j += 1
        return step, i, j

    def _annotate_length(self, seg_tree: ListNeuron, seg_dict: dict[int, list[int]]):
        """
        :param seg_tree: end node id -> nodes on the segment
        :param seg_dict: end node id tree
        :return: a new seg_tree, containing node list and length, with root of inf length.
        """
        seg_tree2 = {}
        child = get_child_dict(seg_tree)
        for t in seg_tree:
            key = t[0]
            v = seg_tree2[key] = list(t)
            x = seg_dict[key][::-1] + [key]
            v.append(x)
            v.append(self._get_seg_len(x))
            if key in child:
                v.append(set(child[key]))
            else:
                v.append(set())
        return seg_tree2

    def _commit_merge(self, branch1, branch2, stop1, stop2):
        par_branch = self.seg_tree[branch1][6]
        nodes1 = self.seg_tree[branch1][7]
        nodes2 = self.seg_tree[branch2][7]
        if stop1 == -1 or stop2 == -1:
            stop1 = max(stop1, 0)
            stop2 = max(stop2, 0)
            p1 = self.new_swc[nodes1[stop1]][2:5]
            p2 = self.new_swc[nodes2[stop2]][2:5]
            r1 = self.new_swc[nodes1[stop1]][5] * self._res.dot((.5,.5,0))
            r2 = self.new_swc[nodes2[stop2]][5] * self._res.dot((.5,.5,0))
            gap = self._dist(p1, p2) - r1 - r2
            if gap > self._radius_gap_factor * (r1 + r2):
                return set()

        kd = KDTree([self.new_swc[i][2:5] for i in nodes1[:stop1 + 1]])
        for i in range(stop2 + 1):
            i = nodes2[i]
            pos = np.array(self.new_swc[i][2:5])
            d, j = kd.query([pos])
            d = d[0][0]
            j = nodes1[j[0][0]]
            r1 = self.new_swc[i][5] * self._res.dot((.5,.5,0))
            r2 = self.new_swc[j][5] * self._res.dot((.5,.5,0))
            self.new_swc[i][2:5] = (pos * r1 + np.array(self.new_swc[j][2:5]) * r2) / (r1 + r2)
            self.new_swc[i][5] = max(r1, r2) / self._res.dot((.5,.5,0))
        # nullify the covered nodes
        for i in range(stop1 + 1):
            i = nodes1[i]
            self.new_swc[i][6] = -2
        self.seg_tree[par_branch][9].remove(branch1)
        merge_parent_branch = len(self.seg_tree[par_branch][9]) < 2

        if stop1 + 1 < len(nodes1) or len(self.seg_tree[branch1][9]) > 0:
            # the remaining of branch1 has to go to the sibling
            to_append = nodes2[stop2]
            if to_append != branch2:      # the anneal point has to be new branch, or to_append == branch2
                temp = nodes2[:stop2 + 1]
                if merge_parent_branch:   # the parent branch can be updated!
                    seg = self.seg_tree[to_append] = self.seg_tree[par_branch]
                    pp = self.seg_tree[par_branch][6]
                    if pp != -1:
                        self.seg_tree[pp][9].remove(par_branch)
                        self.seg_tree[pp][9].add(to_append)
                    del self.seg_tree[par_branch]
                    seg[0] = to_append
                    seg[7].extend(temp)
                    seg[8] = self._get_seg_len(seg[7])
                    self.seg_tree[branch2][6] = to_append
                else:
                    self.seg_tree[to_append] = [*self.new_swc[to_append][:6], par_branch, temp, self._get_seg_len(temp), {branch2}]
                    self.seg_tree[branch2][6] = to_append
                    self.seg_tree[par_branch][9].remove(branch2)
                    self.seg_tree[par_branch][9].add(to_append)
                temp = self.seg_tree[branch2][7] = nodes2[stop2 + 1:]
                self.seg_tree[branch2][8] = self._get_seg_len(temp)
            elif merge_parent_branch:
                # also check if branch2 can be merged to parent
                pp = self.seg_tree[to_append][6] = self.seg_tree[par_branch][6]
                temp = self.seg_tree[to_append][7] = self.seg_tree[par_branch][7] + self.seg_tree[to_append][7]
                self.seg_tree[to_append][8] = self._get_seg_len(temp)
                if pp != -1:
                    self.seg_tree[pp][9].remove(par_branch)
                    self.seg_tree[pp][9].add(to_append)
                del self.seg_tree[par_branch]
            # append branch1 stop1's child to sibling
            if stop1 + 1 < len(nodes1):
                # branch1 isn't over: keep the branch, update it
                self.seg_tree[branch1][6] = self.new_swc[nodes1[stop1 + 1]][6] = to_append
                temp = self.seg_tree[branch1][7] = nodes1[stop1 + 1:]
                self.seg_tree[branch1][8] = self._get_seg_len(temp)
                self.seg_tree[to_append][9].add(branch1)
                # need to check if branch1 and to_append can be merged
                if len(self.seg_tree[to_append][9]) < 2:
                    pp = self.seg_tree[branch1][6] = self.seg_tree[to_append][6]
                    if pp != -1:
                        self.seg_tree[pp][9].remove(to_append)
                        self.seg_tree[pp][9].add(branch1)
                    temp = self.seg_tree[branch1][7] = self.seg_tree[to_append][7] + self.seg_tree[branch1][7]
                    self.seg_tree[branch1][8] = self._get_seg_len(temp)
                    del self.seg_tree[to_append]
                    return {branch1}
                else:
                    return {to_append, branch1, branch2}
            else:
                assert len(self.seg_tree[branch1][9]) > 1
                # branch1 is over but has child segment: remove branch1
                for i in self.seg_tree[branch1][9]:
                    self.seg_tree[i][6] = self.new_swc[self.seg_tree[i][7][0]][6] = to_append
                    self.seg_tree[to_append][9].add(i)
                del self.seg_tree[branch1]
                return self.seg_tree[to_append][9] | {to_append}
        else:
            # here branch1 is fully combined with branch2
            del self.seg_tree[branch1]
            if merge_parent_branch:
                # also check if branch2 can be merged to parent
                pp = self.seg_tree[branch2][6] = self.seg_tree[par_branch][6]
                temp = self.seg_tree[branch2][7] = self.seg_tree[par_branch][7] + self.seg_tree[branch2][7]
                self.seg_tree[branch2][8] = self._get_seg_len(temp)
                if pp != -1:
                    self.seg_tree[pp][9].remove(par_branch)
                    self.seg_tree[pp][9].add(branch2)
                del self.seg_tree[par_branch]
            return {branch2}

    def _commit_merge_daughter(self, branch1, stop1, stop2):
        """

        :param branch1: the daughter branch
        :param stop1: the stop pos on the daughter branch
        :param stop2: the stop pos on the parent branch, reversed, starting from last but 1 node.
        :return:
        """

        branch2 = par_branch = self.seg_tree[branch1][6]      # the parent branch
        pp = self.seg_tree[par_branch][6]           # the parent node of parent branch
        nodes1 = self.seg_tree[branch1][7]          # the nodes of daughter branch
        nodes2 = self.seg_tree[par_branch][7][-2::-1] + [pp]    # the nodes of parent branch, reversed

        if stop1 == -1 or stop2 == -1:      # the stop is before the first point
            stop1 = max(stop1, 0)
            stop2 = max(stop2, 0)
            p1 = self.new_swc[nodes1[stop1]][2:5]
            p2 = self.new_swc[nodes2[stop2]][2:5]
            r1 = self.new_swc[nodes1[stop1]][5] * self._res.dot((.5,.5,0))
            r2 = self.new_swc[nodes2[stop2]][5] * self._res.dot((.5,.5,0))
            gap = self._dist(p1, p2) - r1 - r2
            if gap > self._radius_gap_factor * (r1 + r2):
                return set()

        # merge, modify the parent branch nodes
        kd = KDTree([self.new_swc[i][2:5] for i in nodes1[:stop1 + 1]])
        for i in range(stop2 + 1):
            i = nodes2[i]
            pos = np.array(self.new_swc[i][2:5])
            d, j = kd.query([pos])
            d = d[0][0]
            j = nodes1[j[0][0]]
            r1 = self.new_swc[i][5] * self._res.dot((.5,.5,0))
            r2 = self.new_swc[j][5] * self._res.dot((.5,.5,0))
            self.new_swc[i][2:5] = (pos * r1 + np.array(self.new_swc[j][2:5]) * r2) / (r1 + r2)
            self.new_swc[i][5] = max(r1, r2) / self._res.dot((.5,.5,0))
        # nullify the covered nodes
        for i in range(stop1 + 1):
            i = nodes1[i]
            self.new_swc[i][6] = -2
        self.seg_tree[par_branch][9].remove(branch1)
        merge_parent_branch = len(self.seg_tree[par_branch][9]) < 2

        if stop1 + 1 < len(nodes1) or len(self.seg_tree[branch1][9]) > 0:
            # the remaining of branch1 has to go to the sibling

            to_append = nodes2[stop2]
            if to_append != pp:      # the anneal point has to be new branch, or to_append == branch2
                temp = self.seg_tree[par_branch][7][:-stop2-1]
                if merge_parent_branch:   # the parent branch can be updated!
                    branch2 = list(self.seg_tree[par_branch][9])[0]  # the other daughter branch
                    self.seg_tree[branch2][6] = to_append
                    self.seg_tree[branch2][7] = self.seg_tree[par_branch][7][-stop2-1:] + self.seg_tree[branch2][7]
                    self.seg_tree[branch2][8] = self._get_seg_len(self.seg_tree[branch2][7])
                    seg = self.seg_tree[to_append] = self.seg_tree[par_branch]
                    if pp != -1:
                        self.seg_tree[pp][9].remove(par_branch)
                        self.seg_tree[pp][9].add(to_append)
                    del self.seg_tree[par_branch]
                    seg[0] = to_append
                    seg[7] = temp
                    seg[8] = self._get_seg_len(seg[7])
                else:
                    self.seg_tree[to_append] = [*self.new_swc[to_append][:6], pp, temp, self._get_seg_len(temp), {par_branch}]
                    self.seg_tree[par_branch][6] = to_append
                    self.seg_tree[pp][9].remove(par_branch)
                    self.seg_tree[pp][9].add(to_append)
                    temp = self.seg_tree[branch2][7] = self.seg_tree[branch2][7][-stop2-1:]
                    self.seg_tree[branch2][8] = self._get_seg_len(temp)
            elif merge_parent_branch:       # no need to make new branch for to_append
                branch2 = list(self.seg_tree[par_branch][9])[0]     # the other daughter branch
                # also check if branch2 can be merged to parent
                self.seg_tree[branch2][6] = pp
                temp = self.seg_tree[branch2][7] = self.seg_tree[par_branch][7] + self.seg_tree[branch2][7]
                self.seg_tree[branch2][8] = self._get_seg_len(temp)
                if pp != -1:
                    self.seg_tree[pp][9].remove(par_branch)
                    self.seg_tree[pp][9].add(branch2)
                del self.seg_tree[par_branch]

            # append branch1 stop1's child to sibling
            if stop1 + 1 < len(nodes1):
                # branch1 isn't over: keep the branch, update it
                self.seg_tree[branch1][6] = self.new_swc[nodes1[stop1 + 1]][6] = to_append
                temp = self.seg_tree[branch1][7] = nodes1[stop1 + 1:]
                self.seg_tree[branch1][8] = self._get_seg_len(temp)
                self.seg_tree[to_append][9].add(branch1)
                return {to_append, branch1, branch2}
            else:
                assert len(self.seg_tree[branch1][9]) > 1
                # branch1 is over but has child segment: remove branch1
                for i in self.seg_tree[branch1][9]:
                    self.seg_tree[i][6] = self.new_swc[self.seg_tree[i][7][0]][6] = to_append
                    self.seg_tree[to_append][9].add(i)
                del self.seg_tree[branch1]
                return self.seg_tree[to_append][9] | {to_append}

        else:
            # here branch1 is fully combined with parental branch
            del self.seg_tree[branch1]
            if merge_parent_branch:
                branch2 = list(self.seg_tree[par_branch][9])[0]     # the other doughter branch
                # also check if branch2 can be merged to parent
                pp = self.seg_tree[branch2][6] = self.seg_tree[par_branch][6]
                temp = self.seg_tree[branch2][7] = self.seg_tree[par_branch][7] + self.seg_tree[branch2][7]
                self.seg_tree[branch2][8] = self._get_seg_len(temp)
                if pp != -1:
                    self.seg_tree[pp][9].remove(par_branch)
                    self.seg_tree[pp][9].add(branch2)
                del self.seg_tree[par_branch]
                return {branch2}
            return {par_branch}

    def run(self):
        # prepare data
        self.new_swc = dict((t[0], list(t)) for t in self.morph.tree)
        self.seg_tree = self._annotate_length(*self.morph.convert_to_topology_tree())
        # 6: parent, 7: nodes, 8: length, 9: children

        # init
        pq = PriorityQueue()
        for k, v in self.seg_tree.items():
            pq.add_task(k, v[8])        # ordered by their length

        # bfs: merge segments from short to long
        while not pq.empty():
            head = pq.pop_task()
            if head is None or head not in self.seg_tree:
                continue

            # get the sibling to merge with
            p = self.seg_tree[head][6]
            if p == -1:
                continue
            sib = [c for c in self.seg_tree[p][9] if c != head]
            if len(sib) == 0:
                pp = self.seg_tree[head][6] = self.seg_tree[p][6]
                temp = self.seg_tree[head][7] = self.seg_tree[p][7] + self.seg_tree[head][7]
                self.seg_tree[head][8] = self._get_seg_len(temp)
                if pp != -1:
                    self.seg_tree[pp][9].remove(p)
                    self.seg_tree[pp][9].add(head)
                continue
            updated = set()
            head_interp = self._get_interp(self.seg_tree[head][7])
            if head_interp[0] is None:
                lengths = [self.seg_tree[c][8] for c in sib]     # get all lengths of sibling segments
                to_merge = sib[np.argmin(lengths)]          # choose the shortest to merge with
                updated = self._commit_merge(head, to_merge, len(self.seg_tree[head][7]) - 1, 0)
            else:
                # try merge all siblings (and parent)
                steps = []
                j1 = []
                j2 = []
                par = self.seg_tree[head][6]
                pcomp = False
                if self.seg_tree[par][6] != -1 and len(self.seg_tree[par][7]) > 0:
                    pcomp = True
                    pn = self.seg_tree[par][7][-2::-1] + [self.seg_tree[par][6]]
                    par_interp = self._get_interp(pn, par)
                    step, i1, i2 = self._try_merge(*head_interp, *par_interp)
                    steps.append(step)
                    j1.append(i1)
                    j2.append(i2)
                for s in sib:
                    sib_interp = self._get_interp(self.seg_tree[s][7])
                    # sib must be longer than head, so no need to worry...
                    step, i1, i2 = self._try_merge(*head_interp, *sib_interp)
                    steps.append(step)
                    j1.append(i1)
                    j2.append(i2)
                merge_ord = np.argsort(steps)[::-1]
                for ord in merge_ord:
                    if steps[ord] >= self._min_step:
                        if pcomp and ord == 0:
                            updated = self._commit_merge_daughter(head, j1[ord], j2[ord])
                        else:
                            if pcomp:
                                ss = sib[ord - 1]
                            else:
                                ss = sib[ord]
                            updated = self._commit_merge(head, ss, j1[ord], j2[ord])
                        break

            for i in updated:       # successfully commit
                # check the length
                pq.add_task(i, self.seg_tree[i][8])

        for v in self.seg_tree.values():
            if v[8] < self._drop_len and len(v[9]) == 0:
                for i in v[7]:
                    self.new_swc[i][6] = -2

        out_swc = []
        for v in self.new_swc.values():
            if v[6] != -2:
                out_swc.append(tuple(v))

        return out_swc