import sys
import numpy as np
import os
import math
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d
from .morphology import Morphology


class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def region_gray_level(img, ct, win_radius):
    """
    if actual window is too small return -1
    """
    ind = [np.clip([a - b, a + b + 1], 0, c - 1) for a, b, c in zip(ct, win_radius, img.shape[-1:-4:-1])]
    ind = np.array(ind, dtype=int)
    stat = img[ind[2][0]:ind[2][1], ind[1][0]:ind[1][1], ind[0][0]:ind[0][1]].flatten()
    return stat.mean() if stat.size <= np.sum(win_radius) else -1


class ErrorPruning:

    def __init__(self, res=(.25, .25, 1.), soma_radius=10., anchor_reach=(2., 10.), gap_thr_ratio=1., epsilon=1e-7):
        """

        :param res: image resolution in micrometers, (x, y, z)
        :param soma_radius: expected soma radius in micrometers, within which errors are not counted
        :param anchor_reach: the distances of the anchor to the branch node, the anchor is meant to accurately
        estimate angles. (near, far). near: the near end of the anchor, far: the far end of the anchor,
        :param gap_thr_ratio:
        :param epsilon: value lower than this will be regarded as 0
        """
        self._res = res
        self._soma_radius = soma_radius
        self._near_anchor = anchor_reach[0]
        self._far_anchor = anchor_reach[1]
        self._gap_thr_ratio = gap_thr_ratio
        self._eps = epsilon

    def _length(self, p1, p2=(0, 0, 0), axis=None):
        if not isinstance(p1, np.ndarray):
            p1 = np.array(p1)
        return np.linalg.norm((p1 - p2) * self._res, axis=axis)

    def _vector_angles(self, p, ch):
        """
        modified from Jingzhou's code
        :param p: coordinate for the parent anchor
        :param ch: coordinates for the furcation anchors
        """
        vec_p = p * self._res
        vec_ch = [coord * self._res for coord in ch]
        cos_ch = [vec_p.dot(vec) / (np.linalg.norm(vec_p) * np.linalg.norm(vec)) for vec in vec_ch]
        out = [*map(lambda x: math.acos(max(min(x, 1), -1)) * 180 / math.pi, cos_ch)]
        return np.array(out)

    def _find_point(self, morph: Morphology, pt: np.ndarray, idx: np.ndarray, is_parent: bool,
                    dist_thr: float, pt_rad: float, return_center_point: bool):
        """
        Find the point of exact `dist` to the start pt on tree structure. args are:
        - pt: the start point, [coordinate]
        - anchor_idx: the first node on swc tree to trace, first child or parent node
        - is_parent: whether the anchor_idx is the parent of `pt`, otherwise child.
                     if a furcation points encounted, then break
        - morph: Morphology object for current tree
        - dist: distance threshold
        - return_center_point: whether to return the point with exact distance or
                     geometric point of all traced nodes
        """

        d = 0
        pts = [pt]
        rad = [pt_rad]
        while True:
            new_p = np.array(morph.pos_dict[idx][2:5])
            new_r = morph.pos_dict[idx][5]
            d += self._length(new_p, pts[-1])
            pts.append(new_p)
            rad.append(new_r)
            if d >= dist_thr:
                break
            if is_parent:
                idx = morph.pos_dict[idx][6]
                if idx == -1:
                    break
            else:
                if idx not in morph.unifurcation:
                    break
                else:
                    idx = morph.child_dict[idx][0]

        # interpolate to find the exact point
        dd = d - dist_thr
        if dd < 0:
            pt_a = new_p
        else:  # extrapolate
            dcur = self._length(new_p, pts[-1])
            ratio = (dcur - dd) / (dcur + self._eps)
            pt_a = pts[-1] + (new_p - pts[-1]) * ratio
            r_a = rad[-1] + (new_r - pts[-1]) * ratio
            rad.append(r_a)
            pts.append(pt_a)

        if return_center_point:
            pt_a = np.mean(pts, axis=0)

        return pt_a, pts, rad

    def _get_anchors(self, morph: Morphology, ind: list[int] | int, dist_thr: float, step_size=0.5):
        """
        get anchors for a set of swc nodes to calculate angles, suppose they are one,
        their center is their mean coordinate,
        getting anchors requires removing redundant protrudes
        :param morph: the morphology wrapped swc
        :param dist_thr: path distance thr
        :param ind: array of coordinates
        """
        if isinstance(ind, int):
            ind = [ind]
        center = np.mean([morph.pos_dict[i][2:5] for i in ind], axis=0)
        center_radius = np.mean([morph.pos_dict[i][5] for i in ind])
        protrude = set()
        for i in ind:
            if i in morph.child_dict:
                protrude |= set(morph.child_dict[i])
        com_line = None
        for i in ind:
            line = [i]
            while line[-1] != -1:
                line.append(morph.pos_dict[line[-1]][6])
            line.reverse()
            protrude -= set(line)
            if com_line is None:
                com_line = line
            else:
                for j in range(min(len(com_line), len(line))):
                    if com_line[j] != line[j]:
                        com_line = com_line[:j]
                        break
        # nearest common parent of all indices
        com_node = com_line[-1]
        protrude = np.array(list(protrude))
        # com_node == center can cause problem for spline
        # for finding anchor_p, you must input sth different from the center to get the right pt list
        if self._length(center, morph.pos_dict[com_node][2:5]) <= self._eps:
            p = morph.pos_dict[com_node][6]
            # p can be -1 if the com_node is root
            # but when this happens, com_node can hardly == center
            # this case is dispelled when finding crossings
        else:
            p = com_node
        anchor_p, pts_p, rad_p = self._find_point(morph, center, p, True, dist_thr, center_radius, False)
        res = [self._find_point(morph, center, i, False, dist_thr, center_radius, False) for i in protrude]
        anchor_ch, pts_ch, rad_ch = [i[0] for i in res], [i[1] for i in res], [i[2] for i in res]
        gap_thr = np.mean(rad_p) * self._gap_thr_ratio
        interp_ch = []
        for pts in pts_ch:
            pp = [pts[0]]
            j = 1
            dist_cum = [0]
            while j < len(pts):
                new_d = self._length(pts[j], pp[-1])
                if new_d > self._eps and new_d + dist_cum[-1] != dist_cum[-1]:
                    dist_cum.append(dist_cum[-1] + new_d)
                    pp.append(pts[j])
                j += 1
            if len(pp) > 1:
                f = interp1d(dist_cum, pp, 'quadratic' if len(pp) > 2 else 'linear', 0, fill_value='extrapolate')
                interp_ch.append(f)
        step = step_size
        while step <= dist_thr:
            pts = [i(step) for i in interp_ch]
            if len(pts) < 2:
                break
            gap = distance_matrix(pts, pts)
            gap = np.median(gap[np.triu_indices_from(gap, 1)])
            if gap > gap_thr:
                break
            center = np.mean(pts, axis=0)
            step += step_size
        return center, anchor_p, anchor_ch, protrude, rad_p, rad_ch

    def branch_prune(self, morph, angle_thr=80, radius_amp=1.5):
        """
        Prune branches by an angle threshold and radius rise.

        :param morph: morphology wrapped swc tree
        :param angle_thr: the minimum turning angle in degrees
        :param radius_amp: max ratio of radius amplification.
        :return: nodes to prune.
        """
        rm_ind = set()
        cs = np.array(morph.pos_dict[morph.idx_soma][2:5])
        for n in morph.bifurcation | morph.multifurcation:
            if self._length(cs, morph.pos_dict[n][2:5]) <= self._soma_radius:
                continue
            # branch, no soma, away from soma
            center, anchor_p, anchor_ch, protrude, rad_p, rad_ch = self._get_anchors(morph, n, self._far_anchor)
            _, near_p, near_ch, _, _, _ = self._get_anchors(morph, n, self._near_anchor)
            vec_p = np.array(anchor_p) - near_p
            vec_ch = np.array(anchor_ch) - near_ch
            if self._length(vec_p) < self._eps:
                vec_p = np.array(anchor_p) - center
            mask = self._length(vec_ch, axis=-1) < self._eps
            vec_ch[mask] = vec_ch[mask] - center

            # strange anchor distance can't be considered for pruning
            angles = self._vector_angles(vec_p, vec_ch)
            radius_p = np.median(rad_p)
            radius_ch = np.array([np.median(rad) for rad in rad_ch])
            rm_ind |= set(protrude[(angles < angle_thr) | (radius_ch > radius_p * radius_amp)])
        return rm_ind

    def _find_mega_crossing(self, morph: Morphology, dist_thr, include_bifurcation=False):
        chains = []
        topo_tree, seg_dict = morph.convert_to_topology_tree()
        topo_morph = Morphology(topo_tree)
        dists = topo_morph.get_distances_to_soma(self._res)
        visited = dict.fromkeys(topo_morph.pos_dict.keys(), False)
        branch = topo_morph.bifurcation | topo_morph.multifurcation
        for tid in topo_morph.tips:
            chain = []
            if tid == topo_morph.p_soma:
                continue
            idx = topo_morph.pos_dict[tid][6]       # starting from the last branch
            while idx != topo_morph.p_soma:
                if chain:
                    v = [topo_morph.pos_dict[chain[-1]][2:5],
                         *(morph.pos_dict[n][2:5] for n in seg_dict[chain[-1]]),
                         topo_morph.pos_dict[idx][2:5]]
                    if self._length(v[1:], v[:-1], axis=-1).sum() > dist_thr or \
                            dists[topo_morph.index_dict[idx]] <= self._soma_radius or idx not in branch:
                        if len(chain) > 1 or chain[0] in topo_morph.multifurcation or include_bifurcation:
                            chains.append(chain)
                        chain = []
                        if visited[idx]:
                            break
                if dists[topo_morph.index_dict[idx]] > self._soma_radius and idx in branch:
                    chain.append(idx)
                visited[idx] = True
                idx = topo_morph.pos_dict[idx][6]
            if len(chain) > 1 or chain and (chain[0] in topo_morph.multifurcation or include_bifurcation):
                chains.append(chain)
        # merge
        return [set.union(*[set(t) for t in chains if t[-1] == head]) for head in np.unique([i[-1] for i in chains])]

    def crossover_prune(self, morph: Morphology, dist_thr=5., angle_thr=120):
        """
        Prune crossovers by angle.

        :param morph: morphology wrapped swc tree
        :param dist_thr: the max distance between nearby branch nodes in a mega crossover.
        :param angle_thr: branches less than this angle will only be removed when aligned with another branch
        :return: nodes to prune.
        """
        crossings = self._find_mega_crossing(morph, dist_thr, True)
        rm_ind = set()
        for x in crossings:
            # angle
            center, anchor_p, anchor_ch, protrude, _, _ = self._get_anchors(morph, x, self._far_anchor)
            _, near_p, near_ch, _, _, _ = self._get_anchors(morph, x, self._near_anchor)
            vec_p = np.array(anchor_p) - near_p
            vec_ch = np.array(anchor_ch) - near_ch
            if self._length(vec_p) < self._eps:
                vec_p = np.array(anchor_p) - center
            mask = self._length(vec_ch, axis=-1) < self._eps
            vec_ch[mask] = vec_ch[mask] - center

            angles = self._vector_angles(vec_p, vec_ch)
            rm = set()
            order = np.argsort(angles)
            for i in order:    # starting from the worst angle
                if protrude[i] in rm:
                    continue
                if angles[i] <= angle_thr:
                    new_angles = self._vector_angles(vec_ch[i], vec_ch)
                    for k in np.flip(np.argsort(new_angles ** 2 / angles)):
                        if new_angles[k] > angles[k] and protrude[k] not in rm:
                            rm |= set(protrude[[i, k]])
                else:
                    break
            left = len(angles) - len(rm)
            if left > 2:       # ensure bifurcation
                for i in order:
                    if protrude[i] in rm:
                        continue
                    rm.add(protrude[i])
                    left -= 1
                    if left <= 2:
                        break
            rm_ind |= rm
        return rm_ind
