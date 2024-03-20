import numpy as np
import math
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d
from .morphology import Morphology
from sklearn.decomposition import PCA


class ErrorPruning:

    def __init__(self, res=(.25, .25, 1.), soma_radius=10., anchor_dist=10, gap_thr_ratio=1., epsilon=1e-7):
        """

        :param res: image resolution in micrometers, (x, y, z)
        :param soma_radius: expected soma radius in micrometers, within which errors are not counted
        :param anchor_dist: the distances of the anchor to the branch node
        :param gap_thr_ratio:
        :param epsilon: value lower than this will be regarded as 0
        """
        self._res = res
        self._soma_radius = soma_radius
        # self._near_anchor = anchor_reach[0]
        self._far_anchor = anchor_dist
        self._gap_thr_ratio = gap_thr_ratio
        self._eps = epsilon

    def _length(self, p1: list | np.ndarray, p2=(0, 0, 0), axis=None):
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

    def _find_point(self, morph: Morphology, ct: np.ndarray, idx: np.ndarray, is_parent: bool, dist_thr: float, ct_rad: float):
        """
        Find the point of exact `dist` to the start pt on tree structure. args are:
        - pt: the start point, [coordinate]
        - anchor_idx: the first node on swc tree to trace, first child or parent node
        - is_parent: whether the anchor_idx is the parent of `pt`, otherwise child.
                     if a furcation points encounted, then break
        - morph: Morphology object for current tree
        - dist: distance threshold
        """

        # init
        d = 0
        if is_parent and morph.pos_dict[idx][6] != -1:
            pts = [np.array(morph.pos_dict[idx][2:5])]
            rad = [np.array(morph.pos_dict[idx][5])]
            idx = morph.pos_dict[idx][6]
        elif not is_parent and idx in morph.unifurcation:
            pts = [np.array(morph.pos_dict[idx][2:5])]
            rad = [np.array(morph.pos_dict[idx][5])]
            idx = morph.child_dict[idx][0]
        else:
            pts = [ct]
            rad = [ct_rad]
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

        return pts, rad, idx

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
        pts_p, rad_p, last_p = self._find_point(morph, center, com_node, True, dist_thr, center_radius)
        res = [self._find_point(morph, center, i, False, dist_thr, center_radius) for i in protrude]
        pts_ch, rad_ch, last_ch = [i[0] for i in res], [i[1] for i in res], [i[2] for i in res]
        return com_node, pts_p, pts_ch, protrude, rad_p, rad_ch, last_p, last_ch

    @staticmethod
    def line_fit_pca(pts_list: list[np.ndarray]) -> np.ndarray:
        """
        fit 3D points to a straight line.
        :param pts_list: a list of 3D connected points
        :return: a 3D vector fitted to the list
        """
        pca = PCA(n_components=1)
        pca.fit(pts_list)
        line_direction = pca.components_[0]
        temp = pts_list[-1] - pts_list[0]
        if temp.dot(line_direction) < 0:
            line_direction = -line_direction
        return line_direction

    def get_angle(self, pts_list1: list[np.ndarray], pts_list2: list[np.ndarray]):
        """
        The angle between 2 vectors (fitted from 2 point lists), but supplementary.
        the vectors share the start point, but to make it fit for scoring, its supplementary is returned.
        so a smaller angle means a more straight connection.

        :param pts_list1: a list of 3D points for one branch
        :param pts_list2: a list of 3D points for another branch
        :return: an angle in arc
        """
        vec1 = self.line_fit_pca(pts_list1) * self._res
        vec2 = self.line_fit_pca(pts_list2) * self._res
        cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return math.acos(max(min(cos, 1), -1)) * 180 / math.pi

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
            _, pts_p, pts_ch, protrude, rad_p, rad_ch, _, _ = self._get_anchors(morph, n, self._far_anchor)
            angles = np.array([self.get_angle(pts_p, c) for c in pts_ch])
            radius_p = np.mean(rad_p)
            radius_ch = np.array([np.mean(rad) for rad in rad_ch])
            rm_ind |= set(protrude[(angles < angle_thr) | (radius_ch > radius_p * radius_amp)])
        return rm_ind

    def _find_mega_crossing(self, morph: Morphology, dist_thr):
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
                        if len(chain) > 1 or chain[0] in topo_morph.multifurcation:
                            chains.append(chain)
                        chain = []
                        if visited[idx]:
                            break
                if dists[topo_morph.index_dict[idx]] > self._soma_radius and idx in branch:
                    chain.append(idx)
                visited[idx] = True
                idx = topo_morph.pos_dict[idx][6]
            if len(chain) > 1 or chain and chain[0] in topo_morph.multifurcation:
                chains.append(chain)
        # merge
        return [set.union(*[set(t) for t in chains if t[-1] == head]) for head in np.unique([i[-1] for i in chains])]

    def crossover_prune(self, morph: Morphology, dist_thr=2., angle_thr1=60, angle_thr2=90, check_bif=False,
                        no_multi=True, short_tips_thr=5.):
        """
        Prune crossovers by angle.

        :param morph: morphology wrapped swc tree
        :param dist_thr: the max distance between nearby branch nodes in a mega crossover.
        :param angle_thr1: branches less than this angle will take away a best fit branch
        :param angle_thr2: branches less than this angle will take away a best fit branch than parent
        :param check_bif: if checking bifurcation, in this mode only bifurcation will be checked and pruning
        is not forced.
        :param no_multi: ensure no multifurcation, start removing from tips
        :param short_tips_thr: drop short tips below this threshold before pruning.
        :return: nodes to prune.
        """
        crossings = self._find_mega_crossing(morph, dist_thr)
        cs = np.array(morph.pos_dict[morph.idx_soma][2:5])
        if check_bif:
            to_check = [i for i in morph.bifurcation - set.union(*crossings)
                        if self._length(cs, morph.pos_dict[i][2:5]) > self._soma_radius]
        else:
            to_check = crossings
        rm_ind = set()
        for x in to_check:
            # angle
            com_node, pts_p, pts_ch, protrude, rad_p, rad_ch, last_p, last_ch = self._get_anchors(morph, x, self._far_anchor)
            rad_p = np.mean(rad_p)
            rad_ch = [np.mean(c) for c in rad_ch]
            rad_diff = [abs(c - rad_p) for c in rad_ch]
            angles = np.array([self.get_angle(pts_p, c) for c in pts_ch])
            rm = set()

            for i, c, pts in zip(protrude, last_ch, pts_ch):
                if i in rm:
                    continue
                if c in morph.tips:
                    l = self._length(pts[1:], pts[:-1], axis=1).sum()
                    if l < short_tips_thr:
                        rm.add(i)

            order = np.argsort(angles)
            for i in order:    # starting from the worst angle
                if protrude[i] in rm:
                    continue
                if angles[i] < angle_thr1:
                    new_angles = np.array([self.get_angle(pts_ch[i], c) for c in pts_ch])
                    for k in np.argsort(angles - new_angles):
                        new_rad_diff = abs(rad_ch[i] - rad_ch[k])
                        if new_rad_diff < rad_diff[k] and protrude[k] not in rm:
                            rm.add(protrude[i])
                            rm.add(protrude[k])
                            break
                elif angles[i] < angle_thr2:
                    new_angles = np.array([self.get_angle(pts_ch[i], c) for c in pts_ch])
                    for k in np.argsort(angles - new_angles):
                        new_rad_diff = abs(rad_ch[i] - rad_ch[k])
                        if new_angles[k] > angles[k] and new_rad_diff < rad_diff[k] and protrude[k] not in rm:
                            rm.add(protrude[i])
                            rm.add(protrude[k])
                            break
                else:
                    break
            if no_multi:
                left = len(angles) - len(rm)
                if left > 2:       # ensure bifurcation
                    # first, remove short tips
                    for i, c in zip(protrude, last_ch):
                        if i in rm:
                            continue
                        if c in morph.tips:
                            rm.add(i)
                            left -= 1
                            if left <= 2:
                                break
                    else:
                        for i in np.argsort(rad_diff)[::-1]:
                            if protrude[i] in rm:
                                continue
                            rm.add(protrude[i])
                            left -= 1
                            if left <= 2:
                                break
            # remove upstream of protrudes
            for i in list(rm):
                i = morph.pos_dict[i][6]
                while i != -1 and i in morph.unifurcation or set(morph.child_dict[i]).issubset(rm):
                    rm.add(i)
                    i = morph.pos_dict[i][6]
            rm_ind |= rm
        return rm_ind
