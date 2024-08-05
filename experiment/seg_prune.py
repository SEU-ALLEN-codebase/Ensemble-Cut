import sys

sys.path.insert(0, '/public/home/vkzohj/pylib')
from ecut import swc_handler
import numpy as np
import os
from pathlib import Path
from multiprocessing.pool import Pool
from ecut.morphology import Morphology
import math
from neuron_quality.find_break_crossing import find_point_by_distance
from scipy.interpolate import interp1d
from scipy.spatial import distance_matrix
from neuron_quality.find_break_crossing import CrossingFinder
from functools import cmp_to_key
from sklearn.neighbors import KDTree
from skimage import draw
import traceback
from queue import SimpleQueue
import pandas as pd
from tempfile import TemporaryDirectory
import subprocess
import traceback


debug = False
node_downer_limit = 20
res_xy = pd.read_csv(r"D:\rectify\supplement.csv", index_col=1, header=0).iloc[:, 3]
in_dir = Path(r"D:\rectify\my_app2")
out_dir = Path(r"D:\rectify\seg_prune")


def get_anchors(morph: Morphology, ind: list, dist: float, spacing=(1, 1, 1), epsilon=1e-7, gap_thr_ratio=1,
                step_size=0.5, adjust_center=False):
    """
    get anchors for a set of swc nodes to calculate angles, suppose they are one, their center is their mean coordinate,
    getting anchors requires removing redundant protrudes
    :param dist: path distance thr
    :param ind: array of coordinates
    """
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
    protrude = list(protrude)
    # com_node == center can cause problem for spline
    # for finding anchor_p, you must input sth different from the center to get the right pt list
    if np.linalg.norm((center - morph.pos_dict[com_node][2:5]) * spacing) <= epsilon:
        p = morph.pos_dict[com_node][6]
        # p can be -1 if the com_node is root
        # but when this happens, com_node can hardly == center
        # this case is dispelled when finding crossings
    else:
        p = com_node
    anchor_p, pts_p, rad_p = find_point_by_distance(center, p, True, morph, dist, False, spacing=spacing,
                                                    stop_by_branch=False, only_tgt_pt=False, radius=True,
                                                    pt_rad=center_radius)
    res = [find_point_by_distance(center, i, False, morph, dist, False, spacing=spacing,
                                  stop_by_branch=True, only_tgt_pt=False, radius=True, pt_rad=center_radius)
           for i in protrude]
    anchor_ch, pts_ch, rad_ch = [i[0] for i in res], [i[1] for i in res], [i[2] for i in res]
    gap_thr = np.mean(rad_p) * gap_thr_ratio
    if adjust_center:
        interp_ch = []
        for pts in pts_ch:
            pp = [pts[0]]
            j = 1
            dist_cum = [0]
            while j < len(pts):
                new_d = np.linalg.norm((pts[j] - pp[-1]) * spacing)
                if new_d > epsilon and new_d + dist_cum[-1] != dist_cum[-1]:
                    dist_cum.append(dist_cum[-1] + new_d)
                    pp.append(pts[j])
                j += 1
            if len(pp) > 1:
                f = interp1d(dist_cum, np.array(pp), 'quadratic' if len(pp) > 2 else 'linear', 0,
                             fill_value='extrapolate')
                interp_ch.append(f)
        step = step_size
        while step <= dist:
            pts = [i(step) for i in interp_ch]
            if len(pts) < 2:
                break
            gap = distance_matrix(pts, pts)
            gap = np.median(gap[np.triu_indices_from(gap, 1)])
            if gap > gap_thr:
                break
            center = np.mean(pts, axis=0)
            step += step_size
    return center, anchor_p, anchor_ch, protrude, com_node, pts_p, pts_ch, rad_p, rad_ch


class HidePrint:
    def __init__(self, debug=False):
        self.debug = debug

    def __enter__(self):
        if not debug:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not debug:
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


def anchor_angles(center: np.ndarray, p: np.ndarray, ch, spacing=(1, 1, 1)):
    """
    modified from Jingzhou's code
    :param center: coordinate for the multifurcation center
    :param p: coordinate for the parent anchor
    :param ch: coordinates for the furcation anchors
    :param spacing: scale factor for each dimension
    """
    vec_p = (p - center) * spacing
    vec_ch = [(coord - center) * spacing for coord in ch]
    cos_ch = [vec_p.dot(vec) / (np.linalg.norm(vec_p) * np.linalg.norm(vec)) for vec in vec_ch]
    out = [*map(lambda x: math.acos(max(min(x, 1), -1)) * 180 / math.pi, cos_ch)]
    return out


def vector_angles(p: np.ndarray, ch, spacing=(1, 1, 1)):
    """
    modified from Jingzhou's code
    :param center: coordinate for the multifurcation center
    :param p: coordinate for the parent anchor
    :param ch: coordinates for the furcation anchors
    :param spacing: scale factor for each dimension
    """
    vec_p = p * spacing
    vec_ch = [coord * spacing for coord in ch]
    cos_ch = [vec_p.dot(vec) / (np.linalg.norm(vec_p) * np.linalg.norm(vec)) for vec in vec_ch]
    out = [*map(lambda x: math.acos(max(min(x, 1), -1)) * 180 / math.pi, cos_ch)]
    return out


class SegmentPruning:

    def __init__(self, spacing=(1, 1, 1), soma_radius=15, anchor_dist=20, anchor_tolerance=10, near_anchor=5,
                 gap_thr_ratio=1):
        self.spacing = np.array(spacing)
        self.soma_radius = soma_radius
        self.near_anchor = near_anchor
        self.anchor_dist = anchor_dist
        self.anchor_tolerance = anchor_tolerance
        self.gap_thr_ratio = gap_thr_ratio

    # def soma_limit_filter_radius(self, tree, max_count=3, min_radius=5, pass_rate=0.8, min_radius_keep=True, eps=10):
    #     """
    #     limit the number of soma in an swc, based on radius of nodes
    #     """
    #     morph = Morphology(tree)
    #     dist = morph.get_distances_to_soma(self.spacing)
    #     soma_r = np.max([t[5] for t, d in zip(tree, dist) if d <= self.soma_radius])
    #     if soma_r < min_radius:
    #         return min_radius_keep
    #     pass_r = soma_r * pass_rate
    #     db = DBSCAN(eps=eps, min_samples=1)
    #     dt = [t[2:5] for t in tree if t[5] >= pass_r] * self.spacing
    #     labs = db.fit_predict(dt)
    #     cluster = [dt[np.where(labs == l)].mean(axis=0) for l in np.unique(labs) if l != -1]
    #     ct = morph.pos_dict[morph.idx_soma][2:5] * self.spacing
    #     outer_soma = [p for p in cluster if np.linalg.norm(p - ct) > self.soma_radius]
    #     return len(outer_soma) <= max_count - 1

    def soma_prune(self, morph, candidate_radius=5, area_thr=500, non_detect_radius=50):
        dist = morph.get_distances_to_soma(self.spacing)
        ct = morph.pos_dict[morph.idx_soma][2:5]
        maybe_soma = [t for t, d in zip(morph.tree, dist) if d > non_detect_radius and t[5] > candidate_radius]
        other_soma = []
        kd = KDTree([t[2:5] * self.spacing for t in morph.tree])
        for t in maybe_soma:
            win = np.zeros([self.soma_radius * 2] * 2, dtype=bool)
            win2 = win.copy()
            draw.set_color(win2, draw.disk([self.soma_radius] * 2, self.soma_radius), [1])
            for i in kd.query_radius([t[2:5] * self.spacing], self.soma_radius)[0]:
                xy = morph.tree[i][2:4] - np.array(t[2:4]) + [self.soma_radius] * 2
                draw.set_color(win, draw.disk(xy, morph.tree[i][5]), [1])
            if np.sum(win & win2) >= area_thr:
                other_soma.append(t[0])

        path_dist = get_path_dist(morph, self.spacing)

        def comp(x, y):
            pd1 = path_dist[morph.index_dict[x]]
            pd2 = path_dist[morph.index_dict[y]]
            if pd1 > pd2:
                return 1
            elif pd1 < pd2:
                return -1
            else:
                return 0

        other_soma.sort(key=cmp_to_key(comp))

        rm_ind = set()
        cs = np.array(ct)
        for i in other_soma:
            pts = [morph.pos_dict[i]]
            cs_fake = np.array([pts[0][2:5]])
            skip = False
            while pts[-1][6] != -1:
                if pts[-1][0] in rm_ind:
                    skip = True
                    break
                pts.append(morph.pos_dict[pts[-1][6]])
            if skip:
                continue
            if len(pts) < 3:
                rm_ind.add(pts[-1][0])
                continue
            tmp_morph = Morphology(pts)
            gof1 = []
            len1 = []
            gof2 = []
            len2 = []
            gray2 = []
            for j, p in enumerate(pts):
                if j == 0 or j == len(pts) - 1 or np.linalg.norm((p[2:5] - cs) * self.spacing) <= self.soma_radius:
                    gof1.append(0)
                    len1.append(0)
                    gof2.append(0)
                    len2.append(0)
                    gray2.append(0)
                    continue
                center, anchor_p, anchor_ch = get_anchors(tmp_morph, [p[0]], self.anchor_dist, self.spacing,
                                                          gap_thr_ratio=self.gap_thr_ratio)[:3]
                gof1.append(180 - anchor_angles(center, cs, np.array(anchor_ch), spacing=self.spacing)[0])
                len1.append(np.linalg.norm((np.array(p[2:5]) - pts[j - 1][2:5]) * self.spacing))
                gof2.append(180 - anchor_angles(center, anchor_p, cs_fake, spacing=self.spacing)[0])
                len2.append(np.linalg.norm((np.array(p[2:5]) - pts[j + 1][2:5]) * self.spacing))
                # gray2.append(np.exp(10 * (1 - region_gray_level(img, p[2:5], [2, 2, 1]) / 255) ** 2))
            cdf1 = []  # soma to fake
            cdf2 = []  # fake to soma
            gof1.reverse()
            len1.reverse()
            # gray1 = gray2.copy()
            # gray1.reverse()
            for a, b, c, d in zip(gof1, gof2, len1, len2):
                cdf1.append(a * c)
                if len(cdf1) > 1:
                    cdf1[-1] += cdf1[-2]
                cdf2.append(b * d)
                if len(cdf2) > 1:
                    cdf2[-1] += cdf2[-2]
            cdf1.reverse()
            for k, (a, b) in enumerate(zip(cdf1, cdf2)):
                if a < b:
                    break
            if len(morph.child_dict[pts[k][0]]) > 1:
                k -= 1
            k = max(k, 0)
            rm_ind.add(pts[k][0])
        return rm_ind

    def branch_prune(self, morph, angle_thr=80):
        """
        prune any awry child for every branch,
        check only local angle
        """
        rm_ind = set()
        cs = np.array(morph.pos_dict[morph.idx_soma][2:5])
        for n in morph.bifurcation | morph.multifurcation:
            if np.linalg.norm((morph.pos_dict[n][2:5] - cs) * self.spacing) <= self.soma_radius:
                continue
            # branch, no soma, away from soma
            center, anchor_p, anchor_ch, protrude, com_node, pts_p, pts_ch, rad_p, rad_ch = \
                get_anchors(morph, [n], self.anchor_dist, self.spacing, adjust_center=True,
                            gap_thr_ratio=self.gap_thr_ratio)
            near_p, near_ch = \
                get_anchors(morph, [n], self.near_anchor, self.spacing, gap_thr_ratio=self.gap_thr_ratio)[1:3]
            vec_p = np.array(anchor_p) - near_p
            if np.linalg.norm(vec_p * self.spacing) < self.anchor_tolerance:
                vec_p = np.array(anchor_p) - center
            vec_ch = np.array(anchor_ch) - near_ch
            for k, c in enumerate(vec_ch):
                if np.linalg.norm(c * self.spacing) < self.anchor_tolerance:
                    vec_ch[k] = np.array(anchor_ch[k]) - center
            # strange anchor distance can't be considered for pruning
            mask = np.linalg.norm((anchor_ch - center) * self.spacing, axis=-1) > self.anchor_tolerance
            angles = vector_angles(vec_p, vec_ch, spacing=self.spacing)
            protrude = np.array(protrude)
            rm_ind |= set(protrude[((np.array(angles) < angle_thr)) & mask])
        return rm_ind

    def radius_prune(self, morph, radius_amp=1.5):
        """
        prune any awry child for every branch,
        check only local angle
        """
        rm_ind = set()
        cs = np.array(morph.pos_dict[morph.idx_soma][2:5])
        for n in morph.bifurcation | morph.multifurcation:
            if np.linalg.norm((morph.pos_dict[n][2:5] - cs) * self.spacing) <= self.soma_radius:
                continue
            # branch, no soma, away from soma
            center, anchor_p, anchor_ch, protrude, com_node, pts_p, pts_ch, rad_p, rad_ch = \
                get_anchors(morph, [n], self.anchor_dist, self.spacing, adjust_center=False)
            radius_p = np.median(rad_p)
            radius_ch = [np.median(rad) for rad in rad_ch]
            protrude = np.array(protrude)
            rm_ind |= set(protrude[(np.array(radius_ch) > radius_p * radius_amp)])
        return rm_ind

    def crossing_prune(self, morph, dist_thr=5, angle_thr1=80, angle_thr2=120, para_angle_thr=135):
        """
        detect crossings and prune
        """
        cf = CrossingFinder(morph, self.soma_radius, dist_thr)
        crossings = cf.find_mega_crossing(True)
        rm_ind = set()
        for x in crossings:
            # angle
            center, anchor_p, anchor_ch, protrude = \
                get_anchors(morph, x, self.anchor_dist, self.spacing, adjust_center=True,
                            gap_thr_ratio=self.gap_thr_ratio)[:4]
            near_p, near_ch = \
                get_anchors(morph, x, self.near_anchor, self.spacing, gap_thr_ratio=self.gap_thr_ratio)[1:3]
            vec_p = np.array(anchor_p) - near_p
            if np.linalg.norm(vec_p * self.spacing) < self.anchor_tolerance:
                vec_p = np.array(anchor_p) - center
            vec_ch = np.array(anchor_ch) - near_ch
            for k, c in enumerate(vec_ch):
                if np.linalg.norm(c * self.spacing) < self.anchor_tolerance:
                    vec_ch[k] = np.array(anchor_ch[k]) - center
            angles = vector_angles(vec_p, vec_ch, spacing=self.spacing)
            mask = np.linalg.norm(vec_ch * self.spacing, axis=-1) > self.anchor_tolerance
            rm = set()
            for i in np.argsort(angles):
                if protrude[i] in rm or not mask[i]:
                    continue
                if angles[i] <= angle_thr1:
                    new_p = anchor_ch[i]
                    new_angles = np.array(vector_angles(vec_ch[i], vec_ch, spacing=self.spacing))
                    rm.add(protrude[i])
                    for k in np.flip(np.argsort(new_angles ** 2 / angles)):
                        if mask[k] and new_angles[k] > para_angle_thr and new_angles[k] > angles[k] and protrude[
                            k] not in rm:
                            rm.add(protrude[k])
                            break
                elif angles[i] <= angle_thr2:
                    new_p = anchor_ch[i]
                    new_angles = np.array(vector_angles(vec_ch[i], vec_ch, spacing=self.spacing))
                    for k in np.flip(np.argsort(new_angles ** 2 / angles)):
                        if mask[k] and new_angles[k] > para_angle_thr and new_angles[k] > angles[k] and protrude[
                            k] not in rm:
                            rm.add(protrude[i])
                            rm.add(protrude[k])
                            break
                else:
                    break
            rm_ind |= rm
            rm_ind -= set(np.array(protrude)[~mask])
        return rm_ind

    def winding_prune(self, morph, ratio_thr=2):
        path_dist = get_path_dist(morph, self.spacing)
        dist = morph.get_distances_to_soma(self.spacing)
        ratio = [d1 / d2 if d2 > self.soma_radius else 1 for d1, d2 in zip(path_dist, dist)]
        cand = [t[0] for t, r in zip(morph.tree, ratio) if r > ratio_thr]

        def comp(x, y):
            pd1 = path_dist[morph.index_dict[x]]
            pd2 = path_dist[morph.index_dict[y]]
            if pd1 > pd2:
                return 1
            elif pd1 < pd2:
                return -1
            else:
                return 0

        cand.sort(key=cmp_to_key(comp))
        rm_ind = set()
        for i in cand:
            t = i
            j = morph.pos_dict[i][6]
            while j != morph.idx_soma and j in morph.unifurcation:
                t = j
                j = morph.pos_dict[j][6]
            rm_ind.add(t)
        return rm_ind


def get_path_dist(morph, spacing=(1, 1, 1)):
    q = SimpleQueue()
    q.put(morph.idx_soma)
    path_dist = [0] * len(morph.tree)
    while not q.empty():
        head = q.get()
        if head in morph.child_dict:
            for c in morph.child_dict[head]:
                q.put(c)
        p = morph.pos_dict[head][6]
        if p != morph.p_soma:
            path_dist[morph.index_dict[head]] = path_dist[morph.index_dict[p]] + np.linalg.norm(
                (np.array(morph.pos_dict[head][2:5]) - morph.pos_dict[p][2:5]) * spacing)
    return path_dist


def main(swc):
    outpath = out_dir / swc.name

    # prune
    try:
        res = res_xy.at[int(swc.name.split('_')[0])]
        if np.isnan(res):
            res = .25

        tree_for_soma = swc_handler.parse_swc(swc)
        with TemporaryDirectory() as td:
            tp = Path(td) / 't.swc'
            subprocess.run(f'Vaa3D-x /x pruning_swc_simple /f pruning_iterative /i {swc} /o {tp} /p {str(int(20 / res))}', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tree = swc_handler.parse_swc(tp)

        if len(tree) < node_downer_limit:
            return
        morph = Morphology(tree)

        pruner = SegmentPruning(spacing=(1, 1, 4), soma_radius=int(6 / res),
                                anchor_dist=40, anchor_tolerance=20, near_anchor=10, gap_thr_ratio=1)

        with HidePrint(debug):
            tree = swc_handler.prune(tree,
                                     pruner.branch_prune(morph, angle_thr=45) |
                                     pruner.radius_prune(morph, radius_amp=2) |
                                     pruner.crossing_prune(morph, dist_thr=10, angle_thr1=45, angle_thr2=90,
                                                           para_angle_thr=135) |
                                     (pruner.soma_prune(Morphology(tree_for_soma), candidate_radius=int(3 / res),
                                                        area_thr=int(200 / res), non_detect_radius=int(20 / res)) &
                                      set([t[0] for t in tree])) |
                                     pruner.winding_prune(morph, ratio_thr=3)
                                     )

        swc_handler.write_swc(tree, outpath)
    except Exception as e:
        traceback.print_exc()


if __name__ == '__main__':
    from tqdm import tqdm
    files = sorted(in_dir.glob('*.swc'))
    with Pool(16) as p:
        for i in tqdm(p.imap(main, files), total=len(files)):
            pass
