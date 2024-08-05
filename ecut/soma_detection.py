import numpy as np
from skimage import filters, morphology, measure
from scipy import ndimage
import pandas as pd
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
from tqdm import tqdm
from .base_types import ListNeuron
from skimage import draw
from sklearn.neighbors import KDTree


def merge_points(pts, merge_dist) -> list[np.ndarray]:
    db = DBSCAN(merge_dist, min_samples=1)
    db.fit(pts)
    labels = db.labels_  # Get cluster labels.

    # Initialize an empty list to store the centers of clusters.
    centers = []

    # For each unique label (except -1 which denotes noise), calculate and store the mean point.
    for label in set(labels):
        if label != -1:  # Ignore noise.
            members = pts[labels == label]  # Get all points in the current cluster.
            center = np.median(members, axis=0)  # Calculate the center of the current cluster.
            centers.append(center)

    return centers


class DetectImage:
    """
    Modified from brainlit soma detection algorithm,

    Improvements:
    1. adaptive thresholding as preprocessing
    2. otsu -> triangle thresholding
    """

    def __init__(self, processing_res=1., diam_range=(5, 20)):
        """

        :param processing_res: the image size during processing, in x, y, z.
        Downsize the image to the optimal range for morphological operations. the z size should be small as possible
        :param diam_range: the minimum and maximum diameter in micrometers allowed for the detected somata.
        """
        self._processing_res = processing_res
        self._diam_range = diam_range

    def predict(self, img, res: list[float], thr=None) -> list[np.ndarray]:
        """

        :param img: the image for detection, indexed by (z, y, x)
        :param res: the resolution of x, y, z in micrometers, for computing radius.
        :param thr: manually set the threshold
        :return: a list of soma centers and the 3D mask
        """
        assert len(res) == img.ndim == 3, "Only 3D images are supported"
        assert res[0] > 0 and res[1] > 0 and res[2] > 0, "Resolution must be positive"
        zoom_factors = np.array(res[::-1]) / self._processing_res
        out = ndimage.zoom(img, zoom=zoom_factors)
        m = np.max(out.flatten())
        if m == 0:
            return []
        out = out / m
        win_size = int(self._diam_range[1] / self._processing_res * 2)
        if win_size % 2 == 0:
            win_size += 1
        t = filters.threshold_local(out, win_size)
        out = (out - t).clip(0)
        if thr is None:
            thr = filters.threshold_triangle(out)
        else:
            thr = thr / m
        out = out > thr
        selem_size = np.amax(np.ceil(zoom_factors)).astype(int)
        clean_selem = morphology.octahedron(selem_size)
        out = morphology.binary_erosion(out, clean_selem)
        clean_selem = morphology.octahedron(int(self._diam_range[0] / self._processing_res / 2))
        out = morphology.binary_opening(out, clean_selem)
        out, num_labels = morphology.label(out, background=0, return_num=True)
        properties = ["label", "equivalent_diameter"]
        props = measure.regionprops_table(out, properties=properties)

        df_props = pd.DataFrame(props)
        centers = []
        for _, row in df_props.iterrows():
            l, d = row[properties]
            dmu = d * self._processing_res
            if self._diam_range[0] <= dmu <= self._diam_range[1]:
                ids = np.where(out == l)
                centroid = np.round([np.median(u) for u in ids])
                centroid = np.divide(centroid, zoom_factors)    # original pixel pos
                centers.append(centroid)
        return centers


class DetectTiledImage:
    def __init__(self, tile_size=(256, 256, 64), omit_border=(16, 16, 4), merge_dist=15,
                 base_detector=DetectImage(), nproc=None):
        """

        :param tile_size: The size of a single tile, indexed by x, y, z.
        :param omit_border: The range of the border to omit, any detection falling on the border will be omitted.
        :param merge_dist: The max distance between somata to merge, in micrometers.
        :param base_detector: The tile soma detector.
        """
        self._find_soma = base_detector
        self._tile_size = np.array(tile_size[::-1])
        self._merge_dist = merge_dist
        self._omit_border = np.array(omit_border[::-1])
        self._nproc = nproc

    @staticmethod
    def process_find_soma(mod, img, res, thr, s, ob, ts):
        pts = mod.predict(img, res, thr)
        return [p + s for p in pts if np.all((ob < p) & (p < ts - ob))]

    def predict(self, img: np.ndarray, res: list[float]) -> list[np.ndarray]:
        """

        :param img: the image for detection, indexed by (z, y, x)
        :param res: the resolution of x, y, z in micrometers, for computing radius.
        :return: a list of soma centers and the 3D mask
        """
        assert len(res) == img.ndim == 3, "Only 3D images are supported"
        assert res[0] > 0 and res[1] > 0 and res[2] > 0, "Resolution must be positive"
        thr = filters.threshold_mean(img)
        steps = np.ceil(img.shape / (self._tile_size - 2 * self._omit_border)).astype(int)
        hf = self._tile_size // 2
        z = np.linspace(hf[0], img.shape[0] - hf[0], steps[0], dtype=int)
        y = np.linspace(hf[1], img.shape[1] - hf[1], steps[1], dtype=int)
        x = np.linspace(hf[2], img.shape[2] - hf[2], steps[2], dtype=int)

        jobs = []
        prefilter = []
        if self._nproc is not None:
            with Pool(self._nproc) as p:
                for zz in z:
                    for yy in y:
                        for xx in x:
                            s = (zz, yy, xx) - hf
                            e = (zz, yy, xx) + hf
                            tile = img[s[0]: e[0], s[1]: e[1], s[2]: e[2]]
                            jobs.append(p.apply_async(DetectTiledImage.process_find_soma,
                                                      (self._find_soma, tile, res, thr, s, self._omit_border, self._tile_size)))

                for i in tqdm(jobs):
                    prefilter.extend(i.get())
        else:
            for zz in z:
                for yy in y:
                    for xx in x:
                        s = (zz, yy, xx) - hf
                        e = (zz, yy, xx) + hf
                        tile = img[s[0]: e[0], s[1]: e[1], s[2]: e[2]]
                        prefilter.extend(self.process_find_soma(self._find_soma, tile, res, thr, s, self._omit_border, self._tile_size))

        if len(prefilter) == 0:
            return []
        res = res[::-1]
        centers = merge_points(np.array(prefilter) * res, self._merge_dist)
        centers = [c / res for c in centers]
        return centers


class DetectTracingMask:
    def __init__(self, min_radius=2., merge_dist=15., diam_range=(5., 20.)):
        """

        :param min_radius: the minimum radius of the swc nodes to consider, in micrometer
        :param merge_dist: the max distance between somata to merge, in micrometers.
        :param diam_range: the minimum and maximum diameter in micrometers allowed for the detected somata.
        """
        self._min_radius = min_radius
        self._merge_dist = merge_dist
        self._diam_range = diam_range

    def predict(self, swc: ListNeuron, res: list[float]) -> list[np.ndarray]:
        """

        :param swc: the swc list neuron.
        :param res: the resolution of x, y, z in micrometers, for computing radius.
        :return: a list of soma centers and the 3D mask
        """
        sf = (res[0] + res[1]) / 2
        candid = [t for t in swc if t[5] * sf >= self._min_radius]
        pos = np.array([t[2:5] for t in candid])
        rad = np.array([t[5] for t in candid])
        if len(pos) == 0:
            return []
        db = DBSCAN(self._merge_dist, min_samples=1)
        db.fit(pos * res)
        labels = db.labels_  # Get cluster labels.

        # Initialize an empty list to store the centers of clusters.
        centers = []

        win_size = (np.array([self._diam_range[1]] * 2) / sf).astype(int)

        # For each unique label (except -1 which denotes noise), calculate and store the mean point.
        for label in set(labels):
            if label != -1:  # Ignore noise.
                members = pos[labels == label] # Get all points in the current cluster.
                mem_rad = rad[labels == label]
                center = np.mean(members, axis=0)        # Calculate the center of the current cluster.
                win = np.zeros(win_size, dtype=np.uint8)
                mask = win.copy()
                draw.set_color(mask, draw.disk(tuple(win_size // 2), win_size[0] // 2), 255)
                for p, r in zip(members, mem_rad):
                    p = (p - center).astype(int)[:2] + win_size // 2
                    draw.set_color(win, draw.disk(p, r), 255)
                win &= mask

                dmu = (np.sum(win > 0) / np.pi) ** .5 * 2 * sf

                if self._diam_range[0] <= dmu <= self._diam_range[1]:
                    centers.append(center)

        return centers


class DetectDistanceTransform:
    def __init__(self, processing_size=(160, 160, 50), diam_range=(5, 20)):
        """

        :param processing_size: the image size during processing.
        Downsize the image to the optimal range for morphological operations.
        :param diam_range: the minimum and maximum diameter in micrometers allowed for the detected somata.
        """
        self._processing_size = processing_size
        self._diam_range = diam_range

    def predict(self, img, res: list[float], thr=None) -> list[np.ndarray]:
        """

        :param img: the image for detection, indexed by (z, y, x)
        :param res: the resolution of x, y, z in micrometers, for computing radius.
        :param thr: manually set the threshold
        :return: a list of soma centers and the 3D mask
        """
        assert len(res) == img.ndim == 3, "Only 3D images are supported"
        assert res[0] > 0 and res[1] > 0 and res[2] > 0, "Resolution must be positive"
        desired_size = np.array(self._processing_size[::-1])
        zoom_factors = desired_size / img.shape
        res = np.divide(res[::-1], zoom_factors)
        sf = res[1:].mean()
        out = ndimage.zoom(img, zoom=zoom_factors)
        m = np.max(out.flatten())
        if m == 0:
            return []
        out = out / m
        win_size = self._diam_range[1] / sf * 2
        if win_size % 2 == 0:
            win_size += 1
        t = filters.threshold_local(out, win_size)
        out = (out - t).clip(0)
        if thr is None:
            thr = filters.threshold_triangle(out)
        else:
            thr = thr / m
        out = out > thr
        clean_selem = morphology.octahedron(int(self._diam_range[0] / sf / 4))
        out = morphology.binary_opening(out, clean_selem)
        dt = ndimage.distance_transform_edt(out, res)
        mask = dt > (self._diam_range[0] / 2)

        label, num_labels = morphology.label(mask, background=0, return_num=True)
        centers = []
        for i in range(1, num_labels + 1):
            ids = np.where(label == i)
            dmu = max(dt[ids]) * 2
            if self._diam_range[0] <= dmu <= self._diam_range[1]:
                weighted_center = np.average(np.argwhere(label == i), weights=dt[ids], axis=0)
                centers.append(weighted_center / zoom_factors)

        return centers


def soma_consensus(*centers, radius=10, min_vote=None, merge_dist=30, res=(1,1,1)):
    """

    :param centers:
    :param radius:
    :param min_vote:
    :param merge_dist:
    :param res:
    :return:
    """
    centers = [c for c in centers if len(c) > 0]
    if min_vote is None:
        min_vote = len(centers)
    kd_trees = [KDTree(c) for c in centers]
    ans = []
    for i, c in enumerate(centers):
        vote = [1] * len(c)
        for dists in [k.query(c, 1)[0] for j, k in enumerate(kd_trees) if j != i]:
            for j, d in enumerate(dists):
                if d < radius:
                    vote[j] += 1
        for j, v in enumerate(vote):
            if v >= min_vote:
                ans.append(c[j])
    if len(ans) > 0:
        ans = merge_points(np.array(ans) * res, merge_dist)
        ans = [p / res for p in ans]
    return ans
