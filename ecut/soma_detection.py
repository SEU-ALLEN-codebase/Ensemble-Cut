import numpy as np
from skimage import filters, morphology, measure
from scipy import ndimage
import pandas as pd
from sklearn.cluster import DBSCAN


class DetectImage3D:
    """
    Modified from brainlit soma detection algorithm
    """

    def __init__(self, processing_size=(50, 160, 160), diam_range=(5, 20)):
        """

        :param processing_size: the image size during processing. Downsize the image to the optimal range for
            morphological operations.
        :param diam_range: the minimum and maximum diameter in micrometers allowed for the detected somata.
        """
        self._processing_size = processing_size
        self._volume_range = diam_range

    def __call__(self, img, res: list[float, float, float]) -> list[np.ndarray]:
        """

        :param img: the image for detection, indexed by (z, y, x)
        :param res: the resolution of z, y, x in micrometers, for computing radius.
        :return: a list of soma centers and the 3D mask
        """
        assert len(res) == img.ndim == 3, "Only 3D images are supported"
        assert res[0] > 0 and res[1] > 0 and res[2] > 0, "Resolution must be positive"

        desired_size = np.array(self._processing_size)
        zoom_factors = desired_size / img.shape
        res = np.divide(res, zoom_factors)
        out = ndimage.zoom(img, zoom=zoom_factors)
        out = out / np.max(out.flatten())
        t = filters.threshold_otsu(out)
        out = out > t
        selem_size = np.amax(np.ceil(zoom_factors)).astype(int)
        clean_selem = morphology.octahedron(selem_size)
        out = morphology.erosion(out, clean_selem)
        out, num_labels = morphology.label(out, background=0, return_num=True)
        properties = ["label", "equivalent_diameter"]
        props = measure.regionprops_table(out, properties=properties)

        df_props = pd.DataFrame(props)
        out = []
        for _, row in df_props.iterrows():
            l, d = row[properties]
            dmu = d * np.mean(res[:1])
            if self._volume_range[0] <= dmu <= self._volume_range[1]:
                ids = np.where(out == l)
                centroid = np.round([np.median(u) for u in ids])
                centroid = np.divide(centroid, zoom_factors)
                out.append(centroid)
        return out


class TiledDetectImage3D:
    def __init__(self, tile_size=(64, 256, 256), omit_border=(8, 16, 16), merge_distance=30,
                 base_detector=DetectImage3D()):
        """

        :param tile_size: The size of a single tile, indexed by z, y, x.
        :param omit_border: The range of the border to omit, any detection falling on the border will be omitted.
        :param base_detector: the tile detector.
        """
        self._find_soma = base_detector
        self._tile_size = np.array(tile_size)
        self._merge_distance = merge_distance
        self._omit_border = np.array(omit_border)

    def __call__(self, img: np.ndarray, res: list[float, float, float]):
        assert len(res) == img.ndim == 3, "Only 3D images are supported"
        assert res[0] > 0 and res[1] > 0 and res[2] > 0, "Resolution must be positive"

        steps = np.ceil(img.shape / (self._tile_size - 2 * self._omit_border)).astype(int)
        hf = self._tile_size // 2
        z = np.linspace(hf[0], img.shape[0] - hf[0], steps, dtype=int)
        y = np.linspace(hf[1], img.shape[1] - hf[1], steps, dtype=int)
        x = np.linspace(hf[2], img.shape[2] - hf[2], steps, dtype=int)

        out = []
        for zz in z:
            for yy in y:
                for xx in x:
                    s = (zz, yy, xx) - hf
                    e = (zz, yy, xx) + hf
                    tile = img[s[0]: e[0], s[1]: e[1], s[2]: e[2]]
                    pts = self._find_soma(tile, res)
                    pts = [s + p for p in pts if ((self._omit_border < p) & (p < self._tile_size - self._omit_border)).all()]
                    out.extend(pts)

        DBSCAN(self._merge_distance, )