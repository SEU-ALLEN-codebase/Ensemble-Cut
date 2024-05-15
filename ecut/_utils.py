from sklearn.decomposition import PCA
import numpy as np
from .morphology import Morphology


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


def get_angle(pts_list1: list[np.ndarray], pts_list2: list[np.ndarray], res):
    """
    The angle between 2 vectors (fitted from 2 point lists), but supplementary.
    the vectors share the start point, but to make it fit for scoring, its supplementary is returned.
    so a smaller angle means a more straight connection.

    :param pts_list1: a list of 3D points for one branch
    :param pts_list2: a list of 3D points for another branch
    :return: an angle in arc
    """
    vec1 = -line_fit_pca(pts_list1) * res
    vec2 = line_fit_pca(pts_list2) * res
    vec3 = (pts_list2[0] - pts_list1[0]) * res
    cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    a1 = np.arccos(np.clip(cos, -1, 1))
    if np.linalg.norm(vec3) == 0:
        return a1, 0
    cos1 = vec1.dot(vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
    cos2 = vec2.dot(vec3) / (np.linalg.norm(vec2) * np.linalg.norm(vec3))
    return a1, np.pi - (np.arccos(np.clip(cos1, -1, 1)) + np.arccos(np.clip(cos2, -1, 1))) / 2


def get_gof(pts_list: list[np.ndarray], soma: np.ndarray, res, eps) -> float:
    pts_list = np.array(pts_list)
    tan = (pts_list[1:] - pts_list[:-1]) * res
    tan_norm = np.linalg.norm(tan, axis=1, keepdims=True)
    tan /= tan_norm + eps
    ps = (pts_list[:-1] - soma) * res
    ps_norm = np.linalg.norm(ps, axis=1, keepdims=True)
    ps /= ps_norm + eps
    proj = np.clip(np.sum(tan * ps, axis=1), -1, 1)
    gof = np.mean(np.arccos(proj))
    return gof
