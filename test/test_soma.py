import unittest
from ecut.soma_detection import *
from ecut.swc_handler import parse_swc
from v3dpy.loaders import PBD
import matplotlib.pyplot as plt


class TestSomaDetection(unittest.TestCase):
    def test_brainlit_algorithm(self):
        path = r"D:\rectify\crop_8bit\18453_9442_3817_6561.v3dpbd"
        img = PBD().load(path)[0]
        mod = DetectImage()
        centers = mod.predict(img, [.2, .2, 1.])
        print(centers)
        fig, ax = plt.subplots()
        ax.imshow(img.max(axis=0), cmap='gray')
        for p in centers:
            ax.plot(p[2], p[1], '.r', markersize=15)
        plt.show()

    def test_tile_blocks(self):
        path = r"D:\rectify\crop_8bit\18453_9442_3817_6561.v3dpbd"
        img = PBD().load(path)[0]
        mod = DetectTiledImage(nproc=16)
        centers = mod.predict(img, [.2, .2, 1.])
        print(centers)
        fig, ax = plt.subplots()
        ax.imshow(img.max(axis=0), cmap='gray')
        for p in centers:
            ax.plot(p[2], p[1], '.r', markersize=15)
        plt.show()

    def test_swc(self):
        path = r"D:\rectify\my_app2\18453_9442_3817_6561.swc"
        swc = parse_swc(path)
        img = PBD().load(r"D:\rectify\crop_8bit\18453_9442_3817_6561.v3dpbd")[0].max(axis=0)
        mod = DetectTracingMask()
        centers = mod.predict(swc, [.2, .2, 1.])
        print(centers)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        for p in centers:
            ax.plot(p[0], p[1], '.r', markersize=15)
        plt.show()


if __name__ == '__main__':
    unittest.main()
