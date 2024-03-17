import unittest
from ecut.soma_detection import *
from ecut.swc_handler import parse_swc
from v3dpy.loaders import PBD
import matplotlib.pyplot as plt


class TestSomaDetection(unittest.TestCase):
    def setUp(self):
        self.swc = parse_swc(r"D:\rectify\my_app2\18453_9442_3817_6561.swc")
        self.img = PBD().load(r"D:\rectify\crop_8bit\18453_9442_3817_6561.v3dpbd")[0]
        self.res = [.25, .25, 1]

    def test1_brainlit_algorithm(self):
        mod = DetectImage()
        centers = mod.predict(self.img, self.res)
        print(centers)
        fig, ax = plt.subplots()
        ax.imshow(self.img.max(axis=0), cmap='gray')
        for p in centers:
            ax.plot(p[2], p[1], '.r', markersize=15)
        plt.show()

    def test2_tile_brainlit(self):
        mod = DetectTiledImage([300, 300, 200], nproc=16)
        centers = mod.predict(self.img, self.res)
        print(centers)
        fig, ax = plt.subplots()
        ax.imshow(self.img.max(axis=0), cmap='gray')
        for p in centers:
            ax.plot(p[2], p[1], '.r', markersize=15)
        plt.show()

    def test3_swc(self):
        mod = DetectTracingMask()
        centers = mod.predict(self.swc, self.res)
        print(centers)
        fig, ax = plt.subplots()
        ax.imshow(self.img.max(axis=0), cmap='gray')
        for p in centers:
            ax.plot(p[0], p[1], '.r', markersize=15)
        plt.show()

    def test4_dt_algorithm(self):
        mod = DetectDistanceTransform()
        centers = mod.predict(self.img, self.res)
        print(centers)
        fig, ax = plt.subplots()
        ax.imshow(self.img.max(axis=0), cmap='gray')
        for p in centers:
            ax.plot(p[2], p[1], '.r', markersize=15)
        plt.show()

    def test5_tile_dt(self):
        mod = DetectTiledImage(nproc=16, base_detector=DetectDistanceTransform())
        centers = mod.predict(self.img, self.res)
        print(centers)
        fig, ax = plt.subplots()
        ax.imshow(self.img.max(axis=0), cmap='gray')
        for p in centers:
            ax.plot(p[2], p[1], '.r', markersize=15)
        plt.show()

    def test6_consensus(self):
        centers_list = []
        centers = DetectImage().predict(self.img, self.res)
        centers_list.append(centers)
        centers = DetectTiledImage([300, 300, 200], nproc=16).predict(
            self.img, [.25, .25, 1.])
        centers_list.append(centers)
        centers = DetectTracingMask().predict(self.swc, self.res)
        centers_list.append(centers)
        centers = DetectDistanceTransform().predict(self.img, self.res)
        centers_list.append(centers)
        centers = (DetectTiledImage(nproc=16, base_detector=DetectDistanceTransform()).
                   predict(self.img, self.res))
        centers_list.append(centers)
        centers = soma_consensus(*centers_list, res=self.res)
        print(centers)
        path = r"D:\rectify\crop_8bit\18453_9442_3817_6561.v3dpbd"
        img = PBD().load(path)[0]
        fig, ax = plt.subplots()
        ax.imshow(img.max(axis=0), cmap='gray')
        for p in centers:
            ax.plot(p[2], p[1], '.r', markersize=15)
        plt.show()


if __name__ == '__main__':
    unittest.main()
