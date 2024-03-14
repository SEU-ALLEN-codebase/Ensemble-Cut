from .base_types import BaseSomaDetector


class ImageBasedDetector(BaseSomaDetector):

    def __init__(self, swc, img,):
        super().__init__(swc)
        self._img = img

    def run(self):
        pass