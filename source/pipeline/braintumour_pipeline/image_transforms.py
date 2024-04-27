import cv2
import numpy as np
from detectron2.data.transforms import Transform

class GaussianBlurTransform(Transform):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def apply_image(self, img):
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
    
    def apply_coords(self, coords):
        # Since blurring does not affect coordinates, just return them unchanged
        return coords

class AddGaussianNoiseTransform(Transform):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def apply_image(self, img):
        noise = np.random.normal(0, self.sigma, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def apply_coords(self, coords):
        # Since adding noise does not affect coordinates, just return them unchanged
        return coords