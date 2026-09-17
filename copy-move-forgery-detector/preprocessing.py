import cv2
import numpy as np
import logging
from typing import Tuple
from utils import InvalidImageError, timeit

logger = logging.getLogger(__name__)

@timeit
def load_image(path: str) -> np.ndarray:
    """
    Loads an image from the given path.
    Raises InvalidImageError if the file does not exist or cannot be read.
    """
    image = cv2.imread(path)
    if image is None:
        raise InvalidImageError(f"Could not load image from {path}. Check if file exists and is a valid image.")
    logger.info(f"Loaded image from {path} with shape {image.shape}")
    return image

@timeit
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

@timeit
def resize_if_large(image: np.ndarray, max_dimension: int = 1000) -> np.ndarray:
    """
    Resizes the image if its largest dimension exceeds max_dimension,
    maintaining the aspect ratio. This keeps SIFT performant.
    """
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h} (scale: {scale:.2f})")
        return resized
    return image
