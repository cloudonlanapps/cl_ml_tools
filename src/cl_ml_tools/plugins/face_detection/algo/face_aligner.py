from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def align_and_crop(
    img: NDArray[np.uint8],
    landmarks: list[tuple[float, float]] | NDArray[np.float32],
    image_size: int = 112,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """
    Align and crop the face from the image based on the given landmarks.

    Args:
        img (np.ndarray): The full image (not the cropped bounding box).
        landmarks (List[tuple] or np.ndarray): List of 5 keypoints (x, y).
            Order: [Right Eye, Left Eye, Nose, Right Mouth, Left Mouth] (Person relative)
            which corresponds to [Left side of img, Right side of img, ...]
        image_size (int, optional): The size to which the image should be resized. Defaults to 112.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The aligned face image and the transformation matrix.
    """
    # Define the reference keypoints used in ArcFace model (112x112)
    # [38.29, 51.69] is roughly 34% width (Left side of image -> Person Right Eye)
    _arcface_ref_kps = np.array(
        [
            [38.2946, 51.6963],  # Person Right Eye (Image Left)
            [73.5318, 51.5014],  # Person Left Eye (Image Right)
            [56.0252, 71.7366],  # Nose
            [41.5493, 92.3655],  # Person Right Mouth (Image Left)
            [70.7299, 92.2041],  # Person Left Mouth (Image Right)
        ],
        dtype=np.float32,
    )

    if isinstance(landmarks, list):
        landmarks = np.array(landmarks, dtype=np.float32)

    # Ensure the input landmarks have exactly 5 points
    assert len(landmarks) == 5

    # Validate that image_size is divisible by either 112 or 128
    assert image_size % 112 == 0 or image_size % 128 == 0

    # Adjust the scaling factor (ratio)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio  # Horizontal shift for 128 scaling

    # Apply the scaling and shifting to the reference keypoints
    dst = _arcface_ref_kps * ratio
    dst[:, 0] += diff_x

    # Estimate the similarity transformation matrix
    M, inliers = cv2.estimateAffinePartial2D(landmarks, dst, ransacReprojThreshold=1000)
    # verify inliers is not None and all true?
    # cv2.estimateAffinePartial2D returns (M, inliers).
    # If it fails, M might be None.
    _ = inliers

    # if M is None:
    #    raise ValueError("Failed to estimate affine transformation")

    # Apply the affine transformation
    aligned_img = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

    val0: NDArray[np.uint8] = cast(NDArray[np.uint8], aligned_img)
    val1: NDArray[np.float64] = cast(NDArray[np.float64], M)

    return val0, val1
