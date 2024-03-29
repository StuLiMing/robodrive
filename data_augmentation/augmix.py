import numpy as np
from PIL import Image

from augmentations import augmentations
from config import *

def apply_op(image, op, severity):
	image = np.clip(image * 255., 0, 255).astype(np.uint8)
	pil_img = Image.fromarray(image)  # Convert to PIL.Image
	pil_img = op(pil_img, severity)
	return np.asarray(pil_img) / 255.

def normalize(image):
	"""Normalize input image channel-wise to zero mean and unit variance."""
	image = image.transpose(2, 0, 1)  # Switch to channel-first
	mean, std = np.array(MEAN), np.array(STD)
	image = (image - mean[:, None, None]) / std[:, None, None]
	return image.transpose(1, 2, 0)

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
	"""Perform AugMix augmentations and compute mixture.

	Args:
		image: Raw input image as float32 np.ndarray of shape (h, w, c)
		severity: Severity of underlying augmentation operators (between 1 to 10).
		width: Width of augmentation chain
		depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
			from [1, 3]
		alpha: Probability coefficient for Beta and Dirichlet distributions.

	Returns:
		mixed: Augmented and mixed image.
	"""
	ws = np.float32(
			np.random.dirichlet([alpha] * width))
	m = np.float32(np.random.beta(alpha, alpha))

	mix = np.zeros_like(image)
	for i in range(width):
		image_aug = image.copy()
		d = depth if depth > 0 else np.random.randint(1, 4)
		for _ in range(d):
			op = np.random.choice(augmentations)
			image_aug = apply_op(image_aug, op, severity)
		# Preprocessing commutes since all coefficients are convex
		mix += ws[i] * normalize(image_aug)

	mixed = (1 - m) * normalize(image) + m * mix
	return mixed