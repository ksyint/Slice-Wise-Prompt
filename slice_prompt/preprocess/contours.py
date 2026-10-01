"""Per-slice connected components and consistent contour rendering."""
import numpy as np
from scipy import ndimage

COLORS = {"blue": (0, 0, 1), "red": (1, 0, 0)}


def largest_component(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def slice_contours(mask, thickness=1, largest=True):
    if thickness < 1 or thickness % 2 == 0:
        raise ValueError('Contour thickness must be an odd positive integer.')
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError('Expected mask (Z,Y,X).')
    contours = np.zeros_like(mask)
    for index, region in enumerate(mask):
        region = largest_component(region) if largest else region
        boundary = region & ~ndimage.binary_erosion(region, structure=np.ones((3, 3)), border_value=0)
        if thickness > 1:
            boundary = ndimage.binary_dilation(boundary, iterations=thickness // 2, structure=np.ones((3, 3)))
        contours[index] = boundary
    return contours


def render_prompt(volume, mask, color='blue', thickness=1, center=-600, width=1500):
    if volume.shape != mask.shape or width <= 0:
        raise ValueError('Volume/mask shapes must match and window width must be positive.')
    if color not in COLORS:
        raise ValueError(f'Unsupported color: {color}')
    gray = np.clip((volume.astype('float32') - center + width / 2) / width, 0, 1)
    rgb = np.repeat(gray[:, :, :, None], 3, axis=-1)
    contours = slice_contours(mask, thickness)
    rgb[contours] = COLORS[color]
    return rgb, contours
