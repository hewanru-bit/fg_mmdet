import matplotlib.pyplot as plt
import numpy as np
import torch
def create2dGaussian(mu, sigma, nx, ny):
    x, y = np.meshgrid(np.linspace(-nx / 2.0, +nx / 2.0, nx), np.linspace(-ny / 2.0, +ny / 2.0, ny))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    # just for debugging:
    np.set_printoptions(precision=1, suppress=True)
    print(g.shape)
    # print(g)
    plt.imshow(g, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.show()

    return g
# create2dGaussian(1, 10, 25, 25) # seems to work
# create2dGaussian(1, 5, 25, 25) # the middle is not quite the peak anymore
# create2dGaussian(1, 1, 25, 25) # the above problem more clearly visible
# create2dGaussian(1, 1, 5, 5) # here it is extrem as the middle is now only 0.6
#
# ga = create2dGaussian(1, 10, 25, 25) # mean is still 1 and not 5


def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(
        radius, sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[ radius - top:radius + bottom,
                                      radius - left:radius + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap



heatmap = torch.zeros(50,50)
center = (25, 25)
radius =10

map = gen_gaussian_target(heatmap, center, radius)
print(map.shape)
plt.imshow(map)
plt.show()