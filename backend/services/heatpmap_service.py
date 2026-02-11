import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class HeatmapService:
    def __init__(self, gaussian_wh=200):
        self.gaussian_wh = gaussian_wh
        self.kernel = self._generate_gaussian_kernel(gaussian_wh, gaussian_wh / 6)

    def _generate_gaussian_kernel(self, size, sd):
        """Vectorized Gaussian generation (much faster than nested loops)"""
        x = np.arange(0, size)
        y = np.arange(0, size)
        x, y = np.meshgrid(x, y)
        xo, yo = size / 2, size / 2
        
        # Original logic: np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy))))
        kernel = np.exp(-1.0 * (((x - xo)**2 / (2 * sd**2)) + ((y - yo)**2 / (2 * sd**2))))
        return kernel

    def create_heatmap(self, gazepoints, dispsize, background_img=None, alpha=0.6):
        """
        Refactored from generate_heatmap_figure in your friend's code.
        Returns a Base64 string of the final image.
        """
        width, height = dispsize

        if background_img is not None:
            screen = np.array(background_img).astype('float32') / 255.0
            if screen.shape[2] == 4: screen = screen[:, :, :3]
        else:
            screen = np.zeros((height, width, 3), dtype='float32')

        strt = int(self.gaussian_wh / 2)
        heatmap_padded = np.zeros((height + 2 * strt, width + 2 * strt), dtype=float)


        for p in gazepoints:
            x_pos = strt + int(p[0]) - strt
            y_pos = strt + int(p[1]) - strt
            
            if (0 < x_pos < width) and (0 < y_pos < height):
                heatmap_padded[y_pos:y_pos + self.gaussian_wh, x_pos:x_pos + self.gaussian_wh] += self.kernel * p[2]

        heatmap = heatmap_padded[strt:height + strt, strt:width + strt]

        fig = plt.figure(figsize=(width/100, height/100), dpi=100, frameon=False)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        if background_img is not None:
            ax.imshow(background_img, alpha=0.8)
        
        if np.any(heatmap > 0):
            lowbound = np.mean(heatmap[heatmap > 0]) * 0.5
            heatmap_masked = np.ma.masked_where(heatmap < lowbound, heatmap)
            
            ax.imshow(heatmap_masked, cmap='turbo', alpha=0.8, interpolation='gaussian')
        
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

heatmap_service = HeatmapService()