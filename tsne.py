from keras.datasets import cifar10
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from scipy.misc import imresize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class TSNEViz(object):
    def __init__(self, model):
        self.model = model
        _, (x, _) = cifar10.load_data()

        # Selecting only n images
        n = 5
        self.x_orig = np.zeros((n, x.shape[1], x.shape[2], 3))
        self.x = np.zeros((n, 224, 224, 3))
        # Get random indicies
        indicies = np.random.randint(0, high=x.shape[0], size=n)
        # And resize the images
        for i in range(n):
            self.x_orig[i, :] = x[indicies[i], :]
            self.x[i, :] = imresize(x[indicies[i], :], (224, 224))

    def plot(self):
        embeddings = self.model.predict(self.x)
        tsne = TSNE().fit_transform(embeddings)
        fig, ax = plt.subplots()
        t_x, t_y = tsne[:, 0], tsne[:, 1]
        self.imscatter(t_x, t_y, self.x_orig, ax=ax)
        plt.show()

    def imscatter(self, x, y, images, ax=None, zoom=1):
        if ax is None:
            ax = plt.gca()
        x, y = np.atleast_1d(x, y)
        artists = []
        for i in range(x.shape[0]):
            im = OffsetImage(images[i, :], zoom=zoom)
            ab = AnnotationBbox(
                im, (x[i], y[i]), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists
