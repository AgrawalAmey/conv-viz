from keras.datasets import cifar100
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data


class TSNE(object):
    def __init__(self, model):
        _, (x, _) = cifar100.load_data(label_mode='fine')
        self.x = x[:100, :]

    def plot(self):
        embeddings = model.predict(self.x)
        tsne = TSNE(n_components=2, perplexity=30,
                    verbose=2).fit_transform(embeddings)
        fig, ax = plt.subplots()
        t_x, t_y = tsne[:, 0], tsne[:, 1]
        self.imscatter(t_x, t_y, self.x, ax=ax)
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
