from song.song import SONG
from sklearn.decomposition import IncrementalPCA
import umap

__factory = {
    'SONG': SONG,
    'UMAP': umap.UMAP,
    'PCA': IncrementalPCA
}


def create(name, *args, **kwargs):
    """
    Create a DR model instance.

    Parameters
    ----------
    name : str
        the name of model
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)

    return __factory[name](*args, **kwargs)
