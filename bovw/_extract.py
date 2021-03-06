import cv2
import tables
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from scipy.sparse import lil_matrix


FLANN_INDEX_COMPOSITE = 3
FLANN_DIST_L2 = 1


class _Codebook(object):
    def __init__(self, hdffile):
        clusterf = tables.open_file(hdffile)
        self._clusters = np.array(clusterf.get_node('/clusters'))
        clusterf.close()

        self._clusterids = np.array(xrange(0, self._clusters.shape[0]), dtype=np.int)
        self.n_clusters = self._clusters.shape[0]

        self._flann = cv2.flann_Index(self._clusters,
                                      dict(algorithm=FLANN_INDEX_COMPOSITE,
                                           distance=FLANN_DIST_L2,
                                           iterations=10,
                                           branching=16,
                                           trees=50))

    def predict(self, Xdesc):
        """
        Takes Xdesc a (n,m) numpy array of n img descriptors length m and returns
        (n,1) where every n has been assigned to a cluster id.
        """
        (_, m) = Xdesc.shape
        (_, cm) = self._clusters.shape
        assert m == cm

        result, dists = self._flann.knnSearch(Xdesc, 1, params={})
        return result


def rootSIFT_descriptor(f):
    """
    Extract root sift descriptors from image stored in file f

    :param: f : str or unicode filename
    :return: desc : numpy array [n_sift_desc, 128]
    """
    img = cv2.imread(f)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, desc = sift.detectAndCompute(img_gray, None)

    if desc is None:
        print('Warning: No SIFT features found in {}'.format(f))
        return None

    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc)

    return desc


def _term_counts(f, codebook):
    """
    Takes a list of SIFT vectors from an image and matches
    each SIFT vector to its nearest equivalent in the codebook

    :param: f : Image file path
    :return: countvec : sparse vector of counts for each visual-word in the codebook
    """
    desc = rootSIFT_descriptor(f)
    if desc is None:
        # if no sift features found return 0 count vector
        return lil_matrix((1, codebook.n_clusters), dtype=np.int)

    matches = codebook.predict(desc)
    unique, counts = np.unique(matches, return_counts=True)

    countvec = lil_matrix((1, codebook.n_clusters), dtype=np.int)
    countvec[0, unique] = counts
    return countvec


class CountVectorizer(object):
    def __init__(self, vocabulary_file, n_jobs=1):
        self._codebook = _Codebook(hdffile=vocabulary_file)
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

    def transform(self, raw_images):
        """
        Transform image files to a visual-word count matrix.
        :param: raw_images : iterable
                    An iterable of str or unicode filenames
        :return: X : sparse matrix, [n_images, n_visual_words]
                     visual-word count matrix
        """

        sparse_rows = Parallel(backend='threading', n_jobs=self.n_jobs)(
            (delayed(_term_counts)(f, self._codebook) for f in raw_images)
        )

        X = lil_matrix((len(raw_images), self._codebook.n_clusters),
                       dtype=np.int)

        for i, sparse_row in enumerate(sparse_rows):
            X[i] = sparse_row

        return X.tocsr()

