import tables
import pyflann

import cv2
import numpy as np
from scipy.sparse import lil_matrix


class _Codebook(object):
    def __init__(self, hdffile):
        clusterf = tables.open_file(hdffile)
        self._clusters = np.array(clusterf.get_node('/clusters'))
        clusterf.close()

        self._clusterids = np.array(xrange(0, self._clusters.shape[0]), dtype=np.int)
        self.n_clusters = self._clusters.shape[0]
        self.__flann = pyflann.FLANN()
        self.__flann_params = self.__flann.build_index(self._clusters,
                                                       algorithm='autotuned',
                                                       target_precision=0.9,
                                                       log_level='info')

    def predict(self, Xdesc):
        """
        Takes Xdesc a (n,m) numpy array of n img descriptors length m and returns
        (n,1) where every n has been assigned to a cluster id.
        """
        (_, m) = Xdesc.shape
        (_, cm) = self._clusters.shape
        assert m == cm

        result, dists = self.__flann.nn_index(Xdesc, num_neighbours=1,
                                              checks=self.__flann_params['checks'])
        return result


class CountVectorizer(object):
    def __init__(self, vocabulary_file):
        self._codebook = _Codebook(hdffile=vocabulary_file)

    def _rootsift_from_file(self, f):
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
            raise Exception('No SIFT features found in {}'.format(f))

        desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc)

        return desc

    def _term_counts(self, desc):
        """
        Takes a list of SIFT vectors from an image and matches
        each SIFT vector to its nearest equivalent in the codebook

        :param: desc : List of SIFT descriptors from a single image
        :return: countvec : sparse vector of counts for each visual-word in the codebook
        """
        matches = self._codebook.predict(desc)
        unique, counts = np.unique(matches, return_counts=True)

        countvec = lil_matrix((1, self._codebook.n_clusters), dtype=np.int)
        countvec[0, unique] = counts
        return countvec

    def transform(self, raw_images):
        """
        Transform image files to a visual-word count matrix.
        :param: raw_images : iterable
                    An iterable of str or unicode filenames
        :return: X : sparse matrix, [n_images, n_visual_words]
                     visual-word count matrix
        """
        X = lil_matrix((len(raw_images), self._codebook.n_clusters),
                       dtype=np.int)

        for row, f in enumerate(raw_images):
            X[row] = self._term_counts(self._rootsift_from_file(f))

        return X.tocsr()

