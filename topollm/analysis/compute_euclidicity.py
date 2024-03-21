# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################

# This is a script to calculate and store euclidicity
# scores (as defined in
# `https://proceedings.mlr.press/v202/von-rohrscheidt23a.html`)
# for two given arrays for a comparison
# of embeddings of a base model and a corresponding
# fine-tuned variant. To obtain these arrays, the
# `data_prep.py` may be used.

# third party imports
import gudhi as gd
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree
from gph import ripser_parallel

# provide names of numpy array to be used for dimension estimation
data_name = "sample_embeddings_data-multiwoz21_split-test_ctxt-dataset_entry_base_no_paddings.npy"
data_name_finetuned = "sample_embeddings_data-multiwoz21_split-test_ctxt-dataset_entry_finetuned_no_paddings.npy"

arr_no_pad = np.load(data_name)
arr_no_pad_finetuned = np.load(data_name_finetuned)

# adjust parameters for euclidicity calculations:
# `subsample_size`: number of random points in the data for which euclidicity is calculated
# `k`: number of nearest neighbor to consider
# `max_dim`: maximum dimension for the simplex tree thiat is built
# `n_steps`: number of steps in order to vary the locality for the eucldicity values

subsample_size = 200
k = 50
max_dim = 4
n_steps = 10

class GUDHI:
    """Wrapper for GUDHI persistent homology calculations."""

    def __call__(self, X, max_dim):
        """Calculate persistent homology.
        Parameters
        ----------
        X : np.array of shape ``(N, d)``
            Input data set.
        max_dim : int
            Maximum dimension for calculations
        Returns
        -------
        np.array
            Full barcode (persistence diagram) of the data set.
        """
        barcodes = (
            gd.RipsComplex(points=X)
            .create_simplex_tree(max_dimension=max_dim)
            .persistence()
        )

        if len(barcodes) == 0:
            return None, -1

        # TODO: Check whether this is *always* a feature of non-zero
        # persistence.
        max_dim = np.max([d for d, _ in barcodes])

        # TODO: We are throwing away dimensionality information; it is
        # thus possible that we are matching across different dimensions
        # in any distance calculation.
        barcodes = np.asarray([np.array(x) for _, x in barcodes])

        return barcodes, max_dim

    def distance(self, D1, D2):
        """Calculate Bottleneck distance between two persistence diagrams."""
        return gd.bottleneck_distance(D1, D2)


class Ripser:
    def __call__(self, X, max_dim):
        if len(X) == 0:
            return [], -1

        diagrams = ripser_parallel(
            X, maxdim=max_dim, collapse_edges=True, n_threads=-1
        )

        diagrams = diagrams["dgms"]

        max_dim = np.max([d for d, D in enumerate(diagrams) if len(D) > 0])

        diagrams = np.row_stack(diagrams)

        return diagrams, max_dim

    def distance(self, D1, D2):
        return gd.bottleneck_distance(D1, D2)


def sample_from_ball(n=100, d=2, r=1, seed=None):
    """Sample `n` data points from a `d`-ball in `d` dimensions.
    Parameters
    -----------
    n : int
        Number of data points in ball.
    d : int
        Dimension of the ball. Notice that there is an inherent shift in
        dimension if you compare a ball to a sphere.
    r : float
        Radius of ball.
    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.
    Returns
    -------
    np.array of shape `(n, d)`
        Array of sampled coordinates.
    References
    ----------
    .. [Voelker2007] A. Voelker et al, Efficiently sampling vectors and
    coordinates from the $n$-sphere and $n$-ball, Technical Report,
    2017. http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    rng = np.random.default_rng(seed)

    # This algorithm was originally described in the following blog
    # post:
    #
    # http://extremelearning.com.au/how-to-generate-uniformly-random-points
    # -on-n-spheres-and-n-balls/
    #
    # It's mind-boggling that this works but it's true!
    U = rng.normal(size=(n, d + 2))
    norms = np.sqrt(np.sum(np.abs(U) ** 2, axis=-1))
    U = r * U / norms[:, np.newaxis]
    X = U[:, 0:d]

    return np.asarray(X)


def sample_from_annulus(n, r, R, d=2, seed=None):
    """Sample points from an annulus.
    This function samples `N` points from an annulus with inner radius `r`
    and outer radius `R`.
    Parameters
    ----------
    n : int
        Number of points to sample
    r : float
        Inner radius of annulus
    R : float
        Outer radius of annulus
    d : int
        Dimension of the annulus. Technically, for higher dimensions, we
        should call the resulting space a "hyperspherical shell." Notice
        that the algorithm for sampling points in higher dimensions uses
        rejection sampling, so its efficiency decreases as the dimension
        increases.
    seed : int, instance of `np.random.Generator`, or `None`
        Seed for the random number generator, or an instance of such
        a generator. If set to `None`, the default random number
        generator will be used.
    Returns
    -------
    np.array of shape `(n, 2)`
        Array containing sampled coordinates.
    """
    if r >= R:
        raise RuntimeError(
            "Inner radius must be less than or equal to outer radius"
        )

    rng = np.random.default_rng(seed)

    if d == 2:
        thetas = rng.uniform(0, 2 * np.pi, n)

        # Need to sample based on squared radii to account for density
        # differences.
        radii = np.sqrt(rng.uniform(r ** 2, R ** 2, n))

        X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    else:
        X = np.empty((0, d))

        while True:
            sample = sample_from_ball(n, d, r=R, seed=rng)
            norms = np.sqrt(np.sum(np.abs(sample) ** 2, axis=-1))

            X = np.row_stack((X, sample[norms >= r]))

            if len(X) >= n:
                X = X[:n, :]
                break

    return X


"""Euclidicity implementation."""


class Euclidicity:
    """Functor for calculating Euclidicity of a point cloud."""

    def __init__(
            self,
            max_dim,
            r=None,
            R=None,
            s=None,
            S=None,
            n_steps=10,
            data=None,
            method="gudhi",
            model_sample_fn=None,
    ):
        """Initialise new instance of functor.
        Sets up a new instance of the Euclidicity functor and stores
        shared parameters that will be used for the calculation. The
        client has the choice of either providing global parameters,
        or adjusting them on a per-point basis.
        Parameters
        ----------
        max_dim : int
            Maximum dimension for persistent homology approximations.
            This is the *only* required parameter.
        r : float, optional
            Minimum inner radius of annulus
        R : float, optional
            Maximum inner radius of annulus
        s : float, optional
            Minimum outer radius of annulus
        S : float, optional
            Maximum outer radius of annulus
        n_steps : int, optional
            Number of steps for the radius parameter grid of the
            annulus. Note that the complexity of the function is
            quadratic in the number of steps.
        data : np.array or None
            If set, prepares a tree for nearest-neighbour and radius
            queries on the input data set. This can lead to substantial
            speed improvements in practice.
        method : str
            Persistent homology calculation method. At the moment, only
            "gudhi" and "ripser" are supported. "gudhi" is better for a
            small, low-dimensional data set, while "ripser" scales well
            to larger, high-dimensional point clouds.
        model_sample_fn : callable
            Function to be called for sampling from a comparison space.
            The function is being supplied with the number of samples,
            the radii of the annulus, and the intrinsic dimension. Its
            output must be a point cloud representing the annulus. If no
            sample function is provided, the class will default to
            compare the topological features with those of fixed
            Euclidean annulus.
        """
        self.r = r
        self.R = R
        self.s = s
        self.S = S

        self.n_steps = n_steps
        self.max_dim = max_dim

        self.model_sample_fn = model_sample_fn

        if method == "gudhi":
            self.vr = GUDHI()
        elif method == "ripser":
            self.vr = Ripser()
        else:
            raise RuntimeError("No persistent homology calculation selected.")

        # Prepare KD tree to speed up annulus calculations. We make this
        # configurable to permit both types of workflows.
        if data is not None:
            self.tree = KDTree(data)
        else:
            self.tree = None

    def __call__(self, X, x, **kwargs):
        """Calculate Euclidicity of a specific point.
        Parameters
        ----------
        X : np.array or tensor of shape ``(N, d)``
            Input data set. Must be compatible with the persistent
            homology calculations.
        x : np.array, tensor, or iterable of shape ``(d, )``
            Input point.
        Other Parameters
        ----------------
        r : float, optional
            Minimum inner radius of annulus. Will default to global `r`
            parameter if not set.
        R : float, optional
            Maximum inner radius of annulus. Will default to global `R`
            parameter if not set.
        s : float, optional
            Minimum outer radius of annulus. Will default to global `s`
            parameter if not set.
        S : float, optional
            Maximum outer radius of annulus. Will default to global `S`
            parameter if not set.
        Returns
        -------
        Tuple of np.array, np.array
            1D array containing Euclidicity estimates. The length of the
            array depends on the number of scales. The second array will
            contain the persistent intrinsic dimension (PID) values.
        """
        r = kwargs.get("r", self.r)
        R = kwargs.get("R", self.R)
        s = kwargs.get("s", self.s)
        S = kwargs.get("S", self.S)

        bottleneck_distances = []
        dimensions = []

        for r in np.linspace(r, R, self.n_steps):
            for s in np.linspace(s, S, self.n_steps):
                if r < s:
                    dist, dim = self._calculate_euclidicity(
                        r, s, X, x, self.max_dim
                    )

                    bottleneck_distances.append(dist)
                    dimensions.append(dim)

        return np.asarray(bottleneck_distances), np.asarray(dimensions)

    # Auxiliary method for performing the 'heavy lifting' when it comes
    # to Euclidicity calculations.
    def _calculate_euclidicity(self, r, s, X, x, d):
        if self.tree is not None:
            inner_indices = self.tree.query_radius(x.reshape(1, -1), r)[0]
            outer_indices = self.tree.query_radius(x.reshape(1, -1), s)[0]

            annulus_indices = np.setdiff1d(outer_indices, inner_indices)
            annulus = X[annulus_indices]
        else:
            annulus = np.asarray(
                [
                    np.asarray(p)
                    for p in X
                    if np.linalg.norm(x - p) <= s
                       and np.linalg.norm(x - p) >= r
                ]
            )

        barcodes, max_dim = self.vr(annulus, d)

        if max_dim < 0:
            return np.nan, max_dim

        if self.model_sample_fn is not None:
            euclidean_annulus = self.model_sample_fn(
                n=len(annulus), r=r, R=s, d=d
            )
            barcodes_euclidean, _ = self.vr(euclidean_annulus, d)

        # No sampling function has been specified. Compare to a fixed
        # annulus with known persistent homology.
        #
        # TODO: Technically, the single feature should be put into
        # a persistence diagram of the right dimension. Let us not
        # do that for now (since we stack diagrams anyway).
        else:
            barcodes_euclidean = np.asarray([[0, np.inf], [r, s]])

        if barcodes_euclidean is None:
            return np.nan, max_dim

        dist = self.vr.distance(barcodes, barcodes_euclidean)
        return dist, max_dim


X = arr_no_pad
X_finetuned = arr_no_pad_finetuned

sub_idx = np.random.choice(range(len(X)), replace=False, size=subsample_size)

query_points = X[sub_idx]
query_points_finetuned = X_finetuned[sub_idx]


def estimate_scales(X, query_points, k_max):
    """Perform simple scale estimation of the data set.
    Parameters
    ----------
    k_max : int
        Maximum number of neighbours to consider for the local scale
        estimation.
    Returns
    --------
    List of dict
        A list of dictionaries consisting of the minimum and maximum
        inner and outer radius, respectively.
    """
    from sklearn.neighbors import KDTree

    tree = KDTree(X)
    distances, _ = tree.query(query_points, k=k_max, return_distance=True)

    # Ignore the distance to ourself, as we know that one already.
    distances = distances[:, 1:]

    scales = [
        {
            "r": dist[0],
            "R": dist[round(k_max / 3)],
            "s": dist[round(k_max / 3)],
            "S": dist[-1],
        }
        for dist in distances
    ]

    return scales


scales = estimate_scales(X, query_points, k)
scales_finetuned = estimate_scales(X_finetuned, query_points_finetuned, k)

euclidicity = Euclidicity(
    max_dim=max_dim,
    n_steps=n_steps,
    method="ripser",
    data=X,
)

euclidicity_finetuned = Euclidicity(
    max_dim=max_dim,
    n_steps=n_steps,
    method="ripser",
    data=X_finetuned,
)


def _process(x, scale=None):
    values = euclidicity(X, x, **scale)
    score = np.mean(np.nan_to_num(values))

    s = " ".join(str(a) for a in x)
    s += f" {score}"

    print(s)
    return score

def _process_finetuned(x, scale=None):
    values = euclidicity_finetuned(X, x, **scale)
    score = np.mean(np.nan_to_num(values))

    s = " ".join(str(a) for a in x)
    s += f" {score}"

    print(s)
    return score


scores = joblib.Parallel(n_jobs=None)(
    joblib.delayed(_process)(x, scale)
    for x, scale in zip(query_points, scales)
)

scores_finetuned = joblib.Parallel(n_jobs=None)(
    joblib.delayed(_process_finetuned)(x, scale)
    for x, scale in zip(query_points_finetuned, scales_finetuned)
)


euc_frame = pd.DataFrame({
                         'euclidicity_finetuned':scores_finetuned,
                         'euclidicity':scores
                         })

print(euc_frame.corr())

sns.scatterplot(x = scores,y = scores_finetuned)
plt.show()

euc_frame.to_pickle('euclidicity_base_vs_finetuned_'+data_name[:-4])