import numpy as np
from scipy import stats
from scipy.stats import linregress


def stat_test_linear(data1, data2):
    """
    Function that returns the p-value corresponding
    to a statistical hypothesis test for two cohorts
    data1 and data2 that have a linear relationship.
    The null hypothesis is that the linear relationship
    between data1 and data2 has slope 1.

        Parameters
        ----------
        data1: 1-D numpy arrray
        data2: 1-D numpy arrray

        Returns
        -------
        p-value corresponding to the statistical test

    """

    slope, intercept, r_value, p_value, std_err = linregress(data1, data2)
    std_norm = [(data1[i] - data2[i]) for i in range(len(data1))]

    std_norm = np.array(std_norm)
    std_norm = std_norm / std_norm.std()

    tt = sum(std_norm) / len(std_norm)
    n = len(std_norm)

    pval = stats.t.sf(np.abs(tt), n - 1) * 2
    return pval
