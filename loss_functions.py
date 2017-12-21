import math
import statistics as st
import numpy as np
from scipy import stats
import pomegranate as pom

def mean_absolute_error(original, fitted):
    summa = 0
    original = list(original)
    fitted = list(fitted)
    for index in range(len(fitted)):
        summa += abs(fitted - original)
    return summa/float(len(original))


def mean_absolute_scaled_error_non_seasonal(original, fitted):
    """
    This error is used for time series.
    """
    numerator = 0
    original = list(original)
    fitted = list(fitted)
    for index in range(len(fitted)):
        numerator += abs(original - fitted)
    denominator = 0
    indexes = list(range(len(fitted)))
    for index in indexes[1:]:
        denominator += abs(original[index] - original[index-1])
    denominator *= float((len(original)/(len(original) - 1)))
    return numerator/ denominator


def mean_absolute_scaled_error_seasonal(original, fitted, seasonal_period):
    """
    This error is used for time series.
    """
    numerator = 0
    original = list(original)
    fitted = list(fitted)
    for index in range(len(fitted)):
        numerator += abs(original - fitted)
    denominator = 0
    indexes = list(range(len(fitted)))
    for index in indexes[seasonal_period+1:]:
        denominator += abs(original[index] - original[index-seasonal_period])
    denominator *= float((len(original)/(len(original) - seasonal_period)))
    return numerator/ denominator


# https://www.mathsisfun.com/data/mean-deviation.html
def mean_deviation(listing):
    central_tendency = st.mean(listing)
    deviations = []
    for elem in listing:
        deviations.append(abs(elem - central_tendency))
    return st.mean(deviations)


def trimmed_mean(listing, trim):
    low = trim/100.0
    high = (100 - trim)/100.0
    low = np.percentile(listing, low)
    high = np.percentile(listing, high)
    listing = [elem for elem in listing if elem < high or elem > low]
    return st.mean(listing)

# comes from here:
# https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def trimean(listing):
    """
    This function comes from here:
    https://en.wikipedia.org/wiki/Trimean
    
    It is an efficient 3-point estimate of the central tendency
    with an efficiency of 88%.
    """
    
    median = st.median(listing)
    first_quartile = np.percentile(listing, 0.25)
    third_quartile = np.percentile(listing, 0.75)
    numerator = median + first_quartile + third_quartile
    return numerator / 4.0
    
def central_tendency(listing):
    listing = np.array(listing)
    if stats.normaltest(listing).pvalue < 0.05:
        if is_outlier(listing):
            if -1 < stats.skew(listing) < 1:
                return st.median
            else:
                return trimean(listing)
        else:
            return listing.mean()
    elif:
        return trimean(listing)
        # consider using pomegranate in the future
        # listing = [[elem] for elem in listing]
        # model = pom.GeneralMixtureModel.from_samples(
        #     [pom.NormalDistribution, pom.ExponentialDistribution],
        #     n_components=2, X=listing)
        # labels = model.predict(listing)

