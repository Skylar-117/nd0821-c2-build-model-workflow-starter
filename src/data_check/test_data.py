"""Data quality testing
"""
import logging
import pandas as pd
import numpy as np
import scipy.stats

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_column_names(data):
    """Test column names

    Parameters
    ----------
    data: pandas dataframe
        Input dataframe
    """
    expected_colums = [
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    actual_columns = data.columns.values
    logger.info("Test column names: expected column names "
                f"{list(expected_colums)}")
    logger.info("Test column names: actual column names in the dataset "
                f"{list(actual_columns)}")

    # This also enforces the same order
    assert list(expected_colums) == list(actual_columns)
    logger.info("Test column names: SUCCESS")


def test_neighborhood_names(data):
    """Test neighborhood names

    Parameters
    ----------
    data: pandas dataframe
        Input dataframe
    """
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)
    logger.info("Test neighborhood names: SUCCESS")


def test_proper_boundaries(data):
    """Test proper longitude and latitude boundaries

    Parameters
    ----------
    data: pandas dataframe
        Input dataframe
    """
    latitudes = data['latitude'].between(40.5, 41.2)
    longitude = data['longitude'].between(-74.25, -73.50)
    idx = latitudes & longitude
    assert np.sum(~idx) == 0
    logger.info("Test proper latitude and longitude boundaries: SUCCESS")


def test_similar_neigh_distrib(data, ref_data, kl_threshold):
    """Apply a threshold on the KL divergence to detect if the distribution
       of the new data is significantly different than that of
       the reference dataset

    Parameters
    ----------
    data: pandas dataframe
        Input dataframe
    ref_data: pandas dataframe
        Input dataframe
    kl_threshold: float
        The threshold of KL divergence.
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()
    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold
    logger.info("Test similar distribution: SUCCESS")


def test_row_count(data):
    """Test the number of rows

    Parameters
    ----------
    data: pandas dataframe
        This is the pandas dataframe.
    """
    assert 15000 < data.shape[0] < 1000000
    logger.info("Test rows count: SUCCESS")


def test_price_range(data, min_price, max_price):
    """Test price range

    Parameters
    ----------
        data: pandas dataframe
            Input datafarme
        min_price: float
            Minimum price value
        max_price: float
            Maximum price value
    """
    rows_within_range = data["price"].between(min_price, max_price).shape[0]
    assert data.shape[0] == rows_within_range
    logger.info("Test price range: SUCCESS")
