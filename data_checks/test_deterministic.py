"""
Creator: Lucas Torres Marques
Date: 30 Jun. 2022
Realize deterministic data checks to ensure we are using
a valid clean dataset.
"""
import pandas as pd

def test_dataset_size(full_data):
    """Check if Dataset has more than 3000 rows"""
    assert len(full_data) > 3000, "The choosen Dataset is smaller than 3000 rows."

def test_column_presence_and_type(full_data):
    """Check if dataset has all needed columns with the correct type"""

    required_columns = {
        "Entire home/apt": pd.api.types.is_float_dtype,
        "Private room": pd.api.types.is_float_dtype,
        "Shared room": pd.api.types.is_float_dtype,
        "Hotel room": pd.api.types.is_float_dtype,
        "accommodates": pd.api.types.is_float_dtype,
        "bathrooms_text": pd.api.types.is_float_dtype,
        "bedrooms": pd.api.types.is_float_dtype,
        "beds": pd.api.types.is_float_dtype,
        "price": pd.api.types.is_float_dtype,
        "number_of_reviews": pd.api.types.is_float_dtype,
        "review_scores_rating": pd.api.types.is_float_dtype,
        "review_scores_accuracy": pd.api.types.is_float_dtype,
        "review_scores_cleanliness": pd.api.types.is_float_dtype,
        "review_scores_checkin": pd.api.types.is_float_dtype,
        "review_scores_communication": pd.api.types.is_float_dtype,
        "review_scores_location": pd.api.types.is_float_dtype,
        "review_scores_value": pd.api.types.is_float_dtype
    }

    # Check column presence
    assert set(full_data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(full_data[col_name]), \
        f"Column {col_name} failed test {format_verification_funct}"

def test_column_ranges(full_data):
    """Check if all columns are normalized"""

    ranges = {
        "Entire home/apt": (0, 1),
        "Private room": (0, 1),
        "Shared room": (0, 1),
        "Hotel room": (0, 1),
        "accommodates": (0, 1),
        "bathrooms_text": (0, 1),
        "bedrooms": (0, 1),
        "beds": (0, 1),
        "price": (0, 1),
        "number_of_reviews": (0, 1),
        "review_scores_rating": (0, 1),
        "review_scores_accuracy": (0, 1),
        "review_scores_cleanliness": (0, 1),
        "review_scores_checkin": (0, 1),
        "review_scores_communication": (0, 1),
        "review_scores_location": (0, 1),
        "review_scores_value": (0, 1)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert full_data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={full_data[col_name].min()} and max={full_data[col_name].max()}"
        )
