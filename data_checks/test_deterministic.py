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
        "room_type": pd.api.types.is_object_dtype,
        "accommodates": pd.api.types.is_int64_dtype,
        "bathrooms": pd.api.types.is_float_dtype,
        "bedrooms": pd.api.types.is_float_dtype,
        "beds": pd.api.types.is_float_dtype,
        "price": pd.api.types.is_float_dtype,
        "number_of_reviews": pd.api.types.is_int64_dtype,
        "review_scores_rating": pd.api.types.is_float_dtype,
        "review_scores_accuracy": pd.api.types.is_float_dtype,
        "review_scores_cleanliness": pd.api.types.is_float_dtype,
        "review_scores_checkin": pd.api.types.is_float_dtype,
        "review_scores_communication": pd.api.types.is_float_dtype,
        "review_scores_location": pd.api.types.is_float_dtype,
        "review_scores_value": pd.api.types.is_float_dtype
    }

    # Check column presence
    assert set(full_data.columns.values).issuperset(
        set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(full_data[col_name]), \
            f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(full_data):
    """Check that only the known classes are present"""
    known_classes = [
        'Entire home/apt',
        'Private room',
        'Shared room',
        'Hotel room'
    ]

    assert full_data["room_type"].isin(known_classes).all()


def test_column_ranges(full_data):
    """Check if all columns have meaningful data ranges"""

    ranges = {
        "accommodates": (1, 999),
        "bathrooms": (0.0, 99.0),
        "bedrooms": (1.0, 99.0),
        "beds": (1.0, 99.0),
        "price": (0.0, 999999.0),
        "number_of_reviews": (0, 999999),
        "review_scores_rating": (0.0, 5.0),
        "review_scores_accuracy": (0.0, 5.0),
        "review_scores_cleanliness": (0.0, 5.0),
        "review_scores_checkin": (0.0, 5.0),
        "review_scores_communication": (0.0, 5.0),
        "review_scores_location": (0.0, 5.0),
        "review_scores_value": (0.0, 5.0)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert full_data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={full_data[col_name].min()} and max={full_data[col_name].max()}"
        )
