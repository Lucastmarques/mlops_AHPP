"""
Creator: Lucas Torres Marques
Date: 30 Jun. 2022
Realize hypothesis test to ensure we are using
a valid train/test splitted dataset.
"""
import scipy.stats

def test_kolmogorov_smirnov(splitted_data, ks_alpha):
    """Check if we have corrects train/test splitted datasets"""
    sample1, sample2 = splitted_data

    numerical_columns = [
        "Entire home/apt", "Private room", "Shared room", "Hotel room",
        "accommodates", "bathrooms_text", "bedrooms", "beds", "price",
        "number_of_reviews", "review_scores_rating",
        "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication",
        "review_scores_location", "review_scores_value"
    ]

    # Bonferroni correction for multiple hypothesis testing
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:
        _, p_value = scipy.stats.ks_2samp(
            sample1[col],
            sample2[col],
            alternative='two-sided'
        )

        assert p_value > alpha_prime
