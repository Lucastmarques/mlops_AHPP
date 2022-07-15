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

    numerical_columns = ['accommodates', 'bedrooms', 'beds', 'price']

    # Bonferroni correction for multiple hypothesis testing
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:
        _, p_value = scipy.stats.ks_2samp(
            sample1[col],
            sample2[col],
            alternative='two-sided'
        )

        assert p_value > alpha_prime
