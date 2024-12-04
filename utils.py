from scipy import stats
from scipy import integrate


def find_closest(sorted_array, target):
    """
    Find the closest value to target in a sorted array using binary search.
    Time complexity: O(log n)

    Args:
        sorted_array: List of integers in ascending order
        target: Number to find closest value to

    Returns:
        Tuple of (closest_value, index) found in the array
    """
    if not sorted_array:
        raise ValueError("Array cannot be empty")

    # If target is beyond array bounds, return the boundary value and index
    if target <= sorted_array[0]:
        return (sorted_array[0], 0)
    if target >= sorted_array[-1]:
        return (sorted_array[-1], len(sorted_array) - 1)

    left, right = 0, len(sorted_array) - 1

    while left <= right:
        mid = (left + right) // 2

        # If we find the exact value, return it and its index
        if sorted_array[mid] == target:
            return (sorted_array[mid], mid)

        if sorted_array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # At this point, right points to the largest value smaller than target
    # and left points to the smallest value larger than target
    # Compare these two values to find the closest
    if left >= len(sorted_array):
        return (sorted_array[right], right)
    if right < 0:
        return (sorted_array[left], left)

    # Return the closer value and its index
    if target - sorted_array[right] <= sorted_array[left] - target:
        return (sorted_array[right], right)
    return (sorted_array[left], left)


def normal_ccdf_area(mu, std, lower, upper):
    """
    Compute the area under the curve of (1 - CDF of normal) between 0 and n.

    Parameters:
    -----------
    n : float
        Upper bound of integration
    mu : float
        Mean of the normal distribution
    std : float
        Standard deviation of the normal distribution

    Returns:
    --------
    float : Area under the CCDF curve
    """

    # Define the function to integrate (1 - CDF)
    def integrand(x):
        return 1 - stats.norm.cdf(x, mu, std)

    # Compute the integral from 0 to n
    area, error = integrate.quad(integrand, lower, upper)

    return area
