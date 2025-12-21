from .parse import get_answer, strip_string


def is_equiv(str1, str2, verbose: bool = False):
    """Compare two math answers after normalization."""
    str1 = get_answer(str1)

    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    str1 = strip_string(str1)
    str2 = strip_string(str2)

    if verbose:
        print(f"Comparing: {str1} vs {str2}")

    return str1 == str2 or str1 in str2

