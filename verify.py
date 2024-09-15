
def check(pred, answer):
    """
    This function compares the prediction and the ground truth (answer),
    and returns True if they are the same, otherwise returns False.

    Args:
    pred: The predicted value or output from a model or function.
    answer: The ground truth to compare against.

    Returns:
    bool: True if the prediction is correct, otherwise False.
    """
    return pred == answer
