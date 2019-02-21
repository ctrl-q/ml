import numpy as np


def softmax(x, axis=1):
    """Numerically-stable softmax function

    Args:
        x
        axis (int or tuple of ints): Axis or axes along which the computation is performed

    Returns:
        output (numpy.array): softmax output
    """
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)


def relu(x):
    """RELU activation

    Args:
        x: input

    Returns:
        f(x): RELU ouput
    """
    return (x > 0) * x


def crossentropy(x1, x2):
    """Cross-entropy of x1 and x2

    Args:
        x1
        x2

    Returns:
        f(x): cross-entropy output
    """
    return (x1 * (-np.log(x2))).sum(axis=1)
