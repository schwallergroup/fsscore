from typing import List


def one_hot_encoding(value: int, choices: List) -> List:
    """
    Apply one hot encoding
    :param value:
    :param choices:
    :return: A one-hot encoding for given index and length
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding
