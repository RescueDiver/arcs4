from core.grid_utils import count_matches, grids_equal


def score_prediction(predicted, expected):
    if predicted is None:
        return -10**9

    if predicted == expected:
        return 10**6

    score = count_matches(predicted, expected)

    same_shape = (
        len(predicted) == len(expected)
        and len(predicted[0]) == len(expected[0])
    ) if predicted and expected else False

    if same_shape:
        score += 100

    return score