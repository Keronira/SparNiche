import numpy as np

from scripts import run_experiments


class _Adata:
    def __init__(self, values):
        self.X = np.asarray(values)


def test_input_kind_detects_integer_nonnegative_counts():
    assert run_experiments.detect_input_kind(_Adata([[0, 1], [2, 3]])) == "raw_counts"


def test_input_kind_detects_fractional_normalized_matrix():
    assert run_experiments.detect_input_kind(_Adata([[0.0, 0.5], [1.2, 2.0]])) == "normalized"
