from src.benchmark import filter_ambiguous_ground_truth


def test_filter_ambiguous_ground_truth_matches_banksy_unknown_rule():
    assert filter_ambiguous_ground_truth(["B cell", "unknown", "", None]) == ["B cell"]
