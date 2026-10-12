import glasswing_train


def test_version_matches_engine():
    # The Rust engine and the training package are released together.
    assert glasswing_train.__version__ == "0.1.0"
