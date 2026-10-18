from src.features.feature_sets import FEATURES, LEAKAGE_MAP

NEVER = {"ReservationStatus", "ReservationStatusDate", "AssignedRoomType", "BookingChanges", "IsCanceled"}


def test_no_leak_columns_in_regression_or_classification():
    for task in ("regression", "classification"):
        used = set(FEATURES[task]["numeric"] + FEATURES[task]["categorical"])
        assert not used & NEVER, (task, used & NEVER)


def test_doubt_columns_excluded_from_main_classifier():
    used = set(FEATURES["classification"]["numeric"] + FEATURES["classification"]["categorical"])
    doubt = {v for v, (a, c, k, _) in LEAKAGE_MAP.items() if c == "doubt"}
    assert not used & doubt


def test_leakage_map_has_all_31_variables():
    assert len(LEAKAGE_MAP) == 31
