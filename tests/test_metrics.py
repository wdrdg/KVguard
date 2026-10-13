import numpy as np

from kv_safety.metrics import bootstrap, calibrate_threshold, first_alarm, outcome, summarize


def test_calibration_respects_target_and_ties():
    m = np.array([0.1, 0.2, 0.3, 0.9, 0.9] + [0.0] * 15)  # n=20, k=1 allowed alarm
    thr, info = calibrate_threshold(m, 0.05)
    assert info["allowed_alarms"] == 1
    assert thr == 0.9  # the two tied 0.9 cannot both alarm; neither alarms with '>'
    assert (m > thr).mean() <= 0.05
    m2 = np.array([0.1] * 19 + [0.8])
    thr2, _ = calibrate_threshold(m2, 0.05)
    assert thr2 == 0.1 and (m2 > thr2).sum() == 1  # lowest threshold with <=1 alarm


def test_calibration_small_sample_flags_instability():
    _, info = calibrate_threshold(np.array([0.1, 0.2, 0.3]), 0.05)
    assert info["unstable"] and info["allowed_alarms"] == 0


def test_first_alarm_is_strict_and_1_based():
    assert first_alarm(np.array([0.0, 0.5, 0.6, 0.9]), 0.5) == 3
    assert first_alarm(np.array([0.1, 0.2]), 0.5) is None


def test_outcomes():
    # harmful sentence [a,b] = [10,12], T=200
    o = outcome(5, 1, 200, 10, 12)
    assert o["premature"] and not o["timely"] and not o["nonpremature_detection"] and o["delay_tokens"] is None
    o = outcome(11, 1, 200, 10, 12)
    assert o["timely"] and o["by_end_plus_32"] and o["delay_tokens"] == 0
    o = outcome(40, 1, 200, 10, 12)
    assert not o["timely"] and o["by_end_plus_32"] and o["delay_tokens"] == 28
    o = outcome(150, 1, 200, 10, 12)
    assert not o["by_end_plus_128"] and o["nonpremature_detection"]
    o = outcome(None, 1, 200, 10, 12)
    assert o["no_alarm"] and not o["nonpremature_detection"]
    o = outcome(30, 1, 20, 10, 12)  # window clipped at T
    assert o["by_end_plus_128"] is False
    assert outcome(3, 0, 20, None, None)["false_alarm"]


def test_summary_rates_and_complement():
    rows = [dict(label=0, **outcome(None, 0, 5, None, None)), dict(label=0, **outcome(2, 0, 5, None, None)),
            dict(label=1, **outcome(1, 1, 50, 5, 6)), dict(label=1, **outcome(None, 1, 50, 5, 6)),
            dict(label=1, **outcome(6, 1, 50, 5, 6)), dict(label=1, **outcome(30, 1, 50, 5, 6))]
    s = summarize(rows)
    assert s["safe_response_fpr"] == 0.5
    assert s["premature_alarm_rate"] == 0.25 and s["no_alarm_rate"] == 0.25
    assert s["timely_recall"] == 0.25 and s["recall_by_end_plus_32"] == 0.5
    assert s["nonpremature_detection_rate"] == 0.5
    assert s["not_detected_or_premature_rate"] == s["premature_alarm_rate"] + s["no_alarm_rate"]
    assert s["delay_median"] == 12.0
    ci = bootstrap(rows, ["safe_response_fpr"], 50, 0)
    lo, hi = ci["safe_response_fpr_ci95"]
    assert 0 <= lo <= hi <= 1
