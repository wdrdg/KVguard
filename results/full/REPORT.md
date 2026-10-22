# Report: full

Controlled comparison of native KV-cache running means and hidden-state features on frozen `Qwen/Qwen3-4B` (reference block index 11). This is not a reproduction of SIREN, ShieldHead or Kelp. Labels are a sentence-level proxy for cumulative prefix harm, not human token-level ground truth. FineHarm numbers are teacher-forced replays, not free generations of this backbone.

## Execution status

| stage | artifact | status |
|---|---|---|
| environment probe | environment.json | NOT EXECUTED |
| data audit | data_audit.json | EXECUTED |
| split manifest | split_manifest.jsonl | EXECUTED |
| label audit (pending human review) | label_audit.jsonl | EXECUTED |
| sampled-position feature extraction | feature_manifest.json | EXECUTED |
| classifier training | checkpoints/train_summary.json | EXECUTED |
| calibration thresholds | metrics/calibration_summary.csv | EXECUTED |
| dev streaming evaluation | metrics/quality_summary.csv | EXECUTED |
| held-out test evaluation | metrics/quality_summary_test.csv | NOT EXECUTED |
| CUDA memory/latency benchmark | metrics/memory_latency_raw.csv | EXECUTED |

## Data audit

Train may be class-balanced. Dev keeps the cleaned class mix. Calibration thresholds use only safe responses. Prompt safety comes from FineHarm source filenames (`_HP_` / `_BP_`); it is not inferred from the response, and a safe response is not a refusal.

```json
{
 "construction": "pilot",
 "leakage": {
  "exact_duplicates_removed": {},
  "leaked_group_records_removed": {
   "val": 183,
   "train": 2914
  },
  "leaked_groups": 2704
 },
 "near_duplicates": {
  "rule": "within a prompt group, responses equal after NFKC + casefold + whitespace collapse, but not exact string duplicates (those are already removed)",
  "near_duplicate_groups": 0,
  "near_duplicate_extra_records": 0,
  "action": "audited_not_removed"
 },
 "heldout_test": {
  "status": "prepared_not_scored",
  "eligible": 2898,
  "class_eligible": {
   "n": 2898,
   "safe": 1623,
   "harmful": 1275,
   "harmful_ratio": 0.43995859213250516,
   "n_prompt_groups": 2796
  },
  "excluded_too_long": 9,
  "alignment_failures": 0,
  "scored": false,
  "reason": "rows materialised for Phase D; scoring waits for protocol_lock after calibration"
 },
 "label_audit": {
  "n": 100,
  "review_status": "pending_human_review",
  "categories": {
   "first_sentence_harmful": 25,
   "later_harmful": 25,
   "harmful_prompt_safe_response": 25,
   "safe_response": 25
  },
  "note": "[a, b] is an uncertain span inside the first harmful sentence, not a confirmed first harmful token. Cumulative labels are not the current-sentence label. No model self-check is treated as human review."
 },
 "manifest_counts": {
  "train": 20319,
  "dev": 1353,
  "calibration": 750,
  "test": 2898
 }
}
```

## GPU alignment

```json
{
  "cache_k_equals_recomputed_post_rope": true,
  "chunk128_vs_tokenwise_max_rel_err": {
    "h_last": 0.46311789751052856,
    "h_mean": 0.5885912179946899,
    "k_mean": 0.10213993489742279,
    "v_mean": 0.14413200318813324
  },
  "chunk128_vs_tokenwise_mean_rel_err": {
    "h_last": 0.01547932531684637,
    "h_mean": 0.011215743608772755,
    "k_mean": 0.005261322017759085,
    "v_mean": 0.011677305214107037
  },
  "chunk128_vs_tokenwise_min_cosine": {
    "h_last": 0.9999145234110217,
    "h_mean": 0.9999650548856539,
    "k_mean": 0.9999894518935798,
    "v_mean": 0.9999380368041387
  },
  "prefix_invariance_max_rel_err": {
    "h_last": 0.636185884475708,
    "h_mean": 0.38580191135406494,
    "k_mean": 0.0984501838684082,
    "v_mean": 0.11846841126680374
  },
  "prefix_invariance_mean_rel_err": {
    "h_last": 0.013721955940127373,
    "h_mean": 0.010613277554512024,
    "k_mean": 0.0052740140818059444,
    "v_mean": 0.010866180062294006
  },
  "prefix_invariance_min_cosine": {
    "h_last": 0.9999346108835598,
    "h_mean": 0.9999632243820769,
    "k_mean": 0.9999881181234084,
    "v_mean": 0.9999379676156821
  },
  "online_vs_offline_max_rel_err": {
    "h_last": 0.0,
    "h_mean": 5.030058673582971e-06,
    "k_mean": 1.5838153331060312e-06,
    "v_mean": 6.50233857868443e-07,
    "kv_mean": 2.412235062365653e-06
  }
}
```

The max-element ratio is inflated by BF16 residual accumulation: layer 0 differs by about one ulp, and the causal mask still lets the first query row attend only to the prefix plus itself. The alignment gate is mean relative error below 0.02 and cosine above 0.98. Tokenwise online features match chunk-size-1 offline features. Chunk size 128 is what training and dev replay use.

## Calibration

Thresholds use only safe calibration responses. Dev and test labels are not used to choose them. n_safe=[750], empirical FPR range 0.0493–0.0493, unstable heads=0. A head is unstable when n_safe < 500. The Wilson interval is for this calibration count only.

## Development streaming quality

Read false-positive rate and premature alarms before recall. A method with a higher recall and a much higher dev FPR is not a win at the same false-positive rate. `recall_by_end_plus_32` excludes alarms before the proxy boundary. `first_harm_sentence_alarm_rate` is the share of alarms inside `[a, b]`, not a claim about the true first harmful token. These dev numbers were used for checkpoint selection, so they are development feedback, not an unbiased test.

### Reference layer index 11, mean and sd over seeds

| layer | method | feature_dim | parameter_count | safe_response_fpr_mean | safe_response_fpr_std | premature_alarm_rate_mean | premature_alarm_rate_std | recall_by_end_plus_32_mean | recall_by_end_plus_32_std | first_harm_sentence_alarm_rate_mean | first_harm_sentence_alarm_rate_std | recall_by_end_plus_128_mean | recall_by_end_plus_128_std | nonpremature_detection_rate_mean | nonpremature_detection_rate_std | no_alarm_rate_mean | no_alarm_rate_std | invalid_or_missed_rate_mean | invalid_or_missed_rate_std | delay_after_sentence_end_p50_mean | delay_after_sentence_end_p50_std | delay_after_sentence_end_p90_mean | delay_after_sentence_end_p90_std | detection_fraction_p50_mean | detection_fraction_p50_std | stream_response_auroc_mean | stream_response_auroc_std | final_prefix_auroc_mean | final_prefix_auroc_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | h_mean | 2560 | 327937 | 0.058 | 0.009 | 0.258 | 0.030 | 0.659 | 0.018 | 0.555 | 0.007 | 0.709 | 0.023 | 0.726 | 0.029 | 0.016 | 0.003 | 0.274 | 0.029 | 0.000 | 0.000 | 29.100 | 14.839 | 0.097 | 0.006 | 0.995 | 0.001 | 0.996 | 0.001 |
| 11 | kv_mean | 2048 | 262401 | 0.049 | 0.004 | 0.225 | 0.037 | 0.655 | 0.020 | 0.486 | 0.024 | 0.723 | 0.046 | 0.745 | 0.044 | 0.031 | 0.009 | 0.255 | 0.044 | 0.000 | 0.000 | 43.867 | 13.378 | 0.109 | 0.013 | 0.995 | 0.001 | 0.996 | 0.000 |
| 11 | v_mean | 1024 | 131329 | 0.056 | 0.008 | 0.238 | 0.011 | 0.659 | 0.005 | 0.524 | 0.011 | 0.717 | 0.011 | 0.738 | 0.013 | 0.023 | 0.004 | 0.262 | 0.013 | 0.000 | 0.000 | 35.600 | 4.157 | 0.101 | 0.003 | 0.995 | 0.000 | 0.996 | 0.000 |


Per seed:

| method | seed | threshold | n_safe | n_harm | n_prompt_groups | safe_response_fpr | premature_alarm_rate | recall_by_end_plus_32 | first_harm_sentence_alarm_rate | recall_by_end_plus_128 | nonpremature_detection_rate | no_alarm_rate | invalid_or_missed_rate | delay_after_sentence_end_p50 | delay_after_sentence_end_p90 | detection_fraction_p50 | stream_response_auroc | final_prefix_auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h_mean | 17 | -0.838 | 742 | 611 | 1320 | 0.065 | 0.290 | 0.658 | 0.563 | 0.687 | 0.697 | 0.013 | 0.303 | 0.000 | 12.000 | 0.090 | 0.996 | 0.997 |
| h_mean | 42 | 0.194 | 742 | 611 | 1320 | 0.047 | 0.229 | 0.678 | 0.552 | 0.733 | 0.756 | 0.015 | 0.244 | 0.000 | 36.700 | 0.102 | 0.995 | 0.996 |
| h_mean | 2026 | -0.092 | 742 | 611 | 1320 | 0.062 | 0.255 | 0.642 | 0.552 | 0.705 | 0.725 | 0.020 | 0.275 | 0.000 | 38.600 | 0.098 | 0.995 | 0.996 |
| v_mean | 17 | 0.433 | 742 | 611 | 1320 | 0.058 | 0.239 | 0.656 | 0.529 | 0.718 | 0.741 | 0.020 | 0.259 | 0.000 | 38.000 | 0.102 | 0.995 | 0.997 |
| v_mean | 42 | 0.551 | 742 | 611 | 1320 | 0.047 | 0.227 | 0.664 | 0.512 | 0.728 | 0.750 | 0.023 | 0.250 | 0.000 | 38.000 | 0.103 | 0.995 | 0.996 |
| v_mean | 2026 | 0.215 | 742 | 611 | 1320 | 0.063 | 0.249 | 0.656 | 0.532 | 0.705 | 0.723 | 0.028 | 0.277 | 0.000 | 30.800 | 0.098 | 0.994 | 0.996 |
| kv_mean | 17 | 0.917 | 742 | 611 | 1320 | 0.044 | 0.227 | 0.668 | 0.501 | 0.730 | 0.748 | 0.025 | 0.252 | 0.000 | 42.200 | 0.109 | 0.995 | 0.996 |
| kv_mean | 42 | 1.116 | 742 | 611 | 1320 | 0.050 | 0.187 | 0.664 | 0.458 | 0.766 | 0.787 | 0.026 | 0.213 | 0.000 | 58.000 | 0.122 | 0.995 | 0.996 |
| kv_mean | 2026 | 0.412 | 742 | 611 | 1320 | 0.051 | 0.260 | 0.632 | 0.499 | 0.674 | 0.699 | 0.041 | 0.301 | 0.000 | 31.400 | 0.095 | 0.993 | 0.996 |


### Layer ranking on dev

For each of h_mean, v_mean and kv_mean, rank layers by mean recall_by_end_plus_32 over seeds on dev. Ties break to lower premature_alarm_rate, then lower safe_response_fpr, then the smaller layer index. Block 11 stays the reference comparison and is not replaced by this ranking. Dev was used for early stopping, so this is development selection, not a test result.


`h_mean` selected layer index 11:

| layer | n_seeds | recall_by_end_plus_32 | premature_alarm_rate | safe_response_fpr |
|---|---|---|---|---|
| 11.000 | 3.000 | 0.659 | 0.258 | 0.058 |


`v_mean` selected layer index 11:

| layer | n_seeds | recall_by_end_plus_32 | premature_alarm_rate | safe_response_fpr |
|---|---|---|---|---|
| 11.000 | 3.000 | 0.659 | 0.238 | 0.056 |


`kv_mean` selected layer index 11:

| layer | n_seeds | recall_by_end_plus_32 | premature_alarm_rate | safe_response_fpr |
|---|---|---|---|---|
| 11.000 | 3.000 | 0.655 | 0.225 | 0.049 |


## Paired differences versus same-layer h_mean

| layer | method | seed | metric | difference | ci95_lo | ci95_hi |
|---|---|---|---|---|---|---|
| 11 | v_mean | 17 | safe_response_fpr | -0.007 | -0.022 | 0.008 |
| 11 | v_mean | 17 | recall_by_end_plus_32 | -0.002 | -0.032 | 0.026 |
| 11 | v_mean | 17 | premature_alarm_rate | -0.051 | -0.075 | -0.027 |
| 11 | v_mean | 42 | safe_response_fpr | 0.000 | -0.013 | 0.013 |
| 11 | v_mean | 42 | recall_by_end_plus_32 | -0.013 | -0.042 | 0.015 |
| 11 | v_mean | 42 | premature_alarm_rate | -0.002 | -0.024 | 0.024 |
| 11 | v_mean | 2026 | safe_response_fpr | 0.001 | -0.012 | 0.015 |
| 11 | v_mean | 2026 | recall_by_end_plus_32 | 0.015 | -0.013 | 0.045 |
| 11 | v_mean | 2026 | premature_alarm_rate | -0.007 | -0.030 | 0.017 |
| 11 | kv_mean | 17 | safe_response_fpr | -0.020 | -0.035 | -0.005 |
| 11 | kv_mean | 17 | recall_by_end_plus_32 | 0.010 | -0.022 | 0.040 |
| 11 | kv_mean | 17 | premature_alarm_rate | -0.062 | -0.087 | -0.038 |
| 11 | kv_mean | 42 | safe_response_fpr | 0.003 | -0.011 | 0.017 |
| 11 | kv_mean | 42 | recall_by_end_plus_32 | -0.013 | -0.043 | 0.017 |
| 11 | kv_mean | 42 | premature_alarm_rate | -0.043 | -0.065 | -0.019 |
| 11 | kv_mean | 2026 | safe_response_fpr | -0.011 | -0.027 | 0.005 |
| 11 | kv_mean | 2026 | recall_by_end_plus_32 | -0.010 | -0.039 | 0.020 |
| 11 | kv_mean | 2026 | premature_alarm_rate | 0.005 | -0.015 | 0.026 |

Intervals are prompt-group bootstrap intervals conditional on this training run and the fixed calibration threshold. They are not a full calibration uncertainty.

## Memory and latency

| method | role | batch | prompt_tokens | response_tokens | n_ok | n_oom | n_error | peak_allocated_mib_median | extra_peak_mib | extra_peak_reliable | persistent_state_bytes | decode_ms_per_token_median | latency_overhead_pct | tokens_per_second_total_median | prefill_ms_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_no_detector | main | 1 | 128 | 512 | 5 | 0 | 0 | 7843.529 | 0.000 | False | 0 | 27.589 | 0.000 | 36.247 | 39.252 |
| baseline_no_detector | main | 1 | 128 | 2048 | 5 | 0 | 0 | 8044.810 | 0.000 | False | 0 | 26.916 | 0.000 | 37.153 | 38.486 |
| h_mean | main | 1 | 128 | 512 | 5 | 0 | 0 | 7842.557 | -0.972 | False | 10248 | 29.053 | 6.057 | 34.420 | 41.818 |
| h_mean | main | 1 | 128 | 2048 | 5 | 0 | 0 | 8046.092 | 1.282 | False | 10248 | 28.049 | 5.047 | 35.652 | 38.449 |
| kv_mean | main | 1 | 128 | 512 | 5 | 0 | 0 | 7844.555 | 1.026 | False | 8200 | 30.282 | -0.331 | 33.022 | 38.031 |
| kv_mean | main | 1 | 128 | 2048 | 5 | 0 | 0 | 8045.836 | 1.026 | False | 8200 | 28.187 | 4.191 | 35.477 | 38.107 |
| v_mean | main | 1 | 128 | 512 | 5 | 0 | 0 | 7844.043 | 0.515 | False | 4104 | 29.247 | -0.723 | 34.192 | 39.887 |
| v_mean | main | 1 | 128 | 2048 | 5 | 0 | 0 | 8045.325 | 0.515 | False | 4104 | 27.445 | 1.271 | 36.436 | 40.664 |

Negative extra peak is kept. `extra_peak_reliable=False` means no reliable difference was measured relative to run-to-run spread. Decode time is one forward per response token, including the final flush, and excludes prefill. Throughput counts response tokens across the batch. `kv_mean_matched` is a capacity-control cost, reported with `role=capacity_control` when that column is present.

## Theoretical accumulator sizes (sanity check only)

| method | fp32_bytes_per_sample | KiB |
|---|---|---|
| h_mean | 10240 | 10.000 |
| k_mean | 4096 | 4.000 |
| v_mean | 4096 | 4.000 |
| kv_mean | 8192 | 8.000 |

The backbone KV cache is shared by every method. H-mean and KV-mean accumulators differ by about 2 KiB per sample, so a large peak-memory reduction is not expected from the accumulator alone.

## Held-out test

Official test, scored once after `protocol_lock.json`. Thresholds and checkpoints were fixed on calibration and dev. These numbers are not used to choose a layer, a threshold, or a filter. Read FPR and premature alarms before recall.

### Reference layer index 11, mean and sd over seeds

| layer | method | feature_dim | parameter_count | safe_response_fpr_mean | safe_response_fpr_std | premature_alarm_rate_mean | premature_alarm_rate_std | recall_by_end_plus_32_mean | recall_by_end_plus_32_std | first_harm_sentence_alarm_rate_mean | first_harm_sentence_alarm_rate_std | recall_by_end_plus_128_mean | recall_by_end_plus_128_std | nonpremature_detection_rate_mean | nonpremature_detection_rate_std | no_alarm_rate_mean | no_alarm_rate_std | invalid_or_missed_rate_mean | invalid_or_missed_rate_std | delay_after_sentence_end_p50_mean | delay_after_sentence_end_p50_std | delay_after_sentence_end_p90_mean | delay_after_sentence_end_p90_std | detection_fraction_p50_mean | detection_fraction_p50_std | stream_response_auroc_mean | stream_response_auroc_std | final_prefix_auroc_mean | final_prefix_auroc_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | h_mean | 2560 | 327937 | 0.044 | 0.006 | 0.278 | 0.033 | 0.650 | 0.022 | 0.539 | 0.012 | 0.692 | 0.031 | 0.705 | 0.031 | 0.017 | 0.002 | 0.295 | 0.031 | 0.000 | 0.000 | 25.000 | 6.557 | 0.091 | 0.007 | 0.996 | 0.000 | 0.997 | 0.000 |
| 11 | kv_mean | 2048 | 262401 | 0.042 | 0.001 | 0.233 | 0.033 | 0.643 | 0.006 | 0.485 | 0.032 | 0.720 | 0.028 | 0.740 | 0.034 | 0.028 | 0.001 | 0.260 | 0.034 | 0.000 | 0.000 | 44.967 | 13.107 | 0.111 | 0.012 | 0.995 | 0.000 | 0.995 | 0.000 |
| 11 | v_mean | 1024 | 131329 | 0.046 | 0.005 | 0.253 | 0.016 | 0.645 | 0.008 | 0.500 | 0.026 | 0.702 | 0.011 | 0.721 | 0.015 | 0.026 | 0.004 | 0.279 | 0.015 | 0.000 | 0.000 | 34.800 | 8.374 | 0.102 | 0.005 | 0.995 | 0.000 | 0.995 | 0.000 |


Per seed:

| method | seed | threshold | n_safe | n_harm | n_prompt_groups | safe_response_fpr | premature_alarm_rate | recall_by_end_plus_32 | first_harm_sentence_alarm_rate | recall_by_end_plus_128 | nonpremature_detection_rate | no_alarm_rate | invalid_or_missed_rate | delay_after_sentence_end_p50 | delay_after_sentence_end_p90 | detection_fraction_p50 | stream_response_auroc | final_prefix_auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h_mean | 17 | -0.838 | 1623 | 1275 | 2796 | 0.049 | 0.315 | 0.626 | 0.536 | 0.657 | 0.669 | 0.016 | 0.331 | 0.000 | 19.000 | 0.083 | 0.996 | 0.997 |
| h_mean | 42 | 0.194 | 1623 | 1275 | 2796 | 0.037 | 0.254 | 0.656 | 0.528 | 0.713 | 0.727 | 0.019 | 0.273 | 0.000 | 32.000 | 0.097 | 0.996 | 0.997 |
| h_mean | 2026 | -0.092 | 1623 | 1275 | 2796 | 0.046 | 0.264 | 0.667 | 0.552 | 0.707 | 0.719 | 0.016 | 0.281 | 0.000 | 24.000 | 0.091 | 0.996 | 0.997 |
| v_mean | 17 | 0.433 | 1623 | 1275 | 2796 | 0.044 | 0.247 | 0.653 | 0.515 | 0.706 | 0.722 | 0.031 | 0.278 | 0.000 | 31.000 | 0.102 | 0.995 | 0.996 |
| v_mean | 42 | 0.551 | 1623 | 1275 | 2796 | 0.042 | 0.241 | 0.636 | 0.470 | 0.711 | 0.735 | 0.024 | 0.265 | 0.000 | 44.400 | 0.107 | 0.995 | 0.995 |
| v_mean | 2026 | 0.215 | 1623 | 1275 | 2796 | 0.052 | 0.271 | 0.645 | 0.515 | 0.690 | 0.706 | 0.024 | 0.294 | 0.000 | 29.000 | 0.097 | 0.995 | 0.995 |
| kv_mean | 17 | 0.917 | 1623 | 1275 | 2796 | 0.041 | 0.230 | 0.648 | 0.481 | 0.723 | 0.744 | 0.027 | 0.256 | 0.000 | 46.900 | 0.116 | 0.995 | 0.995 |
| kv_mean | 42 | 1.116 | 1623 | 1275 | 2796 | 0.043 | 0.202 | 0.646 | 0.456 | 0.746 | 0.771 | 0.027 | 0.229 | 0.000 | 57.000 | 0.121 | 0.995 | 0.995 |
| kv_mean | 2026 | 0.412 | 1623 | 1275 | 2796 | 0.043 | 0.267 | 0.636 | 0.520 | 0.690 | 0.704 | 0.029 | 0.296 | 0.000 | 31.000 | 0.097 | 0.995 | 0.995 |


KV is 2048-d and K or V is 1024-d, so equal hidden width is not equal parameter count. A difference is not by itself complementary information.


### Paired differences KV−V, KV−K, V−K

| method | reference | seed | metric | difference | ci95_lo | ci95_hi |
|---|---|---|---|---|---|---|
| v_mean | h_mean | 17 | safe_response_fpr | -0.005 | -0.012 | 0.004 |
| v_mean | h_mean | 17 | recall_by_end_plus_32 | 0.027 | 0.005 | 0.048 |
| v_mean | h_mean | 17 | premature_alarm_rate | -0.068 | -0.086 | -0.052 |
| v_mean | h_mean | 42 | safe_response_fpr | 0.005 | -0.003 | 0.013 |
| v_mean | h_mean | 42 | recall_by_end_plus_32 | -0.020 | -0.041 | 0.002 |
| v_mean | h_mean | 42 | premature_alarm_rate | -0.013 | -0.031 | 0.002 |
| v_mean | h_mean | 2026 | safe_response_fpr | 0.006 | -0.001 | 0.013 |
| v_mean | h_mean | 2026 | recall_by_end_plus_32 | -0.022 | -0.042 | -0.003 |
| v_mean | h_mean | 2026 | premature_alarm_rate | 0.006 | -0.010 | 0.023 |
| kv_mean | h_mean | 17 | safe_response_fpr | -0.009 | -0.017 | 0.001 |
| kv_mean | h_mean | 17 | recall_by_end_plus_32 | 0.022 | -0.002 | 0.045 |
| kv_mean | h_mean | 17 | premature_alarm_rate | -0.085 | -0.105 | -0.066 |
| kv_mean | h_mean | 42 | safe_response_fpr | 0.006 | -0.002 | 0.014 |
| kv_mean | h_mean | 42 | recall_by_end_plus_32 | -0.010 | -0.033 | 0.014 |
| kv_mean | h_mean | 42 | premature_alarm_rate | -0.053 | -0.071 | -0.035 |
| kv_mean | h_mean | 2026 | safe_response_fpr | -0.003 | -0.012 | 0.006 |
| kv_mean | h_mean | 2026 | recall_by_end_plus_32 | -0.031 | -0.052 | -0.011 |
| kv_mean | h_mean | 2026 | premature_alarm_rate | 0.002 | -0.015 | 0.020 |

## Limits

- Dev selected the checkpoints, so dev metrics are development feedback.

- Official test was scored after the protocol lock and was not used to change the method.

- Sentence-boundary `[a, b]` is an uncertain span. Delay after `b` is not leaked-token count.

- Label audit rows are pending human review.

- Calibration Wilson intervals describe the calibration counts only.


## Figures
- `figures/memory_vs_length.png`
- `figures/quality_vs_overhead.png`
