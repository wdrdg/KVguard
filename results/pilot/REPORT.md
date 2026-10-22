# Report: pilot

Controlled comparison of native KV-cache running means and hidden-state features on frozen `Qwen/Qwen3-4B` (reference block index 23). This is not a reproduction of SIREN, ShieldHead or Kelp. Labels are a sentence-level proxy for cumulative prefix harm, not human token-level ground truth. FineHarm numbers are teacher-forced replays, not free generations of this backbone.

## Execution status

| stage | artifact | status |
|---|---|---|
| environment probe | environment.json | EXECUTED |
| data audit | data_audit.json | EXECUTED |
| split manifest | split_manifest.jsonl | EXECUTED |
| label audit (pending human review) | label_audit.jsonl | EXECUTED |
| sampled-position feature extraction | feature_manifest.json | EXECUTED |
| classifier training | checkpoints/train_summary.json | EXECUTED |
| calibration thresholds | metrics/calibration_summary.csv | EXECUTED |
| dev streaming evaluation | metrics/quality_summary.csv | EXECUTED |
| CUDA memory/latency benchmark | metrics/memory_latency_raw.csv | EXECUTED |

## Environment

- OS: Linux-5.15.0-25-generic-x86_64-with-glibc2.35
- Python: 3.12.15
- GPUs: [{'index': 0, 'name': 'NVIDIA GeForce RTX 4080 SUPER', 'total_mib': 32230, 'capability': '8.9'}]
- torch: 2.14.1, transformers: 5.19.0

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
  "status": "held_out",
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
  "reason": "pilot does not extract or score the official test split"
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
  "train": 5000,
  "dev": 1000,
  "calibration": 750
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

### Reference layer index 23, mean and sd over seeds

| layer | method | feature_dim | parameter_count | safe_response_fpr_mean | safe_response_fpr_std | premature_alarm_rate_mean | premature_alarm_rate_std | recall_by_end_plus_32_mean | recall_by_end_plus_32_std | first_harm_sentence_alarm_rate_mean | first_harm_sentence_alarm_rate_std | recall_by_end_plus_128_mean | recall_by_end_plus_128_std | nonpremature_detection_rate_mean | nonpremature_detection_rate_std | no_alarm_rate_mean | no_alarm_rate_std | invalid_or_missed_rate_mean | invalid_or_missed_rate_std | delay_after_sentence_end_p50_mean | delay_after_sentence_end_p50_std | delay_after_sentence_end_p90_mean | delay_after_sentence_end_p90_std | detection_fraction_p50_mean | detection_fraction_p50_std | stream_response_auroc_mean | stream_response_auroc_std | final_prefix_auroc_mean | final_prefix_auroc_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 23 | h_last | 2560 | 327937 | 0.069 | 0.008 | 0.111 | 0.017 | 0.569 | 0.032 | 0.267 | 0.029 | 0.756 | 0.013 | 0.849 | 0.012 | 0.041 | 0.009 | 0.151 | 0.012 | 9.333 | 2.309 | 142.933 | 25.963 | 0.191 | 0.016 | 0.990 | 0.002 | 0.989 | 0.001 |
| 23 | h_mean | 2560 | 327937 | 0.064 | 0.004 | 0.330 | 0.031 | 0.603 | 0.026 | 0.514 | 0.010 | 0.642 | 0.029 | 0.656 | 0.031 | 0.015 | 0.001 | 0.344 | 0.031 | 0.000 | 0.000 | 20.233 | 6.957 | 0.086 | 0.005 | 0.997 | 0.000 | 0.997 | 0.000 |
| 23 | k_mean | 1024 | 131329 | 0.071 | 0.001 | 0.277 | 0.034 | 0.614 | 0.016 | 0.485 | 0.008 | 0.684 | 0.029 | 0.704 | 0.032 | 0.019 | 0.003 | 0.296 | 0.032 | 0.000 | 0.000 | 43.367 | 17.457 | 0.104 | 0.008 | 0.995 | 0.000 | 0.996 | 0.000 |
| 23 | kv_mean | 2048 | 262401 | 0.070 | 0.001 | 0.319 | 0.038 | 0.605 | 0.011 | 0.509 | 0.004 | 0.660 | 0.038 | 0.667 | 0.039 | 0.014 | 0.003 | 0.333 | 0.039 | 0.000 | 0.000 | 28.600 | 15.935 | 0.094 | 0.008 | 0.997 | 0.000 | 0.996 | 0.000 |
| 23 | v_mean | 1024 | 131329 | 0.066 | 0.004 | 0.336 | 0.010 | 0.597 | 0.009 | 0.502 | 0.007 | 0.643 | 0.010 | 0.656 | 0.010 | 0.009 | 0.000 | 0.344 | 0.010 | 0.000 | 0.000 | 27.633 | 9.111 | 0.087 | 0.004 | 0.997 | 0.000 | 0.996 | 0.000 |


Per seed:

| method | seed | threshold | n_safe | n_harm | n_prompt_groups | safe_response_fpr | premature_alarm_rate | recall_by_end_plus_32 | first_harm_sentence_alarm_rate | recall_by_end_plus_128 | nonpremature_detection_rate | no_alarm_rate | invalid_or_missed_rate | delay_after_sentence_end_p50 | delay_after_sentence_end_p90 | detection_fraction_p50 | stream_response_auroc | final_prefix_auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h_last | 17 | 3.901 | 548 | 452 | 974 | 0.078 | 0.108 | 0.595 | 0.277 | 0.770 | 0.843 | 0.049 | 0.157 | 8.000 | 114.000 | 0.180 | 0.989 | 0.989 |
| h_last | 42 | 3.608 | 548 | 452 | 974 | 0.068 | 0.128 | 0.577 | 0.290 | 0.743 | 0.841 | 0.031 | 0.159 | 8.000 | 150.600 | 0.183 | 0.992 | 0.990 |
| h_last | 2026 | 3.980 | 548 | 452 | 974 | 0.062 | 0.095 | 0.533 | 0.235 | 0.754 | 0.863 | 0.042 | 0.137 | 12.000 | 164.200 | 0.210 | 0.990 | 0.988 |
| h_mean | 17 | -0.674 | 548 | 452 | 974 | 0.068 | 0.365 | 0.573 | 0.513 | 0.608 | 0.619 | 0.015 | 0.381 | 0.000 | 13.100 | 0.081 | 0.997 | 0.997 |
| h_mean | 42 | -0.709 | 548 | 452 | 974 | 0.060 | 0.316 | 0.619 | 0.524 | 0.662 | 0.670 | 0.013 | 0.330 | 0.000 | 20.600 | 0.089 | 0.997 | 0.997 |
| h_mean | 2026 | -0.642 | 548 | 452 | 974 | 0.066 | 0.308 | 0.615 | 0.504 | 0.655 | 0.677 | 0.015 | 0.323 | 0.000 | 27.000 | 0.089 | 0.997 | 0.997 |
| k_mean | 17 | -0.009 | 548 | 452 | 974 | 0.071 | 0.312 | 0.606 | 0.478 | 0.655 | 0.670 | 0.018 | 0.330 | 0.000 | 30.800 | 0.099 | 0.995 | 0.996 |
| k_mean | 42 | 0.068 | 548 | 452 | 974 | 0.071 | 0.277 | 0.633 | 0.493 | 0.684 | 0.706 | 0.018 | 0.294 | 0.000 | 36.000 | 0.099 | 0.995 | 0.996 |
| k_mean | 2026 | 0.405 | 548 | 452 | 974 | 0.069 | 0.243 | 0.604 | 0.485 | 0.712 | 0.735 | 0.022 | 0.265 | 0.000 | 63.300 | 0.113 | 0.995 | 0.996 |
| v_mean | 17 | -0.427 | 548 | 452 | 974 | 0.064 | 0.336 | 0.600 | 0.502 | 0.644 | 0.655 | 0.009 | 0.345 | 0.000 | 24.000 | 0.085 | 0.996 | 0.996 |
| v_mean | 42 | -0.967 | 548 | 452 | 974 | 0.064 | 0.345 | 0.604 | 0.509 | 0.633 | 0.646 | 0.009 | 0.354 | 0.000 | 20.900 | 0.084 | 0.997 | 0.996 |
| v_mean | 2026 | -0.429 | 548 | 452 | 974 | 0.071 | 0.325 | 0.586 | 0.496 | 0.653 | 0.666 | 0.009 | 0.334 | 0.000 | 38.000 | 0.091 | 0.997 | 0.996 |
| kv_mean | 17 | -0.499 | 548 | 452 | 974 | 0.071 | 0.350 | 0.593 | 0.507 | 0.624 | 0.633 | 0.018 | 0.367 | 0.000 | 19.500 | 0.086 | 0.996 | 0.996 |
| kv_mean | 42 | -0.379 | 548 | 452 | 974 | 0.069 | 0.330 | 0.613 | 0.507 | 0.657 | 0.659 | 0.011 | 0.341 | 0.000 | 19.300 | 0.093 | 0.997 | 0.997 |
| kv_mean | 2026 | 0.045 | 548 | 452 | 974 | 0.069 | 0.277 | 0.611 | 0.513 | 0.699 | 0.710 | 0.013 | 0.290 | 0.000 | 47.000 | 0.103 | 0.997 | 0.996 |


### Capacity control `kv_mean_matched` (wider head, not the main comparison)

| method | seed | threshold | n_safe | n_harm | n_prompt_groups | safe_response_fpr | premature_alarm_rate | recall_by_end_plus_32 | first_harm_sentence_alarm_rate | recall_by_end_plus_128 | nonpremature_detection_rate | no_alarm_rate | invalid_or_missed_rate | delay_after_sentence_end_p50 | delay_after_sentence_end_p90 | detection_fraction_p50 | stream_response_auroc | final_prefix_auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kv_mean_matched | 17 | -0.781 | 548 | 452 | 974 | 0.077 | 0.374 | 0.575 | 0.504 | 0.604 | 0.608 | 0.018 | 0.392 | 0.000 | 11.600 | 0.080 | 0.996 | 0.996 |
| kv_mean_matched | 42 | -0.507 | 548 | 452 | 974 | 0.068 | 0.330 | 0.613 | 0.511 | 0.650 | 0.655 | 0.015 | 0.345 | 0.000 | 17.500 | 0.091 | 0.997 | 0.997 |
| kv_mean_matched | 2026 | -0.071 | 548 | 452 | 974 | 0.071 | 0.270 | 0.628 | 0.520 | 0.708 | 0.717 | 0.013 | 0.283 | 0.000 | 41.400 | 0.102 | 0.997 | 0.996 |


### Other scheduled layers (h_mean, v_mean, kv_mean)

| layer | method | feature_dim | parameter_count | safe_response_fpr_mean | safe_response_fpr_std | premature_alarm_rate_mean | premature_alarm_rate_std | recall_by_end_plus_32_mean | recall_by_end_plus_32_std |
|---|---|---|---|---|---|---|---|---|---|
| 11 | h_mean | 2560 | 327937 | 0.054 | 0.005 | 0.256 | 0.019 | 0.619 | 0.015 |
| 11 | kv_mean | 2048 | 262401 | 0.047 | 0.005 | 0.243 | 0.015 | 0.624 | 0.012 |
| 11 | v_mean | 1024 | 131329 | 0.047 | 0.007 | 0.251 | 0.014 | 0.608 | 0.018 |
| 35 | h_mean | 2560 | 327937 | 0.067 | 0.003 | 0.338 | 0.018 | 0.580 | 0.008 |
| 35 | kv_mean | 2048 | 262401 | 0.064 | 0.002 | 0.280 | 0.041 | 0.615 | 0.036 |
| 35 | v_mean | 1024 | 131329 | 0.064 | 0.004 | 0.281 | 0.021 | 0.604 | 0.010 |


### Layer ranking on dev

For each of h_mean, v_mean and kv_mean, rank layers by mean recall_by_end_plus_32 over seeds on dev. Ties break to lower premature_alarm_rate, then lower safe_response_fpr, then the smaller layer index. Block 23 stays the reference comparison and is not replaced by this ranking. Dev was used for early stopping, so this is development selection, not a test result.


`h_mean` selected layer index 11:

| layer | n_seeds | recall_by_end_plus_32 | premature_alarm_rate | safe_response_fpr |
|---|---|---|---|---|
| 11.000 | 3.000 | 0.619 | 0.256 | 0.054 |
| 23.000 | 3.000 | 0.603 | 0.330 | 0.064 |
| 35.000 | 3.000 | 0.580 | 0.338 | 0.067 |


`v_mean` selected layer index 11:

| layer | n_seeds | recall_by_end_plus_32 | premature_alarm_rate | safe_response_fpr |
|---|---|---|---|---|
| 11.000 | 3.000 | 0.608 | 0.251 | 0.047 |
| 35.000 | 3.000 | 0.604 | 0.281 | 0.064 |
| 23.000 | 3.000 | 0.597 | 0.336 | 0.066 |


`kv_mean` selected layer index 11:

| layer | n_seeds | recall_by_end_plus_32 | premature_alarm_rate | safe_response_fpr |
|---|---|---|---|---|
| 11.000 | 3.000 | 0.624 | 0.243 | 0.047 |
| 35.000 | 3.000 | 0.615 | 0.280 | 0.064 |
| 23.000 | 3.000 | 0.605 | 0.319 | 0.070 |


## Paired differences versus same-layer h_mean

| layer | method | seed | metric | difference | ci95_lo | ci95_hi |
|---|---|---|---|---|---|---|
| 23 | h_last | 17 | safe_response_fpr | 0.011 | -0.013 | 0.033 |
| 23 | h_last | 17 | recall_by_end_plus_32 | 0.022 | -0.035 | 0.079 |
| 23 | h_last | 17 | premature_alarm_rate | -0.257 | -0.298 | -0.218 |
| 23 | h_last | 42 | safe_response_fpr | 0.007 | -0.014 | 0.029 |
| 23 | h_last | 42 | recall_by_end_plus_32 | -0.042 | -0.097 | 0.015 |
| 23 | h_last | 42 | premature_alarm_rate | -0.188 | -0.227 | -0.149 |
| 23 | h_last | 2026 | safe_response_fpr | -0.004 | -0.028 | 0.020 |
| 23 | h_last | 2026 | recall_by_end_plus_32 | -0.082 | -0.140 | -0.022 |
| 23 | h_last | 2026 | premature_alarm_rate | -0.212 | -0.251 | -0.174 |
| 23 | k_mean | 17 | safe_response_fpr | 0.004 | -0.017 | 0.023 |
| 23 | k_mean | 17 | recall_by_end_plus_32 | 0.033 | -0.002 | 0.068 |
| 23 | k_mean | 17 | premature_alarm_rate | -0.053 | -0.086 | -0.020 |
| 23 | k_mean | 42 | safe_response_fpr | 0.011 | -0.009 | 0.030 |
| 23 | k_mean | 42 | recall_by_end_plus_32 | 0.013 | -0.022 | 0.050 |
| 23 | k_mean | 42 | premature_alarm_rate | -0.040 | -0.069 | -0.009 |
| 23 | k_mean | 2026 | safe_response_fpr | 0.004 | -0.015 | 0.023 |
| 23 | k_mean | 2026 | recall_by_end_plus_32 | -0.011 | -0.056 | 0.029 |
| 23 | k_mean | 2026 | premature_alarm_rate | -0.064 | -0.096 | -0.032 |
| 23 | v_mean | 17 | safe_response_fpr | -0.004 | -0.018 | 0.010 |
| 23 | v_mean | 17 | recall_by_end_plus_32 | 0.027 | -0.005 | 0.054 |
| 23 | v_mean | 17 | premature_alarm_rate | -0.029 | -0.055 | 0.000 |
| 23 | v_mean | 42 | safe_response_fpr | 0.004 | -0.011 | 0.018 |
| 23 | v_mean | 42 | recall_by_end_plus_32 | -0.015 | -0.051 | 0.019 |
| 23 | v_mean | 42 | premature_alarm_rate | 0.029 | -0.002 | 0.062 |
| 23 | v_mean | 2026 | safe_response_fpr | 0.005 | -0.010 | 0.021 |
| 23 | v_mean | 2026 | recall_by_end_plus_32 | -0.029 | -0.065 | 0.007 |
| 23 | v_mean | 2026 | premature_alarm_rate | 0.018 | -0.011 | 0.050 |
| 23 | kv_mean | 17 | safe_response_fpr | 0.004 | -0.012 | 0.021 |
| 23 | kv_mean | 17 | recall_by_end_plus_32 | 0.020 | -0.015 | 0.052 |
| 23 | kv_mean | 17 | premature_alarm_rate | -0.015 | -0.044 | 0.014 |
| 23 | kv_mean | 42 | safe_response_fpr | 0.009 | -0.007 | 0.025 |
| 23 | kv_mean | 42 | recall_by_end_plus_32 | -0.007 | -0.038 | 0.025 |
| 23 | kv_mean | 42 | premature_alarm_rate | 0.013 | -0.015 | 0.044 |
| 23 | kv_mean | 2026 | safe_response_fpr | 0.004 | -0.010 | 0.017 |
| 23 | kv_mean | 2026 | recall_by_end_plus_32 | -0.004 | -0.042 | 0.030 |
| 23 | kv_mean | 2026 | premature_alarm_rate | -0.031 | -0.058 | -0.002 |
| 11 | v_mean | 17 | safe_response_fpr | -0.004 | -0.020 | 0.012 |
| 11 | v_mean | 17 | recall_by_end_plus_32 | -0.013 | -0.050 | 0.020 |
| 11 | v_mean | 17 | premature_alarm_rate | -0.013 | -0.039 | 0.014 |
| 11 | v_mean | 42 | safe_response_fpr | -0.013 | -0.027 | 0.000 |
| 11 | v_mean | 42 | recall_by_end_plus_32 | -0.042 | -0.079 | -0.004 |
| 11 | v_mean | 42 | premature_alarm_rate | -0.002 | -0.029 | 0.024 |
| 11 | v_mean | 2026 | safe_response_fpr | -0.004 | -0.018 | 0.010 |
| 11 | v_mean | 2026 | recall_by_end_plus_32 | 0.022 | -0.013 | 0.056 |
| 11 | v_mean | 2026 | premature_alarm_rate | 0.002 | -0.022 | 0.026 |
| 11 | kv_mean | 17 | safe_response_fpr | -0.007 | -0.022 | 0.007 |
| 11 | kv_mean | 17 | recall_by_end_plus_32 | 0.013 | -0.023 | 0.049 |
| 11 | kv_mean | 17 | premature_alarm_rate | -0.024 | -0.055 | 0.005 |
| 11 | kv_mean | 42 | safe_response_fpr | -0.007 | -0.023 | 0.009 |
| 11 | kv_mean | 42 | recall_by_end_plus_32 | -0.015 | -0.048 | 0.022 |
| 11 | kv_mean | 42 | premature_alarm_rate | 0.011 | -0.016 | 0.036 |
| 11 | kv_mean | 2026 | safe_response_fpr | -0.005 | -0.023 | 0.011 |
| 11 | kv_mean | 2026 | recall_by_end_plus_32 | 0.018 | -0.016 | 0.052 |
| 11 | kv_mean | 2026 | premature_alarm_rate | -0.027 | -0.051 | -0.004 |
| 35 | v_mean | 17 | safe_response_fpr | 0.000 | -0.017 | 0.016 |
| 35 | v_mean | 17 | recall_by_end_plus_32 | 0.035 | -0.000 | 0.072 |
| 35 | v_mean | 17 | premature_alarm_rate | -0.088 | -0.123 | -0.059 |
| 35 | v_mean | 42 | safe_response_fpr | -0.009 | -0.027 | 0.009 |
| 35 | v_mean | 42 | recall_by_end_plus_32 | 0.027 | -0.011 | 0.062 |
| 35 | v_mean | 42 | premature_alarm_rate | -0.058 | -0.086 | -0.029 |
| 35 | v_mean | 2026 | safe_response_fpr | 0.000 | -0.016 | 0.016 |
| 35 | v_mean | 2026 | recall_by_end_plus_32 | 0.009 | -0.027 | 0.042 |
| 35 | v_mean | 2026 | premature_alarm_rate | -0.024 | -0.053 | 0.006 |
| 35 | kv_mean | 17 | safe_response_fpr | -0.002 | -0.018 | 0.014 |
| 35 | kv_mean | 17 | recall_by_end_plus_32 | 0.004 | -0.028 | 0.039 |
| 35 | kv_mean | 17 | premature_alarm_rate | -0.033 | -0.063 | -0.005 |
| 35 | kv_mean | 42 | safe_response_fpr | -0.004 | -0.022 | 0.015 |
| 35 | kv_mean | 42 | recall_by_end_plus_32 | 0.040 | 0.002 | 0.077 |
| 35 | kv_mean | 42 | premature_alarm_rate | -0.055 | -0.089 | -0.024 |
| 35 | kv_mean | 2026 | safe_response_fpr | -0.004 | -0.021 | 0.015 |
| 35 | kv_mean | 2026 | recall_by_end_plus_32 | 0.060 | 0.022 | 0.095 |
| 35 | kv_mean | 2026 | premature_alarm_rate | -0.084 | -0.116 | -0.053 |
| 23 | kv_mean_matched | 17 | safe_response_fpr | 0.009 | -0.007 | 0.026 |
| 23 | kv_mean_matched | 17 | recall_by_end_plus_32 | 0.002 | -0.028 | 0.034 |
| 23 | kv_mean_matched | 17 | premature_alarm_rate | 0.009 | -0.019 | 0.036 |
| 23 | kv_mean_matched | 42 | safe_response_fpr | 0.007 | -0.009 | 0.023 |
| 23 | kv_mean_matched | 42 | recall_by_end_plus_32 | -0.007 | -0.039 | 0.024 |
| 23 | kv_mean_matched | 42 | premature_alarm_rate | 0.013 | -0.014 | 0.045 |
| 23 | kv_mean_matched | 2026 | safe_response_fpr | 0.005 | -0.009 | 0.019 |
| 23 | kv_mean_matched | 2026 | recall_by_end_plus_32 | 0.013 | -0.023 | 0.051 |
| 23 | kv_mean_matched | 2026 | premature_alarm_rate | -0.038 | -0.067 | -0.007 |

Intervals are prompt-group bootstrap intervals conditional on this training run and the fixed calibration threshold. They are not a full calibration uncertainty.

## Memory and latency

| method | role | batch | prompt_tokens | response_tokens | n_ok | n_oom | n_error | peak_allocated_mib_median | extra_peak_mib | extra_peak_reliable | persistent_state_bytes | decode_ms_per_token_median | latency_overhead_pct | tokens_per_second_total_median | prefill_ms_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_no_detector | main | 1 | 128 | 512 | 3 | 0 | 0 | 7843.529 | 0.000 | False | 0 | 27.520 | 0.000 | 36.338 | 33.403 |
| baseline_no_detector | main | 1 | 128 | 2048 | 3 | 0 | 0 | 8044.810 | 0.000 | False | 0 | 26.443 | 0.000 | 37.818 | 48.959 |
| h_last | main | 1 | 128 | 512 | 3 | 0 | 0 | 7842.557 | -0.972 | False | 10248 | 26.971 | -0.930 | 37.077 | 32.465 |
| h_last | main | 1 | 128 | 2048 | 3 | 0 | 0 | 8046.092 | 1.282 | False | 10248 | 26.775 | 1.258 | 37.348 | 33.252 |
| h_mean | main | 1 | 128 | 512 | 3 | 0 | 0 | 7842.557 | -0.972 | False | 10248 | 27.211 | -1.383 | 36.750 | 32.899 |
| h_mean | main | 1 | 128 | 2048 | 3 | 0 | 0 | 8046.092 | 1.282 | False | 10248 | 27.335 | 3.423 | 36.584 | 37.489 |
| k_mean | main | 1 | 128 | 512 | 3 | 0 | 0 | 7844.043 | 0.515 | False | 4104 | 27.382 | -1.572 | 36.521 | 47.460 |
| k_mean | main | 1 | 128 | 2048 | 3 | 0 | 0 | 8045.325 | 0.515 | False | 4104 | 26.876 | 1.589 | 37.208 | 32.885 |
| kv_mean | main | 1 | 128 | 512 | 3 | 0 | 0 | 7844.555 | 1.026 | False | 8200 | 27.574 | 0.658 | 36.266 | 35.667 |
| kv_mean | main | 1 | 128 | 2048 | 3 | 0 | 0 | 8045.836 | 1.026 | False | 8200 | 27.273 | 3.091 | 36.666 | 32.988 |
| kv_mean_matched | capacity_control | 1 | 128 | 512 | 3 | 0 | 0 | 7842.552 | -0.977 | False | 8200 | 27.022 | -1.809 | 37.007 | 33.248 |
| kv_mean_matched | capacity_control | 1 | 128 | 2048 | 3 | 0 | 0 | 8046.087 | 1.277 | False | 8200 | 27.107 | 2.462 | 36.891 | 49.836 |
| v_mean | main | 1 | 128 | 512 | 3 | 0 | 0 | 7844.043 | 0.515 | False | 4104 | 27.342 | 1.354 | 36.574 | 33.105 |
| v_mean | main | 1 | 128 | 2048 | 3 | 0 | 0 | 8045.325 | 0.515 | False | 4104 | 26.300 | -0.587 | 38.022 | 48.395 |

Negative extra peak is kept. `extra_peak_reliable=False` means no reliable difference was measured relative to run-to-run spread. Decode time is one forward per response token, including the final flush, and excludes prefill. Throughput counts response tokens across the batch. `kv_mean_matched` is a capacity-control cost, reported with `role=capacity_control` when that column is present.

## Theoretical accumulator sizes (sanity check only)

| method | fp32_bytes_per_sample | KiB |
|---|---|---|
| h_mean | 10240 | 10.000 |
| k_mean | 4096 | 4.000 |
| v_mean | 4096 | 4.000 |
| kv_mean | 8192 | 8.000 |

The backbone KV cache is shared by every method. H-mean and KV-mean accumulators differ by about 2 KiB per sample, so a large peak-memory reduction is not expected from the accumulator alone.

## Limits

- Dev selected the checkpoints, so dev metrics are development feedback.

- The official test split was not scored in the pilot.

- Sentence-boundary `[a, b]` is an uncertain span. Delay after `b` is not leaked-token count.

- Label audit rows are pending human review.

- Calibration Wilson intervals describe the calibration counts only.


## Figures
- `figures/memory_vs_length.png`
- `figures/quality_vs_overhead.png`
- `figures/quality_vs_layer.png`
