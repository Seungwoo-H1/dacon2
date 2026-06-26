# Future Work

Concrete, prioritized directions for the next iteration of this (or a similar lifelog)
competition. Ordered by expected value.

## 1. Validation-first methodology  *(highest value, lowest cost)*
Build the **time-blocked forward validator before any feature work**, and additionally a
held-out *probe* that mimics the future-extension split. The biggest losses this project
came from trusting validators (random-CV, subject-nested CV) whose split geometry did not
match the test set. Rule to adopt: *date-level features are validated by time, subject-level
features by subject.*

## 2. Treat the extension half as domain adaptation
The test set is ~half future-extension with no nearby labeled day. Instead of one i.i.d.
model, explicitly model the gap-to-nearest-neighbor and switch between:
- **interleaved rows** → recency dominates;
- **extension rows** → fall back to subject prior + any genuinely time-stable signal.
A meta-model on `[recency, gap, prior]` was tried (R-series) but overfit with only 10
subjects; revisit with proper time-blocked tuning and stronger regularization.

## 3. Sleep-metric priors from same-device external data
`docs/external_data_analysis.md` shows the **2018–2020 Withings** sleep data shares the
sensor family behind S2–S4. Even without subject overlap, it could provide:
- a distribution prior for sleep-efficiency / latency / WASO thresholds;
- a pretrained sleep-stage / TST estimator to improve the phone-derived `sleep_v3` features.
Use as a **prior / pretraining source**, never a label join.

## 4. Better feature engineering (only after 1–2)
- `mAmbience` (audio scene) and `mGps` (mobility / time-away-from-home) are unused; they
  could carry context for mood targets — but must clear forward-time CV.
- Lagged sleep → next-day mood (TST deviation → Q) is physiologically motivated.

## 5. Stronger ensembling — with discipline
Hill-climb / stacking only over models that each independently beat recency on forward CV,
and only if the ensemble survives the extension probe. Diversity from *genuinely different
signals* (e.g. sleep-prior model vs recency), not from re-skinned GBMs.

## 6. Representation learning / automated features
Subject embeddings, self-supervised pretraining on raw sensor sequences, automated feature
search. Low priority here (10 subjects, weak day-level signal) but worth it on the larger
50-subject ETRI 2024 release.

## 7. External label join (only if data returns)
The honest route below ~0.59 was joining the public ETRI test-period labels. The host
(nanum.etri.re.kr) shut down and the 10-subject file is currently unobtainable; the
50-subject e-PreTX release is a different cohort. `archive/exp_scripts/exp_R29_external_join.py`
is ready to run the instant a matching `(subject_id, date, labels)` file appears.

## 8. Hyper-parameter optimization
Deferred deliberately — with this little signal, HPO mostly fits the validator. Worth doing
only once 1–2 give a trustworthy validator.
