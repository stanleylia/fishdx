# results/ — artefact inventory

Not every JSON here is an evaluation-of-record artefact. The manuscript's
reported numbers come from the files listed under **Evaluation of record**
below. The remaining files are intermediate, audit, or exploratory artefacts
retained for transparency; they use earlier data-filtering thresholds or
different subset sizes and are **not** the evaluation of record.

## Embedding caches (inputs the headline scripts replay)
- `_cache_d1_gallery.npz` — D1 reference gallery (1,639 × 512: visual + caption).
- `_cache_d2_query.npz`   — D2 query embeddings (3,473 × 512).
- `_cache_newdx_gallery.npz` — new-disease (field) embeddings.

## Evaluation of record (cited in the manuscript)
- `reviewer5/selective_predictions_d2final.jsonl` and `reviewer5/run_manifest.json`
  — the released 14,412 sample–method records requested in review, including
  predictions, similarities, margins, Stage-3 scores, abstention outputs and
  reason codes, plus the provenance manifest.
- `reviewer5/matched_coverage_knn.json`, `reviewer5/per_class_abstention.json`,
  `reviewer5/stage3_document_ablation.json` — the matched-coverage, class-level
  selective-prediction and candidate-conditioned evidence-pool analyses.
- `reviewer5/diagnostic_phrase_substitution.json` and `.jsonl` — aggregate and
  per-caption diagnostic-phrase intervention results. BP/CP/DP are the primary
  prepended conditions; B/C/D are the secondary appended conditions.
- `d2_final_full.json`, `d2_final_no_calib.json` — D2-final (n = 2,402): Table 3
  (six configs) and Table 4 (θ_margin sweep; Decisive DA 0.928 @ 54.4% coverage).
- `d2_clean_tables.json` — cross-dataset modality comparison (caption 0.495 /
  visual 0.893 / fusion 0.897; overlap set n = 2,176).
- `eus_theta_calibration.json` — θ* selection (226 calib / 226 test) + EUS
  interception 71.2% (161/226).
- `eus56_margin_and_latency.json` — EUS interception + latency (559 ms / 1.2 GB).
- `eus_onboarding_modality.json` — gradient-free EUS onboarding (0.696 @ n_ref=50).
- `newdx_ood_interception.json`, `newdx_onboarding.json` — field new-disease
  (Argulus/Oodinium) interception 69.5% (98/141) and onboarding.
- `second_vlm_sca.json`, `qwen_full_pipeline.json` — Qwen2-VL-2B second-VLM control.
- `d2_clean_mcnemar_k5.json` — the single exploratory McNemar test (k=5, p=0.036).
- `verification_penalty_proxy.json` — verification-loop penalty rates
  (3.4/47.2/94.0%) + Mann–Whitney U (p=2.12e-169), from
  `exp18_verification_adversarial.py`. NOTE: a retrieval-confidence proxy
  (top-1 similarity / margin), not live Florence-2 grounding — see the script
  header and REPRODUCIBILITY_MAP.md §1.

## Audit artefacts (support the leakage/provenance claims)
- `d1_d2_overlap_audit.json`, `d1_md5_audit.json`, `e1_dedup_report.json`,
  `d2_embedding_dedup.json`, `d2_dedup_sensitivity.json`, `kb_leakage_audit.json`.

## Intermediate / superseded (NOT evaluation of record; retained for transparency)
- `d2_clean_final.json`, `d2_clean_decisive.json`, `d2_clean_rerun.json` — earlier
  D2-clean pool (n = 2,628 / 1,941 / 1,499), superseded by the D2-final files above.
- `lambda_d2_ablation.json` (n = 1,639/1,242), `ablation_*_retrieval*.json`,
  `class_prototype_ablation.json`, `pareidolia_keyword_ablation.json` — exploratory
  ablations on earlier subsets.
- `mcnemar_d2_reproduced*.json`, `mcnemar_d2_panelA_identification.json`,
  `mcnemar_d2_margin_scale_diagnostic.json` (n = 3,453) — earlier McNemar
  reproductions on the pre-dedup set; see Supplementary Note S5.
- `e2_*.json`, `e3_*.json` — caption/coverage logs and quality reports.

Provenance anchor for all artefacts: the deposit commit in `../PROVENANCE.txt`.
