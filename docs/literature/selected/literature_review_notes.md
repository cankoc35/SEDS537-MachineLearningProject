# Literature Review Notes

## Review Date

May 7, 2026

## Project Fit Criteria

The project focuses on hallucination detection for QA-style LLM outputs using uncertainty signals, self-consistency, and retrieval/evidence agreement. Papers were selected if they directly help with at least one of these needs:

- HaluEval or TruthfulQA-style hallucination benchmarks.
- Token probability, log-probability, entropy, calibration, or uncertainty-based detection.
- Self-consistency, semantic entropy, or multi-generation uncertainty.
- Evidence-aware or judge-style hallucination detection relevant to a multi-signal detector.
- Practical baseline design for binary hallucination classification.

Original PDFs were left in `docs/literature/`. The strongest papers were copied into `docs/literature/selected/`.

## Selected Core Papers

### 1. HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models

- File: `2023.emnlp-main.397.pdf`
- Priority: Must read.
- Why selected: This is the original HaluEval benchmark paper, and our main processed dataset comes from HaluEval QA.
- Main contribution: Introduces HaluEval, a large benchmark of generated and human-annotated hallucinated samples across QA, dialogue, summarization, and general user-query settings.
- Project use: Needed for dataset description, label interpretation, and justification for using HaluEval as the main development dataset.

### 2. Detecting Hallucinations in Large Language Models Using Semantic Entropy

- File: `s41586-024-07421-0.pdf`
- Priority: Must read.
- Why selected: Foundational paper for entropy-based hallucination detection.
- Main contribution: Proposes semantic entropy, which clusters multiple generated answers by meaning and computes uncertainty over semantic clusters rather than surface token forms.
- Project use: Supports the self-consistency and entropy parts of our proposed method, and explains why token-level entropy alone can be insufficient.

### 3. Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus

- File: `2023.emnlp-main.58v2.pdf`
- Priority: Must read.
- Why selected: Directly studies uncertainty-based hallucination detection without retrieval or multiple sampled answers.
- Main contribution: Improves token-probability-based detection by focusing on informative tokens, unreliable context tokens, and token properties such as type and frequency.
- Project use: Strong reference for our first feature extraction stage using token log-probabilities, entropy, and confidence features.

### 4. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs

- File: `2406.15927v1.pdf`
- Priority: High.
- Why selected: Important follow-up to semantic entropy that reduces multi-sampling cost.
- Main contribution: Trains probes on hidden states to approximate semantic entropy from a single generation or even before generation.
- Project use: Useful for discussing the cost of semantic entropy and possible future white-box model extensions.

### 5. A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation

- File: `2025.naacl-long.158.pdf`
- Priority: High.
- Why selected: Strong recent detection method using probabilistic reasoning over related claims.
- Main contribution: Builds a belief tree of logically related claims and uses probabilistic inference to correct noisy or inconsistent model beliefs.
- Project use: Useful for the multi-signal section and for explaining why a single confidence score may be unreliable.

### 6. ChainPoll: A High Efficacy Method for LLM Hallucination Detection

- File: `2310.18344v1.pdf`
- Priority: High.
- Why selected: Relevant for LLM-as-judge and real-world hallucination metric design.
- Main contribution: Proposes ChainPoll and RealHall, emphasizing adherence and correctness as hallucination evaluation dimensions.
- Project use: Useful for comparing uncertainty-only detection with judge-style or reasoning-based detection.

### 7. Hallucination Detection in Large Language Models via Multi-Granular Uncertainty Quantification

- File: `Paper3-17665-proof.pdf`
- Priority: High, but verify publication quality before heavy citation.
- Why selected: Extremely close to our planned method.
- Main contribution: Combines token-level, sequence-level, temporal, and distributional uncertainty features from a single generation.
- Project use: Useful for feature ideas such as mean entropy, entropy variance, top-k probability mass, confidence gap, length-normalized uncertainty, and temporal entropy dynamics.

### 8. Detecting Hallucinations in LLM Responses Using Token-Level Log-Probability Signals

- File: `54_Detecting_Hallucinations_in.pdf`
- Priority: Medium-high, but verify publication quality before heavy citation.
- Why selected: Directly matches our immediate next development step.
- Main contribution: Uses token-level log-probability statistics such as mean, variance, percentiles, and low-confidence token ratios for lightweight hallucination detection.
- Project use: Practical reference for the first baseline feature table.

### 9. Hallucination Detection and Confidence Calibration for Large Language Model Outputs: Reproducible Experiments on HaluEval

- File: `Hallucination+Detection+and+Confidence+Calibration+for+Large+Language+Model+Outputs+Reproducible+Experiments+on+HaluEval.pdf`
- Priority: Medium-high, but verify venue quality before heavy citation.
- Why selected: Directly uses HaluEval and binary hallucination classification.
- Main contribution: Studies TF-IDF, linear models, calibration, expected calibration error, and domain-conditional Platt scaling on HaluEval.
- Project use: Useful for baseline design, calibration discussion, and explaining why AUROC/F1 should be reported with confidence calibration metrics.

### 10. Uncertainty Quantification for Hallucination Detection in Large Language Models: Foundations, Methodology, and Future Directions

- File: `2510.12040v1.pdf`
- Priority: Medium-high.
- Why selected: Good conceptual review of uncertainty quantification for hallucination detection.
- Main contribution: Organizes UQ methods into token probability, output consistency, internal representation, and self-checking approaches.
- Project use: Useful for literature review structure and terminology: aleatoric uncertainty, epistemic uncertainty, black-box vs white-box access, and single-output vs multi-output methods.

## Useful But Not Selected

These papers are relevant but not central enough for the selected folder.

- `2403.04307v3.pdf` — HaluEval-Wild. Useful benchmark background, but our current project uses standard HaluEval QA and TruthfulQA.
- `2024.findings-emnlp.529.pdf` — DiaHalu. Dialogue-level benchmark; useful future work, but not central to QA-only scope.
- `2406.07070v1.pdf` — HalluDial. Dialogue-level hallucination benchmark; not needed for current QA pipeline.
- `2406.11267v1.pdf` — Faithful Finetuning. Mitigation/fine-tuning paper, not a detection pipeline for our current implementation.
- `24110_Efficient_Hallucination_.pdf` — Efficient hallucination detection with uncertainty-aware attention heads. Relevant white-box UQ idea, but anonymous under-review status makes it weaker for a term-paper core citation.
- `2508.10192v1.pdf` — Prompt-response semantic divergence metrics. Interesting but less central than semantic entropy and token uncertainty papers.
- `2508.14496v3.pdf` — Semantic Energy. Relevant advanced uncertainty method, but beyond the first project scope.
- `2511.16275v3.pdf` — SeSE structural entropy. Advanced black-box UQ; useful later but not necessary for the first implementation.
- `Semantic_Reformulation_Entropy_for_Robust_Hallucination_Detection_in_QA_Tasks.pdf` — Relevant QA semantic entropy extension, but less foundational than the Nature semantic entropy paper.
- `Predicting_Truthfulness_of_Large_Language_Model_Generated_Answers_Using_TruthfulQA.pdf` — Uses TruthfulQA and uncertainty/self-evaluation ideas, but appears basic and secondary compared with stronger papers.
- `2024.acl-long.483.pdf` — TruthX. Good ACL paper, but it is mainly hallucination mitigation by representation editing, not detection.
- `2601.09929v1.pdf` — Operational detection/mitigation framework. Useful industry framing, but too broad for core technical review.

## Not Selected / Low Priority

These are broad, duplicated, thesis-style, or too far from the current implementation.

- `2508.01781v1.pdf` — General taxonomy. Too broad for core project work.
- `2601.20026v1.pdf` — Quantum tensor network semantic UQ. Too specialized and unnecessary for this project.
- `BestettiElisa.pdf` — Master thesis literature review. Useful for background, but not a primary paper.
- `FULLTEXT01.pdf` — Master thesis on Bayesian neural network ensembling. Interesting but too broad and not central.
- `YSChuang_PHD-Thesis.pdf` — Broad PhD thesis on factual/trustworthy LLMs. Good background, but too large for core review.
- `ZHENG-THESIS-2024.pdf` — Thesis on factuality in QA systems. Background only.
- `preprints202510.0540.v2.pdf` — Broad hallucination survey and not peer-reviewed.

## Missing But Recommended To Add

The current folder does not appear to include the original TruthfulQA paper. Add it if possible:

- TruthfulQA: Measuring How Models Mimic Human Falsehoods.

Also useful if not already present:

- SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.
- FacTool or a similar retrieval/evidence-based factuality checking paper.

## Suggested Reading Order

1. `2023.emnlp-main.397.pdf`
2. `s41586-024-07421-0.pdf`
3. `2023.emnlp-main.58v2.pdf`
4. `2406.15927v1.pdf`
5. `Paper3-17665-proof.pdf`
6. `54_Detecting_Hallucinations_in.pdf`
7. `Hallucination+Detection+and+Confidence+Calibration+for+Large+Language+Model+Outputs+Reproducible+Experiments+on+HaluEval.pdf`
8. `2025.naacl-long.158.pdf`
9. `2310.18344v1.pdf`
10. `2510.12040v1.pdf`

## How These Papers Support Our Project

- Dataset justification: HaluEval paper.
- Uncertainty feature extraction: Stronger Focus, Multi-Granular UQ, Token-Level Log-Probability Signals.
- Entropy/self-consistency: Semantic Entropy and Semantic Entropy Probes.
- Multi-signal detector motivation: Belief Tree Propagation and ChainPoll.
- Calibration and baselines: HaluEval calibration paper.
- Literature review framing: UQ foundations survey.
