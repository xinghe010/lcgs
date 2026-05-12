# LCGS — Logic-Constrained Graph System

Reference implementation for the paper

> **System-Level Logic-Constrained Graph Learning under Structural Heterogeneity.**

LCGS is a *model-agnostic* neural-symbolic framework that injects first-order knowledge throughout the entire graph-learning pipeline — data construction, normalization, similarity, and optimization — rather than only as a late-stage constraint. The same framework is exercised here on **two structurally heterogeneous tasks** (premise selection and code clone detection) across **six backbones** (graph-based and pre-trained).

---

## Tasks at a glance

| Task                       | Datasets                | Backbones touched by LCGS                          | Headline number                |
|----------------------------|-------------------------|----------------------------------------------------|--------------------------------|
| First-order premise selection | MPTP2078, CNF        | TWGNN, ASTWGNN                                     | +3.0 pp acc / +8.3 pp recall   |
| End-to-end ATP (E-prover)  | MPTP2078 (280 problems) | TWGNN-based selector                               | 229 single pass / 251 with feedback |
| Code clone detection       | BigCloneBench (BCB), POJ-104 | ASTNN, GMN, GGNN, CodeBERT, GraphCodeBERT     | Consistent gains across all five |

A separate sub-tree exists for each task; their entry points are listed in §4.

---

## 1. Framework components

The shared core lives in `code_clone/models/LCGS_core/` and provides the five rule families R1–R5 from the paper:

| Module                      | Paper rule | Role                                                      |
|-----------------------------|-----------|-----------------------------------------------------------|
| `augmentation.py`           | R1        | Relation-closure (transitivity) data augmentation         |
| `normalization.py`          | R2        | Semantics-preserving graph rewriting                      |
| `compatibility.py`          | R3        | Type-hierarchical compatibility for node matching         |
| `tptf.py`                   | R4 (def. 1–2) | TP-TF: type-hierarchical position-sensitive term frequency |
| `cwj.py`                    | R4 (def. 3) | Composite Weighted Jaccard similarity                     |
| `lcgs_trainer.py`           | R5        | Constraint-driven optimization (hard-margin + structure-consistency losses) |

Theoretical properties (monotonicity, variance stability, boundedness, robustness) are derived in the paper; the unit-style sanity checks ship inline next to each module.

---

## 2. Repository layout

```
supplementary material_LCGS/
├── ATP_experiment/                    End-to-end E-prover loop
│   ├── code/                          Selector model (re-used by feedback_loop.py)
│   ├── dataset/                       Theorem set used for proof attempts
│   └── scripts/                       run.py, feedback_loop.py, evaluate_*.py
├── premise_selection/                 Standalone classification benchmark
│   ├── MPTP/                          LCGS_TWGNN.py, LCGS_ASTWGNN.py
│   └── CNF/                           LCGS_TW_CNF.py, LCGS_ASTW_CNF.py
└── code_clone/                        Code-clone detection
    ├── data/{BCB,BCB_LCGS,POJ,POJ_LCGS}
    └── models/
        ├── LCGS_core/                 Framework components (see §1)
        ├── LCGS_ASTNN/                run_astnn_lcgs.py, pipeline_lcgs.py
        ├── LCGS_GMN/                  gmn_lcgs.py
        ├── LCGS_GGNN/                 ggnn_lcgs.py
        ├── LCGS_CodeBERT/             run.py
        ├── LCGS_GraphCodeBERT/        run.py + parser/DFG.py
        └── GNN_utils/                 Shared graph construction utilities
```

The `*_LCGS` data folders contain the LCGS-augmented and -normalized variants of BCB / POJ; the bare `BCB` / `POJ` folders are used to reproduce the unconstrained baselines.

---

## 3. Installation

```bash
pip install -r requirements.txt
```

Notes:

* The premise-selection and ATP pipelines depend only on the PyTorch / PyTorch Geometric stack plus `lark` for first-order parsing.
* The code-clone backbones add the heavier dependencies — `transformers`, `tree-sitter`, `javalang`, `pycparser`, `gensim`, `anytree`. If you only intend to reproduce the ATP results, those packages can safely be omitted.
* For end-to-end ATP, **E-prover** must be installed and discoverable on `$PATH`.

---

## 4. Per-task usage

### 4.1 Premise selection (MPTP / CNF)

```bash
# MPTP, plain term-walk backbone
cd premise_selection/MPTP
python LCGS_TWGNN.py        # train + evaluate
python LCGS_ASTWGNN.py      # attention-enhanced variant

# CNF counterpart
cd ../CNF
python LCGS_TW_CNF.py
python LCGS_ASTW_CNF.py
```

### 4.2 End-to-end ATP with E-prover

```bash
cd ATP_experiment/scripts
python run.py
python feedback_loop.py   # → 229 single pass, 251 after feedback
python evaluate_simple.py
python evaluate_fixed_k.py
```

### 4.3 Code clone detection

| Backbone        | Entry point                                                       |
|-----------------|-------------------------------------------------------------------|
| ASTNN           | `code_clone/models/LCGS_ASTNN/run_astnn_lcgs.py`                  |
| GMN             | `code_clone/models/LCGS_GMN/gmn_lcgs.py`                          |
| GGNN            | `code_clone/models/LCGS_GGNN/ggnn_lcgs.py`                        |
| CodeBERT        | `code_clone/models/LCGS_CodeBERT/run.py`                          |
| GraphCodeBERT   | `code_clone/models/LCGS_GraphCodeBERT/run.py`                     |

Each script picks up the corresponding `*_LCGS` data folder automatically.

---

## 5. Reproducibility notes

* The 60 %-data ablation (LCGS surpassing the strongest full-data graph baseline) is reproduced by passing `--train_fraction 0.6` to the premise-selection scripts.
* All randomness is seeded; minor variation (≤ 0.3 pp) across hardware is expected because of non-deterministic scatter operations.
* Pre-trained backbones (CodeBERT, GraphCodeBERT) are downloaded from HuggingFace at first run — set `HF_HOME` if you want to control the cache location.

---

## 6. Citation

Please cite the accompanying paper. The BibTeX entry will be added once the work is officially published.
