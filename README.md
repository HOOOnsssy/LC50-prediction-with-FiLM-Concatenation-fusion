# Multimodal aquatic LC50 prediction

Model definitions, input data, trained weights and numerical interpretation code for Water Research manuscript WR115321.

Repository: https://github.com/HOOOnsssy/LC50-prediction-with-FiLM-Concatenation-fusion

## Research use and citation

The original code and trained checkpoints are publicly shared for **non-commercial academic research** under the [WR115321 Academic Research License](LICENSE). Permitted research includes reproduction, validation, comparison and methodological extension, without an individual access request. Commercial product development, paid services and proprietary commercial research require separate prior written permission. Direct requests to the repository owner using the contact email on the [owner's GitHub profile](https://github.com/HOOOnsssy).

Public research outputs using the code, checkpoints or their results must cite **both the associated study and this repository**. Redistributed modified versions must identify the original repository, the modifier, the modification date and the principal changes. Modified source files must carry change notices, and modified checkpoints must have an accompanying model note. Retain the original copyright and license notices. The full terms are in `LICENSE`; this is a custom academic research license, not MIT or an OSI-approved open-source license.

The code license does not apply to the underlying ECOTOX records or other files in `data/`. Applicable EPA and third-party terms remain unchanged. See [DATA_NOTICE.md](DATA_NOTICE.md) for sources, filtering, aggregation and the distinction between source data and model results.

[CITATION.cff](CITATION.cff) contains the software and study citation metadata. Cite the software version `2026.10.24` or the version actually used, together with the repository URL. The associated study is currently cited as:

Yeom, J., Kim, M., Kim, S. D., Jeong, H., and Cho, K. H. (2026). *Advancing Structural Interpretability in Multispecies Aquatic Toxicity Prediction through Feature-Wise Conditioning in Multimodal Graph Neural Networks*. Manuscript under review at Water Research (WR115321).

Use the final published study citation when it becomes available. No publication DOI or acceptance status is implied by this release.

## Study

The main experiment contains 2,943 LC50 records for 226 chemicals and 18 species. Four molecular encoders (descriptor MLP, 2D GINE, 3D GINE and SchNet) are combined with three fusion methods (concatenation, FiLM and FiLM-Concat). Each of these 12 models has ten compound-level split checkpoints. The splits overlap and are not ten disjoint folds. Predictions are log10(LC50 in mg/L).

## Contents

```text
models/         Model definitions, molecular features and interpretation methods
data/           Development/external records, atomic features and SDF structures
checkpoints/    Model-specific weight archives and a small lookup index
predict.py      Load a checkpoint and predict
explain.py      GNNExplainer, Fidelity and fixed-graph SHAP
requirements.txt
LICENSE         Academic research terms for original code and checkpoints
CITATION.cff    Software and associated-study citation metadata
DATA_NOTICE.md  Source-data notices and curation scope
```

The 120 main checkpoints are supplemented by 30 checkpoints used in the small-model, no-species and training-species-permutation comparisons. The `main` scope is the default. Checkpoints contain the fitted weights, architecture settings, fitted scalers, categorical mappings and chemical split assignments required for inference. Numerical model definitions and fitted tensors are preserved from the study.

This release focuses on loading and evaluating trained models. It does not include a training/HPO workflow, manuscript production code or project operation logs. Classical baselines and the separate split-ratio training experiment are outside this neural-checkpoint package. The primary 80-record external dataset is included and uses the existing main checkpoints.

## Environment

Validated environment: Python 3.9.23, PyTorch 2.4.0 with CUDA 11.8, PyG 2.6.1 and torch-cluster 1.6.3. Other package versions are pinned in `requirements.txt`. In an isolated Python environment:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 --only-binary=:all: -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
python -m pip install -r requirements.txt
```

These commands follow the [PyTorch version-specific installation instructions](https://pytorch.org/get-started/previous-versions/) and [PyG extension installation instructions](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html). The source environment was validated; a new environment installation was not performed as part of packaging.

The default `--device auto` uses CUDA when available. Use `--device cuda` for reproduction of the reported numerical results. In particular, SchNet's radius-neighbor construction can select different neighbors on CPU for crowded neighborhoods. CPU inference is available, but is not an exact substitute for the study's CUDA SchNet evaluation. Stochastic attribution and floating-point reductions can also vary across devices and library versions.

## Prediction

Run commands from this directory. Keep the model ZIP files compressed: the loader reads weights directly from them. No manual archive joining or hash verification step is required.

```bash
python predict.py --architecture 3D_GINE --fusion film_concat --seed 0 --device cuda --output outputs/test_predictions.csv
python predict.py --all --device cuda --output outputs/main_test_metrics.csv
python predict.py --architecture SchNet --fusion concat --seed 0 --device cuda --output outputs/schnet_test.csv
python predict.py --dataset external --fusion film_concat --seed 0 --device cuda --output outputs/external_predictions.csv
```

Architecture keys are `3D_MLP`, `2D_GINE`, `3D_GINE`, and `SchNet`. `3D_MLP` is the descriptor MLP in the manuscript. Fusion keys are `concat`, `film`, and `film_concat`. Seed values are 0–9. The default evaluates the checkpoint's held-out chemical split. `--split train` and `--split val` select the other assignments. For external data, all supplied external records are evaluated.

The `--all` command evaluates all 120 main checkpoints and writes one metrics row per checkpoint. `data/expected_test_metrics.csv` contains the corresponding study values, including the three controls. Small numerical discrepancies from float32 computation are expected.

The auxiliary checkpoints use the same loading command:

```bash
python predict.py --scope small --seed 0 --device cuda --output outputs/small.csv
python predict.py --scope no_species --seed 0 --device cuda --output outputs/no_species.csv
python predict.py --scope permute_train --seed 0 --device cuda --output outputs/permuted_training_species.csv
```

`--all --scope small`, for example, evaluates the ten small-model checkpoints. Species labels remain unpermuted at inference for the training-species-permutation control, as in the study.

## Interpretation

The paper's interpretation analyses use the three **3D GINE** fusion models. The interpretation command implements that scope, rather than implying that every encoder has the same atom-level explainer.

```bash
python explain.py gnn --fusion film_concat --seed 0 --row-id 36 --device cuda --output outputs/gnn_example
python explain.py shap --fusion film_concat --seed 0 --row-id 36 --device cuda --output outputs/shap_example
```

Omit `--row-id` to process all held-out records of a checkpoint. Use `--limit 1` for a short execution check. Run separately for each fusion and split seed when reproducing the complete analysis.

GNNExplainer uses 200 mask-optimization epochs and explainer seed 0 by default. The model weights are not fitted. Outputs include node-feature masks, edge masks, node scores, grouped atomic-feature contributions, Gini, Shannon entropy, and Fidelity at 10%, 25%, and 50% atom retention. Explicit hydrogens are included. Non-graph inputs remain unchanged. FidelityMinus measures the absolute prediction change after retaining the selected subgraph; FidelityPlus measures the change after removing it. The masks and numerical outputs can be used to draw molecular maps without journal-specific plotting code.

For the supplementary equal-size random-retention control:

```bash
python explain.py gnn --fusion film_concat --seed 0 --row-id 36 --random-repeats 100 --device cuda --output outputs/random_retention
```

This uses uniform atom permutations from NumPy PCG64 with `SeedSequence([20261021, split_seed, row_id])`. Fractions use nested prefixes of each permutation and the same permutations are used across fusions. `RandomRetentionFidelityMinus` is the within-record mean over the random draws. Aggregate records within each split before comparing the ten split means. Random draws and overlapping held-out records are not independent replicates.

For species-conditioned attribution, supply a species name from `data/development.csv`, keeping the same record, split and other settings:

```bash
python explain.py gnn --fusion film_concat --seed 0 --row-id 36 --species "Daphnia magna" --device cuda --output outputs/species_daphnia
```

Use separate output directories for different species or explainer seeds. The `pair_metrics` helper in `models/interpretation.py` compares node-score vectors. The paper's species-context Jaccard analysis selects the top 25% of **heavy atoms**, whereas the primary Fidelity analysis uses all atoms, including hydrogens. Interpretation describes model attribution, not a biological mechanism.

Use `--exposure Static --duration 1` when fixing that exposure setting. Each saved mask file includes `heavy_atom_indices`. Apply these indices to the node-score arrays before using `pair_metrics` for heavy-atom comparisons. Context overrides change model inputs only and do not turn a hypothetical context into an observed measurement.

SHAP uses a fixed molecular graph representation, 100 training-background records and GradientExplainer with 200 samples. The background RNG is `91000 + split_seed`; the record RNG follows the original held-out record order. Categorical dimensions are summed as signed SHAP values before taking the absolute value. Outputs include signed and absolute values for the six grouped non-graph inputs. To obtain the paper's cross-seed summary, first average absolute values over held-out records in each seed, then calculate the mean and sample SD over the ten seeds.

## Data and upload

The SDF archive is expanded automatically into `.cache/structures` when needed. Input CSVs retain scientific record identifiers for connecting predictions and attribution to the supplied records. They exclude internal group IDs and processing flags. The separate journal Supplementary Data workbook contains the measurements, geometric-mean calculations and first-stage filtered ECOTOX records.

All upload files are below **25,000,000 bytes**. Large SchNet model groups are stored in two independent model archives; the lookup index selects the correct archive automatically. Upload this directory's files and the `models`, `data`, and `checkpoints` folders, keeping this structure. Do not unzip checkpoint archives before uploading. Do not upload generated `.cache`, `__pycache__` or `outputs` folders. GitHub's browser uploader accepts up to 100 files per operation and 25 MiB per file ([GitHub documentation](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)).
