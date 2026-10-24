# Data sources and use

The academic research restrictions in `LICENSE` apply to the original software and model checkpoints. They do **not** apply to the underlying ECOTOX source records or impose a new license on the files in `data/` or the journal Supplementary Data workbook. This notice does not itself grant rights in third-party materials or create restrictions on public-domain facts.

## Toxicity records

The underlying toxicity records were obtained from the [U.S. EPA ECOTOX Knowledgebase](https://cfpub.epa.gov/ecotox/). This repository does not claim ownership of those source records or alter any applicable EPA or third-party terms. Refer to the [ECOTOX documentation and limitations](https://cfpub.epa.gov/ecotox/help.cfm) and the [EPA notices](https://www.epa.gov/web-policies-and-procedures/epa-disclaimers), including any conditions attached to particular source materials. The repository does not assert that EPA owns every underlying published study or that one uniform license applies to all source materials.

The study authors filtered the records, standardized concentration units, applied documented corrections, and aggregated eligible measurements by geometric mean. These transformations are the authors' work and are not an official EPA dataset or an EPA-endorsed analysis. The selected records are a study subset, not the complete or continuously updated ECOTOX Knowledgebase.

| Supplied data | Scope |
| --- | --- |
| `data/development.csv` | 2,943 aggregated LC50 records for 226 development chemicals and 18 species |
| `data/external.csv` | 80 aggregated LC50 records for 27 development-excluded chemicals |
| Journal Supplementary Data, `ECOTOX raw` | 10,767 source records after initial chemical, LC50 endpoint and study-species filters, before measurement-level eligibility filtering |
| Journal Supplementary Data, `Measurements` | 6,805 eligible development measurements and 143 eligible external measurements, with reported experimental information and source-row links |
| Journal Supplementary Data, `Geometric means` | 2,943 development and 80 external aggregates, with calculation formulas |

The initial source-record filter covers the 226 development and 27 external-evaluation chemicals and the 18 study species. The workbook retains source fields, including reported life stage and exposure information. Not every first-stage source record entered an aggregate or model fit. Life stage was preserved as reported information; it was not an additional grouping variable in the fitted experiment.

When describing use of these records, acknowledge ECOTOX, retain the available original-study references, and distinguish the authors' filtered/aggregated dataset from the source database. Please also cite the associated study when using its curation or aggregation. These attribution requests do not create new restrictions on underlying facts or displace source terms. A database reference is:

Olker, J. H., et al. (2022). The ECOTOXicology knowledgebase: A curated database of ecologically relevant toxicity tests to support environmental research and risk assessment. *Environmental Toxicology and Chemistry*, 41(6), 1520–1539. https://doi.org/10.1002/etc.5324

## Other supplied data

`data/structures.zip` contains the molecular structures used for model evaluation, and `data/atomic_features.xlsx` contains the elemental feature table used by the models. The manuscript identifies PubChem as the source of the static elemental properties and RDKit as the tool used for calculated molecular features. These scientific inputs are excluded from the academic code license. Applicable source rights, notices and dependency licenses remain unchanged; the repository does not assert that EPA controls these non-ECOTOX inputs.

`data/expected_test_metrics.csv` contains reference numerical model results for checking inference. It is not a collection of original ECOTOX measurements. Please identify the study and software version when reporting comparisons with these results.

## Responsibility

Source data can contain reporting limitations, missing fields and subsequent corrections. Users should consult the original study references and source documentation when interpreting individual records. No EPA endorsement of the code, model predictions, modifications or conclusions is implied.
