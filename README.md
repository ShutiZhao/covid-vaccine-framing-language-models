# Politicization or Medicalization?  
## Partisan Framing of COVID-19 Vaccines on Twitter Using Language Models and Embeddings

This repository contains the full data, code, and outputs for a text-as-data study examining how U.S. political actors frame COVID-19 vaccines on Twitter. The project combines structural topic modeling (STM), large language model (LLM) classification, and embedding-based semantic similarity to analyze partisan differences in vaccine discourse.

The analysis focuses on whether vaccine-related tweets are framed primarily in **medical/scientific** terms or **political/ideological** terms, and how these framings vary by party, vaccine brand, and time.

---

## Repository Structure

```
.
├── data/
│ ├── raw/
│ │ └── tweets_congress.csv # Raw tweet data (tracked with Git LFS)
│ └── processed/ # Intermediate and final datasets
│
├── scripts/ # Main analysis pipeline (run in order)
│ ├── 01_setup_and_filter.R
│ ├── 02_stm_prep_and_model.R
│ ├── 03_embeddings.R
│ ├── 04_llm_topic_labels.R
│ ├── 05_llm_frame_classification.R
│ ├── 06_merge_master_dataset.R
│ └── 07_descriptive_and_models.R
│
├── experiments/ # Alternative embedding specifications
│ └── 03_embeddings_multianchor_experiment.R
│
├── outputs/
│ ├── figures/ # All figures used in the paper
│ └── tables/ # Summary tables and model outputs
│
├── report/
│ ├── Politicization_or_Medicalization_.pdf
│ ├── Politicization_or_Medicalization_.tex
│ └── References.bib
│
├── project.Rproj
├── .gitignore
└── README.md
```

---

## Data

- **Raw data**:  
  `data/raw/tweets_congress.csv`  
  A dataset of COVID-19 vaccine–related tweets from U.S. members of Congress.  
  This file exceeds GitHub’s size limit and is therefore tracked using **Git Large File Storage (LFS)**.

- **Processed data** (`data/processed/`):  
  Includes cleaned tweet datasets, STM outputs, LLM-based frame labels, embedding representations, and final merged datasets used for analysis.

Key final datasets:
- `vax_master_dataset.rds`
- `vax_master_dataset.csv`

---

## Methods Overview

1. **Preprocessing & Filtering**  
   Tweets are filtered to vaccine-related content and enriched with metadata (party, vaccine brand, time).

2. **Structural Topic Modeling (STM)**  
   Topics are estimated to identify dominant themes in vaccine discourse.

3. **LLM-Based Classification**  
   Large language models are used to label topics and tweets by:
   - Frame (e.g., medical, political, rights-based, conspiratorial)
   - Stance (pro-vaccine, anti-vaccine, neutral)

4. **Embedding-Based Semantic Similarity**  
   Sentence embeddings are used to place tweets on a continuous spectrum from **medicalization** to **politicization**, using both data-driven and multi-anchor approaches.

5. **Statistical Analysis & Visualization**  
   The final scripts generate descriptive statistics, regression models, and all figures used in the paper.

---

## Replication

To reproduce the analysis:

1. Open the project using `project.Rproj`
2. Install required R packages (recommended via `renv` if prompted)
3. Run scripts in order:

scripts/01_setup_and_filter.R
scripts/llm_helper.R
scripts/02_stm_prep_and_model.R
scripts/03_embeddings.R
scripts/04_llm_topic_labels.R
scripts/05_llm_frame_classification.R
scripts/06_merge_master_dataset.R
scripts/07_descriptive_and_models.R

All figures and tables will be regenerated in the `outputs/` directory.

---

## Author

**Shuti Zhao**  
Georgetown University  
Data Science for Public Policy (DSPP)

---

## Notes

- Large files are managed using **Git LFS**
- This repository is intended for academic research and course submission purposes


