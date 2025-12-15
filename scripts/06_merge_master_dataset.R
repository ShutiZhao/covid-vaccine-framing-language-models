###############################################################################
# File: 06_merge_master_dataset.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
#
# Description:
#   Builds a single tweet-level *master dataset* by merging:
#     (1) Raw tweet data (vax_tweets.rds)
#     (2) STM document-level metadata (stm_input_vax.rds$meta)
#     (3) LLM frame + stance labels (from 05_llm_frame_classification.R)
#     (4) Embedding-based similarity scores (vax_embed_sim.rds), if available
#
#   IMPORTANT:
#   - I do NOT merge the huge raw embedding vectors (tweet_embeddings.rds).
#     Instead I only merge the small, interpretable similarity scores.
#
# Output:
#   - vax_master_dataset.rds
#   - vax_master_dataset.csv
#
# Usage (from console, in project directory):
#   source("02_stm_prep_and_model.R")       # creates STM objects
#   source("03_embeddings.R")               # creates vax_embed_sim.rds (optional)
#   source("05_llm_frame_classification.R") # creates vax_tweet_frames_llm.rds
#   source("06_merge_master_dataset.R")
#
###############################################################################

# 0. Setup
library(tidyverse)
library(lubridate)

# 1. File paths
# Core tweet + STM objects
path_tweets    <- "vax_tweets.rds"      # full raw tweet-level data
path_stm_input <- "stm_input_vax.rds"   # STM input object (docs, meta, vocab)

# LLM frame / stance annotations
path_frames_llm <- "vax_tweet_frames_llm.rds"

# Embedding-based similarity scores (small + interpretable)
path_embed_sim  <- "vax_embed_sim.rds"

# Output files
out_master_rds  <- "vax_master_dataset.rds"
out_master_csv  <- "vax_master_dataset.csv"

# 2. Load core data
message("Loading base tweet data and STM meta...")

vax_tweets <- readRDS(path_tweets)

stm_input  <- readRDS(path_stm_input)
meta_df    <- stm_input$meta    # document-level metadata used in STM

frames_llm <- readRDS(path_frames_llm)

# 3. Load embedding similarity scores (not raw embeddings)
use_embeddings <- file.exists(path_embed_sim)

if (use_embeddings) {
  message("Loading embedding similarity scores...")
  embed_sim <- readRDS(path_embed_sim)   # id + sim_* columns
} else {
  message("No embedding similarity file found")
}

# 4. Harmonize ID types and merge into master
# Make sure 'id' is character everywhere before joining
vax_tweets <- vax_tweets %>% mutate(id = as.character(id))
meta_df    <- meta_df    %>% mutate(id = as.character(id))
frames_llm <- frames_llm %>% mutate(id = as.character(id))

if (use_embeddings) {
  embed_sim <- embed_sim %>% mutate(id = as.character(id))
}

# Build master tweet-level dataset
master <- vax_tweets %>%
  left_join(meta_df,    by = "id") %>%   # STM document-level meta
  left_join(frames_llm, by = "id")       # LLM frame + stance

# Add embedding similarity scores if available
if (use_embeddings) {
  master <- master %>%
    left_join(embed_sim, by = "id")
}

# 5. Basic checks
message("  Number of rows: ", nrow(master))
message("  Number of columns: ", ncol(master))

# Coverage checks for LLM frame and stance labels
n_frame <- sum(!is.na(master$frame))
n_stance <- sum(!is.na(master$stance))
prop_frame <- mean(!is.na(master$frame))

message("Frames coded: ", n_frame)
message("Stance coded: ", n_stance)
message("Total tweets: ", nrow(master))
message("Share with frame labels: ", round(prop_frame, 3))

# List similarity columns (if any)
sim_cols <- grep("^sim_", names(master), value = TRUE)
message("Similarity columns: ", paste(sim_cols, collapse = ", "))

# Note on missing LLM frame/stance annotations:
# In the final master dataset, 1,795 tweets (~9.6% of 18,652) do not have
# LLM frame/stance labels. This is due to OpenAI API request-per-day (RPD)
# limits and partial reruns of the frame-classification script.
# All analyses that use frame or stance should therefore filter to tweets with
# non-missing labels (e.g., filter(!is.na(frame))).

# 6. Save master dataset
saveRDS(master, out_master_rds)
write_csv(master, out_master_csv)

message("Saved master dataset to:")
message("  - ", out_master_rds)
message("  - ", out_master_csv)


