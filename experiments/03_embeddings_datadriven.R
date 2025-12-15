###############################################################################
# File: 03_embeddings_datadriven.R
# Project: PPOL 6801 – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
#
# Purpose:
#   Create *data-driven* concept vectors using LLM frame labels, and then
#   recompute tweet similarity scores in embedding space.
#
#   NO new API calls are made. We reuse:
#     - tweet_embeddings.rds (N x D matrix from 03_embeddings.R)
#     - vax_embed_sim.rds    (tweet-level metadata + id, from 03_embeddings.R)
#     - vax_master_dataset.rds (merged master with LLM frame/stance)
#
# Outputs:
#   - vax_embed_sim_datadriven.rds
#       (analysis_df rows + new similarity scores:
#        sim_hoax_data, sim_mandate_data, sim_safety_data, sim_politicization_data)
#
#   - plot_similarity_by_party_datadriven.png  (optional diagnostic plot)
###############################################################################

library(dplyr)
library(tibble)
library(stringr)
library(ggplot2)

cat("---- 04_embeddings_datadriven.R ----\n")

#-------------------------#
# 1. Load data & sanity checks
#-------------------------#

if (!file.exists("tweet_embeddings.rds")) {
  stop("tweet_embeddings.rds not found in working directory.")
}
if (!file.exists("vax_embed_sim.rds")) {
  stop("vax_embed_sim.rds not found in working directory.")
}
if (!file.exists("vax_master_dataset.rds")) {
  stop("vax_master_dataset.rds not found in working directory.")
}

tweet_emb_mat <- readRDS("tweet_embeddings.rds")   # N x D numeric matrix
vax_embed_sim <- readRDS("vax_embed_sim.rds")      # tibble/data.frame with id
master        <- readRDS("vax_master_dataset.rds") # full master

if (!is.matrix(tweet_emb_mat)) {
  stop("tweet_embeddings.rds should be a numeric matrix.")
}

cat("Embeddings matrix dim:", paste(dim(tweet_emb_mat), collapse = " x "), "\n")
cat("vax_embed_sim rows:", nrow(vax_embed_sim), "\n")
cat("master rows:", nrow(master), "\n")

# Ensure id types match (character)
vax_embed_sim <- vax_embed_sim %>% mutate(id = as.character(id))
master        <- master        %>% mutate(id = as.character(id))

# Recreate analysis_df (same logic as in 07_descriptive_and_models.R)
analysis_df <- master %>%
  filter(!is.na(frame),
         !is.na(stance)) %>%
  mutate(
    party_simple = case_when(
      Party %in% c("D", "Democrat", "DEM")       ~ "Democrat",
      Party %in% c("R", "Republican", "GOP")     ~ "Republican",
      TRUE                                       ~ "Other"
    ),
    party_simple = factor(party_simple,
                          levels = c("Democrat", "Republican", "Other"))
  )

cat("analysis_df rows (non-missing frame & stance):", nrow(analysis_df), "\n")

#-------------------------#
# 2. Align analysis_df with embedding rows via id
#-------------------------#

# vax_embed_sim rows are in the same order as tweet_emb_mat
embed_index <- vax_embed_sim %>%
  mutate(embed_row = row_number()) %>%
  select(id, embed_row)

analysis_df_aligned <- analysis_df %>%
  left_join(embed_index, by = "id") %>%
  filter(!is.na(embed_row)) %>%
  arrange(embed_row)

cat("analysis_df_aligned rows (with embeddings):",
    nrow(analysis_df_aligned), "\n")

aligned_emb <- tweet_emb_mat[analysis_df_aligned$embed_row, , drop = FALSE]

if (nrow(aligned_emb) != nrow(analysis_df_aligned)) {
  stop("Row mismatch after alignment – something went wrong in indexing.")
}

#-------------------------#
# 3. Build data-driven prototypes from frame labels
#-------------------------#

frame_lower <- tolower(analysis_df_aligned$frame)

mask_hoax <- str_detect(frame_lower, "conspiracy") |
  str_detect(frame_lower, "misinfo") |
  str_detect(frame_lower, "misinformation")

mask_mandate <- str_detect(frame_lower, "mandate") |
  str_detect(frame_lower, "rights")

mask_med <- str_detect(frame_lower, "medical") |
  str_detect(frame_lower, "safety")

cat("Hoax/conspiracy frame tweets:",   sum(mask_hoax), "\n")
cat("Mandate/rights frame tweets:",   sum(mask_mandate), "\n")
cat("Medical/safety frame tweets:",   sum(mask_med), "\n")

if (sum(mask_hoax) == 0L | sum(mask_mandate) == 0L | sum(mask_med) == 0L) {
  stop("One or more frame masks have zero tweets. Check frame naming in analysis_df.")
}

row_normalize <- function(mat) {
  norms <- sqrt(rowSums(mat^2))
  norms[norms == 0] <- 1
  mat / norms
}

aligned_emb_norm <- row_normalize(aligned_emb)

proto_hoax    <- colMeans(aligned_emb_norm[mask_hoax,    , drop = FALSE])
proto_mandate <- colMeans(aligned_emb_norm[mask_mandate, , drop = FALSE])
proto_med     <- colMeans(aligned_emb_norm[mask_med,     , drop = FALSE])

proto_mat <- rbind(
  hoax    = proto_hoax,
  mandate = proto_mandate,
  safety  = proto_med
)

proto_mat_norm <- row_normalize(proto_mat)

#-------------------------#
# 4. Compute cosine similarity scores
#-------------------------#

sim_mat <- aligned_emb_norm %*% t(proto_mat_norm)

colnames(sim_mat) <- c(
  "sim_hoax_data",
  "sim_mandate_data",
  "sim_safety_data"
)

sim_tbl <- as_tibble(sim_mat) %>%
  mutate(
    sim_politicization_data = sim_hoax_data + sim_mandate_data
  )

#-------------------------#
# 5. Attach to analysis_df_aligned and save
#-------------------------#

vax_embed_sim_datadriven <- analysis_df_aligned %>%
  bind_cols(sim_tbl)

cat("Final vax_embed_sim_datadriven rows:",
    nrow(vax_embed_sim_datadriven), "\n")

saveRDS(vax_embed_sim_datadriven, "vax_embed_sim_datadriven.rds")
cat("Saved vax_embed_sim_datadriven.rds\n")

#-------------------------#
# 6. Optional quick diagnostics: similar plot by party
#-------------------------#

if ("party_simple" %in% names(vax_embed_sim_datadriven)) {
  
  party_sim_dd <- vax_embed_sim_datadriven %>%
    filter(party_simple %in% c("Democrat", "Republican")) %>%
    group_by(party_simple) %>%
    summarise(
      sim_hoax_data           = mean(sim_hoax_data,           na.rm = TRUE),
      sim_mandate_data        = mean(sim_mandate_data,        na.rm = TRUE),
      sim_safety_data         = mean(sim_safety_data,         na.rm = TRUE),
      sim_politicization_data = mean(sim_politicization_data, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    )
  
  print(party_sim_dd)
  
  party_sim_long_dd <- party_sim_dd %>%
    tidyr::pivot_longer(
      cols = c(
        sim_hoax_data,
        sim_mandate_data,
        sim_safety_data,
        sim_politicization_data
      ),
      names_to  = "concept",
      values_to = "mean_sim"
    ) %>%
    mutate(
      concept = dplyr::recode(
        concept,
        "sim_hoax_data"           = "Hoax/Misinformation (data-driven)",
        "sim_mandate_data"        = "Mandate/Rights (data-driven)",
        "sim_safety_data"         = "Medical/Safety (data-driven)",
        "sim_politicization_data" = "Politicization = Hoax + Mandate"
      )
    )
  
  p_party_sim_datadriven <- ggplot(
    party_sim_long_dd,
    aes(x = concept, y = mean_sim, fill = party_simple)
  ) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(
      values = c(
        "Democrat"   = "#1f78b4",
        "Republican" = "#e31a1c"
      )
    ) +
    labs(
      x = "Concept",
      y = "Average cosine similarity",
      title = "Data-driven semantic proximity to key narratives, by party",
      fill = "Party"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 20, hjust = 1)
    )
  
  # Show in Plots pane
  p_party_sim_datadriven
  
  # Save to file in project root (change path if you want it in outputs/figures)
  ggsave(
    filename = "plot_similarity_by_party_datadriven.png",
    plot     = p_party_sim_datadriven,
    width    = 7,
    height   = 4.5,
    dpi      = 300
  )
  
  cat("Saved plot_similarity_by_party_datadriven.png\n")
}

cat("---- 04_embeddings_datadriven.R completed ----\n")
