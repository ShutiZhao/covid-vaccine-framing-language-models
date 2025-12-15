###############################################################################
# File: 03_embeddings_multianchor.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
#
# Description (v2):
#   Recompute similarity scores using multiple anchor phrases per frame:
#     - Hoax/misinformation
#     - Mandate/rights (politicization)
#     - Medical/safety
#   Then create a combined "sim_politicization" = sim_hoax + sim_mandate.
#
#   IMPORTANT: This script DOES NOT overwrite your original vax_embed_sim.rds.
#   It saves a new file: vax_embed_sim_multianchor.rds
###############################################################################

library(httr)
library(jsonlite)
library(dplyr)
library(tibble)
library(stringr)
library(purrr)
library(tidyr)
library(lubridate)
library(ggplot2)

# 0. Make sure vax_tweets exists (from 01_setup_and_filter.R)
if (!exists("vax_tweets")) {
  stop("vax_tweets not found. Run source('01_setup_and_filter.R') first.")
}

# 1. Check API key
if (!nzchar(Sys.getenv("OPENAI_API_KEY"))) {
  stop("OPENAI_API_KEY is not set. Add it to .Renviron and restart R.")
}

# 2. Helper: get embeddings in mini-batches ----------------------------------
get_embeddings_batch <- function(texts,
                                 model = "text-embedding-3-small") {
  
  api_key <- Sys.getenv("OPENAI_API_KEY")
  
  if (!is.character(texts)) {
    stop("texts must be a character vector.")
  }
  
  payload <- list(
    model = model,
    input = as.list(texts)
  )
  
  res <- httr::POST(
    url = "https://api.openai.com/v1/embeddings",
    httr::add_headers(
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ),
    body = jsonlite::toJSON(payload, auto_unbox = TRUE),
    encode = "json"
  )
  
  if (httr::http_error(res)) {
    cat("Status code:", httr::status_code(res), "\n")
    cat("Raw content:\n", httr::content(res, "text", encoding = "UTF-8"), "\n")
    stop("Embedding request failed.")
  }
  
  raw <- httr::content(res, "parsed", encoding = "UTF-8")
  emb_list <- raw$data
  
  emb_mat <- do.call(
    rbind,
    lapply(emb_list, function(x) as.numeric(x$embedding))
  )
  emb_mat
}

# 3. Prepare tweets for embedding --------------------------------------------
vax_embed_base <- vax_tweets %>%
  mutate(
    datetime = parse_date_time(
      date,
      orders = "a b d H:M:S z Y",
      quiet = TRUE
    ),
    date_clean = as.Date(datetime),
    quarter    = floor_date(date_clean, unit = "quarter"),
    party_factor = case_when(
      Party == "D" ~ "Democrat",
      Party == "R" ~ "Republican",
      TRUE         ~ "Other"
    ),
    brand = case_when(
      str_detect(text_lower, "pfizer") ~ "Pfizer",
      str_detect(text_lower, "moderna") ~ "Moderna",
      str_detect(text_lower, "astrazeneca|astra zeneca|az") ~ "AstraZeneca",
      str_detect(text_lower, "johnson & johnson|j&j") ~ "JandJ",
      TRUE ~ "Other"
    ),
    brand_main = if_else(
      brand %in% c("Pfizer", "Moderna", "AstraZeneca"),
      brand,
      "Other"
    )
  )

n_total <- nrow(vax_embed_base)
n_embed <- n_total

set.seed(6801)
vax_for_embed <- vax_embed_base %>%
  slice_sample(n = n_embed)

texts_to_embed <- vax_for_embed$text
batch_size     <- 100

batches <- split(
  texts_to_embed,
  ceiling(seq_along(texts_to_embed) / batch_size)
)

embedding_mats <- vector("list", length(batches))
for (i in seq_along(batches)) {
  cat("Embedding batch", i, "of", length(batches), "...\n")
  Sys.sleep(0.4)
  embedding_mats[[i]] <- get_embeddings_batch(batches[[i]])
}

tweet_emb_mat <- do.call(rbind, embedding_mats)
dim(tweet_emb_mat)  # should be n_embed x embedding_dim

# Save as a *new* file so original is untouched
saveRDS(tweet_emb_mat, file = "tweet_embeddings_multianchor.rds")

# 4. Multiple anchor phrases per frame ---------------------------------------

concepts_tbl <- tibble::tibble(
  frame   = c(
    rep("hoax",    3),
    rep("mandate", 3),
    rep("safety",  3)
  ),
  phrase = c(
    # hoax / misinformation anchors
    "covid vaccine is a hoax and a scam",
    "lies about vaccines and rigged health data",
    "fake pandemic microchip conspiracy around vaccines",
    
    # mandate / rights anchors
    "vaccine mandates violate medical freedom and personal liberty",
    "government should not force anyone to take a vaccine",
    "no vaccine passports my body my choice",
    
    # medical / safety anchors
    "covid vaccine safety and effectiveness protect public health",
    "clinical trial data on vaccine efficacy and side effects",
    "vaccination reduces hospitalization and death from covid"
  )
)

concept_emb_mat_all <- get_embeddings_batch(concepts_tbl$phrase)
dim(concept_emb_mat_all)  # 9 x embedding_dim

# Average anchors within each frame → 3 prototype vectors
frame_list <- split(
  as.data.frame(concept_emb_mat_all),
  concepts_tbl$frame
)

prototype_mat <- do.call(
  rbind,
  lapply(frame_list, function(df) {
    colMeans(as.matrix(df))
  })
)

# Row order/rownames: hoax, mandate, safety
prototype_mat <- as.matrix(prototype_mat)
prototype_mat

# 5. Cosine similarity: tweets vs. frame prototypes --------------------------

row_normalize <- function(mat) {
  norms <- sqrt(rowSums(mat^2))
  norms[norms == 0] <- 1
  mat / norms
}

tweet_mat_norm   <- row_normalize(tweet_emb_mat)
concept_mat_norm <- row_normalize(prototype_mat)

# n_tweets x 3 (hoax, mandate, safety)
sim_mat <- tweet_mat_norm %*% t(concept_mat_norm)
dim(sim_mat)

colnames(sim_mat) <- c("sim_hoax", "sim_mandate", "sim_safety")
sim_tbl <- as_tibble(sim_mat)

# Attach similarity scores back to tweets + create politicization
vax_embed_sim_multianchor <- vax_for_embed %>%
  bind_cols(sim_tbl) %>%
  mutate(
    sim_politicization = sim_hoax + sim_mandate
  )

# Save new similarity object
saveRDS(vax_embed_sim_multianchor, "vax_embed_sim_multianchor.rds")
readr::write_csv(vax_embed_sim_multianchor, "vax_embed_sim_multianchor.csv")

# 6. Summaries: similarity by party and by brand -----------------------------

# 6.1 Average similarity by party
party_sim <- vax_embed_sim_multianchor %>%
  group_by(party_factor) %>%
  summarise(
    sim_politicization = mean(sim_politicization, na.rm = TRUE),
    sim_hoax           = mean(sim_hoax,           na.rm = TRUE),
    sim_safety         = mean(sim_safety,         na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

party_sim_long <- party_sim %>%
  pivot_longer(
    cols = c(sim_politicization, sim_hoax, sim_safety),
    names_to = "concept",
    values_to = "mean_sim"
  ) %>%
  mutate(
    concept = recode(
      concept,
      "sim_politicization" = "Politicization (Hoax + Mandate)",
      "sim_hoax"           = "Hoax/Misinformation",
      "sim_safety"         = "Medical/Safety"
    )
  )

ggplot(party_sim_long,
       aes(x = concept, y = mean_sim, fill = party_factor)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(
    values = c(
      "Democrat"   = "#1f78b4",  # blue
      "Republican" = "#e31a1c",  # red
      "Other"      = "gray60"
    )
  ) +
  labs(
    x = "Concept",
    y = "Average cosine similarity",
    title = "Semantic proximity to key narratives, by party (multi-anchor)",
    fill = "Party"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title  = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

ggsave("plot_similarity_by_party_multianchor.png",
       width = 7, height = 4.5, dpi = 300)

# 6.2 Average similarity by vaccine brand
brand_sim <- vax_embed_sim_multianchor %>%
  group_by(brand_main) %>%
  summarise(
    sim_politicization = mean(sim_politicization, na.rm = TRUE),
    sim_hoax           = mean(sim_hoax,           na.rm = TRUE),
    sim_safety         = mean(sim_safety,         na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

brand_sim_long <- brand_sim %>%
  pivot_longer(
    cols = c(sim_politicization, sim_hoax, sim_safety),
    names_to = "concept",
    values_to = "mean_sim"
  ) %>%
  mutate(
    concept = recode(
      concept,
      "sim_politicization" = "Politicization (Hoax + Mandate)",
      "sim_hoax"           = "Hoax/Misinformation",
      "sim_safety"         = "Medical/Safety"
    )
  )

ggplot(brand_sim_long,
       aes(x = concept, y = mean_sim, fill = brand_main)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  labs(
    x = "Concept",
    y = "Average cosine similarity",
    title = "Framing of vaccine brands in embedding space (multi-anchor)",
    fill = "Vaccine brand"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title  = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

ggsave("plot_similarity_by_brand_multianchor.png",
       width = 7, height = 4.5, dpi = 300)
