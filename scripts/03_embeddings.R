###############################################################################
# File: 03_embeddings.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
#
# Description:
#   This script uses OpenAI sentence embeddings to locate congressional
#   vaccine-related tweets in a semantic space and quantify their proximity
#   to key framing concepts (freedom, mandate, safety, hoax).
#
#   Steps:
#     (1) Load filtered vaccine/misinformation tweets (vax_tweets).
#     (2) Define a helper to call the OpenAI embeddings endpoint in batches.
#     (3) Embed a subset or all tweets.
#     (4) Embed prototype phrases representing framing concepts.
#     (5) Compute cosine similarity between each tweet and each concept.
#     (6) Summarize similarities by party, brand, and over time.
#
# Usage:
#   Run after Script 01 (data) and with API key set in .Renviron:
#       source("01_setup_and_filter.R")   # creates vax_tweets
#       source("03_embeddings.R")
#
# Outputs:
#   - vax_embed_sim: data.frame with tweet-level embeddings + similarity scores
#   - Plots of similarity by party and by brand
#   - Saved objects: tweet_embeddings.rds, concept_embeddings.rds, vax_embed_sim.rds
###############################################################################


# 0. Load required libs
library(httr)
library(jsonlite)
library(dplyr)
library(tibble)
library(stringr)
library(purrr)
library(tidyr)
library(lubridate)
library(ggplot2)

# Make sure vax_tweets exists (from 01_setup_and_filter.R)
if (!exists("vax_tweets")) {
  stop("vax_tweets not found. Did you run source('01_setup_and_filter.R')?")
}

# Check API key
if (!nzchar(Sys.getenv("OPENAI_API_KEY"))) {
  stop("OPENAI_API_KEY is not set. Add it to .Renviron and restart R.")
}


# 1. Helper: get embeddings in mini-batches 

get_embeddings_batch <- function(texts,
                                 model = "text-embedding-3-small") {

  api_key <- Sys.getenv("OPENAI_API_KEY")

  if (!is.character(texts)) {
    stop("texts must be a character vector.")
  }

  payload <- list(
    model = model,
    input = as.list(texts)  # list of strings
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

  # IMPORTANT: no simplifyVector here
  raw <- httr::content(res, "parsed", encoding = "UTF-8")
  # raw$data should be a list, each with an $embedding element
  emb_list <- raw$data

  emb_mat <- do.call(
    rbind,
    lapply(emb_list, function(x) {
      # make sure it's numeric
      as.numeric(x$embedding)
    })
  )
  return(emb_mat)
}


# 2. Prepare tweets for embedding (covariates) 
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

# see the distribution of party and brand
table(vax_embed_base$party_factor, useNA = "ifany")
table(vax_embed_base$brand_main, useNA = "ifany")



# 3. Embed tweets (can limit to a subset)
n_total <- nrow(vax_embed_base)
n_embed <- n_total  

set.seed(6801)
vax_for_embed <- vax_embed_base %>%
  slice_sample(n = n_embed)

nrow(vax_for_embed)

# Batch the texts to respect token limits and be gentle on the API
texts_to_embed <- vax_for_embed$text
batch_size     <- 100

batches <- split(
  texts_to_embed,
  ceiling(seq_along(texts_to_embed) / batch_size)
)

embedding_mats <- vector("list", length(batches))

for (i in seq_along(batches)) {
  cat("Embedding batch", i, "of", length(batches), "...\n")
  # slight delay to avoid hammering API
  Sys.sleep(0.4)
  
  emb_mat_i <- get_embeddings_batch(batches[[i]])
  embedding_mats[[i]] <- emb_mat_i
}

tweet_emb_mat <- do.call(rbind, embedding_mats)
dim(tweet_emb_mat)  # should be n_embed x dimension

# Save raw tweet embeddings 
saveRDS(tweet_emb_mat, file = "tweet_embeddings.rds")


# 4. Create prototype embeddings for framing concepts   
concepts_tbl <- tibble::tibble(
  concept = c("freedom", "mandate", "safety", "hoax"),
  text = c(
    "freedom and individual liberty about vaccines",
    "government vaccine mandate and forced vaccination",
    "vaccine safety and effectiveness and protecting public health",
    "vaccine hoax fake pandemic microchip conspiracy"
  )
)

concept_emb_mat <- get_embeddings_batch(concepts_tbl$text)
dim(concept_emb_mat)  # 4 x dim

# Save prototype embeddings
saveRDS(concept_emb_mat, file = "concept_embeddings.rds")

concepts_tbl

# 5. Cosine similarity: tweets vs concepts
# Helper to L2-normalize rows of a matrix
row_normalize <- function(mat) {
  norms <- sqrt(rowSums(mat^2))
  # avoid division by zero
  norms[norms == 0] <- 1
  mat / norms
}

tweet_mat_norm   <- row_normalize(tweet_emb_mat)
concept_mat_norm <- row_normalize(concept_emb_mat)

# Cosine similarity matrix: n_tweets x n_concepts
sim_mat <- tweet_mat_norm %*% t(concept_mat_norm)
dim(sim_mat)  # n_embed x 4

colnames(sim_mat) <- paste0("sim_", concepts_tbl$concept)

sim_tbl <- as_tibble(sim_mat)

# Attach similarity scores back to vax_for_embed
vax_embed_sim <- vax_for_embed %>%
  bind_cols(sim_tbl)

# Attach similarity scores back to vax_for_embed
vax_embed_sim <- vax_for_embed %>%
  bind_cols(sim_tbl) %>%
  # combine freedom + mandate into a single politicization score
  mutate(sim_politicization = (sim_freedom + sim_mandate) / 2)

# Save combined data
saveRDS(vax_embed_sim, file = "vax_embed_sim.rds")

# 6. Summaries: similarity by party and by brand
# 6.1 Average similarity by party
party_sim_long <- party_sim %>%
  pivot_longer(
    cols = c(sim_politicization, sim_hoax, sim_safety),
    names_to = "concept",
    values_to = "mean_sim"
  ) %>%
  mutate(
    concept = recode(
      concept,
      "sim_politicization" = "Politicization (Freedom/Mandate)",
      "sim_hoax"           = "Hoax/Misinformation",
      "sim_safety"         = "Medical/Safety"
    )
  )

ggplot(party_sim_long,
       aes(x = concept, y = mean_sim, fill = party_factor)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(
    values = c(
      "Democrat"   = "#1f78b4",  # BLUE
      "Republican" = "#e31a1c",  # RED
      "Other"      = "gray60"
    )
  ) +
  labs(
    x = "Concept",
    y = "Average cosine similarity",
    title = "Semantic Proximity to Key Narratives, by Party",
    fill = "Party"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

ggsave("plot_similarity_by_party.png", width = 7, height = 4.5, dpi = 300)

# 6.2 Average similarity by vaccine brand
brand_sim <- vax_embed_sim %>%
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
      "sim_politicization" = "Politicization (Freedom/Mandate)",
      "sim_hoax"           = "Hoax/Misinformation",
      "sim_safety"         = "Medical/Safety"
    )
  )

ggplot(brand_sim_long, aes(x = concept, y = mean_sim, fill = brand_main)) +
  geom_col(position = "dodge") +
  labs(
    x = "Concept",
    y = "Average cosine similarity",
    title = "Framing of vaccine brands in embedding space"
  ) +
  theme_minimal()

ggsave("plot_similarity_by_brand.png", width = 7, height = 4.5)

# 6.3 evolution over time (semantic drift)
vax_embed_sim <- vax_embed_sim %>%
  mutate(quarter = floor_date(date_clean, unit = "quarter"))

time_sim <- vax_embed_sim %>%
  group_by(quarter) %>%
  summarise(
    across(starts_with("sim_"), mean, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

time_sim_long <- time_sim %>%
  pivot_longer(
    cols = starts_with("sim_"),
    names_to = "concept",
    values_to = "mean_sim"
  )

ggplot(time_sim_long, aes(x = quarter, y = mean_sim, color = concept)) +
  geom_line() +
  geom_point() +
  labs(
    x = "Quarter",
    y = "Average cosine similarity",
    title = "Semantic framing of vaccine tweets over time"
  ) +
  theme_minimal()

ggsave("plot_similarity_over_time.png", width = 7, height = 4.5)
