###############################################################################
# File: 02_stm_prep_and_model.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
#
# Description:
#   Prepares vaccine/misinformation tweets for Structural Topic Modeling (STM)
#   and fits an STM with party + time as prevalence covariates.
#
#   Steps:
#     (1) Build a quanteda corpus from vax_tweets.
#     (2) Tokenize & clean (stopwords, URLs, punctuation).
#     (3) Create a trimmed document-feature matrix (dfm).
#     (4) Convert to STM input and fit an STM.
#     (5) Export top words and representative tweets per topic for LLM labeling.
#
# Usage (from the console, in project directory):
#   source("01_setup_and_filter.R")    # creates vax_tweets with id, Party, etc.
#   source("02_stm_prep_and_model.R")  # runs STM and saves outputs
###############################################################################

library(dplyr)
library(stringr)
library(tibble)
library(lubridate)
library(quanteda)
library(stm)

# 0. Safety check
if (!exists("vax_tweets")) {
  stop("vax_tweets not found. Run source('01_setup_and_filter.R') first.")
}

# 1. Add covariates used in STM (party, time, brand)

vax_stm_base <- vax_tweets %>%
  mutate(
    datetime = parse_date_time(
      date,
      orders = "a b d H:M:S z Y",
      quiet = TRUE
    ),
    date_clean = as.Date(datetime),
    year       = year(date_clean),
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

# 2. Build corpus and tokens
# Use original tweet text (not lowercased) for readability in examples
corp <- corpus(
  vax_stm_base,
  text_field  = "text",
  docid_field = "id"
)

summary(corp, 3)

toks <- tokens(
  corp,
  remove_punct   = TRUE,
  remove_numbers = TRUE,
  remove_symbols = TRUE
)

# Remove URLs and common stopwords, lowercase, and stemming
toks <- tokens_tolower(toks)
toks <- tokens_remove(toks, pattern = quanteda::stopwords("en"))
toks <- tokens_remove(toks, pattern = "https?://\\S+")
toks <- tokens_wordstem(toks, language = "en")

# 3. Create dfm and trim
dfm_stm <- dfm(toks)

# Trim rare words
dfm_trimmed <- dfm_trim(
  dfm_stm,
  min_docfreq  = 5,    # was 10
  min_termfreq = 10    # was 20
)

# Drop documents that became empty after trimming
dfm_trimmed <- dfm_trimmed[ntoken(dfm_trimmed) > 0, ]

dfm_trimmed

# 4. Convert to STM format 
stm_input <- quanteda::convert(dfm_trimmed, to = "stm")

# Build metadata for the remaining docs
meta_df <- vax_stm_base %>%
  select(id, party_factor, date_clean, year, quarter, brand_main) %>%
  mutate(id = as.character(id))

# Match on doc IDs (docnames(dfm_trimmed) are character ids)
meta_sub <- meta_df[match(docnames(dfm_trimmed), meta_df$id), ]

# Attach to stm_input
stm_input$meta <- meta_sub

# Run searchK to help choose the number of topics
K_values <- c(8, 10, 12, 15, 18)

res <- searchK(
  documents = stm_input$documents,
  vocab     = stm_input$vocab,
  data      = stm_input$meta,
  K         = K_values
)

plot(res)


# 5. Fit STM
set.seed(6801) # our class number

K <- 12  # you can adjust / later run searchK if you want

stm_fit <- stm(
  documents  = stm_input$documents,
  vocab      = stm_input$vocab,
  data       = stm_input$meta,
  K          = K,
  prevalence = ~ party_factor + s(as.numeric(date_clean)),
  max.em.its = 75,
  init.type  = "Spectral",
  verbose    = TRUE,
  seed       = 6801
)

# Save STM model + input for later use (LLM topic labels, plots, etc.)
saveRDS(stm_fit,   file = "stm_vax_K12.rds")
saveRDS(stm_input, file = "stm_input_vax.rds")


# 6. Inspect topics: top words
topic_labels <- labelTopics(stm_fit, n = 10)
topic_labels

# Put top words into a tidy tibble for your paper & LLM labeling
top_words_tbl <- tibble(
  topic = 1:K,
  top_terms = apply(topic_labels$prob, 1, paste, collapse = ", ")
)

write.csv(top_words_tbl, "stm_topic_top_words.csv", row.names = FALSE)


# 7. Representative tweets for each topic
theta <- stm_fit$theta    # document-topic probabilities
doc_ids <- as.integer(rownames(theta)) %||% as.integer(names(dfm_trimmed))

# For each topic, pick top 5 tweets with highest theta
rep_list <- lapply(1:K, function(k) {
  ord <- order(theta[, k], decreasing = TRUE)
  top_idx <- ord[1:5]
  
  tibble(
    topic   = k,
    rank    = 1:5,
    doc_id  = doc_ids[top_idx],
    theta_k = theta[top_idx, k]
  )
})

rep_docs <- bind_rows(rep_list)

# Join back to original tweet text and some covariates
rep_docs_full <- rep_docs %>%
  left_join(
    vax_stm_base %>%
      select(id, author, Party, party_factor, brand_main, date_clean, text),
    by = c("doc_id" = "id")
  ) %>%
  arrange(topic, rank)

# This CSV is to feed to the LLM to name topics later
write.csv(rep_docs_full, "stm_topic_examples_for_llm.csv", row.names = FALSE)


