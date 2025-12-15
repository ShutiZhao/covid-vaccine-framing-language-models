###############################################################################
# File: 05_llm_topic_labels.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
#
# Description:
#   Uses an LLM (via query_open_ai()) to assign human-readable labels to STM
#   topics. For each topic, the model sees:
#     - the top FREX words
#     - a few example tweets
#   and returns:
#     - topic_id
#     - short_label (2–5 words)
#     - description (1–2 sentences)
#     - category (coarse type of topic)
#     - keywords (5–10 phrases, comma-separated)
#
# Inputs (created in 02_stm_prep_and_model.R):
#   - "stm_topic_top_words.csv"
#   - "stm_topic_examples_for_llm.csv"
#
# Outputs:
#   - "stm_topic_labels_llm.csv"
#   - "stm_topic_labels_llm.rds"
#
# Usage (from Console, in project directory):
#   source("llm_helpers.R")          # defines query_open_ai()
#   source("05_llm_topic_labels.R")  # creates topic_labels object + files
###############################################################################

library(dplyr)
library(readr)
library(stringr)
library(tibble)
library(purrr)

# Safety check
if (!exists("query_open_ai")) {
  stop("query_open_ai() not found. Run source('llm_helpers.R') first.")
}

# 1. Load STM outputs needed for labeling
# Top FREX words per topic (12 rows x 2 cols: topic, top_words)
top_words_tbl <- read_csv("stm_topic_top_words.csv",
                          show_col_types = FALSE)

# Representative example tweets per topic
rep_docs_full <- read_csv("stm_topic_examples_for_llm.csv",
                          show_col_types = FALSE)

# make sure topics are integers
top_words_tbl <- top_words_tbl %>%
  mutate(topic = as.integer(topic))

rep_docs_full <- rep_docs_full %>%
  mutate(topic = as.integer(topic))

# 2. Helper to build the text we send to the LLM
build_topic_context <- function(k, n_examples = 3) {
  # Row(s) for this topic
  tw_row <- top_words_tbl %>%
    filter(topic == k)
  
  if (nrow(tw_row) == 0) {
    frex_words <- "(no FREX words found)"
  } else {
    # Use whichever column is NOT "topic" as the word string column
    word_col <- setdiff(names(top_words_tbl), "topic")[1]
    frex_words <- tw_row[[word_col]]
  }
  
  # Get up to n_examples tweets for this topic
  ex_dat <- rep_docs_full %>%
    filter(topic == k) %>%
    arrange(rank) %>%
    slice_head(n = n_examples)
  
  example_lines <- if (nrow(ex_dat) == 0) {
    "No example tweets available."
  } else {
    paste0(
      "Example tweets:\n",
      paste0(
        "Tweet ", seq_len(nrow(ex_dat)), ": ",
        ex_dat$text,
        collapse = "\n"
      )
    )
  }
  
  paste0(
    "Topic number: ", k, "\n",
    "Top FREX words: ", frex_words, "\n\n",
    example_lines
  )
}

# 3. LLM prompt describing the JSON we want
topic_prompt <- '
You are helping a political text analysis project that used Structural Topic Modeling
on US congressional tweets about COVID-19 vaccines.

I will give you:
- a topic number
- the list of top FREX words for that topic
- several example tweets from that topic.

Based on this information, return ONLY a valid JSON object with the following fields:

- "topic_id": integer, the topic number I gave you.
- "short_label": a very concise 2-5 word label (e.g., "Vaccine safety and efficacy").
- "description": 1-2 sentences describing what this topic is about in plain language.
- "category": one of the following strings:
    "public_health"
    "vaccine_distribution"
    "mandates_politics"
    "misinformation_conspiracy"
    "other"
- "keywords": a comma-separated string with 5-10 important words or short phrases.

Do NOT include any markdown. Return ONLY the JSON object with those keys.
'

# 4. Loop over topics and call the LLM
topic_ids <- sort(unique(top_words_tbl$topic))

topic_results <- vector("list", length(topic_ids))

for (i in seq_along(topic_ids)) {
  k <- topic_ids[i]
  message("Labeling topic ", k, " of ", length(topic_ids), " ...")
  
  post_text <- build_topic_context(k)
  
  # Call the LLM via the generic helper
  out <- tryCatch(
    query_open_ai(topic_prompt, post_text,
                  system_profile = "You are a political science methods TA helping label STM topics."),
    error = function(e) {
      message("Error on topic ", k, ": ", e$message)
      tibble(
        topic_id    = k,
        short_label = NA_character_,
        description = NA_character_,
        category    = NA_character_,
        keywords    = NA_character_
      )
    }
  )
  
  # Ensure topic_id is filled (in case the model forgot)
  if (!"topic_id" %in% names(out)) {
    out$topic_id <- k
  } else if (is.na(out$topic_id[1])) {
    out$topic_id <- k
  }
  
  topic_results[[i]] <- out
}

topic_labels <- bind_rows(topic_results) %>%
  mutate(topic_id = as.integer(topic_id)) %>%
  arrange(topic_id)

# Inspect the LLM labels
topic_labels

# 5. Save outputs
write_csv(topic_labels, "stm_topic_labels_llm.csv")
saveRDS(topic_labels, "stm_topic_labels_llm.rds")

message("Saved LLM topic labels to stm_topic_labels_llm.csv and stm_topic_labels_llm.rds.")
