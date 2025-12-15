###############################################################################
# File: 05_llm_frame_classification.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
#
# Description:
#   Uses the OpenAI API (via query_open_ai()) to assign frame labels to each
#   vaccine-related tweet in vax_tweets.
#
#   For each tweet, the LLM returns:
#     - frame: one of "medical", "mandate_rights", "conspiracy_misinfo", "other"
#     - stance: "pro_vaccine", "anti_vaccine", or "mixed_or_unclear"
#     - reasoning: short sentence explaining the choice
#
#   The script:
#     (1) Checks that vax_tweets and query_open_ai() exist.
#     (2) Optionally downsamples tweets (to control API cost).
#     (3) Optionally resumes from an existing RDS (checkpointing).
#     (4) Loops over tweets and calls query_open_ai().
#     (5) Saves results to:
#           - vax_tweet_frames_llm.rds
#           - vax_tweet_frames_llm.csv
#
# Usage (from console, in project directory):
#   source("01_setup_and_filter.R")  # creates vax_tweets
#   source("llm_helpers.R")          # defines query_open_ai()
#   source("05b_llm_frame_classification.R")
#
###############################################################################

library(dplyr)
library(readr)
library(purrr)
library(tibble)
library(stringr)

# 0. Safety checks

if (!exists("vax_tweets")) {
  stop("vax_tweets not found. Run source('01_setup_and_filter.R') first.")
}

if (!exists("query_open_ai")) {
  stop("query_open_ai() not found. Run source('llm_helpers.R') first.")
}

# 1. Base data
# Keep key columns for context
frames_base <- vax_tweets %>%
  select(
    id,
    author,
    Party,
    State,
    date,
    text,
    text_lower
  )

# downsample to control cost while testing
# Set N_MAX_TWEETS = Inf to use ALL tweets
N_MAX_TWEETS <- Inf

set.seed(6801)
if (is.finite(N_MAX_TWEETS) && N_MAX_TWEETS < nrow(frames_base)) {
  message("Downsampling to ", N_MAX_TWEETS, " tweets for frame coding.")
  frames_base <- frames_base %>% sample_n(N_MAX_TWEETS)
}

# 2. Resume / checkpoint logic
out_path_rds <- "vax_tweet_frames_llm.rds"
out_path_csv <- "vax_tweet_frames_llm.csv"

llm_existing <- NULL
frames_todo  <- frames_base

if (file.exists(out_path_rds)) {
  message("Found existing ", out_path_rds, " — will try to resume and skip already-coded tweets.")
  llm_existing <- readRDS(out_path_rds)
  
  done_ids <- unique(llm_existing$id)
  frames_todo <- frames_base %>%
    filter(!id %in% done_ids)
  
  message("Already coded ", length(done_ids), " tweets; ",
          nrow(frames_todo), " remaining.")
} else {
  message("No existing frame RDS found. Starting from scratch on ",
          nrow(frames_todo), " tweets.")
}

if (nrow(frames_todo) == 0) {
  message("Nothing left to code. Exiting.")
  invisible(NULL)
  return(invisible(NULL))
}

# 3. Frame prompt
frame_prompt <- '
Coding how members of Congress frame vaccines in a tweet.

Return a JSON object with:
- "frame": one of ["medical", "mandate_rights", "conspiracy_misinfo", "other"]
- "stance": one of ["pro_vaccine", "anti_vaccine", "mixed_or_unclear"]
- "reasoning": one short sentence explaining why you chose this frame and stance.

Definitions:
- "medical" = safety, effectiveness, side effects, public health, doctors,
              clinical evidence, protecting others, hospital/ICU, variants.
- "mandate_rights" = requirements, mandates, passports, personal freedom,
                     liberty, government overreach, choice vs. coercion.
- "conspiracy_misinfo" = hoax, microchips, bioweapons, fake pandemic,
                         deep state, rigged data, 5G, or other conspiracies.
- "other" = anything that does not clearly match the above frames.

- If the tweet supports vaccination or encourages people to get vaccinated,
  use "pro_vaccine".
- If the tweet discourages vaccination or spreads doubts/misinformation,
  use "anti_vaccine".
- If the tweet mixes cues or you are unsure, use "mixed_or_unclear".

Return ONLY valid JSON with EXACTLY the keys:
"frame", "stance", and "reasoning".
'

#4. Loop over tweets
n_total <- nrow(frames_todo)
results_list <- vector("list", n_total)

for (i in seq_len(n_total)) {
  row_i <- frames_todo[i, ]
  
  message("Annotating tweet ", i, " of ", n_total,
          " (id = ", row_i$id, ")")
  
  tweet_text <- row_i$text
  
  # Call the LLM via helper
  out <- tryCatch(
    query_open_ai(
      prompt         = frame_prompt,
      post           = tweet_text,
      system_profile = "You are a political science RA coding frames in congressional tweets about vaccines."
    ),
    error = function(e) {
      message("Error on id ", row_i$id, ": ", e$message)
      tibble(
        frame     = NA_character_,
        stance    = NA_character_,
        reasoning = NA_character_
      )
    }
  )
  
  # Attach identifying info & metadata
  out$id     <- row_i$id
  out$author <- row_i$author
  out$Party  <- row_i$Party
  out$State  <- row_i$State
  out$date   <- row_i$date
  
  # Reorder columns a bit
  out <- out %>%
    select(
      id, author, Party, State, date,
      frame, stance, reasoning
    )
  
  results_list[[i]] <- out
  
  # Checkpoint every 100 tweets (merge with existing if present)
  if (i %% 100 == 0) {
    partial_new <- bind_rows(results_list[1:i])
    combined <- bind_rows(llm_existing, partial_new) %>%
      arrange(id) %>%
      distinct(id, .keep_all = TRUE)
    
    saveRDS(combined, out_path_rds)
    message("Checkpoint: saved ", nrow(combined),
            " coded tweets to ", out_path_rds)
  }
}

# quick peak
frames <- readRDS("vax_tweet_frames_llm.rds")
nrow(frames)

# Note on missing LLM frame/stance annotations:
# A substantial number of tweets (N = 1,795; ~9.6% of 18,652) remain unlabeled.
# This occurred because the OpenAI API hit the daily request-per-day (RPD)
# limit for gpt-4o-mini during the final batch of LLM calls.
# These missing cases are concentrated near the end of the corpus (IDs > ~1500)
# but appear otherwise random with respect to party and date.
# All analyses that use LLM frame/stance labels should therefore filter to
# tweets with non-missing values

# 5. Final save
new_results <- bind_rows(results_list)

combined_final <- bind_rows(llm_existing, new_results) %>%
  arrange(id) %>%
  distinct(id, .keep_all = TRUE)

saveRDS(combined_final, out_path_rds)
write_csv(combined_final, out_path_csv)

message("Saved tweet-level LLM frames to:")
message("  - ", out_path_rds)
message("  - ", out_path_csv)
