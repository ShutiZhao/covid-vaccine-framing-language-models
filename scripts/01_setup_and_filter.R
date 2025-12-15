###############################################################################
# File: 01_setup_and_filter.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
# Description:
#   - Loads congressional tweet data
#   - Filters dataset to vaccine/misinformation-related tweets
#   - Sets up API key environment and required packages
#   - Produces vax_tweets object for downstream STM + LLM + embedding analysis
# Usage:
#   Source this file in analysis scripts:
#       source("01_setup_and_filter.R")
###############################################################################

# 01_setup_and_filter.R
library(httr)
library(jsonlite)
library(dplyr)
library(readr)
library(stringr)
library(purrr)
library(tidyr)
library(lubridate)

# 1. API setup 
# Store the API key in the .Renviron file
usethis::edit_r_environ()
# Get the API key from the .Renviron file
api_key <- Sys.getenv("OPENAI_API_KEY")

# 2. Load tweets_congress data 
tweets <- read_csv("/Users/zhaoshuti/Desktop/TAT/project/tweets_congress.csv")
glimpse(tweets)

# 3. Filter to vaccine/misinformation tweets

vax_keywords <- c(
  "vaccine", "vaccines", "vax", "vaxx", 
  "pfizer", "moderna", "astrazeneca", 
  "johnson & johnson", "j&j", "booster", "mrna"
)

misinfo_keywords <- c(
  "hoax", "fake", "bioweapon", "microchip", 
  "plandemic", "cover[- ]?up", "conspiracy"
)

vax_pattern     <- str_c(vax_keywords, collapse = "|")
misinfo_pattern <- str_c(misinfo_keywords, collapse = "|")

vax_tweets <- tweets %>%
  mutate(text_lower = str_to_lower(text)) %>%
  filter(
    str_detect(text_lower, vax_pattern) |
      str_detect(text_lower, misinfo_pattern)
  ) %>%
  mutate(id = row_number())

nrow(vax_tweets)

message("Analysis sample size (non-missing frame & stance): ", nrow(analysis_df))


saveRDS(vax_tweets, "vax_tweets.rds")

