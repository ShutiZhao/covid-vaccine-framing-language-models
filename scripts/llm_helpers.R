###############################################################################
# File: llm_helpers.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
# Description:
#   Defines helper functions for interacting with the OpenAI API.
#   Includes:
#     - query_open_ai(): generic function for LLM classification and topic labeling
# Usage:
#   Source this file after loading data:
#       source("llm_helpers.R")
###############################################################################

# query_open_ai() function
query_open_ai <- function(prompt, post,
                          system_profile = "You are a helpful assistant that annotates congressional tweets about vaccines.") {
  
  # Load the API key from environment
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (!nzchar(api_key)) {
    stop("OPENAI_API_KEY is not set. Check your .Renviron file.")
  }
  
  # Construct user prompt
  full_user_prompt <- paste0(
    prompt,
    "\n\nTweet:\n",
    post,
    "\n\nReturn ONLY a valid JSON object. Do not use code fences."
  )
  
  # Prepare request payload
  payload <- list(
    model = "gpt-4o-mini",
    temperature = 0,
    messages = list(
      list(role = "system", content = system_profile),
      list(role = "user",   content = full_user_prompt)
    ),
    max_tokens = 400
  )
  
  # Send POST request via OpenAI API
  response <- httr::POST(
    url = "https://api.openai.com/v1/chat/completions",
    httr::add_headers(
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ),
    body = jsonlite::toJSON(payload, auto_unbox = TRUE),
    encode = "json"
  )
  
  # Check for errors
  if (httr::http_error(response)) {
    cat("Status code:", httr::status_code(response), "\n")
    cat("Raw content:\n", httr::content(response, "text", encoding = "UTF-8"), "\n")
    stop("API request failed.")
  }
  
  # Extract text content
  raw_text <- httr::content(response, "text", encoding = "UTF-8")
  raw <- jsonlite::fromJSON(raw_text, simplifyVector = FALSE)
  
  msg <- raw$choices[[1]]$message
  
  # Handle cases where API returns list-of-text fragments
  if (is.list(msg$content)) {
    parts <- vapply(msg$content, function(x) x[["text"]], FUN.VALUE = character(1))
    answer_text <- paste(parts, collapse = "")
  } else {
    answer_text <- msg$content
  }
  
  # Clean JSON (remove backticks)
  answer_text <- gsub("```json", "", answer_text)
  answer_text <- gsub("```", "", answer_text)
  answer_text <- trimws(answer_text)
  
  # Convert to tibble
  out <- jsonlite::fromJSON(answer_text)
  
  tibble::as_tibble(out)
}
