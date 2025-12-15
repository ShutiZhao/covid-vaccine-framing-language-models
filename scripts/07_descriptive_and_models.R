###############################################################################
# File: 07_descriptive_and_models.R
# Project: PPOL 6801 Final Project – Vaccine Misinformation & Embedding Analysis
# Author: Shuti Zhao
#
# Description:
#   Uses the merged master tweet-level dataset to:
#     (1) Filter to tweets with non-missing LLM frame / stance labels
#     (2) Produce basic descriptive statistics (frame & stance distributions)
#     (3) Compare frames / stance by party
#     (4) Compare frames / stance by vaccine brand
#     (5) Explore politicization vs medicalization by brand using similarity scores
#     (6) Plot time trends during the COVID period (2020–2022)
#     (7) Run a simple logistic model using embeddings and party
#
# Inputs:
#   - vax_master_dataset.rds (created by 06_merge_master_dataset.R)
#
# Outputs (optional – for tables/figures in the paper):
#   - CSV summary tables in the "outputs" folder (if it exists)
#
###############################################################################

library(tidyverse)
library(lubridate)
library(ggrepel)
library(ggplot2)
library(broom)

# 1. Load master dataset
path_master <- "vax_master_dataset.rds"

message("Loading master dataset...")
master <- readRDS(path_master)

message("Master rows: ", nrow(master))
message("Master columns: ", ncol(master))

# 2. Restrict to tweets with LLM labels + clean party/brand/year
analysis_df <- master %>%
  filter(!is.na(frame),
         !is.na(stance))

message("Analysis sample size (non-missing frame & stance): ", nrow(analysis_df))

# Simpler party factor
analysis_df <- analysis_df %>%
  mutate(
    party_simple = case_when(
      Party %in% c("D", "Democrat", "DEM") ~ "Democrat",
      Party %in% c("R", "Republican", "GOP") ~ "Republican",
      TRUE ~ "Other"
    ),
    party_simple = factor(party_simple,
                          levels = c("Democrat", "Republican", "Other"))
  )

# Clean brand variable
analysis_df <- analysis_df %>%
  mutate(
    brand_main = dplyr::coalesce(
      brand_main.x,
      brand_main.y,
      brand
    )
  )

# 3. Overall distributions
# Frame distribution
frame_dist <- analysis_df %>%
  count(frame) %>%
  mutate(pct = n / sum(n))

print(frame_dist)

# Stance distribution
stance_dist <- analysis_df %>%
  count(stance) %>%
  mutate(pct = n / sum(n))

print(stance_dist)

# 4. Frame & stance by party
frame_by_party <- analysis_df %>%
  filter(party_simple != "Other") %>%
  count(party_simple, frame) %>%
  group_by(party_simple) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup()

print(frame_by_party)

stance_by_party <- analysis_df %>%
  filter(party_simple != "Other") %>%
  count(party_simple, stance) %>%
  group_by(party_simple) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup()

print(stance_by_party)

# Plots: frames & stance by party
p_frame_party <- ggplot(frame_by_party,
                        aes(x = frame, y = pct, fill = party_simple)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(
    values = c(
      "Democrat" = "#1f78b4",
      "Republican" = "#e31a1c",
      "Other" = "gray70"
    )
  ) +
  labs(
    title = "Frame distribution by party",
    x = "Frame",
    y = "Share of tweets",
    fill = "Party"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 25, hjust = 1)
  )

p_frame_party

p_stance_party <- ggplot(stance_by_party,
                         aes(x = stance, y = pct, fill = party_simple)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  
  scale_fill_manual(
    values = c(
      "Democrat" = "#1f78b4",   # blue
      "Republican" = "#e31a1c", # red
      "Other" = "gray70"
    )
  ) +
  
  labs(
    title = "Stance distribution by party",
    x = "Stance",
    y = "Share of tweets",
    fill = "Party"
  ) +
  
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

p_stance_party

# 5. Brand-level patterns: frames & politicization vs medicalization
# Frame distribution by brand
brand_frame <- analysis_df %>%
  filter(!is.na(brand_main)) %>%
  count(brand_main, frame) %>%
  group_by(brand_main) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

print(brand_frame)

p_brand_frames <- ggplot(brand_frame,
                         aes(x = brand_main, y = prop, fill = frame)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Frame distribution by vaccine brand",
    x = "Vaccine brand",
    y = "Share of tweets (within brand)",
    fill = "Frame"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold")
  )

p_brand_frames

# Politicization vs medicalization by brand using similarity scores
brand_politics_medical <- analysis_df %>%
  filter(!is.na(brand_main)) %>%
  group_by(brand_main) %>%
  summarise(
    politicized = mean(sim_hoax + sim_mandate, na.rm = TRUE),
    medicalized = mean(sim_safety, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

print(brand_politics_medical)

gg_brand <- ggplot(brand_politics_medical,
                   aes(x = politicized, y = medicalized, color = brand_main)) +
  geom_point(size = 3) +
  geom_text_repel(aes(label = brand_main),
                  size = 3.5, show.legend = FALSE) +
  scale_x_continuous(
    name = "Politicization (similarity to hoax + mandate)",
    expand = expansion(mult = c(0.05, 0.05))
  ) +
  scale_y_continuous(
    name = "Medicalization (similarity to safety)",
    expand = expansion(mult = c(0.05, 0.05))
  ) +
  labs(
    title = "Politicization vs. medicalization of misinformation by vaccine brand",
    subtitle = "Higher x = more politicized; higher y = more medicalized",
    color = "Vaccine brand"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    legend.position = "bottom"
  )

gg_brand

# 6. Time trends during COVID (2020–2022)
# Anti-vax share by party over time (COVID years only)
stance_time <- analysis_df %>%
  filter(
    party_simple != "Other",
    year >= 2020, year <= 2022   # keep only COVID years
  ) %>%
  mutate(anti_vax = if_else(stance == "anti_vaccine", 1, 0)) %>%
  group_by(year, party_simple) %>%
  summarise(
    n = n(),
    share_anti = mean(anti_vax),
    .groups = "drop"
  )

print(stance_time)

p_time_party <- ggplot(stance_time,
                       aes(x = year, y = share_anti, color = party_simple, group = party_simple)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  scale_color_manual(
    values = c(
      "Democrat" = "#1f78b4",
      "Republican" = "#e31a1c"
    )
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Share of anti-vaccine tweets during COVID period (2020–2022)",
    x = "Year",
    y = "Share anti-vaccine",
    color = "Party"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold")
  )

p_time_party

# Anti-vax share by brand over time (COVID years only)
brand_time <- analysis_df %>%
  filter(
    !is.na(brand_main),
    year >= 2020, year <= 2022
  ) %>%
  mutate(anti_vax = if_else(stance == "anti_vaccine", 1, 0)) %>%
  group_by(year, brand_main) %>%
  summarise(
    n = n(),
    share_anti = mean(anti_vax),
    .groups = "drop"
  )

print(brand_time)

p_time_brand <- ggplot(brand_time,
       aes(x = year, y = share_anti, group = brand_main, color = brand_main)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Share of anti-vaccine tweets by brand during COVID period",
    x = "Year",
    y = "Share anti-vaccine",
    color = "Vaccine brand"
  )

p_time_brand

# 7. Simple embedding-based model
# Here I show a simple logistic regression:
#   anti_vaccine (0/1) ~ party_simple + sim_hoax + sim_mandate + sim_safety

sim_cols <- grep("^sim_", names(analysis_df), value = TRUE)
message("Similarity columns used in model: ", paste(sim_cols, collapse = ", "))

model_df <- analysis_df %>%
  filter(party_simple %in% c("Democrat", "Republican")) %>%
  mutate(anti_vax = if_else(stance == "anti_vaccine", 1, 0)) %>%
  select(anti_vax, party_simple, all_of(sim_cols)) %>%
  drop_na()

message("Model sample size: ", nrow(model_df))

model_fit <- glm(
  anti_vax ~ party_simple + sim_hoax + sim_mandate + sim_safety,
  data = model_df,
  family = binomial(link = "logit")
)

summary(model_fit)

# Coeffient plot
coef_df <- broom::tidy(model_fit) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    term = recode(term,
                  "party_simpleRepublican" = "Republican (vs Democrat)",
                  "sim_hoax" = "Similarity: Hoax/Misinformation",
                  "sim_mandate" = "Similarity: Mandate/Freedom",
                  "sim_safety" = "Similarity: Medical/Safety"
    )
  )

p_coef <- ggplot(coef_df, aes(x = estimate, y = term)) +
  geom_point(size = 3, color = "darkred") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(
    title = "Predicting Anti-Vaccine Tweets Using Embeddings + Partisanship",
    x = "Coefficient (log-odds)",
    y = ""
  ) +
  theme_minimal(base_size = 14)

p_coef

# Predicted probabilities visualization
library(effects)

eff <- effects::effect("sim_hoax", model_fit)

plot(eff,
     main = "Effect of Hoax Similarity on Probability of Anti-Vaccine Tweet",
     xlab = "Similarity to Hoax Narrative",
     ylab = "Predicted Probability of Anti-Vaccine Stance")

# Brand level predict risk
brand_pred <- analysis_df %>%
  filter(
    !is.na(brand_main),
    party_simple %in% c("Democrat", "Republican")
  ) %>%
  group_by(brand_main, party_simple) %>%
  summarise(
    sim_hoax    = mean(sim_hoax,    na.rm = TRUE),
    sim_mandate = mean(sim_mandate, na.rm = TRUE),
    sim_safety  = mean(sim_safety,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    pred_anti = predict(model_fit, newdata = ., type = "response")
  )

brand_pred

p_brand_pred_plot <- ggplot(brand_pred,
       aes(x = brand_main, y = pred_anti,
           fill = party_simple)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_manual(values = c(Democrat = "#2b6cb0", Republican = "#c53030")) +
  labs(
    title = "Predicted probability of anti-vaccine tweets by brand and party",
    x = "Vaccine brand",
    y = "Predicted probability (anti-vaccine)",
    fill = "Party"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

p_brand_pred_plot

# 8. Save key tables and figures
dir.create("outputs")
dir.create(file.path("outputs", "figures"))

# 8.1 Save tables as CSV

# Overall distributions
readr::write_csv(frame_dist,      "outputs/frame_distribution.csv")
readr::write_csv(stance_dist,     "outputs/stance_distribution.csv")

# By party
readr::write_csv(frame_by_party,  "outputs/frame_by_party.csv")
readr::write_csv(stance_by_party, "outputs/stance_by_party.csv")

# Brand-level
readr::write_csv(brand_frame,             "outputs/brand_frame_distribution.csv")
readr::write_csv(brand_politics_medical,  "outputs/brand_politicization_medicalization.csv")

# Time trends
readr::write_csv(stance_time, "outputs/stance_time_trend_party.csv")
readr::write_csv(brand_time,  "outputs/stance_time_trend_brand.csv")

# Embedding-based model + predictions
readr::write_csv(coef_df,    "outputs/logit_embedding_coefficients.csv")
readr::write_csv(brand_pred, "outputs/brand_predicted_anti_probabilities.csv")

# 8.2 Save figures as PNG
ggplot2::ggsave("outputs/figures/frame_by_party.png",
                p_frame_party, width = 7, height = 5, dpi = 300)

ggplot2::ggsave("outputs/figures/stance_by_party.png",
                p_stance_party, width = 7, height = 5, dpi = 300)

ggplot2::ggsave("outputs/figures/frame_distribution_by_brand.png",
                p_brand_frames, width = 7, height = 5, dpi = 300)

ggplot2::ggsave("outputs/figures/politicization_vs_medicalization_brand.png",
                gg_brand, width = 7, height = 6, dpi = 300)

ggplot2::ggsave("outputs/figures/anti_vax_share_over_time_party.png",
                p_time_party, width = 7, height = 5, dpi = 300)

ggplot2::ggsave("outputs/figures/anti_vax_share_over_time_brand.png",
                p_time_brand, width = 7, height = 5, dpi = 300)

ggplot2::ggsave("outputs/figures/embedding_logit_coefficients.png",
                p_coef, width = 7, height = 5, dpi = 300)

ggplot2::ggsave("outputs/figures/predicted_anti_vax_by_brand_party.png",
                p_brand_pred_plot, width = 7, height = 5, dpi = 300)

message("Saved all tables to 'outputs/' and figures to 'outputs/figures/'.")