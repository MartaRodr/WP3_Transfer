
# Prepare packages --------------------------------------------------------
# Name the packages needed
packages <- c("tidyverse", "afex", "emmeans", "sjPlot", "psych", "effsize",
              "brms", "ggplot2", "ggthemes", "lmerTest")

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == F)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = T))

# Set contrast for global environment
set_sum_contrasts()

# Set degrees-of-freedom method for marginal means as asymptotic 
emm_options(lmer.df = "asymptotic")



# -------------------------------------------------------------------------------------------------
                                      # SPATIAL TASK #
---------------------------------------------------------------------------------------------------#
# *- Load and prepare data ------------------------------------------------

dat <- read.csv("/Users/aramendi/Desktop/EscritorioMARTA/WP_Transfer/WP3Project/data/processed/egospatialTask_SpatialScore.csv", stringsAsFactors = T)

# Number of responses prior to exclusion
dim(dat)[1] # N = 9728

# Data processing step 1: Exclude responses with impossibly short RTs
dim(dat[dat$Response.rt <= .005, ])[1] # n = 15 responses
dim(dat[dat$Response.rt <= .005, ])[1]/9728 # 0% exclusion rate
dat <- dat[dat$Response.rt > .005, ]


# Data processing step 2: Log-transform RTs to account for their skewness

# overall distribution is skewed
hist(dat$Response.rt, breaks = dim(dat)[1], xlim = c(0, 8))
hist(dat$meanDistance, breaks = dim(dat)[1], xlim = c(0, 8))

# log-transform
dat$RTlog <- log(dat$Response.rt)

# overall distribution is normal
hist(dat$RTlog, breaks = dim(dat)[1])


# Data processing step 3: Exclude outliers ≥2.5 SD from the grand mean
out_sd_lo <- mean(dat$RTlog, na.rm = T) - 2.5*sd(dat$RTlog, na.rm = T)
out_sd_hi <- mean(dat$RTlog, na.rm = T) + 2.5*sd(dat$RTlog, na.rm = T)
(length(which(dat$RTlog < out_sd_lo)) + 
    length(which(dat$RTlog > out_sd_hi)))/768 
dat <- dat[dat$RTlog > out_sd_lo & dat$RTlog < out_sd_hi, ]

# Centering RT within clusters (CWC) by participant (ida_ID)
dat <- dat %>% 
  group_by(PROLIFIC_PID) %>%
  mutate(RTlog_cm = mean(RTlog, na.rm = T),
         RTlog_cwc = RTlog - RTlog_cm) %>%
  as.data.frame()

library(lme4)
dat$meanDistance_z         <- scale(dat$meanDistance,      center = TRUE, scale = TRUE)
dat$AD_z         <- scale(dat$AD,      center = TRUE, scale = TRUE)
dat$Score_SpatialGeneral_z <- scale(dat$Score_SpatialGeneral, center = TRUE, scale = TRUE)
dat$SelfCorrect_z <- scale(dat$distCorrSelf, center = TRUE, scale = TRUE)


# *- Analysis: Spatial model -------------------------------------------------
# Fit model RTmodel with meanDistancez
mod_spatial <- mixed( RTlog ~  meanDistance_z  * AD_z * Score_SpatialGeneral_z +
                        (meanDistance_z  | PROLIFIC_PID),  data = dat, method = "S")
summary(mod_spatial)

mod_spatial <- mixed( RTlog ~  meanDistance_z * Self_proximity * AD_z  +
                        (meanDistance_z  | PROLIFIC_PID),  data = dat, method = "S")
summary(mod_spatial)

# Extract coeffs
lmer_model <- mod_spatial$full_model
ranefs <- ranef(lmer_model)$PROLIFIC_PID

library(tibble)

df_slopes_LMM <- ranefs %>%
  rownames_to_column("PROLIFIC_PID") %>%
  rename(intercept_random = `(Intercept)`,
         slope_random_meanDistance = meanDistance_z)

# Save results in result path
write.csv(df_slopes_LMM,
          file = "C:/Users/aramendi/Desktop/EscritorioMARTA/WP_Transfer/WP3Project/results/02EgoSpatialTask/CoefsMixedLinearRT_meanDistance_egoTask.csv",
          row.names = FALSE)

#### accuracy model ###
# Fit model Accuracy
mod_spatial_logit2 <- glmer(
  Accuracy ~  meanDistance_z * AD_z * Score_SpatialGeneral_z +
    (meanDistance_z  | PROLIFIC_PID),
  data   = dat,
  family = binomial(link = "logit"),
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl   = list(maxfun = 5e5)
  )
)




mod_spatial_logit2 <- glmer(
  Accuracy ~  meanDistance_z * Self_proximity * Score_SpatialGeneral_z +
    (meanDistance_z  | PROLIFIC_PID),
  data   = dat,
  family = binomial(link = "logit"),
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl   = list(maxfun = 5e5)
  )
)


tab_model(
  mod_spatial,
  mod_spatial_logit2,
  dv.labels   = c("RTlog (REML)", "Accuracy (logit)"),
  show.re.var = TRUE,                 
  string.est  = "Est.", 
  string.se   = "SE",
  string.p    = "p"
)


# coef() returns combined fixed + random effects for each subject:
coefs_df <- coef(mod_spatial_logit2)$PROLIFIC_PID %>%
  tibble::rownames_to_column("PROLIFIC_PID")


write.csv(
  coefs_df,
  file = "C:/Users/aramendi/Desktop/EscritorioMARTA/WP_Transfer/WP3Project/results/02EgoSpatialTask/CoefsMixedLogisticAcc_meanDistance_egoTask.csv",
  row.names = TRUE   
)

### frecuency
# Fit model RTmodel with self correct option
modelSelfCorrect <- mixed( RTlog ~  SelfCorrect_z  * AD_z * Score_SpatialGeneral_z +
                        (SelfCorrect_z  | PROLIFIC_PID),  data = dat, method = "S")
summary(modelSelfCorrect)


# Extract coeffs
lmer_model <- modelSelfCorrect$full_model
ranefs <- ranef(lmer_model)$PROLIFIC_PID

library(tibble)

df_slopes_LMM <- ranefs %>%
  rownames_to_column("PROLIFIC_PID") %>%
  rename(intercept_random = `(Intercept)`,
         slope_selfCorrect = SelfCorrect_z)

# Save results in result path
write.csv(df_slopes_LMM,
          file = "C:/Users/aramendi/Desktop/EscritorioMARTA/WP_Transfer/WP3Project/results/02EgoSpatialTask/CoefsMixedLinearRT_selfCorrectDistance_egoTask.csv",
          row.names = FALSE)
