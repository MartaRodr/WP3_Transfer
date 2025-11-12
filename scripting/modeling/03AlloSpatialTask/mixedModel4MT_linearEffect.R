# Allocentric spatial task mixed model regressions

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




# *- Load and prepare data ------------------------------------------------

dat <- read.csv("/Users/aramendi/Desktop/EscritorioMARTA/WP_Transfer/WP3Project/data/processed/AlloTask_SpatialScore.CSV", stringsAsFactors = T)
dat$RTlog <- log(dat$key_resp_3.rt)
dat$Score_SpatialGeneral_z <- scale(dat$Score_SpatialGeneral, center = TRUE, scale = TRUE)

# Eliminate rts trials outlier
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



# RTmodel - 
mod_spatialRT <- mixed( RTlog ~  bin * Score_SpatialGeneral_z +(1| PROLIFIC_PID),  data = dat, method = "S")
summary(mod_spatialRT)
tab_model(mod_spatialRT$full_model, show.se = T, p.val = "satterthwaite", show.df = T,
          show.stat = T, string.est = "b", string.se = "SE", string.stat = "t",
          col.order = c("est", "se", "df.error", "stat", "p", "ci"), digits = 2)


# Accuracy model-
mod_spatialAcc <- mixed( key_resp_3.corr ~  bin  * Score_SpatialGeneral_z* RTlog_cwc +(1| PROLIFIC_PID),  data = dat, method = "S")
summary(mod_spatialAcc)

tab_model(mod_spatialAcc$full_model, show.se = T, p.val = "satterthwaite", show.df = T,
          show.stat = T, string.est = "b", string.se = "SE", string.stat = "t",
          col.order = c("est", "se", "df.error", "stat", "p", "ci"), digits = 2)
