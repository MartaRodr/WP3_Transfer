
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
                                                  # SOCIAL TASK #
---------------------------------------------------------------------------------------------------#
# *- Load and prepare data ------------------------------------------------
dat <- read.csv("/Users/aramendi/Desktop/EscritorioMARTA/WP_Transfer/WP3Project/data/processed/egosocialtask.csv", stringsAsFactors = T)
dat <- read.csv(Users/aramendi/Desktop/EscritorioMARTA/)
# Number of responses prior to exclusion
dim(dat)[1] # N =9092

# Data processing step 1: Exclude responses with impossibly short RTs
dim(dat[dat$RTothers <= .005, ])[1] # n = 1194 responses
dim(dat[dat$RTothers <= .005, ])[1]/9212 # 12% exclusion rate
dat <- dat[dat$RTothers > .005, ]


# Data processing step 2: Log-transform RTs to account for their skewness

# overall distribution is skewed
hist(dat$RTothers, breaks = dim(dat)[1], xlim = c(0, 8))
hist(dat$RD, breaks = dim(dat)[1], xlim = c(0, 8))

# log-transform
dat$RTlog <- log(dat$RTothers)

# overall distribution is normal
hist(dat$RTlog, breaks = dim(dat)[1])

# Data processing step 3: Exclude outliers ≥2.5 SD from the grand mean
out_sd_lo <- mean(dat$RTlog, na.rm = T) - 2.5*sd(dat$RTlog, na.rm = T)
out_sd_hi <- mean(dat$RTlog, na.rm = T) + 2.5*sd(dat$RTlog, na.rm = T)
(length(which(dat$RTlog < out_sd_lo)) + 
    length(which(dat$RTlog > out_sd_hi)))/9212
dat <- dat[dat$RTlog > out_sd_lo & dat$RTlog < out_sd_hi, ]

# Centering RT within clusters (CWC) by participant (ida_ID)
dat <- dat %>% 
  group_by(PROLIFIC_PID) %>%
  mutate(RTlog_cm = mean(RTlog, na.rm = T),
         RTlog_cwc = RTlog - RTlog_cm) %>%
  as.data.frame()



# *- Analysis: Main model -------------------------------------------------
# Fit model
mod_social <- mixed( RD ~ RTlog_cwc * Individual + (1 | PROLIFIC_PID) + (0+RTlog_cwc || PROLIFIC_PID) ,  data = dat, method = "S")

# Fit model
mod_social <- mixed( RTlog_cwc ~ sim +  (1 | PROLIFIC_PID) + (0+RTlog_cwc || PROLIFIC_PID) ,  data = dat, method = "S")

mod1 <- mixed(RD ~ RTlog_cwc * Individual + (1 | PROLIFIC_PID), data=dat, method="S")

# Check differences between models.
anova(mod1, mod_social) # Compare if adding the random slope improve the models. There is not reason to include the random slopes


## SAve the image for the model without random slopes
summary(mod1)

tab_model(mod1$full_model, show.se = T, p.val = "satterthwaite", show.df = T,
          show.stat = T, string.est = "b", string.se = "SE", string.stat = "t",
          col.order = c("est", "se", "df.error", "stat", "p", "ci"), digits = 2)

# Get VALUES for the random slope model
fm <- mod_social$full_model # Get coef model
df_social <- coef(fm)$PROLIFIC_PID # Get coef participant

df_social <- coef(fm)$PROLIFIC_PID %>%
  rownames_to_column("PROLIFIC_PID")

## Save results from the model with slopes random for each participant
write.csv(df_social,
          file = "/Users/aramendi/Desktop/EscritorioMARTA/WP_Transfer/WP3Project/results/01SocialAnchor/tables/AnchorSlopes_RTlog_byParticipant.csv",
          row.names = FALSE)
