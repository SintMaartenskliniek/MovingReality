## ---------------------------
##
## Script name: LMM_FSA.R
##
## Purpose of script: statistical analysis Movin(g) Reality study
##
## Author: Carmen Ensink
##
## Date Created: 2023-12-20
##
## 
## ---------------------------
##
## Notes:
##   Internal study ID: 1019_Movin(g)Reality_Feedback
## 
## ---------------------------

# load up the needed packages
library(dplyr)
library(lme4)
library(jtools)
library(readxl)
library(tidyr)
library(lmerTest)
library(grafify)

# open data set, assumed the .xlsx file is available from the working directory
filename = "./meanPROP.xlsx"
PROPdata <- read_excel(filename)

# change names
PROPdata$trialtype[(PROPdata$trialtype) == "2Reg"] <- "Reg2"
PROPdata$trialtype[(PROPdata$trialtype) == "FBPO"] <- "FBaPO"
PROPdata$trialtype[(PROPdata$trialtype) == "FBIC"] <- "FBbIC"
PROPdata$trialtype[(PROPdata$trialtype) == "2FB"] <- "FBz2"


# The basemodel only takes the random intercept of participantID into account on the outcome PROP
# Evaluate the model by maximum likelihood, not restricted maximum likelihood (REML), so set REML to FALSE (will allow the "anova" function later)
basemodel = lmer(PROP ~ trialtype + (1|participantID), data=PROPdata, REML=FALSE, na.action=na.exclude) # Only take the random intercept based on participant ID into account as random variable on outcome 'PROP'


# Add group (trial order) as fixed variable (including interaction effect) to the model
# Evaluate the model by maximum likelihood, not restricted maximum likelihood (REML), so set REML to FALSE (will allow the "anova" function later)
model_1 = lmer(PROP ~ trialtype * group + (1|participantID), data=PROPdata, REML=FALSE, na.action=na.exclude)

# Perform Anova between basemodel and extended model.
# In case there is a significant difference (Pf(>Chisq)) and AIC/BIC decrease; the fixed parameter group adds to the accuracy of the basemodel
anovaResult_1 <- anova(basemodel, model_1)
print(anovaResult_1)

# Pr(>Chisq) = 0.02076; and decrease of AIC and BIC > continue with model_1 (adding group as a fixed parameter improves the accuracy of the model)


# Add time as a fixed effect (to estimate learning effect)
model_2 = lmer(PROP ~ time + trialtype * group + (1|participantID), data=PROPdata, REML=FALSE, na.action=na.exclude)
# Model_2 outputs: fixed-effect model matrix is rank deficient so dropping 1 column / coefficient
# Adding time to the model as a fixed effect does not affect the outcome?

anovaResult_2 <- anova(model_1, model_2)
print(anovaResult_2)

# Add time as a random effect, assumed the effect between participants is similar (e.g. +10) but correct for starting at different values (e.g. 0 for participant 1 and 10 for participant 2)
model_3 = lmer(PROP ~ trialtype * group + (1+time|participantID), data=PROPdata, REML=FALSE, na.action=na.exclude)

anovaResult_3 <- anova(model_1, model_3)
print(anovaResult_3)

# Pr(>Chisq) = 0.611; > continue with model_1


# Calculate the estimate, standard error, and t-value for the parameters
results.coefs <-coef(summary(model_1), ddf='Kenward-Roger')
print(summ(model_1, confint = TRUE, digits = 3))



# Results suggest:
#   * No interaction effect (trialtype:group effects are all not-significant), meaning no effect of the order of feedback,
#   * Main effect of trialtype (trialtype effects are all significant), meaning the trialtype has an effect on the mean PROP
#   * No main group effect (group effect is not significant), meaning there is no different intercept between the groups (first push off feedback has a similar mean PROP compared to first FSA feedback, independent of the trialtype)



# Post-hoc pairwise comparisons, p-values corrected for multiple comparisons, degrees of freedom corrected using Kenward-Roger method
posthoc_Pairwise(Model = model_1, 
                 Fixed_Factor = "trialtype")


# # Result suggest:
# #   *  difference between propulsion in 1Reg and FBIC conditions
# #   *  difference between propulsion in 1Reg and FBPO conditions
# #   *  difference between propulsion in 1Reg and 2FB conditions
# #   *  difference between propulsion in 1Reg and 2Reg conditions
# #   *  difference between propulsion in FBPO and 2FB conditions
# #   *  difference between propulsion in FBPO and FBIC conditions


# # Post-hoc paired samples t-tests with bonferroni correction for multiple comparisons
# 
# 
# # Use pivot_wider to transform the data to wide format
# df_wide <- pivot_wider(PROPdata, names_from = trialtype, values_from = PROP, id_cols=participantID)
# # Rename columns
# names(df_wide)[names(df_wide) == "1Reg"] <- "Reg1"
# names(df_wide)[names(df_wide) == "2Reg"] <- "Reg2"
# names(df_wide)[names(df_wide) == "2FB"] <- "FB2"
# 
# 
# # Reg1 - FBIC
# # Perform paired t-tests, use confidence level of 95%
# ttest_Reg1_FBIC <- t.test(df_wide$Reg1, df_wide$FBIC, paired = TRUE, conf.level=0.95)
# # Print the pairwise result
# print(ttest_Reg1_FBIC)
# 
# 
# # Reg1 - FBPO
# # Perform paired t-tests, use confidence level of 95%
# ttest_Reg1_FBPO <- t.test(df_wide$Reg1, df_wide$FBPO, paired = TRUE, conf.level=0.95)
# # Print the pairwise result
# print(ttest_Reg1_FBPO)
# 
# 
# # Reg1 - FB2
# # Perform paired t-tests, use confidence level of 95%
# ttest_Reg1_FB2 <- t.test(df_wide$Reg1, df_wide$FB2, paired = TRUE, conf.level=0.95)
# # Print the pairwise result
# print(ttest_Reg1_FB2)
# 
# 
# # Reg1 - Reg2
# # Perform paired t-tests, use confidence level of 95%
# ttest_Reg1_Reg2 <- t.test(df_wide$Reg1, df_wide$Reg2, paired = TRUE, conf.level=0.95)
# # Print the pairwise result
# print(ttest_Reg1_Reg2)
# 
# 
# # FBPO - FB2
# # Perform paired t-tests, use confidence level of 95%
# ttest_FBPO_FB2 <- t.test(df_wide$FBPO, df_wide$FB2, paired = TRUE, conf.level=0.95)
# # Print the pairwise result
# print(ttest_FBPO_FB2)
# 
# 
# # FBPO - FBIC
# # Perform paired t-tests, use confidence level of 95%
# ttest_FBPO_FBIC <- t.test(df_wide$FBPO, df_wide$FBIC, paired = TRUE, conf.level=0.95)
# # Print the pairwise result
# print(ttest_FBPO_FBIC)
# 
# 
# # Place all p-values in one variable for Bonferroni correction
# p_values <- c(ttest_Reg1_FBIC$p.value, ttest_Reg1_FBPO$p.value, ttest_Reg1_FB2$p.value, ttest_Reg1_Reg2$p.value, ttest_FBPO_FB2$p.value, ttest_FBPO_FBIC$p.value)
# num_comparisons <- length(p_values)
# 
# # Apply Bonferroni correction for multiple comparisons
# adjusted_p_values <- p.adjust(p_values, method = "bonferroni")
# print(adjusted_p_values)
# 
# 
# # Result suggest:
# #   * No significant difference between propulsion in 1Reg and FBIC conditions
# #   * Significant difference between propulsion in 1Reg and FBPO conditions
# #   * Significant difference between propulsion in 1Reg and 2FB conditions
# #   * Significant difference between propulsion in 1Reg and 2Reg conditions
# #   * No significant difference between propulsion in FBPO and 2FB conditions
# #   * No significant difference between propulsion in FBPO and FBIC conditions
