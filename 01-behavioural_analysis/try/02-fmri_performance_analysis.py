#Behavioral progress during fMRI scanning sessions
#The code below allows to visualize and analize data from 6-week working memory training study.
#Participants were scanned four times while performing dual n-back (Jaeggi et al., 2018).

#Three performance measures were calculated:

#accuracy
#d' (dprime)
#penallized reaction time (prt)
#We additionally calculated block-to-block standard diviation for each of this measures.

#Last edited: 21-11-2019

#Step 0: Setup

# Loading packages
library(psych)
library(tidyverse)
library(data.table)
library(nlme)
library(broom)

# Customizing theme for plotting
theme_training <- theme_bw() + theme(axis.text.y = element_text(size=22, colour='#262626ff'),
          axis.text.x = element_text(size=22, colour='#262626ff'),
          axis.title.y = element_text(size=22, colour='#262626ff'),
          axis.title.x  = element_text(size=22, colour='#262626ff'),
          plot.title = element_text(hjust=0.5, size=22),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.line = element_line(colour="#262626ff"),
          panel.border = element_rect(colour = "#262626ff", fill=NA, size=1.8),
          panel.background = element_rect(fill="transparent",colour=NA),
          plot.background = element_rect(fill="transparent",colour=NA),
          legend.key = element_rect(fill= "transparent", colour="transparent"),
          strip.background =element_rect(fill="transparent", colour=NA),
          strip.text = element_text(size=25),
          axis.ticks = element_line(colour="#262626ff", size=1, 2),
          axis.ticks.length = unit(.15, "cm"),
          aspect.ratio = 1)


#%%Step 1: Preparing data

# Setting working directory
setwd("~/Dropbox/Projects/LearningBrain/")

# Loading data
performance <- read.csv("data/behavioral/WM_fmri_behaviour_mean_tidy.csv")
performance$Group <- factor(performance$Group, levels = c('Control', 'Experimental'))
performance$Session <- factor(performance$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))

# Removing subjects with high motion
performance <- performance %>% filter(!(Subject %in% c('sub-13', 'sub-21', 'sub-23', 'sub-50')))
head(performance)

#%%Step 2: Multilevel modelling (MLM): d-prime change

baseline <- lme(Dprime ~ 1, random = ~ 1|Subject/Session/Condition, data = performance, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
condition <- update(baseline, .~. + Condition)
session <- update(condition, .~. + Session)
group <- update(session, .~. + Group)

condition_session <- update(group, .~. + Condition:Session)
condition_group <- update(condition_session, .~. + Condition:Group)
session_group <- update(condition_group, .~. + Session:Group)
condition_session_group <- update(session_group, .~. + Condition:Session:Group)

anova(baseline, condition, session, group, condition_session, condition_group, session_group, condition_session_group)

#%%
summary(condition_session_group)

#%%

performance$Group <- factor(performance$Group, levels = c('Experimental', 'Control'))

p <- ggplot(performance, aes(x = Session, y = Dprime, color = Condition)) +
    stat_summary(fun.y = mean, geom = 'point', size = 3) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Condition)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
    scale_colour_manual(values=c('#919649', '#fc766a')) +
    theme_training +
    ylab('D-prime') +
    xlab('') +
    facet_wrap('~Group')

p

#ggsave("figures/dprime_lineplot.pdf", plot = p, width = 12, height = 6, dpi = 300)

#%%Step 4: T-tests: d-prime change

# Differences between first and last sessions for each group and task conditions
beh_mean_nl <- performance %>% filter(Session %in% c('Naive', 'Late'))

exp1 <-  beh_mean_nl %>% filter(Group == 'Experimental') %>% filter(Condition == '1-back')
exp2 <-  beh_mean_nl %>% filter(Group == 'Experimental') %>% filter(Condition == '2-back')
con1 <-  beh_mean_nl %>% filter(Group == 'Control') %>% filter(Condition == '1-back')
con2 <-  beh_mean_nl %>% filter(Group == 'Control') %>% filter(Condition == '2-back')

t.test(exp1$Dprime ~ exp1$Session, paired = TRUE)
t.test(exp2$Dprime ~ exp2$Session, paired = TRUE)
t.test(con1$Dprime ~ con1$Session, paired = TRUE)
t.test(con2$Dprime ~ con2$Session, paired = TRUE)

# Multiply by 4 to correct for mulriple comparisons (Bonferroni)

#%%
t.test(exp2[exp2$Session == 'Naive', ]$Dprime - exp2[exp2$Session == 'Late', ]$Dprime,
con2[con2$Session == 'Naive', ]$Dprime - con2[con2$Session == 'Late', ]$Dprime, paired = FALSE)

t.test(exp1[exp1$Session == 'Naive', ]$Dprime - exp1[exp1$Session == 'Late', ]$Dprime,
con1[con1$Session == 'Naive', ]$Dprime - con1[con1$Session == 'Late', ]$Dprime, paired = FALSE)

# Multiply by 2 to correct for mulriple comparisons (Bonferroni)

#%%
# Difference between 1-back and 2-back pRT in the last session (Experimental)
t.test(exp1[exp1$Session == 'Late', ]$Dprime, exp2[exp2$Session == 'Late', ]$Dprime, paired = TRUE)

# Difference between 1-back and 2-back pRT in the last session (Control)
t.test(con1[con1$Session == 'Late', ]$Dprime, con2[con2$Session == 'Late', ]$Dprime, paired = TRUE)

# Correct for 2 tests

#%%
p <- performance %>% filter(Session %in% c('Naive', 'Late')) %>%
    ggplot(aes(x = Group, y = Dprime, fill = Session)) + 
    geom_point(aes(col = Session), position=position_jitterdodge(dodge.width=0.9), alpha = 0.6, size = 4) +
    geom_boxplot(alpha = 0.4, outlier.shape = NA, position=position_dodge(width=0.8), size = 1) + 
    scale_fill_manual(values=c('#daa03d', '#755841')) +
    scale_color_manual(values=c('#daa03d', '#755841')) +
    facet_wrap(~Condition) +
    ylim(0, 6.2) +
    ylab('D-prime') +
    xlab(' ') +
    theme_training
p

ggsave("figures/dprime_ttests.pdf", plot = p, width = 12, height = 6, dpi = 300)

#%%Step 2: Multilevel modelling (MLM): pRT change

# Differences between first and last sessions for each group and task conditions
beh_mean_nl <- performance %>% filter(Session %in% c('Naive', 'Late'))

exp1 <-  beh_mean_nl %>% filter(Group == 'Experimental') %>% filter(Condition == '1-back')
exp2 <-  beh_mean_nl %>% filter(Group == 'Experimental') %>% filter(Condition == '2-back')
con1 <-  beh_mean_nl %>% filter(Group == 'Control') %>% filter(Condition == '1-back')
con2 <-  beh_mean_nl %>% filter(Group == 'Control') %>% filter(Condition == '2-back')

#%%
baseline <- lme(pRT ~ 1, random = ~ 1|Subject/Session/Condition, data = performance, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
condition <- update(baseline, .~. + Condition)
session <- update(condition, .~. + Session)
group <- update(session, .~. + Group)

condition_session <- update(group, .~. + Condition:Session)
condition_group <- update(condition_session, .~. + Condition:Group)
session_group <- update(condition_group, .~. + Session:Group)
condition_session_group <- update(session_group, .~. + Condition:Session:Group)

anova(baseline, condition, session, group, condition_session, condition_group, session_group, condition_session_group)
summary(condition_session_group)

#%%
performance$Group <- factor(performance$Group, levels = c('Experimental', 'Control'))

p <- ggplot(performance, aes(x = Session, y = pRT, color = Condition)) +
    stat_summary(fun.y = mean, geom = 'point', size = 3) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Condition)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
    scale_colour_manual(values=c('#919649', '#fc766a')) +
    theme_training +
    ylab('pRT(ms)') +
    xlab('') +
    facet_wrap('~Group')
p

ggsave("figures/prt_lineplot.pdf", plot = p, width = 12, height = 6, dpi = 300)

#%%Step 3: T-tests: pRT change

t.test(exp1$pRT ~ exp1$Session, paired = TRUE)
t.test(exp2$pRT ~ exp2$Session, paired = TRUE)
t.test(con1$pRT ~ con1$Session, paired = TRUE)
t.test(con2$pRT ~ con2$Session, paired = TRUE)

# All p-values should be multiplied by 4, to correct for multiple comparisons (Bonferroni correction)

#%%
# Differences in changes of behavior between groups

t.test(exp2[exp2$Session == 'Naive', ]$pRT - exp2[exp2$Session == 'Late', ]$pRT,
con2[con2$Session == 'Naive', ]$pRT - con2[con2$Session == 'Late', ]$pRT, paired = FALSE)

t.test(exp1[exp1$Session == 'Naive', ]$pRT - exp1[exp1$Session == 'Late', ]$pRT,
con1[con1$Session == 'Naive', ]$pRT - con1[con1$Session == 'Late', ]$pRT, paired = FALSE)

# All p-values should be multiplied by 2, to correct for multiple comparisons (Bonferroni correction)

#%%
# Difference between 1-back and 2-back pRT in the last session (Experimental)
t.test(exp1[exp1$Session == 'Late', ]$pRT, exp2[exp2$Session == 'Late', ]$pRT, paired = TRUE)

# Difference between 1-back and 2-back pRT in the last session (Control)
t.test(con1[con1$Session == 'Late', ]$pRT, con2[con2$Session == 'Late', ]$pRT, paired = TRUE)

# All p-values should be multiplied by 2, to correct for multiple comparisons (Bonferroni correction)

#%%
summary_performance <- performance %>% 
    group_by(Session, Group, Condition) %>% filter(Session %in% c('Naive', 'Late'))%>%
    summarize(mpRT = mean(pRT), mDprime = mean(Dprime), mAcc = mean(Accuracy))

summary_performance

#%%
# Percentage of improvement
1 - (summary_performance %>% filter(Session=='Naive') / summary_performance %>% filter(Session=='Late')) %>% 
select(mpRT, mDprime, mAcc)

#%%
p <- performance %>% filter(Session %in% c('Naive', 'Late')) %>%
    ggplot(aes(x = Group, y = pRT, fill = Session)) + 
    geom_point(aes(col = Session), position=position_jitterdodge(dodge.width=0.9), alpha = 0.6, size = 4) +
    geom_boxplot(alpha = 0.4, outlier.shape = NA, position=position_dodge(width=0.8), size = 1) + 
    scale_fill_manual(values=c('#daa03d', '#755841')) +
    scale_color_manual(values=c('#daa03d', '#755841')) +
    facet_wrap(~Condition) +
    ylim(400, 2500) +
    ylab('pRT(ms)') +
    xlab(' ') +
    theme_training
p

ggsave("figures/prt_ttests.pdf", plot = p, width = 12, height = 6, dpi = 300)

#%%Block-to-block variability

head(performance)

performance_variability <- read.csv('data/behavioral/WM_fmri_behaviour_variability_tidy.csv')
performance_variability$Group <- factor(performance_variability$Group, levels = c('Control', 'Experimental'))
performance_variability$Session <- factor(performance_variability$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))

performance_variability <- performance_variability %>% 
                            filter(!(Subject %in% c('sub-13', 'sub-21', 'sub-23', 'sub-50')))


head(performance_variability)

#%%pRT variability

baseline <- lme(pRT_std ~ 1, random = ~ 1|Subject/Session, data = performance_variability, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
session <- update(baseline, .~. + Session)
group <- update(session, .~. + Group)
session_group <- update(group, .~. + Session:Group)

anova(baseline, session, group, session_group)

#%%
summary(group)

#%%
p <- ggplot(performance_variability, aes(x = Session, y = pRT_std, col = Group)) +
    stat_summary(fun.y = mean, geom = 'point', size = 3) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Group)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
    scale_colour_manual(values=c('#379dbc','#ee8c00')) +
    theme_training +
    ylab(expression(paste(" Behavioral variability (", sigma, "pRT)"))) +
    xlab('') 
p

p <- ggplot(performance_variability, aes(x = Session, y = pRT_std,col=Group)) +
    geom_boxplot(size=1) +
    ylab(expression(paste(" Behavioral variability (", sigma, "pRT)"))) +
    xlab('') + 
    scale_colour_manual(values=c('#379dbc','#ee8c00')) +
    theme_training
p

#%%D-prime variability

baseline <- lme(Dprime_std ~ 1, random = ~ 1|Subject/Session, data = performance_variability, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
session <- update(baseline, .~. + Session)
group <- update(session, .~. + Group)
session_group <- update(group, .~. + Session:Group)

anova(baseline, session, group, session_group)

#%%
summary(session)

p <- ggplot(performance_variability, aes(x = Session, y = Dprime_std, col = Group)) +
    stat_summary(fun.y = mean, geom = 'point', size = 3) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Group)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
    scale_colour_manual(values=c('#379dbc','#ee8c00')) +
    theme_training +
    ylab(expression(paste(" Behavioral variability (", sigma, "d')"))) +
    xlab('') 
p

p <- ggplot(performance_variability, aes(x = Session, y = Dprime_std, col = Group)) +
    geom_boxplot(size=1) +
    ylab(expression(paste(" Behavioral variability (", sigma, "d')"))) +
    xlab('') + 
    scale_colour_manual(values=c('#379dbc','#ee8c00')) +
    theme_training
p

#%%

















