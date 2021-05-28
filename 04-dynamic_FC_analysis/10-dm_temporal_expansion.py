#DM temporal expansion
#Last edited: 20-10-2020

Step 0: Setup
In [1]:
# Loading packages
library(psych)
library(tidyverse)
library(data.table)
library(nlme)
library(broom)

#customizing theme for plotting
theme_training <- theme_bw() + theme(axis.text.y = element_text(size=25, colour='#262626ff'),
          axis.text.x = element_text(size=25, colour='#262626ff'),
          axis.title.y = element_text(size=25, colour='#262626ff'),
          axis.title.x  = element_text(size=25, colour='#262626ff'),
          plot.title = element_text(hjust=0.5, size=25),
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

#%%Step 1: Loading data

setwd("~/Dropbox/Projects/LearningBrain/")
fc_cartography_win = read.csv('data/neuroimaging/03-modularity/dynamic/04-recruitment_integration/whole-brain_power_raw_mean_allegiance_all_windows_tidy.csv')
dualnback_exclude = c('sub-13', 'sub-21', 'sub-23', 'sub-50') # higly motion subjects in one of four sessions

fc_cartography_win$Condition <- NA
fc_cartography_win$Condition[fc_cartography_win$Window %% 2 == 0] <- '2-back'
fc_cartography_win$Condition[fc_cartography_win$Window %% 2 != 0] <- '1-back'

fc_cartography_win$Session <- factor(fc_cartography_win$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))
fc_cartography_win$Group <- factor(fc_cartography_win$Group, levels = c('Experimental', 'Control'))

fc_cartography_win_clean <- fc_cartography_win %>% filter(!(Subject %in% dualnback_exclude))

dmn_win <- fc_cartography_win_clean %>% filter(Network == 'DM' ) %>% select(Subject:Window, Condition, DM)
head(dmn_win)

#%%Step 2: Multilevel modelling (DMN fluctuations)

baseline <- lme(DM ~ 1, random = ~ 1|Subject/Session/Condition, data = dmn_win, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
condition <- update(baseline, .~. + Condition)
session <- update(condition, .~. + Session)
group <- update(session, .~. + Group)

condition_session <- update(group, .~. + Condition:Session)
condition_group <- update(condition_session, .~. + Condition:Group)
session_group <- update(condition_group, .~. + Session:Group)
condition_session_group <- update(session_group, .~. + Condition:Session:Group)

anova(baseline, condition, session, group, condition_session, condition_group, session_group, condition_session_group)

#%%
write.csv(dmn_win, "DM_fluctuations.csv")

p <- ggplot(dmn_win, aes(x = Window, y = DM, col = Session)) + 
    stat_summary(fun.y = mean, geom = 'point', size = 3, alpha = 0.8) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Session), alpha = 0.8) +
    facet_wrap(~Group, nrow = 2) +
    scale_color_manual(values=c('#daa03d', '#efa978', '#b4530c', '#755841'))+
    ylab('DMN recruitment') +
    theme_training + theme(aspect.ratio = 0.4) + 
    xlab('Task block')
p

ggsave("figures/Figure_DMN_fluctuations.pdf", plot = p, width = 12, height = 9, dpi = 300)

#%%
summary(condition)

#%%
back')
exp2 <- dmn_win %>% filter(Session %in% c('Naive', 'Late')) %>% filter(Group == 'Experimental') %>% filter(Condition == '2-back')
con1 <- dmn_win %>% filter(Session %in% c('Naive', 'Late')) %>% filter(Group == 'Control') %>% filter(Condition == '1-back')
con2 <- dmn_win %>% filter(Session %in% c('Naive', 'Late')) %>% filter(Group == 'Control') %>% filter(Condition == '2-back')

t.test(exp1$DM ~ exp1$Session, paired = TRUE)
t.test(exp2$DM ~ exp1$Session, paired = TRUE)
t.test(con1$DM ~ exp1$Session, paired = TRUE)
t.test(con2$DM ~ exp1$Session, paired = TRUE)

#%%
dmn_win %>% group_by(Condition) %>% summarize(DM = mean(DM))

#%%
p <- ggplot(dmn_win, aes(x = Session, y = DM, col = Condition)) + 
    stat_summary(fun.y = mean, geom = 'point', size = 3, alpha = 0.8) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Condition), alpha = 0.8) +
    stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
    facet_wrap(~Group) +
    scale_colour_manual(values=c('#919649', '#fc766a')) +
    theme_training  + theme(aspect.ratio = 1) + 
    ylab('DMN recruitment')
p

ggsave("figure/Figure_DMN_fluctuations_mean.pdf", plot = p, width = 12, height = 5.5, dpi = 300)

#%%
dmn_win %>% filter(Session %in% c('Naive', 'Late')) %>% 
    group_by(Subject, Group, Session, Condition) %>% 
    summarize(DM = mean(DM)) %>%
    ggplot(aes(x = Group, y = DM, fill = Session)) + 
    geom_point(aes(col = Session), position=position_jitterdodge(dodge.width=0.9), alpha = 0.6, size = 4) +
    #geom_jitter(aes(col = Session), alpha = 0.6, size = 4, position=position_dodge(width=0.8)) +
    geom_boxplot(alpha = 0.4, outlier.shape = NA, position=position_dodge(width=0.8), size = 1) + 
    scale_fill_manual(values=c('#daa03d', '#755841')) +
    scale_color_manual(values=c('#daa03d', '#755841')) +
    facet_wrap(~Condition) +
    ylim(0.25, 0.8) +
    ylab('pRT(s)') +
    xlab(' ') +
    theme_training
    
#%% Step 2: Multilevel modelling (FPN fluctuations)

baseline <- lme(FP ~ 1, random = ~ 1|Subject/Session/Condition, data = fpn_win, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
condition <- update(baseline, .~. + Condition)
session <- update(condition, .~. + Session)
group <- update(session, .~. + Group)

condition_session <- update(group, .~. + Condition:Session)
condition_group <- update(condition_session, .~. + Condition:Group)
session_group <- update(condition_group, .~. + Session:Group)
condition_session_group <- update(session_group, .~. + Condition:Session:Group)

anova(baseline, condition, session, group, condition_session, condition_group, session_group, condition_session_group)
In [7]:
p <- ggplot(fpn_win, aes(x = Window, y = FP, col = Session)) + 
    stat_summary(fun.y = mean, geom = 'point', size = 3, alpha = 0.8) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Session), alpha = 0.8) +
    facet_wrap(~Group, nrow = 2) +
    ylab('FP recruitment') +
    theme_training + theme(aspect.ratio = 0.5) + 
    xlab('Task block')


































