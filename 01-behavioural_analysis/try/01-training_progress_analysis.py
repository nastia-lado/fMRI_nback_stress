#Behavioral progress during training sessions
#Visualize and analize data from 6-week working memory training study.
#Participants performed 18 sessions of dual n-back training (Jaeggi et al., 2008).
#Each session consisted of 20 runs of the task. The level of n increased with at least 80% correct
#trials in previous run.

#Last edited: 11-01-2019

#Step 0: Setup

# Loading packages
library(psych)
library(tidyverse)
library(data.table)
library(nlme)
library(broom)

# Customizing theme for plotting
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
          strip.background = element_rect(fill="transparent", colour=NA),
          strip.text = element_text(size=25),
          axis.ticks = element_line(colour="#262626ff", size=1, 2),
          axis.ticks.length = unit(.15, "cm"),
          aspect.ratio = 0.5)

#%%Step 1: Preparing data

# Setting working directory
setwd("~/Dropbox/Projects/LearningBrain/")
training_tidy_summary <- read_csv('data/behavioral/WM_training_mean_tidy.csv')
glimpse(training_tidy_summary)

# Display summary measures for all subjects
training_tidy_summary %>% group_by(as.factor(Session)) %>% summarize(mean = mean(Nback_mean))


#%%Step 2: Individual growth models

# Testing training progress for experimental group
baseline <- lme(Nback_mean ~ 1, random = ~ 1 + Session|Subject, data = training_tidy_summary, method = 'ML',  control = list(opt = "optim"))
session <- update(baseline, .~. + Session)
session_quadr <- update(session, .~. + I(Session^2))
anova(baseline, session, session_quadr)

summary(session)

#%%
# Place individual predictions and residuals into the dataframe
training_tidy_summary$pred_fl_rl_quadr <- predict(session_quadr)
training_tidy_summary$resid_fl_rl_quadr <- residuals(session_quadr)

# Obtaining predicted scores for prototype
training_tidy_summary$proto_quadr <- predict(session_quadr, level=0)

p <- ggplot(training_tidy_summary, aes(x = Session, y = pred_fl_rl_quadr, group = Subject)) + 
    #geom_point(col = '#ee8c00', alpha = 0.3)+
    geom_line(col = '#ee8c00', alpha = 0.2, size = 1.5) +
    xlab('Training session') + 
    ylab('Predicted mean n level') +
    geom_line(aes(x = Session, y = proto_quadr), color='#262626ff',size=1.5, alpha = 0.5) +
    theme_training
p  

ggsave("figures/Figure_train_prog_pred_nback.pdf", plot = p, width = 10, height = 5, dpi = 300)

#%%
p <- ggplot(training_tidy_summary, aes(x = Session, y = Nback_mean, group = Session)) + 
    geom_boxplot(col = '#262626ff', fill = '#ee8c00', alpha = 0.2, size = 1, outlier.shape = NA)+
    geom_jitter(col = '#ee8c00', alpha = 0.2, size = 4) +
    xlab('Training session') + 
    ylab('Mean n level') +
    theme_training
 p  

ggsave("figures/Figure_train_prog_mean_nback.pdf", plot = p, width = 10, height = 5, dpi = 300)

#%%
p <- ggplot(training_tidy_summary, aes(x = Session, y = pred_fl_rl_quadr, group = Subject)) + 
    geom_point(aes(x = Session,y = Nback_mean), col = '#ee8c00', alpha = 1, size = 2.5)+
    geom_line(col = '#262626ff', alpha = 1, size = 1) +
    xlab('Training session') + 
    ylab('Mean n-back level') +
    ggtitle("Experimental group") +
    theme_training + 
    ylim(1,6) + 
    facet_wrap(~Subject) +
    theme(aspect.ratio = 1)
 p   

ggsave("figures/Figure_train_prog_indiv.pdf", plot = p, width = 10, height = 15, dpi = 300)

#%%

max_n_level = training_tidy_summary %>% group_by(Subject) %>% summarize(Max_n_lev = max(Nback_max))
write.csv(max_n_level, file = "WM_training_max_level.csv", row.names=FALSE)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    