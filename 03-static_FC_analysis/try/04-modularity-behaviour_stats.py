#Modularity-behaviour analysis
#Last edited: 20-10-2020

# Loading packages
library(psych)
library(tidyverse)
library(data.table)
library(nlme)
library(broom)
library(Hmisc)

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
          strip.background =element_rect(fill="transparent", colour=NA),
          strip.text = element_text(size=25),
          axis.ticks = element_line(colour="#262626ff", size=1, 2),
          axis.ticks.length = unit(.15, "cm"),
          aspect.ratio = 1,
          )

#%%Step 1: Preparing data

# Setting working directory
setwd("~/Dropbox/Projects/LearningBrain/")

#---Modularity
Q <- read.csv('./data/neuroimaging/03-modularity/static/Q_normalized_power_tidy.csv')
Q$Session <- factor(Q$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))
Q$Condition <- factor(Q$Condition, levels = c('Rest', '1-back', '2-back'))
Q$Group <- factor(Q$Group, levels = c('Experimental', 'Control'))


Q$Task[Q$Condition == '1-back'] <- 'N-back'
Q$Task[Q$Condition == '2-back'] <- 'N-back'
Q$Task[Q$Condition == 'Rest'] <- 'Rest'

first_session_motion = 'sub-21' # subject with highly motion on first session
dualnback_motion = c('sub-13', 'sub-21', 'sub-23', 'sub-50')

# Selecting 1-back
Q_1back <- Q %>% 
        filter(Condition == '1-back') %>%
        select(-Task) %>% 
        select(-Condition)

colnames(Q_1back)[4] <- 'Q_norm_1back'

# Selecting 2-back
Q_2back <- Q %>% 
           filter(Condition == '2-back') %>% 
           select(-Task) %>% 
           select(-Condition)

colnames(Q_2back)[4] <- 'Q_norm_2back'

# Delta Q
Q_2back_1back <- Q_2back$Q_norm_2back - Q_1back$Q_norm_1back

Q_diff <- Q_1back %>% left_join(Q_2back)
Q_diff$Q_2back_1back <- Q_2back_1back  

head(Q_diff)

#%%
#--- Behaviour
behaviour <- read.csv('./data/behavioral/WM_fmri_behaviour_mean_tidy.csv')
behaviour$Session <- factor(behaviour$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))
behaviour$Group <- factor(behaviour$Group, levels = c('Experimental', 'Control'))

# Behaviour 1-back
behaviour_1back <- behaviour %>% 
                   filter(Condition=='1-back') %>%
                   select(-Condition)

# Behaviour 2-back
behaviour_2back <- behaviour %>% 
                   filter(Condition=='2-back') %>% 
                   select(-Condition)

# # Add 1-back/2-back labels to behaviour columns
colnames(behaviour_1back)[4:6] <- paste(colnames(behaviour_1back)[4:6], "1back", sep = "_")
colnames(behaviour_2back)[4:6] <- paste(colnames(behaviour_2back)[4:6], "2back", sep = "_")

# Calculate change from 1-back to 2-back
behaviour_2back_1back <- behaviour_2back[4:6] - behaviour_1back[4:6]
colnames(behaviour_2back_1back) <- paste(colnames(behaviour)[5:7], "delta", sep = "_")


behaviour_diff <- behaviour_1back %>% 
    left_join(behaviour_2back, by=c("Subject", "Session", "Group")) %>% 
    cbind(behaviour_2back_1back)

#--- Merging modulatiy and behaviour into one table

Q_beh <- left_join(Q_diff, behaviour_diff)
head(Q_beh)

#%% Step 2: Correlation of delta modularity and d-prime / pRT (first session)

naive_diff <- Q_beh %>% 
    filter(Session == 'Naive') %>% 
    filter(Subject != 'sub-21')
   # filter(!Subject %in% dualnback_motion)

cor.test(naive_diff$Q_2back_1back, naive_diff$pRT_delta)
cor.test(naive_diff$Q_2back_1back, naive_diff$Dprime_delta)

#%%
p <- ggplot(naive_diff, aes(y = Q_2back_1back, x = pRT_delta)) + 
    geom_point(col = '#daa03d',size = 4, alpha = 0.8) + 
    geom_smooth(
            method = 'lm', 
            col = '#262626ff', fill = '#daa03d', size = 1.5, alpha = 0.2) +
    theme_training + 
    xlab(expression(paste(Delta, " pRT"))) + 
    ylab(expression(paste(Delta, ' Normalized modularity')))
p

#ggsave("figures/Figure_modu_pRT_naive_corr.pdf", plot = p, width = 8, height = 6, dpi = 300)

#%%Step 3: Correlation of delta modularity and delta pRT change

#--- Preparing data
Q_beh <- Q_beh %>% 
               filter(!(Subject %in% dualnback_motion))

naive <- Q_beh %>% 
        filter(Session %in% c('Naive')) %>%
        select(-Session)

late <- Q_beh %>% 
        filter(Session %in% c('Late')) %>%
        select(-Session)

# Add naive/late labels to columns
colnames(naive)[3:14] <- paste(colnames(naive)[3:14], "naive", sep = "_")
colnames(late)[3:14] <- paste(colnames(late)[3:14], "late", sep = "_")

# Calculate change from Naive to Late
Q_beh_change <- late[3:14] - naive[3:14]
colnames(Q_beh_change) <- paste(colnames(Q_beh%>%select(-Session))[3:14], "change", sep = "_")

Q_beh_diff <- left_join(naive, late) %>% cbind(Q_beh_change)
Q_beh_diff_exp <- Q_beh_diff %>% filter(Group=='Experimental')

#%%

# 2-back pRT change vs. 2-back Q change:

cor.test(Q_beh_diff$pRT_2back_change, Q_beh_diff$Q_norm_2back_change) # all
cor.test(Q_beh_diff_exp$pRT_2back_change, Q_beh_diff_exp$Q_norm_2back_change) # experimental

cor.test(Q_beh_diff$Dprime_2back_change, Q_beh_diff$Q_norm_2back_change) # all
cor.test(Q_beh_diff_exp$Dprime_2back_change, Q_beh_diff_exp$Q_norm_2back_change) # experimental

#%%

p <- Q_beh_diff %>% ggplot(aes(x = pRT_2back_change, y = Q_norm_2back_change)) + 
    geom_point(col = '#daa03d',size = 4, alpha = 0.8) + 
    geom_smooth(
            method = 'lm', 
            col = '#262626ff', fill = '#daa03d', size = 1.5, alpha = 0.2) +
    theme_training + 
    xlab("pRT (2-back)") + 
    ylab("Normalized modularity (2-back)")
p

#%%
#----------DELETE THIS

#--- Preparing data
Q_beh <- Q_beh %>% 
               filter(!(Subject %in% dualnback_motion))

naive <- Q_beh %>% 
        filter(Session %in% c('Naive')) %>%
        select(-Session)

late <- Q_beh %>% 
        filter(Session %in% c('Early')) %>%
        select(-Session)

# Add naive/late labels to columns
colnames(naive)[3:14] <- paste(colnames(naive)[3:14], "naive", sep = "_")
colnames(late)[3:14] <- paste(colnames(late)[3:14], "late", sep = "_")

# Calculate change from Naive to Late
Q_beh_change <- late[3:14] - naive[3:14]
colnames(Q_beh_change) <- paste(colnames(Q_beh%>%select(-Session))[3:14], "change", sep = "_")

Q_beh_diff <- left_join(naive, late) %>% cbind(Q_beh_change)
Q_beh_diff_exp <- Q_beh_diff %>% filter(Group=='Experimental')

# 2-back pRT change vs. 2-back Q change:
cor.test(Q_beh_diff$pRT_2back_change, Q_beh_diff$Q_norm_2back_change) # all
cor.test(Q_beh_diff_exp$pRT_2back_change, Q_beh_diff_exp$Q_norm_2back_change) # experimental

# Delta pRT change vs. Delta Q change:
cor.test(Q_beh_diff$pRT_delta_change, Q_beh_diff$Q_2back_1back_change) # all
cor.test(Q_beh_diff_exp$pRT_delta_change, Q_beh_diff_exp$Q_2back_1back_change) # experimental

#%%







































