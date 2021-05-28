#TRY TO MAKE IT IPYNB
#Static modularity analysis
#Last edited: 18-10-2020

#Step 0: Loading libraries
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

Q <- read.csv('data/neuroimaging/03-modularity/static/Q_normalized_power_tidy.csv')

# Preparing data
Q$Session <- factor(Q$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))
Q$Condition <- factor(Q$Condition, levels = c('Rest', '1-back', '2-back'))
Q$Group <- factor(Q$Group, levels = c('Control', 'Experimental'))

Q$Task[Q$Condition == '1-back'] <- 'N-back'
Q$Task[Q$Condition == '2-back'] <- 'N-back'
Q$Task[Q$Condition == 'Rest'] <- 'Rest'

# Subjects to excluede
dualnback_motion = c('sub-13', 'sub-21', 'sub-23', 'sub-50') # higly motion subjects in one of four sessions
rest_motion = c('sub-21', 'sub-46', 'sub-47') # higly motion subjects in one of four sessions / missing data(20-44)
rest_missing = c('sub-20', 'sub-44')

first_session_motion = 'sub-21' # subject with highly motion on first session

# Setting contrasts
rest_vs_dual <- c(-2, 1, 1)
back2_vs_back1 <- c(0, -1, 1) 

contrasts(Q$Condition) <- cbind(rest_vs_dual, back2_vs_back1)

head(Q)

#%%Step 2: Multilevel modeling (static modularity differences during first session)
naive <- Q %>% filter(Session == 'Naive') %>% 
    #filter(!Subject %in% dualnback_motion)
    filter(Subject != 'sub-21') 

baseline <- lme(Q_norm ~ 1, random = ~ 1|Subject/Condition, data = naive, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
condition <- update(baseline, .~. + Condition)

anova(baseline, condition)

#%%
summary(condition)

#%%
p <- ggplot(naive, aes(x = Condition, y = Q_norm, fill = Condition)) + 
    scale_colour_manual(values=c('#2e5d9f', '#919649', '#fc766a')) +
    scale_fill_manual(values=c('#2e5d9f', '#919649', '#fc766a')) +
    geom_jitter(aes(col = Condition),  alpha = 0.7, size = 4, width = 0.3, height = 0.0) +
    geom_boxplot(size = 1,  alpha = 0.3, outlier.shape = NA) +
    xlab('') +
    ylim(1.5, 5.2) +
    ylab('Normalized modularity') +
    ggtitle('') + 
    theme_training

p

ggsave("figures/Figure_modu_norm_naive.pdf", plot = p, width = 8, height = 6, dpi = 300)

#%%Step 3: Multilevel modeling (static modularity changes during training)

Q_clean <- Q %>% filter(Condition != 'Rest') %>% filter(!(Subject %in% dualnback_motion)) 
contrasts(Q_clean$Condition) <- cbind(back2_vs_back1)

baseline <- lme(Q_norm ~ 1, random = ~ 1|Subject/Session/Condition, data = Q_clean, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
condition <- update(baseline, .~. + Condition)
session <- update(condition, .~. + Session)
group <- update(session, .~. + Group)

condition_session <- update(group, .~. + Condition:Session)
condition_group <- update(condition_session, .~. + Condition:Group)
session_group <- update(condition_group, .~. + Session:Group)
condition_session_group <- update(session_group, .~. + Condition:Session:Group)

anova(baseline, condition, session, group, condition_session, condition_group, session_group, condition_session_group)


t.test(Q_clean$Q_norm ~ Q_clean$Group)

p <- ggplot(Q_clean, aes(x = Session, y = Q_norm, color = Condition)) +
    stat_summary(fun.y = mean, geom = 'point', size = 3) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Condition)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
    scale_colour_manual(values=c('#919649', '#fc766a')) +
    ylab('Normalized modularity') +
    facet_wrap(~Group) +
    xlab(' ') +
    theme_training
p

ggsave("figures/Figure_modu_power_groups.pdf", plot = p, width = 12, height = 6, dpi = 300)

experimental <- Q_clean %>% filter(Group == 'Experimental')
control <- Q_clean %>% filter(Group == 'Control') 
head(experimental)

#%%Step 4: T-tests (Naive vs. Late comparison)

p <- Q_clean %>% filter(Session %in% c('Naive', 'Late')) %>%
    ggplot(aes(x = Group, y = Q_norm, fill = Session)) + 
    geom_point(aes(col = Session), position=position_jitterdodge(dodge.width=0.9), alpha = 0.6, size = 4) +
    geom_boxplot(alpha = 0.4, outlier.shape = NA, position=position_dodge(width=0.8), size = 1) + 
    scale_fill_manual(values=c('#daa03d', '#755841')) +
    scale_color_manual(values=c('#daa03d', '#755841')) +
    facet_wrap(~Condition) +
    ylim(1.5, 5.5) +
    ylab('Normalized modularity') +
    xlab('') + 
    theme_training

p

ggsave("figures/Figure_modu_power_ttests.pdf", plot = p, width = 12, height = 6, dpi = 300)

#%%
library(plyr)

Q_nl <- Q_clean %>% filter(Session %in% c('Naive', 'Late'))
table <-  ddply(Q_nl, c('Condition', 'Session', 'Group'), summarise, 
                              N = length(Q_norm),
                              mean = mean(Q_norm),
                              sd = sd(Q_norm),
                              se = sd/sqrt(N)
                )

table

limits <- aes(ymax = table$mean + table$se,
              ymin = table$mean - table$se)

#%%
exp1 <- Q_nl %>% filter(Group == 'Experimental') %>% filter(Condition == '1-back')
exp2 <- Q_nl %>% filter(Group == 'Experimental') %>% filter(Condition == '2-back')
con1 <- Q_nl %>% filter(Group == 'Control') %>% filter(Condition == '1-back')
con2 <- Q_nl %>% filter(Group == 'Control') %>% filter(Condition == '2-back')

t.test(exp1$Q_norm ~ exp1$Session, paired = TRUE)
t.test(exp2$Q_norm ~ exp2$Session, paired = TRUE)
t.test(con1$Q_norm ~ con1$Session, paired = TRUE)
t.test(con2$Q_norm ~ con2$Session, paired = TRUE)

# Correct for 4 tests

#%%
# Comparison of groups in Naive session

naive_1 <- Q_nl %>% filter(Session == 'Naive') %>% filter(Condition == '1-back')
naive_2 <- Q_nl %>% filter(Session == 'Naive') %>% filter(Condition == '2-back')

t.test(naive_1$Q_norm ~ naive_1$Group, paired = FALSE)
t.test(naive_2$Q_norm ~ naive_2$Group, paired = FALSE)

# Correct for 2 test

#%%Step 4: Multilevel modeling (1-back vs. 2-back changes during training)

Q_1back <- Q %>% filter(Condition == '1-back') %>% select(Q_norm)
Q_2back <- Q %>% filter(Condition == '2-back') %>% select(Q_norm)
Q_diff <- Q_2back-Q_1back

colnames(Q_diff) <- 'Q_diff'

Q_diff_all <- Q %>% filter(Condition == '1-back') 
Q_diff_all <- cbind(Q_diff_all, Q_diff)

# Modelling
baseline <- lme(Q_diff ~ 1, random = ~ 1|Subject/Session, data = Q_diff_all, method = 'ML',  control = list(opt = "optim"), na.action = na.exclude)
session <- update(baseline, .~. + Session)
group <- update(session, .~. + Group)

session_group <- update(group, .~. + Session:Group)

anova(baseline, session, group, session_group)












































