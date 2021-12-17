#Supplementary statistical analyses of training-related changes of brain activity in preselected networks

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

#%%
# Setting working directory
setwd("~/Dropbox/Projects/LearningBrain/")

glm <- read.csv('data/neuroimaging/04-glm/glm_results.csv')
glm$Session <- factor(glm$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))

high_motion <- c('sub-13', 'sub-21', 'sub-23', 'sub-50')
glm <- glm %>% filter(!(Subject %in% high_motion))
unique_networks <- unique(glm$Network)


#Multilevel modeling for each large-scale system
mlm_stats <- data.frame()
mlm_params <- data.frame()

for (i in 1:length(unique_networks)){
    net = glm %>% filter(Network == unique_networks[i])
    
    baseline <- lme(Activation ~ 1, random = ~ 1 |Subject/Session, data = net, method = 'ML',  control=lmeControl(returnObject=TRUE))#control = list(opt = "optim"))
    session <- update(baseline, .~. + Session)
    group <- update(session, .~. + Group)
    session_group <- update(group, .~. + Session:Group)
    
    #print(i)
    stats <- anova(baseline, session, group, session_group)
    #print(stats)
    
    params_interaction <- summary(session_group)
    
    vector_stats <- c(i,
                  stats['L.Ratio'][4,], 
                  stats['p-value'][4,])

    vector_params <- c(i, 
                       params_interaction$tTable[6,1], params_interaction$tTable[7,1], params_interaction$tTable[8,1], # betas
                       params_interaction$tTable[6,4], params_interaction$tTable[7,4], params_interaction$tTable[8,4], # ttests
                       params_interaction$tTable[6,5], params_interaction$tTable[7,5], params_interaction$tTable[8,5]) # pvals

    mlm_stats <- rbind(mlm_stats, vector_stats)
    mlm_params <- rbind(mlm_params, vector_params)

    p <- ggplot(net, aes(x = Session, y = Activation, col = Group)) +
    stat_summary(fun.y = mean, geom = 'point', size = 3) +
    stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Group)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
    ylab(paste(unique_networks[i], 'activation (z-score)')) +    
    scale_colour_manual(values=c('#379dbc','#ee8c00')) +
    theme_training +
    theme(legend.position = "none") +
    xlab(' ') 
    print(p)         

     }

colnames(mlm_stats) <- c('i', 
                         'chi_interaction', 
                         'pval_interaction')
colnames(mlm_params) <- c('i',  
                          'beta12_interaction', 'beta13_interaction', 'beta14_interaction', 
                          'ttest12_interaction', 'ttest13_interaction', 'ttest14_interaction',
                          'pval12_interaction', 'pval13_interaction', 'pval14_interaction')

#%%
mlm_stats$pval_interaction_fdr <- p.adjust(mlm_stats$pval_interaction, method='fdr')
mlm_stats$net <- unique_networks[mlm_stats$i]
mlm_params$net <- unique_networks[mlm_params$i]

filter_stats <- mlm_stats %>% #filter(pval_interaction < 0.05) %>% 
    select(net, chi_interaction, pval_interaction, pval_interaction_fdr) 

filter_stats_all <- left_join(filter_stats, mlm_params[,2:11])
values <- format(round(filter_stats_all[,2:13],6), nsmall = 6)
cbind(filter_stats_all$net, values)




