#SWITCH TO IPYNB
#Multilevel modeling on whole-brain normalized recruitment/integration
#Last edited: 20-10-2020

library(tidyverse)
library(nlme)
library(lattice)
library(RcppCNPy)
library(reticulate)

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

#%%Step1: Loading data

setwd("~/Dropbox/Projects/LearningBrain/")

# Load parcellation
parcellation <- 'power'
filename <- paste('whole-brain', parcellation,'normalized_mean_allegiance_tidy.csv', sep = '_')

networks_allegiance <- read.csv(paste0('data/neuroimaging/03-modularity/dynamic/04-recruitment_integration/', filename))
dualnback_exclude <- c('sub-13', 'sub-21', 'sub-23', 'sub-50') # higly motion subjects in one of four sessions
unique_networks <- unique(networks_allegiance$Network)
networks_allegiance$Session <- factor(networks_allegiance$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))
networks_allegiance$Group <- factor(networks_allegiance$Group, levels = c('Control', 'Experimental'))

networks_allegiance <- networks_allegiance %>% 
                       filter(!(Subject %in% dualnback_exclude))
head(networks_allegiance)

#%%Step 2: MLM modeling for all networks

mlm_stats <- data.frame()
mlm_params <- data.frame()

for (i in 1:length(unique_networks)){
    for (j in i:length(unique_networks)){
        
        network1 <- as.character(unique_networks[i])
        network2 <- as.character(unique_networks[j])
    
        net <- networks_allegiance %>% filter(Network == network1)
        variable <- net[[network2]]


        baseline <- lme(variable ~ 1, random = ~ 1 |Subject/Session, data = net, method = 'ML',  control=lmeControl(returnObject=TRUE))#control = list(opt = "optim"))
        session <- update(baseline, .~. + Session)
        group <- update(session, .~. + Group)
        session_group <- update(group, .~. + Session:Group)

        #print(paste(network1, network2))
        stats <- anova(baseline, session, group, session_group)
        params_session <- summary(session)
        params_interaction <- summary(session_group)
        
        vector_stats <- c(i, j, 
                          stats['L.Ratio'][2,], stats['L.Ratio'][3,], stats['L.Ratio'][4,], 
                          stats['p-value'][2,], stats['p-value'][3,], stats['p-value'][4,])
        
        vector_params <- c(i, j,
                           params_session$tTable[2,1], params_session$tTable[3,1], params_session$tTable[4,1], # betas
                           params_session$tTable[2,4], params_session$tTable[3,4], params_session$tTable[4,4], # ttests
                           params_session$tTable[2,5], params_session$tTable[3,5], params_session$tTable[4,5], # pvals
                           params_interaction$tTable[6,1], params_interaction$tTable[7,1], params_interaction$tTable[8,1], # betas
                           params_interaction$tTable[6,4], params_interaction$tTable[7,4], params_interaction$tTable[8,4], # ttests
                           params_interaction$tTable[6,5], params_interaction$tTable[7,5], params_interaction$tTable[8,5]) # pvals
                                    
        mlm_stats <- rbind(mlm_stats, vector_stats)
        mlm_params <- rbind(mlm_params, vector_params)
     }
}

colnames(mlm_stats) <- c('i', 'j', 
                         'chi_session', 'chi_group', 'chi_interaction', 
                         'pval_session', 'pval_group', 'pval_interaction')
colnames(mlm_params) <- c('i', 'j', 
                          'beta12_session', 'beta13_session', 'beta14_session', 
                          'ttest12_session', 'ttest13_session', 'ttest14_session',
                          'pval12_session', 'pval13_session', 'pval14_session',
                          'beta12_interaction', 'beta13_interaction', 'beta14_interaction', 
                          'ttest12_interaction', 'ttest13_interaction', 'ttest14_interaction',
                          'pval12_interaction', 'pval13_interaction', 'pval14_interaction')

#write.csv(mlm_stats, paste('data/neuroimaging/03-modularity/dynamic/04-recruitment_integration/whole-brain', parcellation, 'normalized_mlm_stats.csv', sep = "_"))
#write.csv(mlm_params, paste('data/neuroimaging/03-modularity/dynamic/04-recruitment_integration/whole-brain', parcellation,'normalized_mlm_params.csv', sep = "_"))

mlm_stats$pval_session_fdr <- p.adjust(mlm_stats$pval_session, method='fdr')
mlm_stats$pval_interaction_fdr <- p.adjust(mlm_stats$pval_interaction, method='fdr')

mlm_stats$net1 <- unique_networks[mlm_stats$i]
mlm_params$net1 <- unique_networks[mlm_params$i]
mlm_stats$net2 <- unique_networks[mlm_stats$j]
mlm_params$net2 <- unique_networks[mlm_params$j]


#%%Step 3: Planned contrasts and post-hocs:
    
library(emmeans)

network1 <- 'DM'
network2 <- 'FP'
net = networks_allegiance %>% filter(Network == network1)
variable <- net[[network2]]


baseline <- lme(variable ~ 1, random = ~ 1 |Subject/Session, data = net, method = 'ML',  control=lmeControl(returnObject=TRUE))#control = list(opt = "optim"))
session <- update(baseline, .~. + Session)
group <- update(session, .~. + Group)
session_group <- update(group, .~. + Session:Group)
anova(baseline, session, group, session_group)
summary(session_group)



contrast(emmeans(session_group, ~ Session*Group),
         interaction = c("consec"), adjust = "bonferroni")

emmeans(session_group, pairwise ~ Session|Group, adjust = "bonferroni")

#%%Line plots

mlm_stats$pval_session_fdr <- p.adjust(mlm_stats$pval_session, method = "fdr")


for (i in 1:nrow(mlm_stats)){
   
    network1 = as.character(unique_networks[mlm_stats$i[i]])
    network2 = as.character(unique_networks[mlm_stats$j[i]])
    measure <- 'integration'
    label <- paste0(network1, '-', network2)
               
    if (mlm_stats$pval_session_fdr[i] < 0.05){    
        net <- networks_allegiance %>% filter(Network == network1)
        variable <- net[[network2]]
        
    if (network1 == network2){
        measure <- 'recruitment'
        label <- network1
        }
        
        p <- ggplot(net, aes(x = Session, y = variable, col = Group)) +
            stat_summary(fun.y = mean, geom = 'point', size = 3) +
            stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Group)) +
            stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
            ylab(paste(label, measure)) +
            scale_colour_manual(values=c('#379dbc','#ee8c00')) +
            theme_training +
            theme(legend.position = "none") +
            xlab(' ') 
        print(p)
        
        #ggsave(paste0("figures/", paste0(label, '_', measure), ".pdf"), plot = p, width = 11, height = 5, dpi = 300)  
        
        }
}


#%%
interactions <- data.frame()

for (i in 1:nrow(mlm_stats)){
   
    network1 = as.character(unique_networks[mlm_stats$i[i]])
    network2 = as.character(unique_networks[mlm_stats$j[i]])
    measure <- 'integration'
    label <- paste0(network1, '-', network2)
               
    if (mlm_stats$pval_interaction[i] < 0.05){    
        net <- networks_allegiance %>% filter(Network == network1)
        variable <- net[[network2]]
        
        inter <- mlm_params[i,]
        inter <- cbind(networks = label, inter)
        #inter$networks <- label
        interactions <- rbind(interactions, inter) 
        
        
    if (network1 == network2){
        measure <- 'recruitment'
        label <- network1
        }
        
        p <- ggplot(net, aes(x = Session, y = variable, col = Group)) +
            stat_summary(fun.y = mean, geom = 'point', size = 3) +
            stat_summary(fun.y = mean, geom = 'line', size = 1.2, aes(group = Group)) +
            stat_summary(fun.data = mean_cl_boot, geom = 'errorbar', width = 0.2, size = 1.2) +
            ylab(paste(label, measure)) +
            scale_colour_manual(values=c('#379dbc', '#ee8c00')) +
            theme_training +
            theme(legend.position = "none") +
            xlab(' ') 
        print(p)
        
        #ggsave(paste0("figures/", paste0(label, '_', measure, '_interaction.pdf')), plot = p, width = 11, height = 5, dpi = 300)  
        
        }
}

#%%Step 4: Group comparison in 'Naive' session

naive_pval <- data.frame()
naive_session <- networks_allegiance %>% filter(Session=='Naive') 

for (i in 1:length(unique_networks)){
    for (j in i:length(unique_networks)){
        
        network1 <- as.character(unique_networks[i])
        network2 <- as.character(unique_networks[j])
    
        net <- naive_session %>% filter(Network == network1)

        stat <- t.test(net[[network2]] ~ net$Group, paired = FALSE)
        x <- c(i, j, stat$p.value, stat$statistic)
        #print(x)
        
        naive_pval <- rbind(naive_pval, x)
        colnames(naive_pval) <- c('i', 'j', 'pval', 't')
        
        }
}

naive_pval$net1 <- unique_networks[naive_pval$i] 
naive_pval$net2 <- unique_networks[naive_pval$j]
naive_pval$pval_fdr <- p.adjust(naive_pval$pval, method = 'fdr')

naive_pval %>% filter (pval < 0.05)


#%%Relationship between modularity change and DM recruitment change

Q <- read.csv('data/neuroimaging/03-modularity/static/Q_normalized_power_tidy.csv') %>% 
     filter(Condition %in% c('1-back', '2-back'))

# Preparing data
Q$Session <- factor(Q$Session, levels = c('Naive', 'Early', 'Middle', 'Late'))
Q$Condition <- factor(Q$Condition, levels = c('1-back', '2-back'))
Q$Group <- factor(Q$Group, levels = c('Control', 'Experimental'))

# Subjects to exclude
dualnback_motion = c('sub-13', 'sub-21', 'sub-23', 'sub-50')

Q_clean <- Q %>% filter(!(Subject %in% dualnback_motion)) %>% filter(Condition %in% c('2-back'))

network1 <- 'DM'
net <- networks_allegiance %>% filter(Network == network1) #%>% filter(Group=='Experimental')
cor.test(net$DM, Q_clean$Q_norm)

results <- data.frame(cbind(net$DM, Q_clean$Q_norm))
names(results)[1:2] <- c('Recruitment', 'Modularity')

p <- results %>% ggplot(aes(Recruitment, Modularity)) +
    geom_point(size = 3, alpha = 0.8, col = '#1f77b4') +
    geom_smooth(method='lm', col = '#1f77b4', fill = '#1f77b4', size = 1.5, alpha = 0.1) +
    theme_training +
    xlab('DM recruitment') +
    ylab('Q (2-back)')  +
    ggtitle('All subjects, all sessions')
p

ggsave("figures/DM_Q_corr_2back_all.pdf", plot = p, width = 12, height = 6, dpi = 300)

#%%
Q <- read_csv('/home/finc/Dropbox/Projects/LearningBrain/Q_beh_diff.csv')
network1 <- 'DM'
net = networks_allegiance %>% filter(Network == network1)
diff = net %>% filter(Session=='Late') %>% select(FP) - net %>% filter(Session=='Naive') %>% select(FP)
cor.test(diff$FP, Q$Q_norm_1back_change)


network1 <- 'DM'
#Q <- Q  %>% filter(Group=='Experimental')
net <- networks_allegiance %>% filter(Network == network1) #%>% filter(Group=='Experimental')
diff <- net %>% filter(Session=='Late') %>% 
        select(DM) - 
        net %>% 
        filter(Session=='Naive') %>% 
        select(DM)
cor.test(diff$DM, Q$Q_norm_1back_change)
cor.test(diff$DM, Q$Q_norm_2back_change)
cor.test(diff$DM, Q$Q_2back_1back_change)

#%%

results <- data.frame(cbind(diff$DM, Q$Q_norm_2back_change))
colnames(results)[1:2] <- c("Recruitment", "Modularity") 

p <- results %>% ggplot(aes(Recruitment, Modularity)) +
    geom_point(size = 3, alpha = 0.8, col = '#1f77b4') +
    geom_smooth(method='lm', col = '#1f77b4', fill = '#1f77b4', size = 1.5, alpha = 0.1) +
    theme_training +
    xlab(expression(Delta*' DM recruitment')) +
    ylab(expression(Delta*' Q (2-back)'))    
p

ggsave("figures/DM_Q_corr_2back.pdf", plot = p, width = 12, height = 6, dpi = 300)

#%% Relationship with behavioral change

#--- Behaviour
behaviour <- read.csv('./data/behavioral/WM_fmri_behaviour_mean_tidy.csv') %>% 
                       filter(!(Subject %in% dualnback_exclude))

# Behaviour 1-back
behaviour_1back <- behaviour %>% 
                   filter(Condition=='1-back') %>%
                   select(-Condition)

# Behaviour 2-back
behaviour_2back <- behaviour %>% 
                   filter(Condition=='2-back') %>% 
                   select(-Condition)

# Add 1-back/2-back labels to behaviour columns
colnames(behaviour_1back)[4:6] <- paste(colnames(behaviour_1back)[4:6], "1back", sep = "_")
colnames(behaviour_2back)[4:6] <- paste(colnames(behaviour_2back)[4:6], "2back", sep = "_")

# Calculate change from 1-back to 2-back
behaviour_2back_1back <- behaviour_2back[4:6] - behaviour_1back[4:6]
colnames(behaviour_2back_1back) <- paste(colnames(behaviour)[5:7], "delta", sep = "_")


behaviour_diff <- behaviour_1back %>% 
    left_join(behaviour_2back, by=c("Subject", "Session", "Group")) %>% 
    cbind(behaviour_2back_1back)

#%%
ses1 <- 'Naive'
ses2 <- 'Late'

behaviour_diff_1 <- behaviour_diff  %>% 
#            filter(Group == 'Experimental') %>% 
            filter(Session == ses1)

behaviour_diff_2 <- behaviour_diff  %>% 
#            filter(Group == 'Experimental') %>% 
            filter(Session == ses2)

networks_allegiance_1 <- networks_allegiance %>% 
#            filter(Group == 'Experimental') %>% 
            filter(Session == ses1)

networks_allegiance_2 <- networks_allegiance %>% 
#            filter(Group == 'Experimental') %>% 
            filter(Session == ses2)

beh1 <- behaviour_diff_1$Dprime_delta
beh2 <- behaviour_diff_2$Dprime_delta

beh_diff <- beh2 - beh1
#--------------------------------------------

results_df <- data.frame()

for (i in 1:length(unique_networks)){
    for (j in i:length(unique_networks)){
        
        network1 <- as.character(unique_networks[i])
        network2 <- as.character(unique_networks[j])
    
        net1 <- networks_allegiance_1  %>% 
            filter(Network == network1) 
        
        net2 <- networks_allegiance_2 %>% 
            filter(Network == network1)
        
        var1 <- net1[[network2]]
        var2 <- net2[[network2]]
        
        net_diff <- var2 - var1
        
        corr <- cor.test(beh_diff, net_diff, method = "pearson")        
        results <- c(i, j, corr$estimate, corr$p.value)
        results_df <- rbind(results_df, results)
    }
}

colnames(results_df)[1:4] <- c('i', 'j', 'estimate', 'pval')
results_df$net1 <- unique_networks[results_df$i]
results_df$net2 <- unique_networks[results_df$j]
results_df$pval_fdr <- p.adjust(results_df$pval)
results_df %>% filter(pval < 0.05) %>% select(net1, net2, estimate, pval, pval_fdr)

#%%
write.csv(results_df, 'correlation_naive_late_dprime_all.csv')
























