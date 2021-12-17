#Correlating behavioral variability with recruitment

library("tidyverse")

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

setwd("~/Dropbox/Projects/LearningBrain/")

performance_variability <- read.csv('data/behavioral/WM_fmri_behaviour_variability_tidy.csv')
networks_recruitment <- read.csv('data/neuroimaging/03-modularity/dynamic/04-recruitment_integration//whole-brain_power_normalized_recruitment_tidy.csv')

#%%
head(networks_recruitment)

#%%
head(performance_variability)

#%%
dualnback_exclude = c('sub-13', 'sub-21', 'sub-23', 'sub-50') # higly motion subjects in one of four sessions

performance_variability <- performance_variability %>% filter(!(Subject %in% dualnback_exclude))
networks_recruitment <- networks_recruitment %>% filter(!(Subject %in% dualnback_exclude))

performance_variability_diff <- performance_variability[performance_variability$Session=='Naive',] - 
                                performance_variability[performance_variability$Session=='Early',]
                                
#%%
unique_networks = unique(networks_recruitment$Network)

for (i in unique_networks){
    
    naive <- networks_recruitment %>% filter(Network==i) %>% filter(Session=='Naive')
    early <- networks_recruitment %>% filter(Network==i) %>% filter(Session=='Early')
    
    recruitment_diff <- early$Recruitment - naive$Recruitment
    print(i)
    #print(cor.test(performance_variability_diff$pRT_std, recritment_diff))
    
    recr_beh <- data.frame(cbind(performance_variability_diff$Dprime_std, recruitment_diff))
    results <- print(cor.test(performance_variability_diff$Dprime_std, recruitment_diff))
    colnames(recr_beh)[1] <- 'beh_var'
    print(results)
    
        if (results$p.value < 0.05){
        
            p <- recr_beh %>% ggplot(aes(beh_var , recruitment_diff)) +
            geom_point(size = 3, alpha = 0.8, col = '#393939') +
            geom_smooth(method='lm', col = '#393939', fill = 'darkgrey', size = 1.5, alpha = 0.3) +
            theme_training +
            ylab(expression(paste(Delta," Behavioral variability (", sigma, "d')"))) +
            xlab(expression(paste(Delta, ' Recruitment'))) 
            #ggtitle('All subjects')
            print(p)
        }
}


















