library(profvis)
library(lme4)
library(lmerTest)
library(MuMIn)
library(sjstats)
library(asremlPlus)
library(car)

#data <- read.csv('D:/Research/code/cross_project/regression2.csv')
profvis({data <- read.csv('D:/Projects/cp_jitdp/R/regression_data/regression_train_after_vif10.csv')
test <- read.csv('D:/Projects/cp_jitdp/R/regression_data//regression_test_after_vif10.csv')

#print(summary(data))
mlm_model <- lmer('roc_auc ~ n_commits + local_roc_auc + ratio +
                  prjA_popularity + prjB_popularity + prjA_age + prjB_age + 
                  same_owner_type + same_license + same_language + text_sim +
                  prjA_n_external + prjB_n_external + n_core_diff + n_external_diff +
                  n_intersection + contribution_entropy_diff + 
                  size_diff + dep_intersection + dep_diff + 
                  (1 | idx_s) + (1 | idx_t)', 
                  data = data) # 
#print(summary(mlm_model),correlation=TRUE)

#print(sum((fitted(mlm_model)-data['roc_auc'])^2))

pred <- predict(mlm_model,newdata=test,allow.new.levels=TRUE)
write.csv(pred, file='D:/pred10.csv')})
#print(predict(mlm_model,newdata=test,allow.new.levels=TRUE))

#r.squaredGLMM(mlm_model)
