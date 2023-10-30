getwd()
#set the directory path
setwd("C:/Users/asus/Desktop/Data analyst/Projects and results of R/Project 2")

#Read the data
store_train=read.csv("store_train.csv",stringsAsFactors = F)
store_test=read.csv("store_test.csv",stringsAsFactors = F)

#import the library
library(dplyr)
library(stringr)
library(visdat)
library(ggplot2)
library(tidymodels)
library(car)
library(vip)
library(rpart.plot)

store_train$store=as.factor(as.numeric(store_train$store==1))

vis_dat(store_train)

#create the recipe
dp_pipe= recipe(store ~. , data= store_train) %>% 
  update_role(sales0,sales1,sales2,sales3,sales4,population,
              new_role = "to numeric")%>%  
  update_role(Id,new_role="drop_vars") %>% 
  step_novel(storecode) %>% 
  update_role(countyname,countytownname,Areaname,
              state_alpha,store_Type,new_role="to_dummies") %>%
  step_mutate_at(State, fn=function(x)as.numeric(as.character(x))) %>% 
  step_mutate_at(country, fn=function(x)as.numeric(as.character(x))) %>%
  step_mutate_at(CouSub, fn=function(x)as.numeric(as.character(x))) %>%
  step_rm(has_role("drop_vars")) %>% 
  step_mutate_at(has_role("to_numeric"), fn = as.numeric) %>%
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.05,other="__other__") %>%
  step_dummy(has_role("to_dummies"))  %>%
  step_impute_median(all_numeric(),-all_outcomes())

#prepare the recipe
dp_pipe=prep(dp_pipe)

#bake the recipe  
train=bake(dp_pipe, new_data = NULL)
test=bake(dp_pipe,new_data=store_test)

vis_dat(train)

# PREDICTIVE MODELLING USING LINEAR REGRESSION

# SPLITTING DATA SET

set.seed(1)

s=sample(1:nrow(train),0.8*nrow(train))

t1=train[s,]

t2=train[-s,]

# FITTING LINEAR MODEL TO TRAINING DATA SET
 
fit = lm(formula = store~ . -country -State 
         -countytownname_X__other__ -store_Type_Supermarket.Type2
         -store_Type_Supermarket.Type3 -store_Type_X__other__ 
         -store_Type_Supermarket.Type1 , data = t1)

summary(fit)

t2.pred = predict(fit , newdata=t2)

errors=t2$store-t2.pred

rmse=errors**2 %>% mean() %>% sqrt()

sort(vif(fit),decreasing = T)

fit=stats::step(fit)

summary(fit)

# PREDICTING TEST SET RESULT

y_pred = predict(fit , newdata = test)


## Decision Tree

tree_model=decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")


folds = vfold_cv(train, v = 5)

tree_grid = grid_regular(cost_complexity(), 
                         tree_depth(),
                         min_n(), 
                         levels = 3)
#View(tree_grid)
doParallel::registerDoParallel()
my_res=tune_grid(
  tree_model,
  store~.,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
  )


autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)
View(fold_metrics)
x=my_res %>% show_best()
write.csv(x,'Decision_Tree_Best5_roc_auc.csv',row.names = F)

final_tree_fit=tree_model %>% 
  finalize_model(select_best(my_res)) %>% 
  fit(store~.,data=train)

score=predict(final_tree_fit,new_data= test, type="prob")[,2]


# feature importance 

final_tree_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# plot the tree

rpart.plot(final_tree_fit$fit)

# predictions
train_pred=predict(final_tree_fit,new_data = train)
test_pred=predict(final_tree_fit,new_data = test)


# Random Forest
install.packages("ranger")

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,30)), trees(c(10,500)),
                       min_n(c(2,10)),levels = 3)

doParallel::registerDoParallel()
my_res1=tune_grid(
  rf_model,
  store~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)
autoplot(my_res1)+theme_light()

fold_metrics=collect_metrics(my_res1)
View(fold_metrics)
x=my_res1 %>% show_best()
write.csv(x,'Random_Forest_roc_auc.csv',row.names = F)

write.csv(fold_metrics,'fold_metrics.csv',row.names = F)


final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res1,"roc_auc")) %>% 
  fit(store~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons
train_pred=predict(final_rf_fit,new_data = train,type="prob") 
test_pred=predict(final_rf_fit,new_data = test,type="prob") 

score=predict(final_rf_fit,new_data= test, type="prob")[,2]


#predict on test data
test.rf.class=predict(final_rf_fit,new_data = test)
write.csv(test.rf.class,'Vishnupriya_samugari_P2_part2_.csv',row.names = F)


#XGBoost

library(xgboost)

xgb_spec = boost_tree(
  trees = 600,
  tree_depth = tune(), 
  min_n = tune(), 
  loss_reduction = tune(),                     
  sample_size = tune(), 
  mtry = tune(),         
  learn_rate = tune(),                         
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")


xgb_grid = grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), train),
  learn_rate(),
  size = 10
)
str(train)

xgb_grid

xgb_wf = workflow() %>%
  add_formula(store~.) %>%
  add_model(xgb_spec)
#xgb_wf

set.seed(2)
property_folds= vfold_cv(train, v=10)


set.seed(2)
xgb_res = tune_grid(
  xgb_wf,
  resamples = property_folds,
  grid = xgb_grid,
  control = control_grid(verbose = T)
)


collect_metrics(xgb_res)

y=show_best(xgb_res, "roc_auc")
write.csv(y,'XGBoost_Best5_roc_auc.csv',row.names = F)

best_roc_auc=select_best(xgb_res,"roc_auc")
show_best(xgb_res, "roc_auc")

#finalize xgboost model

final_xgb=finalize_workflow(xgb_wf,best_roc_auc)

#train final model using full training data
final_xgb_fit=final_xgb %>% fit(data=train) %>% extract_fit_parsnip()

final_xgb_fit %>% vip(goem="point")

#make forecast
test_forecast_xgb=predict(final_xgb_fit,new_data = test)
sum(test_forecast_xgb$.pred_class==1)
is.na(test_forecast_xgb)
write.csv(test_forecast_xgb,'Vishnupriya_samugari_P2.csv',row.names = F)