library(tidyverse)
library(caret)
library(lift)
library(ROCR)
library(rpart)
library(rpart.plot)
library(readr)
set.seed(69)
# DATA PREPARATION -------------------------------------------------------------

heart_failure_clinical_records_dataset <- read_csv("heart_failure_dataset.csv")
any(is.na(heart_failure_clinical_records_dataset))

heart_failure_clinical_records_dataset %>% rename(
    cpk = creatinine_phosphokinase,
    death = DEATH_EVENT
  ) %>% 
  select(-time) -> all_data

all_data$death = as.factor(all_data$death)


# DATA EXPLORATION AND VISUALIZATION -------------------------------------------

#Check Data Type 
unique(all_data$age)        #Continuous
unique(all_data$anaemia)    #Categorical
unique(all_data$cpk)        #Continuous
unique(all_data$diabetes)   #Categorical
unique(all_data$ejection_fraction)   #Continuous
unique(all_data$high_blood_pressure) #Categorical
unique(all_data$platelets)           #Continuous 
unique(all_data$serum_creatinine)    #Continuous 
unique(all_data$serum_sodium)        #Continuous
unique(all_data$sex)     #Categorical
unique(all_data$smoking) #Categorical 
unique(all_data$death)   #Response y is Categorical

#Check Linear? and visualization
ggplot(all_data) + 
  geom_point(aes(x = age, y = cpk, color = death))
ggplot(all_data) + 
  geom_point(aes(x = age, y = ejection_fraction, color = death))
ggplot(all_data) + 
  geom_point(aes(x = age, y = platelets, color = death))
ggplot(all_data) + 
  geom_point(aes(x = cpk, y = ejection_fraction, color = death))
ggplot(all_data) + 
  geom_point(aes(x = cpk, y = platelets, color = death))
ggplot(all_data) + 
  geom_point(aes(x = ejection_fraction, y = platelets, color = death))
ggplot(all_data) + 
  geom_point(aes(x = serum_creatinine, y = serum_sodium, color = death))
ggplot(all_data) + 
  geom_point(aes(x = cpk, y = serum_sodium, color = death))
ggplot(all_data) + 
  geom_point(aes(x = serum_creatinine, y = platelets, color = death))
ggplot(all_data) + 
  geom_point(aes(x = serum_creatinine, y = ejection_fraction, color = death))

#Check Skewed 
summary(all_data$death)
#Choose Model: Decision Tree


# MODEL EXPLANATION AND IMPLEMENT ----------------------------------------------

#Separate train and test data
n <- nrow(all_data)
test_index <- sample(1:n,size =  0.3*n) 
data_train <- all_data[-test_index,] #train 0.7
data_test <- all_data[test_index,]   #test 0.3

#Summary train and test dataset
summary(data_test$death)
summary(data_train$death)

#Train model
model1 <- rpart(death~., data = data_train)
rpart.plot(model1)
model1$variable.importance
summary(model1)$cptable

#Test Model
res1 <- predict(model1, data_test, type = 'class') 
#class type return factor ('0', '1')
head(res1)
confusionMatrix(res1, data_test$death, positive = "1")
confusionMatrix(res1, data_test$death, positive = "1", mode = 'prec_recall')


# MODEL EVALUATION -------------------------------------------------------------

#Lift Calculation of Model1
res1_pos <- predict(model1, data_test)[,'1']
lift_result1 <- data.frame(prob = res1_pos, y = data_test$death)
lift_obj1 <- lift(y ~ prob, data = lift_result1, class = '1')
plot(lift_obj1) #add parameter values (determine %sample found) to find %sample tested

pred1 <- prediction(res1_pos, data_test$death, label.ordering = c('0', '1'))
perf_lift1 <- performance(pred1, 'lift', 'rpp')
plot(perf_lift1)

tdl1 <- TopDecileLift(res1_pos, as.integer(data_test$death)-1) 
tdl1

#Cross Validation
train_control <- trainControl(method = 'cv', number = 10)
model_cv <- train(death~. ,
                  data = data_train, 
                  trControl = train_control, 
                  method = 'rpart')
model_cv
model_cv$finalModel #get new model


model_final <- model_cv$finalModel
rpart.plot(model_final)
model_final$variable.importance

#Test Model from CV
res2 <- predict(model_final, data_test, type = 'class')
head(res2)
confusionMatrix(res2, data_test$death, positive = "1")
confusionMatrix(res2, data_test$death, positive = "1", mode = 'prec_recall')
#------------------------------------------------------------------------------
#Cross Validation LOOCV
train_control_one <- trainControl(method = 'LOOCV', number = 10)
model_cv_one <- train(death~. ,
                  data = all_data, 
                  trControl = train_control_one, 
                  method = 'rpart')
model_cv_one
model_cv_one$finalModel #get new model


model_final_one <- model_cv_one$finalModel
rpart.plot(model_final_one)
model_final_one$variable.importance

#Test Model from CV
res2_one <- predict(model_final_one, data_test, type = 'class')
head(res2_one)
confusionMatrix(res2_one, data_test$death, positive = "1")
confusionMatrix(res2_one, data_test$death, positive = "1", mode = 'prec_recall')
#------------------------------------------------------------------------------

#Lift Calculation of Model from CV
res2_pos <- predict(model_final, data_test)[,'1']
lift_result2 <- data.frame(prob = res2_pos, y = data_test$death)
lift_obj2 <- lift(y ~ prob, data = lift_result2, class = '1')
plot(lift_obj2, values = 60) #add parameter values (determine %sample found) to find %sample tested

pred2 <- prediction(res2_pos, data_test$death, label.ordering = c('0', '1'))
perf_lift2 <- performance(pred2, 'lift', 'rpp')
plot(perf_lift2) 

tdl2 <- TopDecileLift(res2_pos, as.integer(data_test$death)-1) 
tdl2
