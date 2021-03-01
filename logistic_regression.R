#1. Initialization
#Clean the work space #
rm(list = ls()) # :) remove all variables from global environment
cat("\014") # clear the screen
#  Get the data #

## Check where is the currently working directory path
getwd()
##read dataset while using a file path Create file path
file_path_Clients = file.path(getwd(),"ffp_train.csv") 
dataset_Clients = read.csv(file_path_Clients, na.strings="")

#when we check the structure of Data set we can see that some of our variables are considered as int and they should 
#factorial variables - we mean all the variables that their level are between 0 and 1:

data_preparation <- function(dataset) {
  dataset[,"BUYER_FLAG"] = factor(dataset[,"BUYER_FLAG"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"BENEFIT_FLAG"] = factor(dataset[,"BENEFIT_FLAG"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"RETURN_FLAG"] = factor(dataset[,"RETURN_FLAG"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"CALL_FLAG"] = factor(dataset[,"CALL_FLAG"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"STATUS_PANTINUM"] = factor(dataset[,"STATUS_PANTINUM"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"STATUS_GOLD"] = factor(dataset[,"STATUS_GOLD"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"STATUS_SILVER"] = factor(dataset[,"STATUS_SILVER"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"CREDIT_PROBLEM"] = factor(dataset[,"CREDIT_PROBLEM"], levels = c(0,1), labels = c("No", "Yes"))
  return(dataset)
}

calc_ev <-function(predict, threshold = 0.5) {
  y_tr = (testing$BUYER_FLAG == "Yes")
  predict = (predict > threshold)
  #' Calculate the confusion matrix manually
  tp_rf = sum(((predict==TRUE) & (y_tr==TRUE)))   # true positive
  fp_rf = sum(((predict==TRUE) & (y_tr==FALSE)))  # false positive
  fn_rf = sum(((predict==FALSE) & (y_tr==TRUE)))  # false negative
  tn_rf = sum(((predict==FALSE) & (y_tr==FALSE))) # true negative
  #Expected Value
  EV_rf=tp_rf*287+fp_rf*-35+0
  EV_rf
  print(sum(y_tr==TRUE))
  EV_OVR_rf=EV_rf/(sum(y_tr==TRUE)*287)#how much our profit comparing the maximal profit
  EV_OVR_rf
}
dataset_Clients = data_preparation(dataset_Clients)

#3.  divide the training data of the clients into 70%-30% of training and Testing
set.seed(11000)
p = runif(40000, 0, 1) # create uniformly distributed random variable in [0,1]in vector p

training = dataset_Clients[p < 0.7,]  # approximately %70 of Train-set    
testing  = dataset_Clients[p >= 0.7,] # approximately %30 of Train-set  

# package to compute 
# cross - validation methods 
library(caret)

# setting seed to generate a  
# reproducible random sampling 
set.seed(125)  

# as independent variable 
model_training <- train(BUYER_FLAG ~., data =training, method = 'glm') 
prop.table(table(training$BUYER_FLAG))
prop.table(table(testing$BUYER_FLAG))
#we can see that the class distribution of buyer flag is pretty similar between the training and testing. but the problem is that
#only 10% are true which make the data imbalance so we will try to solve it with SMOTE Function


library(DMwR)
balanced.data <- SMOTE(BUYER_FLAG ~. -ID, training, perc.over = 100)
summary(balanced.data)
model_smot <- train(BUYER_FLAG ~. -ID, data =balanced.data, method = 'glm')

library(FSelector)
weights <- information.gain(BUYER_FLAG~., data=balanced.data)
model_ig <- train(BUYER_FLAG ~ LAST_DEAL + ADVANCE_PURCHASE + NUM_DEAL + FARE_L_Y1 + FARE_L_Y4 + FARE_L_Y2, data =balanced.data, method = 'glm') 

library(mboost)
model_boost <- train(BUYER_FLAG ~., data =balanced.data, method = 'glmboost') 
model_boost_ig <- train(BUYER_FLAG ~LAST_DEAL + ADVANCE_PURCHASE + NUM_DEAL + FARE_L_Y1 + FARE_L_Y4 + FARE_L_Y2, data =balanced.data, method = 'glmboost') 


# printing model performance metrics 
# along with other details 
print(model_training)
print(model_smot)
print(model_ig)
print(model_boost)
print(model_boost_ig)

library(ROCR)

predict_training<- predict(model_training, newdata = testing, type="prob")[,2]
predict_smot<- predict(model_smot, newdata = testing, type="prob")[,2]
predict_ig<- predict(model_ig, newdata = testing, type="prob")[,2]
predict_boost<- predict(model_boost, newdata = testing, type="prob")[,2]
predict_boost_ig<- predict(model_boost_ig, newdata = testing, type="prob")[,2]

calc_ev(predict_training, 0.5)
calc_ev(predict_smot, 0.5)
calc_ev(predict_ig, 0.5)
calc_ev(predict_boost, 0.5)
calc_ev(predict_boost_ig, 0.5)

calc_ev(predict_training, 0.7)
calc_ev(predict_smot, 0.7)
calc_ev(predict_ig, 0.7)
calc_ev(predict_boost, 0.7)
calc_ev(predict_boost_ig, 0.7)
