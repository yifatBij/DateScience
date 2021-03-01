#1. Initialization
#Clean the work space #
rm(list = ls()) # :) remove all variables from global environment
cat("\014") # clear the screen
#  Get the data #

## Check where is the currently working directory path
getwd()
##read dataset while using a file path Create file path
file_path_Clients = file.path(getwd(),"train_with_rating.csv") 
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
  dataset[,"REVIEW_RATING"] = factor(dataset[,"REVIEW_RATING"], levels = c("No","Yes"), labels = c("No", "Yes"))
  return(dataset)
}

calc_ev <-function(prediction, threshold = 0.5) {
  y_tr = (testing$BUYER_FLAG == "Yes")
  prediction = (prediction > threshold)
  #' Calculate the confusion matrix manually
  tp_rf = sum(((prediction==TRUE) & (y_tr==TRUE)))   # true positive
  fp_rf = sum(((prediction==TRUE) & (y_tr==FALSE)))  # false positive
  fn_rf = sum(((prediction==FALSE) & (y_tr==TRUE)))  # false negative
  tn_rf = sum(((prediction==FALSE) & (y_tr==FALSE))) # true negative
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
prop.table(table(training$BUYER_FLAG))
prop.table(table(testing$BUYER_FLAG))
#we can see that the class distribution of buyer flag is pretty similar between the training and testing. but the problem is that
#only 10% are true which make the data imbalance so we will try to solve it with SMOTE Function


library(DMwR)
balanced.data <- SMOTE(BUYER_FLAG ~., training, perc.over = 100)

summary(balanced.data)
#now we can see equal class distribution - OK
###################################################################

# package to compute 
# cross - validation methods 
library(caret)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
# setting seed to generate a  
# reproducible random sampling 
set.seed(125)  
# defining training control
cvControl5 <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)
cvControl10 <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)

train_control_oob_parallel <- trainControl(method = "oob",  
                                           number = 5, allowParallel = TRUE) 
train_control_oob_single <- trainControl(method = "oob",  
                                         number = 5) 

boot_train_control_single <- trainControl(method = "boot",  
                                          number = 5) 
boot_train_control_parallel <- trainControl(method = "boot",  
                                            number = 5, allowParallel = TRUE) 


# training the model by assigning bayer_flag column 
# as target variable and rest other column 
# as independent variable 
set.seed(555)
model_rf <- train(BUYER_FLAG ~.-ID, data = balanced.data, method = 'rf')
                              
set.seed(130) 
model_rf_boot_single <- train(BUYER_FLAG ~.-ID, data = balanced.data, method = 'rf', 
                  trControl = boot_train_control_single) 
set.seed(131) 
model_rf_boot_parallel <- train(BUYER_FLAG ~.-ID, data = balanced.data, method = 'rf', 
                              trControl = boot_train_control_parallel) 
set.seed(132) 
model_oob_rf_parallel <- train(BUYER_FLAG ~.-ID, data = balanced.data, method = 'rf', 
                  trControl = train_control_oob_parallel)
set.seed(133) 
model_oob_rf_single <- train(BUYER_FLAG ~.-ID, data = balanced.data, method = 'rf', 
                               trControl = train_control_oob_single)
set.seed(134) 
model_rf_cv_5 <- train(BUYER_FLAG ~.-ID, data = balanced.data, method = 'rf', 
                              trControl = cvControl5) 
set.seed(135) 
model_rf_cv_10 <- train(BUYER_FLAG ~.-ID, data = balanced.data, method = 'rf', 
                                trControl = cvControl10) 

# printing model performance metrics 
# along with other details 
print(model_oob_rf_single)
print(model_oob_rf_parallel)
print(model_rf_boot_single)
print(model_rf_boot_parallel)
print(model_rf_cv_5)
print(model_rf_cv_10)

library(ROCR)

predict_rf_oob_parallel<- predict(model_oob_rf_parallel, newdata = testing, type="prob")[,2]
predict_rf_oob_single<- predict(model_oob_rf_single, newdata = testing, type="prob")[,2]

predict_rf_boot_single<- predict(model_rf_boot_single, newdata = testing, type="prob")[,2]
predict_rf_boot_parallel<- predict(model_rf_boot_parallel, newdata = testing, type="prob")[,2]

predict_rf_cv_5<- predict(model_rf_cv_5, newdata = testing, type="prob")[,2]
predict_rf_cv_10<- predict(model_rf_cv_10, newdata = testing, type="prob")[,2]

#predict_lr_boot<- predict(model_lr, newdata = testing, type="prob")[,2]

calc_ev(predict_rf_boot_single, 0.6)
calc_ev(predict_rf_boot_parallel, 0.6)
calc_ev(predict_rf_oob_single, 0.6)
calc_ev(predict_rf_oob_parallel, 0.6)
calc_ev(predict_rf_cv_5, 0.6)
calc_ev(predict_rf_cv_10, 0.6)

calc_ev(predict_rf, 0.5)
calc_ev(predict_rf_boot_single, 0.5)
calc_ev(predict_rf_boot_parallel, 0.5)
calc_ev(predict_rf_oob_single, 0.5)
calc_ev(predict_rf_oob_parallel, 0.5)
calc_ev(predict_rf_cv_5, 0.5)
calc_ev(predict_rf_cv_10, 0.5)

calc_ev(predict_rf_boot_single, 0.4)
calc_ev(predict_rf_boot_parallel, 0.4)
calc_ev(predict_rf_oob_single, 0.4)
calc_ev(predict_rf_oob_parallel, 0.4)
calc_ev(predict_rf_cv_5, 0.4)
calc_ev(predict_rf_cv_10, 0.4)




