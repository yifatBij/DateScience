data_preparation <- function(dataset) {
  dataset[,"BUYER_FLAG"] = factor(dataset[,"BUYER_FLAG"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"BENEFIT_FLAG"] = factor(dataset[,"BENEFIT_FLAG"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"RETURN_FLAG"] = factor(dataset[,"RETURN_FLAG"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"CALL_FLAG"] = factor(dataset[,"CALL_FLAG"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"STATUS_PANTINUM"] = factor(dataset[,"STATUS_PANTINUM"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"STATUS_GOLD"] = factor(dataset[,"STATUS_GOLD"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"STATUS_SILVER"] = factor(dataset[,"STATUS_SILVER"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"CREDIT_PROBLEM"] = factor(dataset[,"CREDIT_PROBLEM"], levels = c(0,1), labels = c("No", "Yes"))
  dataset[,"REVIEW_RATING"] = factor(dataset[,"REVIEW_RATING"], levels = c("No", "Yes"), labels = c("No", "Yes"))

  return(dataset)
}

calc_ev <-function(y_hat_rf, threshold = 0.5) {
  y_tr = (testing$BUYER_FLAG == "Yes")
  y_hat_rf = (y_hat_rf > threshold)
  #' Calculate the confusion matrix manually
  tp_rf = sum(((y_hat_rf==TRUE) & (y_tr==TRUE)))   # true positive
  fp_rf = sum(((y_hat_rf==TRUE) & (y_tr==FALSE)))  # false positive
  fn_rf = sum(((y_hat_rf==FALSE) & (y_tr==TRUE)))  # false negative
  tn_rf = sum(((y_hat_rf==FALSE) & (y_tr==FALSE))) # true negative
  #Expected Value
  EV_rf=tp_rf*287+fp_rf*-35+0
  EV_rf
  print(sum(y_tr==TRUE))
  EV_OVR_rf=EV_rf/(sum(y_tr==TRUE)*287)#how much our profit comparing the maximal profit
  EV_OVR_rf
}

# Load the file and prepare the data
file_path_train = file.path(getwd(), "train_with_rating.csv")
file_path_rollout = file.path(getwd(), "rollout_with_rating.csv")

dataset_train = read.csv(file_path_train, na.strings = "")
dataset_rollout = read.csv(file_path_rollout, na.strings = "")

dataset_train = data_preparation(dataset_train)
dataset_rollout = data_preparation(dataset_rollout)

# Balance the data
library(DMwR)
balanced.data <- SMOTE(BUYER_FLAG ~.-ID, dataset_train, perc.over = 100)

# Run the selected model
library(caret)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
set.seed(225)
trainControl <- trainControl(method = "oob",
                            number = 5,
                            allowParallel = TRUE)
set.seed(135) 
model <- train(BUYER_FLAG ~., data = balanced.data, method = 'rf', 
                        trControl = trainControl) 
set.seed(131) 
prediction<- predict(model, newdata = dataset_rollout, type="prob")[,2]

labels = ifelse(prediction > 0.5, 1, 0)
dataset_rollout[,"BUYER_FLAG"] = labels
write.csv(dataset_rollout[,c("ID","BUYER_FLAG")], 'recommendations_oob.csv', row.names = FALSE)