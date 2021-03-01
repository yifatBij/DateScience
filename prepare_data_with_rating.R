#1. Initialization
#Clean the work space #
rm(list = ls()) # :) remove all variables from global environment
cat("\014") # clear the screen
#2. first we need to run on the review rolls out the model we buld in review project file #

## Check where is the currently working directory path
getwd()
##read dataset while using a file path Create file path

file_path_train_reviews = file.path(getwd(), "text_training.csv") #this is the dataset from homework assignment
dataset_reviews = read.csv(file_path_train_reviews, na.strings="")
dim(dataset_reviews)     # Dimensions of the dataset
dataset_reviews[,"rating"] = factor(dataset_reviews[,"rating"], levels = c(0,1), labels = c("No", "Yes"))#factoring Rating
class(dataset_reviews$rating) 
###############choose the right attributes

###############################fit the model to data


#3. Random Forest model version 1  we skip on logistic regression a we saw that random forest is better 
# package to compute 
# cross - validation methods 
library(randomForest)

# training the model by assigning buyer_flag column 
# as target variable and rest other column 
# as independent variable 
set.seed(2111)
mdl_rf_Reveiws = randomForest(
  rating ~ .-ID, 
  data=dataset_reviews, ntree=100)

#######now when have trained model we will run it on the review rollout data

##Add rating prediction to train and rollout reviews

file_path_reviews_training = file.path(getwd(), "reviews_training.csv") #this is the dataset for the project
reviews_training = read.csv(file_path_reviews_training, na.strings="")
file_path_reviews_rollout = file.path(getwd(), "reviews_rollout.csv") #this is the dataset for the project
reviews_rollout = read.csv(file_path_reviews_rollout, na.strings="")

reviews_training$REVIEW_RATING<-0 #we create a new column to store the rating results grade column
reviews_training$REVIEW_RATING<-predict(mdl_rf_Reveiws,newdata = reviews_training,type="response")
reviews_rollout$REVIEW_RATING<-0 #we create a new column to store the rating results grade column
reviews_rollout$REVIEW_RATING<-predict(mdl_rf_Reveiws,newdata = reviews_rollout,type="response")

# Export result to file
write.csv(reviews_training[,c("ID","REVIEW_RATING")], 'rating_train.csv', row.names = FALSE)
write.csv(reviews_rollout[,c("ID","REVIEW_RATING")], 'rating_rollout.csv', row.names = FALSE)

####now we add rge rating column to our train and rollout data
getwd()
##read dataset while using a file path Create file path
file_path_train_clients = file.path(getwd(),"ffp_train.csv") 
dataset_train_clients = read.csv(file_path_train_clients, na.strings="")
file_path_train_clients = file.path(getwd(),"ffp_rollout_x.csv") 
dataset_rollout_clients = read.csv(file_path_train_clients, na.strings="")
file_path_train_rating = file.path(getwd(),"rating_train.csv") 
dataset_train_rating = read.csv(file_path_train_rating, na.strings="")
file_path_train_rating = file.path(getwd(),"rating_rollout.csv") 
dataset_rollout_rating = read.csv(file_path_train_rating, na.strings="")

## Add rating column to table
library(tidyverse)
add_rating_to_table <- function(clients_table, rating_table) {
  data_with_rating <-left_join(clients_table, rating_table, by="ID")
  data_with_rating[is.na(data_with_rating$REVIEW_RATING), "REVIEW_RATING"] = "No"# the assumption we take here is that if we don't have value due it's should be no
  table(data_with_rating$REVIEW_RATING)
  
  class(data_with_rating$REVIEW_RATING)# we can see it consider it as character
  data_with_rating[,"REVIEW_RATING"] = factor(data_with_rating[,"REVIEW_RATING"], levels = c("No","Yes"), labels = c("No", "Yes"))
  class(data_with_rating$REVIEW_RATING)# we can see it consider it as factor
  table(data_with_rating$REVIEW_RATING)
  return(data_with_rating)
}

train_with_rating = add_rating_to_table(dataset_train_clients, dataset_train_rating)
rollout_with_rating = add_rating_to_table(dataset_rollout_clients, dataset_rollout_rating)
##create new files with rating
write.csv(train_with_rating, 'train_with_rating.csv', row.names = FALSE)
write.csv(rollout_with_rating, 'rollout_with_rating.csv', row.names = FALSE)

#################################################################