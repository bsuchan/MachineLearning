library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)
#Read training and test datasets
file_training <- "pml-training.csv"
file_testing <- "pml-testing.csv"
data_training <-read.csv(file_training, na.strings=c("NA",""), header=TRUE)
data_testing <- read.csv(file_testing, na.strings=c("NA",""), header=TRUE)
col_names <- names(data_training)
print(col_names)
#Discard columns with NA
col_NA <-names(data_training[,colSums(is.na(data_training)) != 0])
data_training_noNA <-data_training[,colSums(is.na(data_training)) == 0]
data_training_noNA <-data_training[,!(names(data_training) %in% col_NA) ]
names_noNA <-names(data_training_noNA)
print(names_noNA)
#Inspecting the variables the first seven variables seem to be linked to logistics and not performance
data_training_prep <-data_training_noNA[,8:length(names_noNA)]
selected_pred <-names(data_training_prep)
print(selected_pred)
#Throw away the same columns/variables from test data
data_testing <- data_testing[,!names(data_testing) %in% col_NA]
col_testing <-names(data_testing)
print(col_testing)
data_testing_prep <-data_testing[,8:length(col_testing)]
col_testing_prep <-names(data_testing_prep)
print(col_testing_prep)
#Check dimensions of training and testing data after discarding NA items
print(dim(data_training_prep))
print(dim(data_testing_prep))
#Now that the columns have been downselected from training data, and matched in test data
#we will start fitting models to training data. 
set.seed(1026)
##Partitioning trainign data into trainign and Testing 
inTrain <-createDataPartition(y=data_training_prep$classe, p=0.6, list=FALSE)
data_training_sub1 <- data_training_prep[inTrain,]
data_testing_sub1 <- data_training_prep[-inTrain,]
##Since there are more than 2 classes we tried Tree classification and used repeated cross validation
fitControl <-trainControl(method = "repeatedcv",number = 20,repeats = 4)
modFit <- train(data_training_sub1$classe ~ ., data = data_training_sub1 , method="rpart", trControl = fitControl)
print(modFit)
fancyRpartPlot(modFit$finalModel)
predictions <-predict(modFit, newdata=data_testing_sub1)
##Confusion matrix
print(confusionMatrix(predictions, data_testing_sub1$classe))
##We also looked at PCA preprocessing to reduce number of variables
modFit <- train(data_training_sub1$classe ~ ., data = data_training_sub1 , method="rpart", preprocess="pca", trControl = fitControl)
print(modFit)
predictions <-predict(modFit, newdata=data_testing_sub1)
print(confusionMatrix(predictions, data_testing_sub1$classe))
##We got 95% accuracy and 94% concordance. we next we tried preprocessing
## Since Random Forest is supposed to have high accuracy, we tried it next with repeated crossvalidation to avoid overfitting
modFit <-train(data_training_sub1$classe ~ ., method="rf",trControl = trainControl(method="repeatedcv",number=4), data = data_training_sub1)
print(modFit)
fancyRpartPlot(modFit$finalModel)
##We tried to use prox argument to get additional information and look at individual trees, but my
##computer ran out of memory
predictions <-predict(modFit, newdata=data_testing_sub1)
print(confusionMatrix(predictions, data_testing_sub1$classe))
##We got a very high accuracy, so we used this modelfit to predict the performance categories
##in the test data
print(predict(modFit, newdata=data_testing_prep))
##Out of curiosity we also tried predicting based using boosted regression trees
modFit <-train(data_training_sub1$classe ~ ., method="gbm",trControl = trainControl(method="repeatedcv",number=4), data = data_training_sub1)
print(modFit)
predictions <-predict(modFit, newdata=data_testing_sub1)
print(confusionMatrix(predictions, data_testing_sub1$classe))
##It had a lower accuracy than Random forest
## We also tried model based predictions
modFit <-train(data_training_sub1$classe ~ ., method="lda",trControl = trainControl(method="repeatedcv",number=4), data = data_training_sub1)
print(modFit)
predictions <-predict(modFit, newdata=data_testing_sub1)
print(confusionMatrix(predictions, data_testing_sub1$classe))
##As expected linear model based predictions had much lower accuracy
##At this point I called it a day