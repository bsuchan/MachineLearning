##This is the writeup on the project assignment for Machine Learning.
##It is about predicting how well people are performing fitness activities
## based on data from accelerometers on the belt, forearm, arm, and dumbbell
library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)
library(gbm)

file_training <- "pml-training.csv"
file_testing <- "pml-testing.csv"
data_training <-read.csv(file_training, na.strings=c("NA",""), header=TRUE)
data_testing <- read.csv(file_testing, na.strings=c("NA",""), header=TRUE)
col_names <- names(data_training)
print(col_names)
##Discard columns with NA
col_NA <-names(data_training[,colSums(is.na(data_training)) != 0])
data_training_noNA <-data_training[,colSums(is.na(data_training)) == 0]
data_training_noNA <-data_training[,!(names(data_training) %in% col_NA) ]
names_noNA <-names(data_training_noNA)
print(names_noNA)
##Inspecting the variables the first seven variables seem to be linked to logistics and not performance
data_training_prep <-data_training_noNA[,8:length(names_noNA)]
selected_pred <-names(data_training_prep)
print(selected_pred)
##Throw away the same columns/variables from test data
data_testing <- data_testing[,!names(data_testing) %in% col_NA]
col_testing <-names(data_testing)
print(col_testing)
data_testing_prep <-data_testing[,8:length(col_testing)]
col_testing_prep <-names(data_testing_prep)
print(col_testing_prep)
##Check dimensions of training and testing data after discarding NA items
print(dim(data_training_prep))
print(dim(data_testing_prep))
##Now that the columns have been downselected from training data, and matched in test data
##I will start fitting models to training data. 
set.seed(1026)
##I had to take a subdivide the data to reduce computation load
set.seed(666)
inTrainsub <- createDataPartition(y=data_training_prep$classe, p=0.4, list=FALSE)
data_training_prep_sub1 <- data_training_prep[inTrainsub,]
data_training_prep_sub2 <- data_training_prep[-inTrainsub,]
##Partitioning trainign data into trainign and Testing 
inTrain <-createDataPartition(y=data_training_prep_sub1$classe, p=0.6, list=FALSE)
data_training_sub1 <- data_training_prep_sub1[inTrain,]
data_testing_sub1 <- data_training_prep_sub1[-inTrain,]
##Since there are more than 2 classes I tried Tree classification.
set.seed(1026)
modFit <- train(data_training_sub1$classe ~ ., data = data_training_sub1 , method="rpart")
print(modFit)
fancyRpartPlot(modFit$finalModel)
predictions <-predict(modFit, newdata=data_testing_sub1)
##Confusion matrix
print(confusionMatrix(predictions, data_testing_sub1$classe))
## Then I used repeated cross validation with tree classification
fitControl <-trainControl(method = "repeatedcv",number = 20,repeats=3)
modFit <- train(data_training_sub1$classe ~ ., data = data_training_sub1 , method="rpart", trControl = fitControl)
print(modFit)
fancyRpartPlot(modFit$finalModel)
predictions <-predict(modFit, newdata=data_testing_sub1)
##Confusion matrix
print(confusionMatrix(predictions, data_testing_sub1$classe))
## Since Random Forest is supposed to have high accuracy, I tried it next
##with repeated crossvalidation to avoid overfitting. For the markup I couldnt use repeated
##cross validation because my Rsession crashed due to low memory
set.seed(1026)
modFit <-train(data_training_sub1$classe ~ ., method="rf",trControl = trainControl(method = "cv", number = 4), data = data_training_sub1)
print(modFit)

##I tried to use prox argument to get additional information and look at individual trees, but my
##computer ran out of memory
predictions <-predict(modFit, newdata=data_testing_sub1)
print(confusionMatrix(predictions, data_testing_sub1$classe))
##With 4 times repeated 10 folds crossvalidation I got 99% accuracy. Since random forest was giving a fairly very high accuracy, so I used this modelfit to predict the performance categories
##in the test data
print(predict(modFit, newdata=data_testing_prep))
## I wanted to test different K folds and leave one out for crossvalidation, but was struggling with
##computer memory at this point. I also tried gradient boosting and got an accuracy of .966-leaving this from the markdown
## once again because of computation speed and memory issues.
## I also tried model based predictions
modFit <-train(data_training_sub1$classe ~ ., method="lda",trControl = trainControl(method="cv",number=4), data = data_training_sub1)
print(modFit)
predictions <-predict(modFit, newdata=data_testing_sub1)
print(confusionMatrix(predictions, data_testing_sub1$classe))
##I also plotted the 2 variables pitch_forearm and roll_belt against each other, colored
##by Class but as can be seen from the plot there is no easy classifier
qplot(roll_belt,pitch_forearm,color=classe,data=data_training_sub1)
##From this plot it is not surprising that linear model based predictions has lower accuracy than random forest
##for this dataset
##At this point I called it a day
