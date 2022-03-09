
# set up R

if(!require(tidyverse)) install.packages('tidyverse')
if(!require(caret)) install.packages('caret')
if(!require(GGally)) install.packages('GGally')
if(!require(nnet)) install.packages('nnet')
if(!require(reshape2)) install.packages('reshape2')

library(tidyverse)
library(caret)
library(GGally)
library(nnet)
library(reshape2)

#### read in the data ####
# (from https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients )

raw <- read.csv("column_3c_weka.csv") %>% 
  mutate(class = as.factor(class)) #set the dependent variable to a factor

# build a boxplot of the raw data
ggplot(melt(raw), aes(variable, value)) + geom_boxplot() 
# it seems like there may be one error in the data in 'degree_spondylolisthesis' 
# where it is greater than 400. Maybe someone misplaced a decimal? Let's filter it out. 

# filter out the likely error
raw_clean <- raw %>% filter(degree_spondylolisthesis < 400) 

# build another boxplot - that looks much better
ggplot(melt(raw_clean), aes(variable, value)) + geom_boxplot() 

# create training and test sets
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = raw_clean$class, times = 1, p = 0.1, list = FALSE)
train_set <- raw_clean[-test_index,]
test_set <- raw_clean[test_index,]

# display some summary statistics of the training set
str(train_set)
summary(train_set)

#build a scatter plot matrix of the independent variables in the training set
pairs(train_set[,1:5], col = train_set$class)

#build a parallel coordinates plot of the independent variables in the training set
ggparcoord(train_set, columns = c(1:6), groupColumn = 7, scale = "uniminmax")

#### Build some models ####

#set up a results table to keep track of model accuracy 
results_table <- tribble(
  ~set, ~mode, ~regression, ~knn, ~random_forest,
  "train", 1, 1, 1, 1,
  "test", 1, 1, 1, 1
  )

# use the mode dependent variable in the training set as the predictor in the test set
train_freq <- train_set %>% count(class) %>% arrange(desc(n)) #find the mode
train_mu <- train_set %>% mutate(mu = train_freq[1,1]) 
confusionMatrix(train_mu$mu, train_set$class)$overall["Accuracy"]

# add the model accuracy to the results table
results_table[1,2] <- confusionMatrix(train_mu$mu, train_set$class)$overall[["Accuracy"]]



# build a multinominal logistic regression model using the training set 
# to predict the test set
train_multinom <- multinom(class ~ . , data = train_set)
confusionMatrix(predict(train_multinom, train_set), train_set$class)$overall["Accuracy"]

# add the model accuracy to the results table
results_table[1,3] <- confusionMatrix(predict(train_multinom, train_set), train_set$class)$overall[["Accuracy"]]


# fit a knn model to the training set
set.seed(1, sample.kind="Rounding")
train_knn <- train(class ~ ., method = "knn", 
                   data = train_set, 
                   tuneGrid = data.frame(k = seq(4, 44, 2)))
ggplot(train_knn, highlight = TRUE)

# display the model
train_knn

# display the maximum accuracy
max(train_knn$results$Accuracy)

# add the model accuracy to the results table
results_table[1,4] <- max(train_knn$results$Accuracy)



# fit a random forest model to the training set
set.seed(1, sample.kind="Rounding")
train_rf <- train(class ~., method = "rf", data = train_set)

train_rf

ggplot(train_rf, highlight = TRUE)
max(train_rf$results$Accuracy)

# add the model accuracy to the results table
results_table[1,5] <- max(train_rf$results$Accuracy)

#### Test the models ####

# use the mode in the training set to predict the test set and add the results to the table
test_mu <- test_set %>% mutate(mu = train_freq[1,1])
cm_mode <- confusionMatrix(test_mu$mu, test_set$class)
results_table[2,2] <-  cm_mode$overall["Accuracy"] 

# test the regression / "Nominal Logistic Regression" and add the results to the table
cm_regression <- confusionMatrix(predict(train_multinom, test_set), test_set$class)
results_table[2,3] <- cm_regression$overall["Accuracy"]

# test knn and add the results to the table
cm_knn <- confusionMatrix(predict(train_knn, test_set), test_set$class)
results_table[2,4] <- cm_knn$overall["Accuracy"]

# test random forest and add the results to the table
cm_rf <- confusionMatrix(predict(train_rf, test_set), test_set$class)
results_table[2,5] <- cm_rf$overall["Accuracy"]

results_table

# look at the confusion matrix tables for the three 'high-performing' models
cm_regression$table
cm_knn$table
cm_rf$table


