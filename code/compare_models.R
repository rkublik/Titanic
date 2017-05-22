## Predictive Models for Titanic Survival

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(caret)
library(ranger)
library(e1071)
library(caTools)
library(pROC)
library(randomForest)
library(glmnet)

# Set seed for reproducible results
set.seed(234233343)

# load the test dataset
train_data <- read_csv(file.path('.','data','train.csv'))

# Preprocessing, data cleanup the same as for exploratory analysis
# initial pre-processing:
training_data <- train_data %>% 
  mutate(Survived = factor(Survived, levels = c(1, 0), labels = c("yes", "no")),
         Title = factor(str_extract(Name, "[a-zA-z]+\\.")))

# Convert variable names to lowercase
names(training_data) <- tolower(names(training_data))

# Fill in missing values:
#look for missing data
summary(training_data)
sapply(training_data, function(df){mean(is.na(df))})

# Have missing data for age, embarked, cabin.
# ignore cabin (missing 77% of data)

# impute Embarked:
table(training_data$embarked, useNA = "always")
# set missing data to S, as the most common.
training_data$embarked[which(is.na(training_data$embarked))] <- "S"

# impute ages to be the mean of people with same title:
tb <- cbind(training_data$age, training_data$title)
table(tb[is.na(tb[,1]),2])

# get the mean ages for each title
age_dist <- training_data %>% 
  group_by(title) %>% 
  summarize(n = n(),
            n_missing = sum(is.na(age)),
            perc_missing = 100*n_missing/n,
            mean_age = mean(age, na.rm = TRUE),
            sd_age = sd(age, na.rm = TRUE))
age_dist
# missing data for Dr, Master, Miss, Mr, Mrs
# because so many values are missing, impute with values taken from 
# normal distribution

for (key in c("Dr.", "Master.", "Miss.", "Mr.", "Mrs.")) { # can likely do this with group_by.  see Loan Prediction Problem...
  idx_na <- which(training_data$title == key & is.na(training_data$age))
  age_idx <- which(age_dist$title == key)
  training_data$age[idx_na] <- rnorm(length(idx_na), 
                                  age_dist$mean_age[age_idx], 
                                  age_dist$sd_age[age_idx])
}


## Further Data Cleanup:
# reduce the number of titles
training_data %>% 
  group_by(title, sex) %>% 
  summarize(n = n())

# Reduce to 4 titles: Mr, Mrs, Master, Miss: capture Gender, basic age
training_data <- training_data %>% 
  mutate(title = as.character(title),
         title = ifelse((title == "Dr.") & sex == "female", "Mrs.", title),
         title = ifelse(title == "Mlle.", "Miss.", title),
         title = ifelse((title != "Miss.") & sex == "female", "Mrs.", title),
         title = ifelse((title != "Master.") & sex == "male", "Mr.", title))


# split training_data into training, and testing sets for development
train_idx <- createDataPartition(training_data$survived, p = 0.7, list = FALSE)

data.train <- training_data[train_idx,]
data.test <- training_data[-train_idx,]


# Investigate which variables to include in the model:
# pclass:
table(data.train$survived, data.train$pclass)

# sex
table(data.train$survived, data.train$sex)

# title
table(data.train$survived, data.train$title)
# not important

# embarked
table(data.train$survived, data.train$embarked)

# age
ggplot(data.train, aes(x = survived, y = age)) + 
  geom_boxplot()
# doesn't seem to be important

# fare
ggplot(data.train, aes(x = survived, y = fare)) + 
  geom_boxplot()
# include... seems to be a difference between survival or not

# siblings
ggplot(data.train, aes(x = survived, y = sibsp)) + 
  geom_boxplot()
# not important


# parents:
ggplot(data.train, aes(x = survived, y = parch)) + 
  geom_boxplot()
# important

# Now look at each model in turn:

# setup a common training control:
train_control.class_probs <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 10,
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              verboseIter = TRUE)



##########################
# 1. logistic regression: 
##########################
# include all variables with full data for now (except id, ticket, cabin)
glm.model <- train(survived ~ pclass + sex + age + sibsp + parch + fare + embarked + as.integer(title), 
                   data = data.train,
                   method = "glm",
                   metric = "ROC",
                   trControl = train_control.class_probs,
                   family = binomial(link = "logit"))

# Determine the optimal threshold
glm.pred <- predict(glm.model, data.test, type = "prob")
glm.analysis <- roc(response = data.test$survived, predictor = glm.pred$yes)
glm.error <- cbind(glm.analysis$thresholds, glm.analysis$sensitivities + glm.analysis$specificities)
glm.thresh <- subset(glm.error, glm.error[,2] == max(glm.error[,2]))[,1]
glm.thresh

#Plot ROC Curve
plot(1-glm.analysis$specificities,glm.analysis$sensitivities,type="l",
     ylab="Sensitiviy",xlab="1-Specificity",col="black",lwd=2,
     main = "ROC Curve for Simulated Data")
abline(a=0,b=1)
abline(v = glm.thresh) #add optimal t to ROC curve

# Turn probabilities into classes, and look at frequencies:
glm.pred$survived <- ifelse(glm.pred[['yes']] > glm.thresh, 1, 0)
glm.pred$survived <- factor(glm.pred$survived, levels = c(1, 0), labels = c("yes", "no"))
#table(glm.pred$survived)
#table(glm.pred$survived, data.test[['survived']])

# Use caret's confusion matrix:
glm.pred$survived
data.test$survived
glm.conf <- confusionMatrix(glm.pred$survived, data.test$survived)
colAUC(as.integer(glm.pred$survived), data.test$survived, plotROC = TRUE)

# look at important parameters:
glm.imp <- varImp(glm.model, scale = FALSE)
plot(glm.imp)

glm.perf <- c(glm.conf$overall["Accuracy"],
              glm.conf$byClass["Sensitivity"],
              glm.conf$byClass["Specificity"])

###############
# Decision Tree
###############


###################################
# Random Forest: using caret ranger
###################################
ranger.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 10,
                              classProbs = TRUE,
                              verboseIter = TRUE)

ranger.model <- train(survived ~ pclass + sex + age + sibsp + parch + fare + embarked + title, 
                      data.train, 
                      method = "ranger",
                      tuneLength = 10,
                      trControl = ranger.control)

plot(ranger.model)
ranger.model$finalModel$confusion.matrix

# get predictions:
ranger.pred <- predict(ranger.model, data.test)

ranger.pred
ranger.conf <- confusionMatrix(ranger.pred, data.test$survived)
colAUC(as.integer(ranger.pred), data.test$survived, plotROC = TRUE)

ranger.perf <- c(ranger.conf$overall["Accuracy"],
                 ranger.conf$byClass["Sensitivity"],
                 ranger.conf$byClass["Specificity"])


rbind(ranger.perf, glm.perf)

##########
# glmnet #
##########
glmnet.control <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 10,
                               classProbs = TRUE,
                               verboseIter = TRUE)
                               
glmnet.model <- train(survived ~ pclass + sex + age + sibsp + parch + fare + embarked + title, 
                      data.train, 
                      method = "glmnet",
                      trControl = ranger.control)

plot(glmnet.model)

glmnet.pred <- predict(glmnet.model, data.test)
glmnet.pred

glmnet.conf <- confusionMatrix(glmnet.pred, data.test$survived)
colAUC(as.integer(glmnet.pred), data.test$survived, plotROC = TRUE)

glmnet.perf <- c(glmnet.conf$overall["Accuracy"],
                 glmnet.conf$byClass["Sensitivity"],
                 glmnet.conf$byClass["Specificity"])


rbind(ranger.perf, glm.perf, glmnet.perf)

#####
# SVM
#####

############################################
# Look at PCA, what variables are co-linear
############################################



