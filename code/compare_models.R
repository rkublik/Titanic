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

# Set seed for reproducible results
set.seed(234233343)

# load the test dataset
train_data <- read_csv(file.path('.','data','train.csv'))

# Preprocessing, data cleanup the same as for exploratory analysis
# Merge datasets, initial pre-processing
training_data <- train_data %>% 
  mutate(Pclass = factor(Pclass), 
         Survived = factor(Survived, levels = c(1, 0), labels = c("yes", "no")),
         Sex = factor(Sex),
         Embarked = factor(Embarked),
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
# becaus so many values are missing, imput with values taken from 
# normal distribution

for (key in c("Dr.", "Master.", "Miss.", "Mr.", "Mrs.")) {
  idx_na <- which(training_data$title == key & is.na(training_data$age))
  age_idx <- which(age_dist$title == key)
  training_data$age[idx_na] <- rnorm(length(idx_na), 
                                  age_dist$mean_age[age_idx], 
                                  age_dist$sd_age[age_idx])
}

# split training_data into training, and testing sets for development
train_idx <- createDataPartition(training_data$survived, p = 0.7, list = FALSE)

data.train <- training_data[train_idx,]
data.test <- training_data[-train_idx,]


# Now look at each model in turn:

# setup a common training control:
train_control.class_probs <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 10,
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              verboseIter = TRUE)
train_control <- trainControl(method = "repeatedcv",
                                          number = 10,
                                          repeats = 10,
                                          summaryFunction = twoClassSummary,
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
confusionMatrix(glm.pred$survived, data.test$survived)
colAUC(as.integer(glm.pred$survived), data.test$survived, plotROC = TRUE)

# look at important parameters:
glm.imp <- varImp(glm.model, scale = FALSE)
plot(glm.imp)

###################################
# Random Forest: using caret ranger
###################################
str(data.train)
ranger.model <- ranger(survived ~ pclass + sex + age + sibsp + parch + fare + embarked + title,
                      mtry = 4,
                      min.node.size = 12,
                      write.forest = TRUE,
                      data = data.train)
  
ranger.model <- ranger(survived ~ pclass + sex + age + sibsp + parch + fare + embarked + title,
                       mtry = 4,
                       min.node.size = 12,
                       write.forest = TRUE,
                       data = data.train)


rp.model <- rpart(survived ~ pclass + sex + age + sibsp + parch + fare + embarked + title, 
                  data = data.train)
plotcp(rp.model)
rpart.plot(ranger.model)
plot(ranger.model)
fuck you
