# Exploratory data analysis of Titanic dataset
library(dplyr)
library(ggplot2)
library(stringr)

# load the test dataset
train_data <- read.csv(file = file.path('.','data','train.csv'), 
                       na.strings = c("NA", ""))

## Data Cleanup
str(train_data)

# convert column names to lowercase for consistency
names(train_data) <- tolower(names(train_data))

# convert to/from factors
train_data <- train_data %>% 
  mutate(name = as.character(name),
         pclass = factor(pclass),
         ticket = as.character(ticket),
         cabin = as.character(cabin),
         survived = factor(survived),
         title = str_extract(name, "[a-zA-z]+\\."))

#look for missing data
summary(train_data)
sapply(train_data, function(df){mean(is.na(df))})

# Have missing data for age, embarked, cabin.
