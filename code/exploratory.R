# Exploratory data analysis of Titanic dataset
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)

# load the test dataset
train_data <- read_csv(file.path('.','data','train.csv'))

## Data Cleanup
str(train_data)

# convert column names to lowercase for consistency
names(train_data) <- tolower(names(train_data))

# convert to/from factors
train_data <- train_data %>% 
  mutate(pclass = factor(pclass),
         survived = factor(survived),
         title = str_extract(name, "[a-zA-z]+\\."))

#look for missing data
summary(train_data)
sapply(train_data, function(df){mean(is.na(df))})

# Have missing data for age, embarked, cabin.
# ignore cabin (missing 77% of data)

# impute Embarked:
table(train_data$embarked, useNA = "always")
# set missing data to S, as the most common.
train_data$embarked[which(is.na(train_data$embarked))] <- "S"

# impute ages to be the mean of people with same title:
tb <- cbind(train_data$age, train_data$title)
table(tb[is.na(tb[,1]),2])

# get the mean ages for each title
age_dist <- train_data %>% 
  group_by(title) %>% 
  summarize(n = n(),
            n_missing = sum(is.na(age)),
            perc_missing = 100*n_missing/n,
            mean_age = mean(age, na.rm = TRUE),
            sd_age = sd(age, na.rm = TRUE))

# missing data for Dr, Master, Miss, Mr, Mrs
# becaus so many values are missing, imput with values taken from 
# normal distribution

for (key in c("Dr.", "Master.", "Miss.", "Mr.", "Mrs.")) {
  idx_na <- which(train_data$title == key & is.na(train_data$age))
  age_idx <- which(age_dist$title == key)
  train_data$age[idx_na] <- rnorm(length(idx_na), 
                                  age_dist$mean_age[age_idx], 
                                  age_dist$sd_age[age_idx])
}

train_data %>% 
  group_by(title) %>% 
  summarize(ages = mean(age))
  
str(train_data)
# Plot passenger distributions:
ggplot(data = train_data, aes(x = sex)) +  geom_bar()
ggplot(data = train_data, aes(x = pclass)) + geom_bar()
ggplot(data = train_data, aes(x = embarked)) + geom_bar()
ggplot(data = train_data, aes(x = age)) + geom_histogram(binwidth = 10)
ggplot(data = train_data, aes(x = title)) + geom_bar()
ggplot(data = train_data, aes(x = fare)) + geom_histogram(binwidth = 5)
ggplot(data = train_data, aes(x = sibsp)) + geom_histogram(binwidth = 1)
ggplot(data = train_data, aes(x = parch)) + geom_histogram(binwidth = 1)
ggplot(data = train_data, aes(x = survived)) + geom_bar()

# flipping through plots, surprised to see Miss > Mrs.... explore
train_data %>% 
  filter(sex == "female") %>% 
  ggplot(aes(x = title)) + 
  geom_bar()

# how many travelling with parents?
train_data %>% 
  filter(title == "Miss.") %>% 
  ggplot(aes(x = parch)) + 
  geom_bar()

# Hmm.. most traveling without parents.... 
# how many have siblings?
train_data %>% 
  filter(title == "Miss.") %>% 
  ggplot(aes(x = sibsp)) + 
  geom_bar()

# most traveling without siblings

train_data %>% 
  filter(title == "Miss.") %>%
  ggplot(aes(x = age)) + 
  geom_histogram()

train_data %>% 
  filter(sex == "female") %>% 
  ggplot(aes(x = age, fill = title)) +
  geom_histogram(alpha = 0.6, binwidth = 5, position = "identity")

train_data %>% 
  filter(title == "Miss.") %>% 
  mutate(sibsp = factor(sibsp),
         parch = factor(parch)) %>% 
  ggplot(aes(x = age, fill = parch)) + 
  geom_histogram(alpha = 0.6,position = "identity", binwidth = 5) +
  facet_wrap(~ sibsp)

train_data %>% 
  summarize(max_parch = max(parch),
            max_sibs = max(sibsp))

# plot distributions of passengers, breakdown by sex:
ggplot(data = train_data, aes(x = pclass, fill = sex)) + 
  geom_bar(position = "dodge")

ggplot(data = train_data, aes(x = embarked, fill = sex)) + 
  geom_bar(position = "dodge")
ggplot(data = train_data, aes(x = age, fill = sex)) + 
  geom_histogram(position = "identity", alpha = 0.6)

ggplot(data = train_data, aes(x = title)) + geom_bar()

ggplot(data = train_data, aes(x = fare, fill = sex)) + 
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 5)

ggplot(data = train_data, aes(x = sibsp, fill = sex)) + 
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 5)
ggplot(data = train_data, aes(x = parch, fill = sex)) + 
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 5)

ggplot(data = train_data, aes(x = sibsp)) + geom_histogram(binwidth = 1)
ggplot(data = train_data, aes(x = parch)) + geom_histogram(binwidth = 1)
ggplot(data = train_data, aes(x = survived, fill = sex)) + 
  geom_bar(position = "fill")


# now make the same plots for survivors:
train_survived <- train_data %>% 
  filter(survived == 1)
# plot distributions of passengers, breakdown by sex:
ggplot(data = train_survived, aes(x = pclass, fill = sex)) + 
  geom_bar(position = "dodge")

ggplot(data = train_survived, aes(x = embarked, fill = sex)) + 
  geom_bar(position = "dodge")
ggplot(data = train_survived, aes(x = age, fill = sex)) + 
  geom_histogram(position = "identity", alpha = 0.6)

ggplot(data = train_survived, aes(x = title)) + geom_bar()

ggplot(data = train_survived, aes(x = fare, fill = sex)) + 
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 5)

ggplot(data = train_survived, aes(x = sibsp, fill = sex)) + 
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 5)
ggplot(data = train_survived, aes(x = parch, fill = sex)) + 
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 5)

ggplot(data = train_survived, aes(x = sibsp)) + geom_histogram(binwidth = 1)
ggplot(data = train_survived, aes(x = parch)) + geom_histogram(binwidth = 1)
ggplot(data = train_survived, aes(x = survived, fill = sex)) + 
  geom_bar(position = "fill")
