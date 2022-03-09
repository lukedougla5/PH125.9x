
# setup R

if(!require(tidyverse)) install.packages('tidyverse')
if(!require(caret)) install.packages('caret')
if(!require(lubridate)) install.packages('lubridate')
if(!require(data.table)) install.packages('data.table')

library(tidyverse)
library(caret)
library(lubridate)
library(data.table)

options(pillar.sigfig = 6)


### Use the edX provided code to create the edx set and validation set 
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# download the file
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# read the files into the environment
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# using R 4.0 or later, clean up the formatting of the file
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# add in movie ratings
movielens <- left_join(ratings, movies, by = "movieId")

# count the number of users, movies, and genres
movielens %>% summarize(n_users = n_distinct(userId), 
                        n_movies = n_distinct(movieId), 
                        n_genres = n_distinct(genres))


# create validation set with 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# clean up the environment
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# clean up the data set: pull out rating year, movie year, and age at rating
edx_clean <- edx %>% 
  mutate(rating_year = year(as_datetime(timestamp)),
         movie_year = str_extract(title, pattern = "(\\(\\d{4}\\))"),
         movie_year = as.numeric(str_extract(movie_year, pattern = "(\\d{4})")),
         age_at_rating = rating_year - movie_year) %>% 
  filter(age_at_rating >= 0) # filter out a few with negative ages as errors in data

# build training and test sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx_clean$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx_clean[-test_index,]
test_set <- edx_clean[test_index,]

# create a histogram of the training set ratings
hist(train_set$rating)

# build a histogram of movie rating frequency
train_set %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 25) + 
  scale_x_log10()

# build a histogram of movies rated per user
train_set %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 25) + 
  scale_x_log10()

# define an RMSE function to evaluate the models
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#### Build a model that uses the average ####

# find the mean rating of the training set
mu <- mean(train_set$rating) 

# use this mean to predict the test set rating
rmse_table <- tibble(approach = "Use the average", rmse = RMSE(mu, test_set$rating))
rmse_table


#### Build a model that uses movie effects ####

# find the average rating of each movie in the training set
movieID_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# join the movie averages ratings to the test set and use them to predict ratings 
# in the test set
predicted_ratings <- test_set %>% 
  left_join(movieID_avgs, by = 'movieId') %>%
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), #impute missing movies 
                                                                  #with the mean
         pred = mu + b_i) %>% 
  pull(pred) 

# add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
          tibble(approach = "Add movie effects", 
                 rmse = RMSE(predicted_ratings, test_set$rating)))

#### Movie and User effects model ####
# find the average rating by each user in the training set
userID_avgs <- train_set %>% 
  left_join(movieID_avgs, by = 'movieId') %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

# join the movie averages ratings and user averages to the test set and use them to 
# predict ratings in the test set
predicted_ratings <- test_set %>% 
  left_join(movieID_avgs, by = 'movieId') %>%
  left_join(userID_avgs, by = 'userId') %>% 
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), #impute missing movies 
                                                                  #with the mean
         pred = mu + b_i + b_u) %>% 
  pull(pred) 

#add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
                        tibble(approach = "Movie and User effects", 
                               rmse = RMSE(predicted_ratings, test_set$rating)))
rmse_table


#### regularized movie effects model ####
# create a set of potential lambdas
lambdas <- seq(0, 10, 0.25)

# create movie averages and count the number of ratings per movie
train_sums <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

# test the lambdas to see how well the model predicts the test set for each
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(train_sums, by='movieId') %>% 
    mutate(b_i = s/(n_i+l),
           b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), #impute missing movies 
                                                                    #with the mean
           pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# store the best lambda (i.e. smallest RMSE) for use below
lambda <- lambdas[which.min(rmses)]
lambda


#use the best lambda to build the regularized movie averages
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

# predict ratings in the test set using regularized movie effects
predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>%
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), # impute missing movies 
                                                                  # with the mean
         pred = mu + b_i) %>% 
  pull(pred) 

#add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
                        tibble(approach = "Regularized movie effects only", 
                               rmse = RMSE(predicted_ratings, test_set$rating)))

#### Regularized movie effects and user effects ####

# predict ratings in the test set using regularized movie effects and user effects
predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>%
  left_join(userID_avgs, by = 'userId') %>% 
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), # impute missing movies 
                                                                  # with the mean
         pred = mu + b_i + b_u) %>% 
  pull(pred) 

# add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
                        tibble(approach = "Regularized movie effects and user effects", 
                               rmse = RMSE(predicted_ratings, test_set$rating)))
rmse_table


#### build on the course provided logic ####

# find influence of age of movie during rating
age_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  group_by(age_at_rating) %>% 
  summarize(b_a = mean(rating - mu - b_i - b_u))

# predict ratings in the test set using regularized movie, user, and age of movie
predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  left_join(age_avgs, by = 'age_at_rating') %>% 
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), #impute missing movies with the mean
         pred = mu + b_i + b_u + b_a, 
         pred = ifelse(is.na(pred), mu, pred)) %>% #use the mean if there's no match in the training set
  pull(pred)

# add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
                        tibble(approach = "Regularized movie, user, and age of movie effects", 
                               rmse = RMSE(predicted_ratings, test_set$rating))) 

# add in genres to movie averages
genre_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u))

# predict ratings in the test set using regularized movie, user, and genre of movie
predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  left_join(genre_avgs, by = 'genres') %>% 
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), # impute missing movies with the mean
         pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

# add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
                        tibble(approach = "Regularized movie, user, and genre of movie effects", 
                               rmse = RMSE(predicted_ratings, test_set$rating))) 

# try age and genres together
genre_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  left_join(age_avgs, by = 'age_at_rating') %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_a))

# test regularized movie, user, age, and genre of movie effects
predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  left_join(age_avgs, by = 'age_at_rating') %>% 
  left_join(genre_avgs, by = 'genres') %>% 
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), # impute missing movies with the mean
         pred = mu + b_i + b_u + b_a + b_g) %>%
  pull(pred)

# add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
                        tibble(approach = "Regularized movie, user, age, and genre of movie effects", 
                               rmse = RMSE(predicted_ratings, test_set$rating))) 
rmse_table

#### Adding user age - ####
# Maybe people get harsher as they watch more movies and develop their palate?

# find the first year each user rated a movie ('start')
train_start_year <- train_set %>% 
  group_by(userId) %>% 
  summarize(start = min(rating_year))

# find the years that a user had been rating a movie when they made a rating ('rater_age')
train_set_plus_age <- train_set %>% left_join(train_start_year) %>% 
  mutate(rater_age = rating_year - start)

# calculate 'user age' effects
rater_age_avgs <- train_set_plus_age %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  left_join(age_avgs, by = 'age_at_rating') %>% 
  left_join(genre_avgs, by = 'genres') %>% 
  group_by(rater_age) %>% 
  summarize(b_ra = mean(rating - mu - b_i - b_u - b_a - b_g))

# add user ages to the test set
test_start_year <- test_set %>% 
  group_by(userId) %>% 
  summarize(start = min(rating_year)) # one could also add back in the start years 
                                      # from the training set and find a 'true' 
                                      # user start year without overtraining, but 
                                      # they have been left separate here for 
                                      # simplicity's sake
test_set_plus_age <- test_set %>% left_join(test_start_year) %>% 
  mutate(rater_age = rating_year - start)

# predict test set ratings using regularized movie, user, age, genre, and rater age effects
predicted_ratings <- test_set_plus_age %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  left_join(age_avgs, by = 'age_at_rating') %>% 
  left_join(genre_avgs, by = 'genres') %>% 
  left_join(rater_age_avgs, by = 'rater_age') %>% 
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), # impute missing movies 
                                                                  # with the mean
         pred = mu + b_i + b_u + b_a + b_g + b_ra) %>%
  pull(pred)

# add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
                        tibble(approach = "Regularized movie, user, age, genre, and rater age effects", 
                               rmse = RMSE(predicted_ratings, test_set_plus_age$rating))) 
rmse_table

#### validation - how'd we do? ####
# pre-process validation set (adding movie age, remove errors)
validation_clean <- validation %>% 
  mutate(rating_year = year(as_datetime(timestamp)),
         movie_year = str_extract(title, pattern = "(\\(\\d{4}\\))"),
         movie_year = as.numeric(str_extract(movie_year, pattern = "(\\d{4})")),
         age_at_rating = rating_year - movie_year) %>% 
  filter(age_at_rating >= 0)

# calculate user ages for the validation set
validation_start_year <- validation_clean %>% group_by(userId) %>% 
  summarize(start = min(rating_year)) 
validation_set_plus_age <- validation_clean %>% left_join(validation_start_year) %>% 
  mutate(rater_age = rating_year - start)

# predict ratings in the validation set using the final model logic in methods section
predicted_ratings <- validation_set_plus_age %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>% 
  left_join(userID_avgs, by = 'userId') %>% 
  left_join(age_avgs, by = 'age_at_rating') %>% 
  left_join(genre_avgs, by = 'genres') %>% 
  left_join(rater_age_avgs, by = 'rater_age') %>% 
  mutate(b_i = replace(b_i, is.na(b_i), mean(b_i, na.rm = TRUE)), # impute missing movies 
                                                                  # with the mean
         pred = mu + b_i + b_u + b_a + b_g + b_ra) %>%
  pull(pred)

# add the RMSE to the table
rmse_table <- bind_rows(rmse_table,
                        tibble(approach = "vs. Validation set", 
                               rmse = RMSE(predicted_ratings, 
                                           validation_set_plus_age$rating)))
rmse_table


