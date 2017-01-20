# first attempt at mountain car

# load necessary libraries
library(gym)
library(tidyverse)
library(glmnet)
library(FNN)

# adjust env_reset and env_step functions for custom output
my_env_reset <- function(x, instance_id) {
  route <- paste0("/v1/envs/", instance_id, "/reset/", sep = "")
  response <- post_request(x, route)
  observation <- response[["observation"]]
  names(observation) <- paste0("obs", 1:length(observation))
  observation
}

my_env_step <- function(x, instance_id, action, render = FALSE) {
  route <- paste0("/v1/envs/", instance_id, "/step/", sep = "")
  data <- list(instance_id = instance_id, action = action, 
               render = render)
  response <- post_request(x, route, data)
  observation <- response[["observation"]]
  names(observation) <- paste0("obs", 1:length(observation))
  reward <- unlist(response[["reward"]])
  done <- unlist(response[["done"]])
  info <- unlist(response[["info"]])
  list(obs = observation, reward = reward, done = done, 
       info = info)
}

# Environment setup on R side
remote_base <- "http://127.0.0.1:5000"  
client <- create_GymClient(remote_base) 

env_id <- "MountainCar-v0"
instance_id <- env_create(client, env_id)

# function to determine action given state, model, action space and error
my_agent <- function(obs, mod, acts = 0:2, epsi = eps) {
  if(!is.null(mod)) {
    temp <- bind_rows(lapply(acts, function(a) c(obs, action = a)))
    surv_prob <- predict(mod, temp)
    ought_to <- rank(surv_prob, ties.method = "random")[length(acts)] - 1
    ifelse(runif(1) > epsi, acts[acts %in% ought_to], acts[!(acts %in% ought_to)])
  } else sample(acts, 1)
}

gam_pam <- 0.9
eps <- 0.1

episode_data <- NULL
mod <- NULL
obs <- my_env_reset(client, instance_id)
for(i in 1:2000) {
  #action <- sample(0:2, 1)
  action <- my_agent(obs, mod)
  results <- my_env_step(client, instance_id, action, render = TRUE)
  update_list <- c(obs, action = action)
  if(!is.null(episode_data)) {
    nn_dist <- knnx.dist(episode_data %>% select(-reward, -disc_dist),
                         query = as.data.frame(update_list),
                         k = 1) %>% as.numeric()
    episode_data <- episode_data %>% 
      mutate(disc_dist = disc_dist + gam_pam^(nrow(.):1)*nn_dist)
    } else nn_dist <- 0
  
  if(results$done | i == 2000) break
  
  update_list <- c(update_list, reward = results$reward, 
                   disc_dist = nn_dist)
  episode_data <- episode_data %>% 
    bind_rows(update_list)
  
  obs <- results$obs 
  
  mod_data <- episode_data %>% 
        select(-reward)
  
  mod <- lm(disc_dist ~ .*., data = mod_data)
}

episode_data %>% 
  ggplot(aes(obs1, obs2)) +
  #geom_point(aes(color = factor(action)))
  geom_point(aes(color = disc_dist, shape = factor(action)))  

nrow(episode_data)






