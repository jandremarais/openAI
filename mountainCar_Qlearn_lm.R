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

my_knn_agent <- function(obs, mod_data, acts = 0:2, epsi = eps) {
  if(!is.null(mod)) {
    temp <- bind_rows(lapply(acts, function(a) c(obs, action = a)))
    surv_prob <- knn.reg(train = mod_data %>% select(-q), 
                         y = mod_data$q,
                         test = temp, k = 10
                         )$pred
    ought_to <- rank(surv_prob, ties.method = "random")[length(acts)] - 1
    ifelse(runif(1) > epsi, acts[acts %in% ought_to], acts[!(acts %in% ought_to)])
  } else sample(acts, 1)
}

gam_pam <- 0.98
gam_pam2 <- 0.98
eps <- 0.2

all_episodes <- NULL
for(j in 1:100) {

  episode_data <- NULL
  mod <- NULL
  obs <- my_env_reset(client, instance_id)
  results <- NULL
  results$done <- FALSE
  while(!results$done) {
    #action <- sample(0:2, 1)
    #action <- my_agent(obs, mod)
    action <- my_knn_agent(obs, mod_data)
    results <- my_env_step(client, instance_id, action, render = TRUE)
    update_list <- c(obs, action = action)
    if(length(unique(all_episodes$action)) ==3) { #!is.null(episode_dat)
      # nn_dist <- mean((colMeans(bind_rows(all_episodes, episode_data) %>% 
      #                             select(-disc_reward, -disc_dist)) -
      #                    as.data.frame(update_list))^2)
      # nn_dist <- knnx.dist(
      #   bind_rows(all_episodes, episode_data) %>% 
      #     select(-disc_reward, -disc_dist),
      #   query = as.data.frame(update_list),
      #   k = 1) %>% as.numeric()
      nn_dist <- knnx.dist(
          bind_rows(all_episodes, episode_data) %>%
            filter(action == update_list$action) %>% 
            select(-disc_reward, -disc_dist, -action),
          query = as.data.frame(obs),
          k = 1) %>% as.numeric()
      episode_data <- episode_data %>% 
        mutate(disc_dist = disc_dist + gam_pam2^(nrow(.):1)*nn_dist) %>% 
        mutate(disc_reward = disc_reward + gam_pam^(nrow(.):1)*results$reward)
    } else nn_dist <- 0
    
    if(results$done) break
    
    update_list <- c(update_list, disc_reward = results$reward, 
                     disc_dist = nn_dist)
    episode_data <- episode_data %>% 
      bind_rows(update_list)
    
    obs <- results$obs 
    
    mod_data <- bind_rows(all_episodes, episode_data) %>% 
      mutate(q = 5*disc_dist + disc_reward) %>% #disc_dist + disc_reward
      select(-disc_reward, -disc_dist)
    
    #mod <- lm(q ~ .*., data = mod_data)
    #mod <- knn.reg(mod_data %>% select(-q), k = 10, y = mod_data$q)
  }
  print(nrow(episode_data))
  all_episodes <- bind_rows(all_episodes, episode_data)
  
  p <- mod_data %>% 
    ggplot(aes(obs1, obs2)) +
    geom_point(aes(color = q)) +
    facet_wrap(~action)
  print(p)
}


episode_data %>% 
  ggplot(aes(obs1, obs2)) +
  #geom_point(aes(color = factor(action)))
  geom_point(aes(color = disc_dist, shape = factor(action)))  

nrow(episode_data)

mod_data %>% 
  ggplot(aes(obs1, q)) +
  geom_smooth(aes(color = factor(action)))

mod_data %>% 
  ggplot(aes(obs2, q)) +
  geom_smooth(aes(color = factor(action)))

mod_data %>% 
  #mutate(pred = predict(mod, mod_data)) %>% 
  #mutate(pred = knn.pred) %>% 
  ggplot(aes(obs1, obs2)) +
  geom_point(aes(color = q)) +
  facet_wrap(~action)

mod <- glm(q ~ (obs1 + obs2 + action), data = mod_data)
predict(mod, mod_data)


lapply(0:2, function(a) {
  all_episodes %>% 
    filter(action == a) %>% 
    select(obs1, obs2) %>% 
    knnx.dist(query = as.data.frame(obs), k = 1) %>% 
    as.numeric()
}) %>% unlist %>% rank %>% extract2(2)

?extract2





