# load necessary libraries
library(gym)
library(tidyverse)

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

env_id <- "CartPole-v0"
instance_id <- env_create(client, env_id)

# function to determine action given state, model, action space and error
my_agent <- function(obs, mod, acts = 0:1, epsi = eps) {
  if(!is.null(mod)) {
    temp <- obs[nrow(obs), ] %>% 
      bind_rows(obs[nrow(obs), ]) %>% 
      mutate(action = acts)
    surv_prob <- predict(mod, temp)
    ought_to <- rank(surv_prob, ties.method = "random")[2] - 1
    ifelse(runif(1) > epsi, acts[acts %in% ought_to], acts[!(acts %in% ought_to)])
  } else sample(acts, 1)
}

# run to monitor algorithm
outdir <- "~/lm-agent-results"
env_monitor_start(client, instance_id, outdir, force = TRUE, resume = FALSE)


#### Start of algorithm ####

gam_pam <- 0.9  # discounting parameter
eps <- 0.1  # probability of not choosing the estimated best action (decrease over time?)

# create and clear objects to be used in algo
hist_data <- NULL
mod <- NULL
obs <- NULL
score <- NULL
mean_score <- 0
cnt <- 0
# start
while(cnt < 100 | mean_score < 195) {
  cnt <- cnt + 1
  action <- NULL
  obs <- as.data.frame(my_env_reset(client, instance_id))
  for(i in 1:200) {
    action[i] <- my_agent(obs = obs, mod = mod)
    results <- my_env_step(client, instance_id, action[i], render = TRUE)
    if(results$done | i == 200) break else
      obs <- bind_rows(obs, results$obs)
  }
  score <- c(score, nrow(obs))
  mean_score <- mean(tail(score, 100))
  
  episode_data <- obs %>% 
    mutate(action = action,
           reward = 1) %>% 
    mutate(q = rev(cumsum(reward * gam_pam^(0:(length(reward)-1))))) %>% 
    select(-reward)
  
  hist_data <- bind_rows(hist_data, episode_data)
  mod <- lm(q ~ .*., data = hist_data)
}
# end

# stop monitoring
env_monitor_close(client, instance_id)

# upload to openAI
upload(client,
       training_dir = outdir,
       api_key = "sk_mtuzQZRZS1eFaLkGpGZcbg")