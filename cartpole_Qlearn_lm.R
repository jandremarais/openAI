library(gym)
library(tidyverse)

# adjust functions
my_env_reset <- function (x, instance_id) {
  route <- paste0("/v1/envs/", instance_id, "/reset/", sep = "")
  response <- post_request(x, route)
  observation <- response[["observation"]]
  names(observation) <- paste0("obs", 1:length(observation))
  observation
  #unlist(observation)
}

my_env_step <- function (x, instance_id, action, render = FALSE) {
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

remote_base <- "http://127.0.0.1:5000"  
client <- create_GymClient(remote_base) 

env_id <- "CartPole-v0"
instance_id <- env_create(client, env_id)

# function to determine action given state and model
my_agent <- function(obs, mod, acts = 0:1, epsi = eps) {
  if(!is.null(mod)) {
    temp <- obs[nrow(obs), ] %>% 
      bind_rows(obs[nrow(obs), ]) %>% 
      mutate(action = acts)
    #temp <- as.matrix(temp)
    surv_prob <- predict(mod, temp)
    ought_to <- rank(surv_prob, ties.method = "random")[2] - 1
    ifelse(runif(1) > epsi, acts[acts %in% ought_to], acts[!(acts %in% ought_to)])
  } else sample(acts, 1)
  
}

######################################
# Algo
######################################

outdir <- "C:/Users/Jan/Documents/openAI/tmp/random-agent-results"
env_monitor_start(client, instance_id, outdir, force = TRUE, resume = FALSE)
gam_pam <- 0.9
eps <- 0.1

hist_data <- NULL
mod <- NULL
obs <- NULL
score <- NULL
for(j in 1:200) {
  action <- NULL
  obs <- as.data.frame(my_env_reset(client, instance_id))
  for(i in 1:200) {
    action[i] <- my_agent(obs = obs, mod = mod)
    results <- my_env_step(client, instance_id, action[i], render = TRUE)
    if(results$done | i == 200) break else
      obs <- bind_rows(obs, results$obs)
  }
  score <- c(score, nrow(obs))
  # plot(score, type = "l",
  #      main = paste(mean(score[length(score):(ifelse(length(score)>100, 
  #                                                    length(score) - 100,
  #                                                    1))])))
  
  episode_data <- obs %>% 
    mutate(action = action,
           reward = 1) %>% 
    mutate(q = rev(cumsum(reward * gam_pam^(0:(length(reward)-1))))) %>% 
    select(-reward)
  
  hist_data <- bind_rows(hist_data, episode_data)
  mod <- lm(q ~ .*., data = hist_data)
  #mod <- glmnet(x = as.matrix(hist_data %>% select(-q)),
  # y = hist_data$q,
  # family = "gaussian")
  #eps <-eps/(exp(0.005*(i-1)))
  #if(score[length(score)] == max(score)) eps <- 0.999 * eps
  #print(eps)
}
env_monitor_close(client, instance_id)

upload(
  client,
  training_dir = 'C:\\Users\\Jan\\Documents\\openAI\\tmp\\random-agent-results',
  api_key = "sk_mtuzQZRZS1eFaLkGpGZcbg")