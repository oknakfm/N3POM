## ===========================================
## Please specify the path to NNOLM directory
DIR = paste0(GITHUB_DIR,"/NNOLM")
## ===========================================
setwd(DIR)

## =====================
## NN parameters
## =====================
## Parameters for NN
L = 100 #Number of hidden units
activation_type = "tanh" #"tanh", "sigmoid"

## Parameters for training
lr = 0.5 #Initial learning rate
n_GD = 2500 #Number of iterations for GD
sd = 0.1
minibatch_size = 32

## =====================
##  Experiments 
## =====================
test_ratio = 0.1 # = test/n
num_instances = 10
# dataset.names = c("cleveland", "newthyroid", "car", "winequality-white", "winequality-red","plastic","concrete","autoMPG6","autoMPG8","boston-housing")
dataset.names = "real-estate"


source(paste0(DIR,"/scripts/preprocessing_datasets.R"))

settings = expand.grid(1:num_instances, dataset.name)

## creating folders
dir.create(paste0(DIR,"/results"), showWarnings=FALSE)
for(id in 1:nrow(settings)){
  instance_id = settings[id,1]
  dataset.name = settings[id,2]
  dir.create(paste0(DIR,"/results/",dataset.name), showWarnings=FALSE)
  dir.create(paste0(DIR,"/results/",dataset.name,"/instance_id=",instance_id), showWarnings=FALSE)
}




instance_id = 1
DESTINATION_TO_SAVE = paste0(DIR,"/results/",dataset.name,"/instance_id=",instance_id)
load(paste0(DIR,"/datasets/preprocessed/",dataset.name,"/instance_id=",instance_id,".RData"))  


## Loading several functions
source(paste0(DIR,"/scripts/functions.R"), local=TRUE)

# weights (uniform or inverse)
zeta = weights(g=g, type="uniform")

## initialization
theta = theta_initialization(L=L, sd=sd, data=data, type="polr")
theta$W2 = W2_scale(theta=theta, X_upper=X_upper) * theta$W2
save(file=paste0(DESTINATION_TO_SAVE,"/iteration=0.RData"), theta)

## whether do we monitor the progress of GD
do_monitor = TRUE
if(do_monitor){
  monitor = vector("list",2);
  names(monitor) = c("iteration","log_likelihood")
}

## Optimization via full-batch gradient descent
for(iteration in 1:n_GD){
  
  minibatch_ind = sort(sample(1:n)[1:minibatch_size])
  
  theta = theta_sum(theta,
                    theta_scaling(log_likelihood(theta=theta, data=data, grad_level=1,
                                                 minibatch_ind = minibatch_ind,
                                                 gradient_direction=c("V1","V2","W1")), lr))
  
  theta = theta_sum(theta,
                    theta_scaling(log_likelihood(theta=theta, data=data, grad_level=1,
                                                 minibatch_ind = minibatch_ind,
                                                 gradient_direction="W2"), lr))
  
  coef = W2_scale(theta, X_upper=X_upper)
  theta$W2 = coef * theta$W2 ## rescaling of W2 for monotonicity constraint
  
  theta = theta_sum(theta,
                    theta_scaling(log_likelihood(theta=theta, data=data, grad_level=1,
                                                 minibatch_ind = minibatch_ind,
                                                 gradient_direction="alpha"), lr))
  
  save(file=paste0(DESTINATION_TO_SAVE,"/iteration=",iteration,".RData"), theta, coef)
  
  if(do_monitor && (iteration == 1 || iteration %% 10 == 0)){
    monitor$iteration = append(monitor$iteration, iteration)
    monitor$log_likelihood = append(monitor$log_likelihood,
                                    (m1 <- log_likelihood(theta=theta, data=data, grad_level=0)))
  }
  
  if(iteration %% 100 == 0) lr = lr * 0.8
}


plot_dataset(data=data)
plot_monitor(monitor)



u = seq(1,J,0.05)
beta = sapply(u, function(u) neural_network(u=u,theta=theta,derv_level=0))
plot_beta(beta, g=data$g, xax=u)


plot(sigmoid(phi_to_alpha(theta$phi)), type="l")

# plot_beta_pca(beta=beta)

