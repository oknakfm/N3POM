set.seed(123)
require(MASS)
require("parallel"); require("foreach"); require("doParallel"); require("doRNG")
require("ordinalNet")
require("abind"); require("Hmisc")
require("pdist"); require("serp")

# Directory
DIR = getwd()

## ==============
## preprocessing
## ==============
dataset.name = "autoMPG6"
# dataset.name = "real-estate"

## loading dataset
Z = read.csv(paste0(DIR,"/datasets/",dataset.name,".csv"), header=TRUE, sep=";")
if(dataset.name == "real-estate") Z = Z[-271,]

## rescaling
d = ncol(Z)-1; n = nrow(Z); J = 10
X = as.matrix(Z[,1:d]); X = scale(X, center=T, scale=T); 
g = as.vector(Z[,d+1])
X_upper = max(apply(X, 1, function(z) sqrt(mean(z^2)))) + 0.01
min_g = min(g); max_g = max(g)
g = g - min(g); g = g * ((J-1)/max(g)) + 1
cov_names = colnames(Z)[1:d]; res_name = colnames(Z)[d+1]
data = list(X=X, g=g, n=n, d=d, J=J, 
            cov_names = cov_names, res_name = res_name) 

## =============
##  Experiments
## =============

## Neural network parameters
L=50 #Number of hidden units
activation_type="sigmoid" 
n_GD = 5000
minibatch_size=16
decreasing_interval=50
decreasing_rate=0.95
R=20; n_initial=10; 

## Loading several functions
source(paste0(DIR,"/scripts/functions.R"), local=TRUE)

## Initialization
theta0 = theta_initialization(L=L, sd=initial_sd, data=data, type="serp")
g_discrete = round(data$g)

## N3POM
N3POM = NNOLM(data=data, L=L, 
               initial_sd=1, do_monitor=TRUE, exponent=0.5,
               shows_progress_bar=TRUE,
               n_GD=n_GD, 
               minibatch_size=minibatch_size, 
               theta0=theta0,
               decreasing_interval=decreasing_interval,
               decreasing_rate=decreasing_rate)

## =======
##  Plot
## =======
u = seq(1,J,0.05); lu = length(u)
a1 = (max_g-min_g)/(J-1); a2 = min_g-a1
u_original = a1 * u + a2; g_original = a1 * data$g + a2

NNa = a(u=u, alpha=phi_to_alpha(N3POM$theta$phi), j_seq=j_seq, derv_level=0)
NNb = neural_network(u=u, theta=N3POM$theta, derv_level=0)

par(mfrow=c(1,1+data$d))

## Loss trajectory (SGD)
plot(N3POM$monitor$iteration, N3POM$monitor$log_likelihood, 
     xlab="iteration", ylab="log-likelihood", type="l")

## Coefficient functions
for(k in 1:data$d){
  plot(u_original, -NNb[,k], xlab=data$res_name, ylab="sk(u)", main=data$cov_names[k],
       type="l")
}
