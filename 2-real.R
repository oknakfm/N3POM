# Directory
DIR = getwd()

DESTINATION_TO_SAVE = paste0(DIR,"/2-real")
DATASET_DIR = paste0(DIR,"/datasets")
PREPROCESSED_DIR = paste0(DIR,"/preprocessed")

dir.create(DESTINATION_TO_SAVE, showWarnings=FALSE)
dir.create(PREPROCESSED_DIR, showWarnings = FALSE)

require(MASS)
require("parallel"); require("foreach"); require("doParallel"); require("doRNG")
require("ordinalNet")
require("abind"); require("Hmisc")
require("pdist"); require("serp")


## Number of CPU cores used for parallel computation
N_CORES = detectCores()

## =======================
## preprocessing
## =======================

dataset.names = c("autoMPG6", "autoMPG8", "real-estate", 
                  "boston-housing", "concrete", "airfoil")

for(dataset.name in dataset.names){
  dir.create(DESTINATION_TO_SAVE,showWarnings=FALSE)
  
  set.seed(123)
  
  ## loading dataset
  Z = read.csv(paste0(DATASET_DIR,"/",dataset.name,".csv"), header=TRUE, sep=";")
  
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
              cov_names = cov_names, 
              res_name = res_name) # for training
  
  save(file=paste0(PREPROCESSED_DIR,"/",dataset.name,".RData"), 
         data, J, X_upper, min_g, max_g)
}

## ========================
## Experiments
## ========================
## parameters for NN
L = 50 #Number of hidden units
activation_type = "sigmoid" 

## parameters for training NN
n_GD = 5000; minibatch_size = 16

decreasing_interval = 50
decreasing_rate = 0.95

R=20; n_initial=10

for(dataset.name in dataset.names){

  load(paste0(PREPROCESSED_DIR,"/",dataset.name,".RData"))
  
  NN_settings = list(L=L, activation_type=activation_type, n_GD=n_GD, minibatch_size=minibatch_size, 
                     decreasing_interval = decreasing_interval, 
                     decreasing_rate = decreasing_rate)
  
  ## Loading several functions
  source(paste0(DIR,"/scripts/functions.R"), local=TRUE)
  
  set.seed(1)
  
  if(data$n>2000){
    data_for_init = data
    ind = sort(order(runif(data$n))[1:2000])
    data_for_init$X = data$X[ind,]
    data_for_init$g = data$g[ind]
    data_for_init$n = 2000
  }else{
    data_for_init = data
  }
  
  theta0 = theta_initialization(L=L, sd=initial_sd, data=data_for_init, type="serp")
  g_discrete = round(data_for_init$g)
  .polr = polr(formula = as.ordered(g_discrete) ~ data_for_init$X , method="logistic")
  
  save(file=paste0(DESTINATION_TO_SAVE, "/", dataset.name, ".RData"),  
       .polr, theta0)
  
  registerDoParallel(N_CORES)
  registerDoRNG(1)
  
  foreach(initial_id = 1:n_initial, .export=c("polr","ordinalNet","serp")) %dopar% {
    .nnolm = NNOLM(data=data, L=L, initial_sd=1, do_monitor=TRUE, exponent=0.5,
                   n_GD=n_GD, minibatch_size=minibatch_size, theta0=theta0,
                   decreasing_interval=decreasing_interval,
                   decreasing_rate=decreasing_rate)
    save(file=paste0(DESTINATION_TO_SAVE, "/", dataset.name, "_",initial_id,".RData"),  
         .nnolm)
  }
  stopImplicitCluster()
  
}

## =============================
##  Plot (Figure 2, 3(a), 4-7)
## =============================

J=10
n_initial = 10
colors = rainbow(n_initial)
source(paste0(DIR,"/scripts/functions.R"), local=TRUE)

for(dataset.name in dataset.names){
  load(paste0(PREPROCESSED_DIR,"/",dataset.name,".RData"))
  load(paste0(DESTINATION_TO_SAVE, "/", dataset.name,".RData"))
  
  u = seq(1,J,0.05); lu = length(u)
  a1 = (max_g-min_g)/(J-1); a2 = min_g-a1
  u_original = a1 * u + a2
  g_original = a1 * data$g + a2
  
  initial.beta = neural_network(u=u, theta=theta0, derv_level=0)
  polr.beta = -as.vector(.polr$coefficients)
  nnolm.beta = array(dim=c(n_initial, length(u), data$d))
  monitored_likelihood = NULL

  for(initial_id in 1:n_initial){
    load(paste0(DESTINATION_TO_SAVE, "/", dataset.name, "_",initial_id,".RData"))
    
    NNa = a(u=u, alpha=phi_to_alpha(.nnolm$theta$phi), j_seq=j_seq, derv_level=0)
    NNb = neural_network(u=u, theta=.nnolm$theta, derv_level=0)
    NNc = t(t(data$X %*% t(NNb)) + NNa)
    S = NNb * apply(activation(z=NNc, grad_level=1),2,mean)
    
    # nnolm.beta[initial_id,,] = S
    nnolm.beta[initial_id,,] = NNb
    monitored_likelihood = rbind(monitored_likelihood, .nnolm$monitor$log_likelihood)

    if(initial_id == 1) SGD_iteration = .nnolm$monitor$iteration
  }
  
  n_row = ceiling((data$d+1)/4)+1
  pdf(file=paste0(DESTINATION_TO_SAVE,"/exp_",dataset.name,".pdf"),
      height=3*n_row, width=3*4)
  par(mfrow=c(n_row,4))
  par(mar = c(5, 5, 3, 1), oma = c(0,0,0,0))  
  
  plot(x=0:6, y=0:6, type="n", ann=F, bty="n", xaxt = "n", yaxt = "n")
  text(x = 2,y = 4, "Dataset:", cex=2)
  text(x = 3,y = 2, dataset.name, cex=2)
  
  ### PCA plot for X
  X = prcomp(data$X)
  plot(X$x[,1], X$x[,2], xlab="PC1", ylab="PC2", main="PCA plot of X", 
       pch="+", cex.lab=1.5, cex.axis=1.5, cex.main=1.5)
  
  ### Histogram for h
  hist(g_original, breaks=50, xlab=paste0("Response (",data$res_name,")"), 
       main="Histogram of h",
       cex.lab=1.5, cex.axis=1.5, cex.main=1.5)
  
  ### process of learning
  xl = range(0,SGD_iteration); yl = range(monitored_likelihood)
  plot(SGD_iteration, monitored_likelihood[1,], type="n", 
       xlim=xl, ylim=yl, xlab="Iteration", ylab="log-likelihood",
       main="log-likelihood via MPS",
       cex.lab=1.5, cex.axis=1.5, cex.main=1.5)
  for(iteration_id in 1:n_initial){
    par(new=T)
    plot(SGD_iteration, monitored_likelihood[iteration_id,], 
         xlab=" ",ylab=" ",xlim=xl, ylim=yl, xaxt="n",yaxt="n", type="l",
         col=colors[iteration_id],cex.lab=1.5, cex.axis=1.5, cex.main=1.5)  
  }
  
  for(k in 1:data$d){
    tmp_beta = nnolm.beta[,,k]
    xl=range(u_original); yl=range(-tmp_beta, -initial.beta[,k], -polr.beta[k])
    plot(u_original, -initial.beta[,k],xlim=xl,ylim=yl,type="l",xlab=data$res_name,ylab=paste0("s",k,"(u) = -b",k,"(u)"),
         main=paste0(k,". ",data$cov_names[k]), col="black", lty=2, lwd=2,
         cex.lab=1.5, cex.axis=1.5, cex.main=1.5)
    par(new=T)
    plot(u_original, rep(-polr.beta[k], length(u_original)), type="l", col="grey", lty=4,
         xlim=xl,ylim=yl,xlab=" ",ylab=" ",xaxt="n",yaxt="n", lwd=2)
    for(iteration_id in 1:n_initial){
      par(new=T)
      plot(u_original, -tmp_beta[iteration_id,], type="l", col=colors[iteration_id],
           xlim=xl,ylim=yl,xlab=" ",ylab=" ",xaxt="n",yaxt="n", lwd=2)
    }
  }
  
  plot(0, type="n", ann=F, bty="n", xaxt = "n", yaxt = "n")
  legend("topleft", legend=c("POM","Initial state", paste0("N3POM (instance=",1:n_initial,")")), 
         col=c("grey","black",colors), bty="n",
         lty=c(4,2,rep(1,n_initial)), lwd=c(2,2,rep(1,n_initial)),cex=1.1, pt.cex=1
         )
  
  dev.off()
  
}

## ======================
##  Plot (Figure 8-12)
## ======================

require(psych)
for(dataset.name in dataset.names){
  load(paste0(PREPROCESSED_DIR,"/",dataset.name,".RData"))
  
  Z = cbind(data$X,data$g)
  colnames(Z)[1:data$d] = paste0(1:data$d, ". ",colnames(Z)[1:data$d])
  colnames(Z)[data$d+1] = paste0(data$res_name)
  
  if(data$d <= 8) lg = 960 else lg = 120*data$d
  png(file=paste0(DESTINATION_TO_SAVE,"/cor_",dataset.name,".png"), res=200,
      height=lg, width=lg)
  psych::pairs.panels(Z)
  dev.off()
}

## =======================
##  Plot (Figure 3(b))
## =======================

dataset.names = c("autoMPG6", "autoMPG8", "real-estate", 
                  "boston-housing", "concrete", "airfoil")

for(dataset.name in dataset.names){
  load(paste0(PREPROCESSED_DIR,"/",dataset.name,".RData"))

  d = data$d
  u = seq(1,J,0.05); lu = length(u)
  a1 = (max_g-min_g)/(J-1); a2 = min_g-a1
  u_original = a1 * u + a2
  g_original = a1 * data$g + a2
  
  ths <- a1 * c(-1,4,7,12) + a2
  dif <- 3 * a1
  
  pdf(file=paste0(DESTINATION_TO_SAVE,"/3L-",dataset.name,".pdf"), 
      width=16, height=4*ceiling(d/4))
  par(mfrow=c(ceiling(d/4),4))
  par(mar = c(5, 5, 3, 1), oma = c(0,0,0,0))  
  for(k in 1:d){
    xl = range(data$X[,k]); yl = range(g_original)
    plot(0, type="n", xlab=paste0(k, ". ",data$cov_names[k]), ylab=paste0("(response) ",data$res_name), 
         cex=2, cex.lab=2, cex.axis=2, xlim=xl, ylim=yl)
    
    xmin = xl[1]-dif; xmax = xl[2]+dif; ymin = ths[3] ; ymax=ths[4]; 
    z = cbind(c(xmin,ymin), c(xmax,ymin), c(xmax, ymax), c(xmin, ymax))
    polygon(x = z[1,], y = z[2,], col = "#1B98E0FF")
    
    xmin = xl[1]-dif; xmax = xl[2]+dif; ymin = ths[2] ; ymax=ths[3]; 
    z = cbind(c(xmin,ymin), c(xmax,ymin), c(xmax, ymax), c(xmin, ymax))
    polygon(x = z[1,], y = z[2,], col = "#1B98E080")
    
    xmin = xl[1]-dif; xmax = xl[2]+dif; ymin = ths[1] ; ymax=ths[2]; 
    z = cbind(c(xmin,ymin), c(xmax,ymin), c(xmax, ymax), c(xmin, ymax))
    polygon(x = z[1,], y = z[2,], col = "#1B98E00D")
    
    par(new=T)
    plot(data$X[,k], g_original, type="p", pch="+", xlab=" ", ylab=" ", xaxt="n", yaxt="n", 
         cex=2, cex.lab=2, cex.axis=2, xlim=xl, ylim=yl)
  }
  
  dev.off()  
}

