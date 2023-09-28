# Directory
DIR = getwd()
DESTINATION = paste0(DIR,"/1-synthetic")
dir.create(DESTINATION, showWarnings=FALSE)

require(MASS)
require("parallel"); require("foreach"); require("doParallel"); require("doRNG")
require("ordinalNet"); require("serp")
require("abind"); require("Hmisc")
require("progress")

## Number of CPU cores used for parallel computation
N_CORES = detectCores()

J = 7
R = (J-1)*4

## =================================
## synthetic dataset specification
## =================================
# generating the label g via specifying the distribution P(H|X)
a_synth <- function(u) 2*u - 9 
b_synth <- function(u,m=c(0.05,0.05)) c(-1 + m[1]*(u^2), 1 + m[2]*(u^2))

dataset_generation <- function(n=2000, m=c(0.05,0.05), Xtype=1){
  g_generation <- function(v=runif(n=1), x=c(0,0)){
    logit <- function(z) log(z) - log(1-z)
    bound_cut = function(z) min(max(z,1),J)
    z = -9 - x[1] + x[2] - logit(v)
    w = m[1]*x[1] + m[2]*x[2]
    if(w != 0){
      tmp = 4 - 4 * w * z
      if(tmp>0) u = (-2 + sqrt(tmp))/(2 * w) else u=J
    }else{
      u = -z/2
    }
    return(bound_cut(u))
  }
  
  if(Xtype==1){
    lengths = runif(n=n, min=0, max=1); angles = runif(n=n, min=0, max=2*pi)
    X = cbind(lengths * cos(angles), lengths * sin(angles))
  }else{
    X = matrix(rbeta(2*n,shape1=0.5,shape2=0.5),n,2)
  }
  colnames(X) = c("X1","X2")
  
  g = sapply(1:n, function(i) g_generation(x=X[i,]))
  data = list(X=X, g=g, n=n, d=2, J=J)
  return(data)
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

settings = list(L=L, activation_type=activation_type, n_GD=n_GD, minibatch_size=minibatch_size, 
                decreasing_interval = decreasing_interval, 
                decreasing_rate = decreasing_rate)
continuization = FALSE

## Loading several functions
source(paste0(DIR,"/scripts/functions.R"), local=TRUE)
source(paste0(DIR,"/scripts/myplots.R"), local=TRUE)

exp_settings = rbind(c(0.05, -0.05), c(0.05, 0.05), c(0.05,0), c(0,0))

n_exp = 20

for(exp_setting_id in 1:nrow(exp_settings)){
  set.seed(123)
  
  TMP_DESTINATION = paste0(DIR,"/1-synthetic/setting_id=",exp_setting_id)
  dir.create(TMP_DESTINATION, showWarnings=FALSE)
  m = exp_settings[exp_setting_id,]
  
  registerDoParallel(N_CORES)
  registerDoRNG(123)
  
  foreach(exp_id = 1:n_exp, .export=c("polr","ordinalNet","serp")) %dopar% {

    data = data_discrete = dataset_generation(n=1000, m=m, Xtype=1)
    g_discrete = round(data$g); data_discrete$g = g_discrete ## discretized

    .nnolm = NNOLM(data=data, L=L, initial_sd=1, do_monitor=TRUE, exponent = 0.5,
                   n_GD=n_GD, minibatch_size=minibatch_size, 
                   decreasing_interval=decreasing_interval,
                   decreasing_rate=decreasing_rate)
    
    .nnolm.d = NNOLM(data=data_discrete, L=L, initial_sd=1, do_monitor=TRUE, exponent = 0.5,
                            n_GD=n_GD, minibatch_size=minibatch_size, 
                            decreasing_interval=decreasing_interval,
                            decreasing_rate=decreasing_rate)

    .nnolm.d.c = NNOLM(data=data_discrete, L=L, initial_sd=1, do_monitor=TRUE, exponent = 0.5,
                            n_GD=n_GD, minibatch_size=minibatch_size, 
                            decreasing_interval=decreasing_interval,
                            decreasing_rate=decreasing_rate,
                            continuization = TRUE)
    
    .polr = polr(formula = as.ordered(g_discrete) ~ data$X , method="logistic")
    .oNet.ridge = ordinalNet(x = data$X, y = as.ordered(g_discrete),
                                  family = "cumulative", link = "logit",
                                  parallelTerms = FALSE, 
                                  nonparallelTerms=TRUE, alpha=0)
    .oNet.half = ordinalNet(x = data$X, y = as.ordered(g_discrete),
                             family = "cumulative", link = "logit",
                             parallelTerms = FALSE, 
                             nonparallelTerms=TRUE, alpha=0.5)
    .oNet.lasso = ordinalNet(x = data$X, y = as.ordered(g_discrete),
                             family = "cumulative", link = "logit",
                             parallelTerms = FALSE, 
                             nonparallelTerms=TRUE, alpha=1)
    
    .pNet.ridge = ordinalNet(x = data$X, y = as.ordered(g_discrete),
                             family = "cumulative", link = "logit",
                             parallelTerms = TRUE, 
                             nonparallelTerms=TRUE, alpha=0)
    .pNet.half = ordinalNet(x = data$X, y = as.ordered(g_discrete),
                            family = "cumulative", link = "logit",
                            parallelTerms = TRUE, 
                            nonparallelTerms=TRUE, alpha=0.5)
    .pNet.lasso = ordinalNet(x = data$X, y = as.ordered(g_discrete),
                             family = "cumulative", link = "logit",
                             parallelTerms = TRUE, 
                             nonparallelTerms=TRUE, alpha=1)

    .serp = serp(formula = as.ordered(g_discrete) ~ data$X, 
                 link = "logit", slope = "penalize")
    n.serp = length(coef(.serp))/3
    
    u = seq(1,J,0.05); lu = length(u)
    
    ## true coefficients
    true.beta = t(sapply(u,function(u) b_synth(u,m)))
      
    nnolm.beta = neural_network(u=u, theta=.nnolm$theta, derv_level=0)
    nnolm.d.beta = neural_network(u=u, theta=.nnolm.d$theta, derv_level=0)
    nnolm.d.c.beta = neural_network(u=u, theta=.nnolm.d.c$theta, derv_level=0)
    
    levels = as.numeric(names(table(g_discrete)))
    
    polr.beta = t(matrix(rep(-as.vector(.polr$coefficients), lu), 2, lu))
    oNet.ridge.beta = t(rbind(interpolator(u=u, levels=levels, beta=(tmp <- as.matrix(coef(.oNet.ridge, matrix=TRUE)[-1,]))[1,])
            ,interpolator(u=u, levels=levels, beta=tmp[2,])))
    oNet.half.beta = t(rbind(interpolator(u=u, levels=levels, beta=(tmp <- as.matrix(coef(.oNet.half, matrix=TRUE)[-1,]))[1,])
                              ,interpolator(u=u, levels=levels, beta=tmp[2,])))
    oNet.lasso.beta = t(rbind(interpolator(u=u, levels=levels, beta=(tmp <- as.matrix(coef(.oNet.lasso, matrix=TRUE)[-1,]))[1,])
                             ,interpolator(u=u, levels=levels, beta=tmp[2,])))
    

    pNet.ridge.beta = t(rbind(interpolator(u=u, levels=levels, beta=(tmp <- as.matrix(coef(.pNet.ridge, matrix=TRUE)[-1,]))[1,])
                              ,interpolator(u=u, levels=levels, beta=tmp[2,])))
    pNet.half.beta = t(rbind(interpolator(u=u, levels=levels, beta=(tmp <- as.matrix(coef(.pNet.half, matrix=TRUE)[-1,]))[1,])
                             ,interpolator(u=u, levels=levels, beta=tmp[2,])))
    pNet.lasso.beta = t(rbind(interpolator(u=u, levels=levels, beta=(tmp <- as.matrix(coef(.pNet.lasso, matrix=TRUE)[-1,]))[1,])
                              ,interpolator(u=u, levels=levels, beta=tmp[2,])))
        

    serp.beta = t(rbind(interpolator(u=u, levels=levels, beta=coef(.serp)[n.serp + 1:n.serp]),
                        interpolator(u=u, levels=levels, beta=coef(.serp)[2 * n.serp + 1:n.serp]) ))
    
    save(file=paste0(TMP_DESTINATION, "/exp_id=",exp_id,".RData"), 
         data, .nnolm, .nnolm.d, .nnolm.d.c, .polr, .oNet.ridge, .oNet.half, .oNet.lasso, 
         .pNet.ridge, .pNet.half, .pNet.lasso,
         u, true.beta, nnolm.beta, nnolm.d.beta, nnolm.d.c.beta, 
         pNet.ridge.beta, pNet.half.beta, pNet.lasso.beta,
         polr.beta, oNet.ridge.beta, oNet.half.beta, oNet.lasso.beta, serp.beta)
  }
  
  stopImplicitCluster()
}


## ===============
##   plot lines
## ===============

exp_setting_id = 2
exp_id = 1

set.seed(123)

TMP_DESTINATION = paste0(DIR,"/1-synthetic/setting_id=",exp_setting_id)
m = exp_settings[exp_setting_id,]

load(file=paste0(TMP_DESTINATION,"/exp_id=",exp_id,".RData"))


u = seq(1,J,0.05)
xl = range(u)

names = c("True", "Proposal", "POM", "NPOM")
colors = c("gray","blue","red","red"); ltys = c(1,5,2,4); lwds=c(2,2,2,2)

pdf(file=paste0(DIR,"/1-synthetic/illustration.pdf"), width=8, height=4)
par(mfrow=c(1,2))
par(mar = c(4, 4, 2, 1), oma = c(0,0,0,0))

m1 = exp_settings[exp_setting_id,1]
m2 = exp_settings[exp_setting_id,2]

for(k in 1:2){
  
  yl = range(true.beta[,k], nnolm.beta[,k], nnolm.d.beta[,k], nnolm.d.c.beta[,k], 
             polr.beta[,k], serp.beta[,k])
  if(k==1) yl[2] = yl[2] + 0.5 else yl[1]=yl[1]-1.2

  plot(u, true.beta[,k], type="l", xlim=xl, ylim=yl, xlab="u", ylab=paste0("b",k,"(u)"), 
       col=colors[1], lty=ltys[1], lwd=lwds[1])
  par(new=T)
  plot(u, nnolm.beta[,k], type="l", col=colors[2], lty=ltys[2], 
       xlim=xl, ylim=yl, xlab=" ",ylab=" ", xaxt="n", yaxt="n", lwd=lwds[2])
  par(new=T)
  plot(u, polr.beta[,k], type="l", col=colors[3], lty=ltys[3], 
       xlim=xl, ylim=yl, xlab=" ",ylab=" ", xaxt="n", yaxt="n", lwd=lwds[3])
  par(new=T)
  plot(u, oNet.ridge.beta[,k], type="l", col=colors[4], lty=ltys[4], 
       xlim=xl, ylim=yl, xlab=" ",ylab=" ", xaxt="n", yaxt="n", lwd=lwds[4])

  position = if(k==1) "topright" else "bottomright"
  legend(position, legend=names, lty=ltys, col=colors, lwd=lwds)
}
dev.off()
  

## ==================
##  MSE experiments
## ==================

nnolm.MSE = nnolm.d.MSE = nnolm.d.c.MSE = polr.MSE = NULL
pNet.ridge.MSE = pNet.half.MSE = pNet.lasso.MSE = NULL
oNet.ridge.MSE = oNet.half.MSE = oNet.lasso.MSE = serp.MSE = NULL

TB.med = TB.rsd = settings = NULL

for(exp_setting_id in 1:nrow(exp_settings)){
  set.seed(123)
  
  TMP_DESTINATION = paste0(DIR,"/1-synthetic/setting_id=",exp_setting_id)
  m = exp_settings[exp_setting_id,]
  settings = append(settings, m)

  ..true = ..nnolm = ..nnolm.d = ..nnolm.d.c = ..polr = NULL
  ..oNet.ridge = ..oNet.half = ..oNet.lasso = ..serp = NULL
  ..pNet.ridge = ..pNet.half = ..pNet.lasso = NULL
  
  for(exp_id in 1:n_exp){
    load(file=paste0(TMP_DESTINATION,"/exp_id=",exp_id,".RData"))
    ..true = abind(..true, true.beta, along=3)
    ..nnolm = abind(..nnolm, nnolm.beta, along=3)
    ..nnolm.d = abind(..nnolm.d, nnolm.d.beta, along=3)
    ..nnolm.d.c = abind(..nnolm.d.c, nnolm.d.c.beta, along=3)
    ..polr = abind(..polr, polr.beta, along=3)
    ..oNet.ridge = abind(..oNet.ridge, oNet.ridge.beta, along=3)
    ..oNet.half = abind(..oNet.half, oNet.half.beta, along=3)
    ..oNet.lasso = abind(..oNet.lasso, oNet.lasso.beta, along=3)
    ..pNet.ridge = abind(..pNet.ridge, pNet.ridge.beta, along=3)
    ..pNet.half = abind(..pNet.half, pNet.half.beta, along=3)
    ..pNet.lasso = abind(..pNet.lasso, pNet.lasso.beta, along=3)    
    ..serp = abind(..serp, serp.beta, along=3)
  }

  nnolm.MSE[[exp_setting_id]] = apply(..nnolm - ..true, c(2,3), function(z) mean(z^2))
  nnolm.d.MSE[[exp_setting_id]] = apply(..nnolm.d - ..true, c(2,3), function(z) mean(z^2))
  nnolm.d.c.MSE[[exp_setting_id]] = apply(..nnolm.d.c - ..true, c(2,3), function(z) mean(z^2))
  polr.MSE[[exp_setting_id]] = apply(..polr - ..true, c(2,3), function(z) mean(z^2))
  oNet.ridge.MSE[[exp_setting_id]] = apply(..oNet.ridge - ..true, c(2,3), function(z) mean(z^2))
  oNet.half.MSE[[exp_setting_id]] = apply(..oNet.half - ..true, c(2,3), function(z) mean(z^2))
  oNet.lasso.MSE[[exp_setting_id]] = apply(..oNet.lasso - ..true, c(2,3), function(z) mean(z^2))
  pNet.ridge.MSE[[exp_setting_id]] = apply(..pNet.ridge - ..true, c(2,3), function(z) mean(z^2))
  pNet.half.MSE[[exp_setting_id]] = apply(..pNet.half - ..true, c(2,3), function(z) mean(z^2))
  pNet.lasso.MSE[[exp_setting_id]] = apply(..pNet.lasso - ..true, c(2,3), function(z) mean(z^2))
  serp.MSE[[exp_setting_id]] = apply(..serp - ..true, c(2,3), function(z) mean(z^2))
  
  robust_mean <- function(z, trunc_coef=0.01){
    z = z[!is.na(z)]
    n=length(z); m=ceiling(n*trunc_coef)
    mean(sort(z)[(m+1):(n-m)])
    median(z)
  } 

  robust_sd <- function(z,trunc_coef=0.01){
    z = z[!is.na(z)]
    n=length(z); m=ceiling(n*trunc_coef)
    sd(sort(z)[(m+1):(n-m)])
  } 


  for(k in 1:2){

    tmp.med = tmp.rsd = NULL
        
    tmp.med <- append(tmp.med, round(apply(nnolm.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(nnolm.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])

    tmp.med <- append(tmp.med, round(apply(nnolm.d.c.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(nnolm.d.c.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])
    
    tmp.med <- append(tmp.med, round(apply(nnolm.d.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(nnolm.d.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])
    
    tmp.med <- append(tmp.med, round(apply(polr.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(polr.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])
    
    tmp.med <- append(tmp.med, round(apply(oNet.ridge.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(oNet.ridge.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])

    tmp.med <- append(tmp.med, round(apply(oNet.half.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(oNet.half.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])

    tmp.med <- append(tmp.med, round(apply(oNet.lasso.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(oNet.lasso.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])
    
    
    tmp.med <- append(tmp.med, round(apply(pNet.ridge.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(pNet.ridge.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])
    
    tmp.med <- append(tmp.med, round(apply(pNet.half.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(pNet.half.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])
    
    tmp.med <- append(tmp.med, round(apply(pNet.lasso.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(pNet.lasso.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])
    

    tmp.med <- append(tmp.med, round(apply(serp.MSE[[exp_setting_id]],1,robust_mean), digits=3)[k])
    tmp.rsd <- append(tmp.rsd, round(apply(serp.MSE[[exp_setting_id]],1,robust_sd), digits=3)[k])
    
    tmp.med = matrix(tmp.med, ncol=1)
    tmp.rsd = matrix(tmp.rsd, ncol=1)
    
    TB.med = cbind(TB.med, tmp.med)
    TB.rsd = cbind(TB.rsd, tmp.rsd)
  }
}

TB = matrix(paste0(TB.med," (",TB.rsd,")"),11,nrow(exp_settings)*2)
colnames(TB) = settings
rownames(TB) = c("CNPOM h", "CNPOM [h]", "CNPOM C[h]", "POLR", 
                 "oNet (a=0)", "oNet (a=0.5)", "oNet (a=1)", 
                 "pNet (a=0)", "pNet (a=0.5)", "pNet (a=1)", "serp")