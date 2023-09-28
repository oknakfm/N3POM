# require("progress")

R = 10

## points j1, j2, ..., jR
j_seq = seq(1, J, (J-1) / (R-1))

activation <- function(z=0, grad_level=0){
  if(activation_type=="tanh"){
    return(switch(as.character(grad_level),
           "-1" = 1,
           "0" = tanh(z),
           "1" = 1-tanh(z)^2,
           "2" = -2 * tanh(z) * (1-tanh(z)^2),
           stop("invalid grad_level")
    ))
  }else if(activation_type=="sigmoid"){
    sigmoid <- function(z) 1/(1+exp(-z))
    return(switch(as.character(grad_level),
           "-1" = 1/4,
           "0" = sigmoid(z),
           "1" = sigmoid(z) * (1-sigmoid(z)),
           "2" = sigmoid(z) * (1-sigmoid(z)) * (1-2*sigmoid(z)),
           stop("invalid grad_level")
    ))
  }else stop("invalid input")
}

weights <- function(g, type="weighted", exponent=0.5){
  if(is.null(data)) stop("invalid data")
  n = length(g)
  if(type == "uniform"){
    zeta = rep(1/n,n)
  }else if(type == "weighted"){
    realms = c(1,(j_seq[-1]+j_seq[-R])/2,J)
    ind = findInterval(g, realms); ind[which(ind>R)]=R
    num_elements = sapply(1:R, function(j) sum(ind==j)+1) + 0.1 * (n/J)
    prob = (tmp <- 1/num_elements^(exponent))/sum(tmp)
    zeta = (tmp <- prob[ind])/sum(tmp)
  }
  return(zeta)
}

theta_initialization <- function(L = 100, sd = 0.01, 
                                 data = NULL, type = "serp"){
  d=data$d
  if(type=="polr"){
    if(is.null(data)) stop("data is missing")
    g_discrete = round(data$g)
    olm = polr(formula = as.factor(g_discrete) ~ data$X, method="logistic")
    zmin = min(olm$zeta); zmax = max(olm$zeta)
    alpha = zmin + (j_seq - 1) * (zmax-zmin) / (max(j_seq) - min(j_seq))
    phi = as.vector(alpha_to_phi(alpha))
    V1 = matrix(rnorm(n=d*L, mean=0, sd=sd), d, L)
    V2 = -as.vector(olm$coefficients)
    W1 = matrix(rnorm(n=d*L, mean=0, sd=sd), d, L)
    W2 = matrix(rnorm(n=d*L, mean=0, sd=sd), d, L)
  }else if(type=="serp"){
    g_discrete = round(data$g)
    .serp = serp(formula = as.ordered(g_discrete) ~ data$X, 
                 link = "logit", slope = "penalize")
    n.serp = length(coef(.serp))/(1+data$d)
    
    levels = as.numeric(names(table(g_discrete)))
    max_g = max(levels)
    
    alpha = coef(.serp)[1:n.serp]
    
    .beta = NULL; 
    for(k in 1:data$d) .beta = rbind(.beta, coef(.serp)[k * n.serp + 1:n.serp])
    .beta = cbind(.beta, 2 * .beta[,n.serp] - .beta[,n.serp-1])
    
    if(min(levels)>1){
      levels = append(1,levels)
      .beta = cbind(.beta[,1], .beta)
    }
    
    beta=NULL
    for(j in 1:(max_g-1)){
      ind = findInterval(j,levels)
      ratio = (j - levels[ind])/(levels[ind+1] - levels[ind])
      beta = cbind(beta, ratio * .beta[,ind+1] + (1-ratio) * .beta[,ind])
    }
    beta = cbind(beta, 2 * beta[,max_g-1] - beta[,max_g-2])    
    
    const_T = 3
    n_repeats = floor(L/max_g)
    zmin = min(alpha); zmax = max(alpha)
    alpha = zmin + (j_seq - 1) * (zmax-zmin) / (max(j_seq) - min(j_seq))
    phi = as.vector(alpha_to_phi(alpha))
    .V1 = -1 * rep(const_T,d) %*% t(1:J)
    V2 = apply(beta,1,mean)
    .W1 = matrix(const_T,d,J)
    .W2 = as.matrix(beta[,1] - V2)
    for(l in 2:J){
      .v = if(l==2) .W2[,1] else apply(.W2[,1:(l-1)],1,sum)
      .W2 = cbind(.W2, beta[,l] - V2 - .v)
    }
    
    V1 = NULL; for(k in 1:n_repeats) V1 = cbind(V1, .V1)
    W1 = NULL; for(k in 1:n_repeats) W1 = cbind(W1, .W1)
    W2 = NULL; for(k in 1:n_repeats) W2 = cbind(W2, .W2/n_repeats)

    n_remaining = L - max_g * n_repeats
    
    if(n_remaining!=0){
      V1E = matrix(rnorm(mean=0, sd=sd, d*n_remaining), d, n_remaining); V1 = cbind(V1, V1E)
      W1E = matrix(rnorm(mean=0, sd=sd, d*n_remaining), d, n_remaining); W1 = cbind(W1, W1E)
      W2E = matrix(rnorm(mean=0, sd=sd, d*n_remaining), d, n_remaining); W2 = cbind(W2, W2E)
    }
    
  }else if(type=="uniform"){
    phi = rnorm(n=R, mean=0, sd=sd)
    V1 = matrix(rnorm(n=d*L, mean=0, sd=sd), d, L)
    V2 = rnorm(n=d, mean=0, sd=sd)
    W1 = matrix(rnorm(n=d*L, mean=0, sd=sd), d, L)
    W2 = matrix(rnorm(n=d*L, mean=0, sd=sd), d, L)
  }else{
    stop("invalid type")
  }

  theta = list(V1=V1, V2=V2, W1=W1, W2=W2, phi=phi)
  return(theta)
} 

phi_to_alpha <- function(phi){
  alpha = append(phi[1], phi[1] + cumsum(abs(phi[-1])))
  return(alpha)
}

alpha_to_phi <- function(alpha){
  phi = append(alpha[1], sapply(1:(R-1), function(j) abs(alpha[j+1]-alpha[j])))
  return(phi)
}

W2_scale <- function(theta, X_upper=1){
  R = length(theta$phi)
  alpha = phi_to_alpha(theta$phi)
  minimum_gap = min((alpha[-1] - alpha[-R])/(j_seq[-1] - j_seq[-R]))
  rho_inf = activation(grad_level=-1)
  coef = min(1, minimum_gap/(X_upper * rho_inf * sqrt(sum(apply(abs(theta$W1 * theta$W2),1,sum)^2)) + 0.0001 ))
  return(coef)
}

theta_summary <- function(theta){
  d = nrow(theta$W1); L = ncol(theta$W1); R = length(theta$phi)
  cat("[Summary of theta]\n")
  cat("d=",d,"/ L=",L,"/ R=",R,"\n")
  lapply(theta, function(z) summary(as.vector(z)))  
}

theta_scaling <- function(theta, magnitude=1){
  scaled = lapply(theta, function(z) magnitude * z)
  return(scaled)
}

theta_rescaling <- function(theta, X_upper=1){
  coef = W2_scale(theta, X_upper=X_upper)
  theta$W1 = sqrt(coef) * theta$W1
  theta$W2 = sqrt(coef) * theta$W2
  theta$V1 = sqrt(coef) * theta$V1
  return(theta)
}

theta_numparams <- function(theta){
  list(
       V1 = (V1 <- prod(dim(theta$V1))),
       V2 = (V2 <- length(theta$V2)),
       W1 = (W1 <- prod(dim(theta$W1))), 
       W2 = (W2 <- prod(dim(theta$W2))),
       phi = (phi <- length(theta$phi)),
       overall = W1+V2+W1+W2+phi)
}

theta_L2norm <- function(theta, grad_level=0){
  if(grad_level == 0){
    L2norm = do.call("sum", lapply(theta, function(z) sum(z^2)))
    return(L2norm / theta_numparams(theta)$overall)
  }else if(grad_level == 1){
    return(theta_scaling(theta,2/theta_numparams(theta)$overall))
  }else{
    stop("invalid input")
  }
}

theta_sum <- function(theta1, theta2){
  new_theta = list(V1 = theta1$V1 + theta2$V1,
                   V2 = theta1$V2 + theta2$V2,
                   W1 = theta1$W1 + theta2$W1,
                   W2 = theta1$W2 + theta2$W2,
                   phi = theta1$phi + theta2$phi)
  return(new_theta)
}

a <- function(u=1.5, alpha=seq(1,J,0.5), j_seq=seq(1,J,0.5), derv_level=0){
  R = length(alpha); n = length(u)
  r = findInterval(u,j_seq) + 1
  if(derv_level==0){
    .a <- function(u,r){
      if(r<=R){
        rate = (u-j_seq[r-1])/(j_seq[r]-j_seq[r-1])
        tmp <- alpha[r] * rate + alpha[r-1] * (1-rate)
      }else if(r==R+1){
        tmp <- alpha[R]
      }
      as.numeric(tmp)
    }
    a0 = sapply(1:n, function(i) .a(u=u[i],r=r[i]))
    return(a0)
  }else if(derv_level==1){
    grd = (alpha[-1]-alpha[-R])/(j_seq[-1]-j_seq[-R])
    a1 = append(grd, grd[R-1])[r-1]
    return(a1)
  }else{
    stop("invalid derv_level")
  }
}

## Neural network for predicting the coefficients
neural_network <- function(u=1, theta=NULL, derv_level=0){
  if(is.null(theta)) stop("theta is missing")
  if(derv_level==0){
    nn0 = t(sapply(u, function(u) apply(theta$W2 * activation(theta$W1*u + theta$V1),1,sum) + theta$V2))
    return(nn0)
  }else if(derv_level==1){
    nn1 = t(sapply(u, function(u) apply(theta$W2 * theta$W1 * activation(theta$W1*u + theta$V1, grad_level=1),1,sum)))
    return(nn1)
  }else{
    stop("invalid derv_level")
  }
}

## log-likelihood
log_likelihood <- function(theta=NULL, data=NULL, grad_level=0, minibatch_ind=NULL,
                           gradient_direction=c("phi","V1","V2","W1","W2"), 
                           zeta=NULL){
  if(is.null(theta)) stop("theta is missing")
  if(is.null(data)) stop("data is missing")
  if(is.null(minibatch_ind)){
    warning("minibatch indices are missing, so fullbatch is used instead")
    minibatch_ind = 1:data$n
  }
  if(is.null(zeta)){
    warning("uniform weights are used")
    zeta = rep(1/data$n, data$n)
  }
  sigmoid <- function(z, grad_level=0){
    .sigmoid <- function(z) 1/(1+exp(-z))
    return(switch(as.character(grad_level),
                  "0" = .sigmoid(z),
                  "1" = .sigmoid(z) * (1-.sigmoid(z)),
                  "2" = .sigmoid(z) * (1-.sigmoid(z)) * (1-2*.sigmoid(z)),
                  "r21" = 1-2*.sigmoid(z),
                  stop("invalid grad_level")))
  } 
  n = length(minibatch_ind); d = data$d; J = data$J; L = ncol(theta$V1)
  batch = list(X = data$X[minibatch_ind,], g = data$g[minibatch_ind], n = n)
  zeta = zeta[minibatch_ind] 

  diag_product <- function(A,B) sapply(1:batch$n, function(i) A[i,] %*% B[i,])
  trc <- function(z) min(max(z,0),1)
  boxcar <- function(z) if(0<=z && z<1) 1 else 0
  
  alpha = phi_to_alpha(theta$phi)
  A0 = a(u=batch$g, alpha=alpha, j_seq=j_seq, derv_level=0)
  B0 = neural_network(u=batch$g, theta, derv_level=0) 
  f0 = diag_product(B0,batch$X) + A0

  A1= a(u=batch$g, alpha=alpha, j_seq=j_seq, derv_level=1)
  B1 = neural_network(u=batch$g, theta, derv_level=1)
  # f1 = sapply(diag_product(B1,batch$X) + A1, function(z) max(z, 0.1)) ## for robust computation
  f1 = diag_product(B1,batch$X) + A1
  
  if(grad_level == 0){
    log_lik = sum(zeta * log(sigmoid(f0, grad_level=1)*f1))
    return(log_lik)
  }else if(grad_level == 1){
    r21 = sigmoid(f0,grad_level="r21") ## sigmoid(f0,grad_level=2)/sigmoid(f0,grad_level=1)
    doesComputeU = any(c("V1","W1","W2") %in% gradient_direction)
    if(doesComputeU){
      tmp = outer(batch$g,theta$W1) + outer(rep(1,batch$n),theta$V1)
      CORE_U_0 = activation(tmp, grad_level=0)
      CORE_U_1 = activation(tmp, grad_level=1)
      CORE_U_2 = activation(tmp, grad_level=2)
    }
    if("phi" %in% gradient_direction){
      phi = sum(zeta * r21)
      CORE_P_1 = (outer(batch$g,rep(1,R-1))-outer(rep(1,n),j_seq[-R]))/outer(rep(1,n),j_seq[-1]-j_seq[-R])
      CORE_P_2 = apply(CORE_P_1, c(1,2), trc)
      CORE_P_3 = outer(zeta * r21, sign(theta$phi[-1]))
      CORE_P_4 = apply(CORE_P_1, c(1,2), boxcar); 
      # exc_ind = which(batch$g==J); CORE_P_2[exc_ind,]=append(rep(0,R-2),1); CORE_P_4[exc_ind,]=append(rep(0,R-2),1)
      CORE_P_5 = outer(zeta/f1, sign(theta$phi[-1]) / (j_seq[-1]-j_seq[-R]))
      phi = append(phi, apply(CORE_P_2 * CORE_P_3 + CORE_P_4 * CORE_P_5, 2, sum))
      rm(CORE_P_1, CORE_P_2, CORE_P_3, CORE_P_4, CORE_P_5)
    }else{
      phi = rep(0,R)
    }
    if("V1" %in% gradient_direction){
      CORE_V1_1 = outer(zeta * r21, matrix(1,d,L))
      CORE_V1_2 = outer(batch$X, rep(1,L)) * outer(rep(1,n),theta$W2)
      CORE_V1_3 = outer(zeta/f1, matrix(1,d,L))
      CORE_V1_4 = outer(batch$X, rep(1,L)) * outer(rep(1,n),theta$W2*theta$W1)
      V1 = apply(CORE_V1_1 * CORE_V1_2 * CORE_U_1 
                 + CORE_V1_3 * CORE_V1_4 * CORE_U_2, c(2,3), sum)
      rm(CORE_V1_1, CORE_V1_2, CORE_V1_3, CORE_V1_4)
    }else{
      V1 = matrix(0,d,L)
    }
    if("V2" %in% gradient_direction){
      CORE_V2 = outer(zeta * r21, rep(1,d))
      V2 = apply(CORE_V2 * batch$X, 2, sum)
      rm(CORE_V2)
    }else{
      V2 = rep(0,d)
    }
    if("W1" %in% gradient_direction){
      CORE_W1_1 = outer(batch$X, rep(1,L)) * outer(rep(1,n),theta$W2)
      CORE_W1_2 = outer(batch$X, rep(1,L)) * outer(rep(1,n),theta$W2*theta$W1)
      CORE_W1_3 = outer(zeta * r21 * batch$g, matrix(1,d,L))
      CORE_W1_4 = outer((zeta/f1) * batch$g, matrix(1,d,L))
      CORE_W1_5 = outer(zeta/f1, matrix(1,d,L)) 
      W1 = apply(CORE_W1_1 * CORE_W1_3 * CORE_U_1 
                 + CORE_W1_2 * CORE_W1_4 * CORE_U_2 
                 + CORE_W1_1 * CORE_W1_5 * CORE_U_1, c(2,3), sum)
      rm(CORE_W1_1, CORE_W1_2, CORE_W1_3, CORE_W1_4, CORE_W1_5)
    }else{
      W1 = matrix(0,d,L)
    }
    if("W2" %in% gradient_direction){    
      CORE_W2_1 = outer(batch$X, rep(1,L)) 
      CORE_W2_2 = outer(batch$X, rep(1,L)) * outer(rep(1,n),theta$W1)
      CORE_W2_3 = outer(zeta * r21, matrix(1,d,L))
      CORE_W2_4 = outer(zeta/f1, matrix(1,d,L))
      W2 = apply(CORE_W2_1 * CORE_W2_3 * CORE_U_0 
                 + CORE_W2_2 * CORE_W2_4 * CORE_U_1, c(2,3), sum)
      rm(CORE_W2_1, CORE_W2_2, CORE_W2_3, CORE_W2_4)
    }else{
      W2 = matrix(0,d,L)
    }
    if(doesComputeU) rm(CORE_U_0, CORE_U_1, CORE_U_2)
    
    grad_log_lik = list(phi=phi, V1=V1, V2=V2, W1=W1, W2=W2)
    return(grad_log_lik)
    
  }else{
    stop("invalid input")
  }
}

NNOLM <- function(data=NULL, L=100, initial_sd=1, exponent = 0.5, 
                  n_GD=5000, minibatch_size=16, theta0 = NULL,
                  initial_lr = 1,
                  decreasing_interval=100, decreasing_rate=0.85, 
                  do_monitor=TRUE, 
                  continuization=FALSE){
  
  if(is.null(data)) stop("invalid data")
  if(continuization){
    mmcut <- function(u){
      .mmcut <- function(u) max(min(u,J),1)
      sapply(u, .mmcut)
    }
    data$g = mmcut(data$g + runif(n=data$n, min=-0.5, max=0.5))
  }
  
  X_upper = max(apply(data$X, 1, function(z) sqrt(mean(z^2)))) + 0.01
  
  ## weights
  zeta = weights(g=data$g, type="weighted", exponent=exponent)
  
  ## initialization
  if(is.null(theta0)){
    theta0 = theta_initialization(L=L, sd=initial_sd, data=data, type="serp")
  }
  
  theta0 = theta_rescaling(theta=theta0, X_upper=X_upper)
  
  gradient = log_likelihood(theta=theta0, data=data, grad_level=1,
                            minibatch_ind = 1:data$n, zeta=zeta,
                            gradient_direction=c("V1","V2","W1","W2","phi"))
  
  lr = lr0 = list(
    V1 = initial_lr, 
    V2 = initial_lr * median(abs(theta0$V2/gradient$V2)),
    W1 = initial_lr,
    W2 = initial_lr * median(abs(theta0$W2/gradient$W2)),
    phi = initial_lr * median(abs(theta0$phi/gradient$phi)),
    coef = 1
  )

  theta = theta0
  
  ## whether do we monitor the progress of GD
  if(do_monitor){
    monitor = vector("list",2);
    names(monitor) = c("iteration","log_likelihood")
  }
  
  ## Optimization via full-batch gradient descent
  for(iteration in 1:n_GD){

    minibatch_ind = sort(sample(1:data$n)[1:minibatch_size])
    
    theta_tmp = theta
    
    theta = theta_sum(theta,
                      theta_scaling(log_likelihood(theta=theta_tmp, data=data, grad_level=1,
                                                   minibatch_ind = minibatch_ind, zeta=zeta, 
                                                   gradient_direction="V1"), lr$V1))
    
    theta = theta_sum(theta,
                      theta_scaling(log_likelihood(theta=theta_tmp, data=data, grad_level=1,
                                                   minibatch_ind = minibatch_ind, zeta=zeta, 
                                                   gradient_direction="V2"), lr$V2))
    
    theta = theta_sum(theta,
                      theta_scaling(log_likelihood(theta=theta_tmp, data=data, grad_level=1,
                                                   minibatch_ind = minibatch_ind, zeta=zeta,
                                                   gradient_direction="W1"), lr$W1))
    
    theta = theta_sum(theta,
                      theta_scaling(log_likelihood(theta=theta_tmp, data=data, grad_level=1,
                                                   minibatch_ind = minibatch_ind, zeta=zeta, 
                                                   gradient_direction="W2"), lr$W2))
    
    theta = theta_sum(theta,
                      theta_scaling(log_likelihood(theta=theta_tmp, data=data, grad_level=1,
                                                   minibatch_ind = minibatch_ind, zeta=zeta,
                                                   gradient_direction="phi"), lr$phi))
    
    theta = theta_rescaling(theta=theta, X_upper=X_upper)
    
    if(do_monitor && (iteration == 1 || iteration %% 10 == 0)){
      monitor$iteration = append(monitor$iteration, iteration)
      monitor$log_likelihood = append(monitor$log_likelihood,
                                      (m1 <- log_likelihood(theta=theta, data=data, grad_level=0, minibatch_ind=1:data$n, zeta=zeta)))
    }
    
    if(iteration %% decreasing_interval == 0){
      lr = lapply(lr, function(z) decreasing_rate*z)
    }
    
  }
  output = list(theta0=theta0, theta=theta, lr0=lr0, X_upper=X_upper, 
                zeta=zeta, exponent=exponent)
  if(do_monitor) output = append(output, list(monitor=monitor))

  return(output)
}


interpolator <- function(u=seq(1,J,0.05), levels=NULL, beta=1:(J-1)){
  if(is.null(levels)) levels = 1:length(beta)
  
  .interpolator <- function(u){
    ind = findInterval(u, levels)
    if(ind == 0){
      interpolated_beta = beta[1] 
    }else if(levels[ind] < J-1){
      diff = (u - levels[ind]) / (levels[ind+1] - levels[ind])
      .interpolated_beta = diff * beta[ind+1] + (1-diff) * beta[ind]
    }else{
      .interpolated_beta = beta[length(beta)]
    }
  }
  sapply(u, .interpolator)
}

nonparametric_cdf <- function(query=NULL, data=NULL, k=5, h=2){
  if(is.null(query)) stop("invalid query")
  if(is.null(data)) stop("invalid data")
  
  .dist = pdist(query, data$X)@dist
  ind = which(order(.dist)<=k)

  return(list(k = k, h = h, 
              dist_to_kth = sort(.dist)[k], 
              dist_to_farthest = max(.dist), 
              val = mean(data$g[ind] <= h)))
}




NNOLM.cdf <- function(theta=NULL,X=NULL,h=2){
  if(is.null(theta)) stop("invalid theta")
  if(is.null(X)) stop("invalid X")
  
  sigmoid <- function(z) 1/(1+exp(-z))
  diag_product <- function(A,B) sapply(1:nrow(X), function(i) A[i,] %*% B[i,])
  
  alpha = phi_to_alpha(theta$phi)
  A0 = a(u=h, alpha=alpha, j_seq=j_seq, derv_level=0)
  B0 = neural_network(u=h, theta, derv_level=0) 
  prob = sigmoid(as.vector(X %*% t(B0) + A0))
  return(prob)
}



polr.cdf <- function(polr=NULL, X=NULL, h=2){
  if(is.null(.polr)) stop("invalid .polr")
  if(is.null(X)) stop("invalid X")
  
  .lev = as.numeric(polr$lev)
  .ind = findInterval(h, .lev)
  .ratio = (h - .lev[.ind]) / (.lev[.ind+1] - .lev[.ind])
  alpha = polr$zeta[.ind] + .ratio * (polr$zeta[.ind+1]-polr$zeta[.ind])
  beta = -as.vector(polr$coefficients)
  sigmoid <- function(z) 1/(1+exp(-z))
  
  prob = sigmoid(as.vector(alpha + X %*% beta))
  return(prob)
}











