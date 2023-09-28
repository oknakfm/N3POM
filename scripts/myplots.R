plot_dataset <- function(data, sd_X=0.01, sd_g=0.01){
  d = ncol(data$X); n = nrow(data$X)
  
  if(d>4){
    par(mfrow=c(ceiling(d/4),4))    
  }else{
    par(mfrow=c(1,d))
  }
  par(mar = c(4, 4, 2, 1), oma = c(0,0,0,0))
  for(k in 1:d){
    plot(data$X[,k] + rnorm(n,mean=0,sd=sd_X), data$g + rnorm(n,mean=0,sd=sd_g), 
         xlab=colnames(data$X)[k], ylab="g",main=" ",pch="+")
    
    abline(lm(data$g~data$X[,k]))
  }
}

plot_monitor <- function(monitor){
  x = monitor$iteration; xl = range(x); 
  yl = range(monitor$log_likelihood); 
  par(mfrow = c(1,1), mar = c(4, 4, 2, 1), oma = c(0,0,0,0))
  plot(0, type="n", xlim=xl, ylim=yl, xlab="iteration", ylab="objective", main="Progress of training")
  par(new=T)
  myplot <- function(...){
    par(new=T); plot(..., xlim=xl, ylim=yl, xaxt="n", yaxt="n", xlab=" ", ylab=" ")
  }
  myplot(x, monitor$log_likelihood, type="l", lty = 1, col="black", lwd=1)
}

plot_curves <- function(alpha=NULL, alpha_true=NULL, 
                        beta, beta_true = NULL, 
                        xax=seq(1,J,0.05)){
  d = ncol(beta)
  if(d>=4){
    if(is.null(alpha)) par(mfrow=c(ceiling(d/4),4))
    else par(mfrow=c(ceiling((d+1)/4),4))
  }else{
    if(is.null(alpha)) par(mfrow=c(1,d))
    else par(mfrow=c(1,d+1))
  }
  par(mar = c(4, 4, 3, 1), oma = c(0,0,0,0))  
  
  if(!is.null(alpha)){
    yl = range(alpha,alpha_true)
    plot(xax, alpha, type="l", xlab="u", ylab="a(u)", main=" ", ylim=yl)
    if(!is.null(alpha_true)){
      par(new=T)
      plot(xax, alpha_true, type="l", xlab=" ", ylab=" ", main=" ", ylim=yl, col="blue")
      legend("topleft", legend=c("estimated","true"), col=c("black","blue"), lty=c(1,1))
    }
  }
  

  
  for(k in 1:d){
    yl = range(beta[,k])
    plot(xax, beta[,k], type="l", xlab="u", ylab="b(u)", main=colnames(beta)[k], ylim=yl)
    if(!is.null(beta_true)){
      par(new=T)
      plot(xax, beta_true[,k], type="l", xlab=" ", ylab=" ", main=" ", ylim=yl, col="blue")
      legend("topleft", legend=c("estimated","true"), col=c("black","blue"), lty=c(1,1))
    }
  }
  # hist(g, breaks=50)
}






