dir.create(paste0(DIR,"/preprocessed"),showWarnings=FALSE)

for(dataset.name in dataset.names){
  DATASETS = paste0(DIR,"/datasets")
  DESTINATION_TO_SAVE = paste0(DIR,"/preprocessed/", dataset.name,"/"); 
  dir.create(DESTINATION_TO_SAVE,showWarnings=FALSE)

  set.seed(123)
  
  ## loading dataset
  Z = read.csv(paste0(DATASETS,"/",dataset.name,".csv"), header=TRUE, sep=";")

  ## rescaling
  d = ncol(Z)-1
  
  if(dataset.name == "real-estate") Z = Z[-271,]
  
  X = as.matrix(Z[,1:d]); X = scale(X, center=T, scale=T); 
  X_upper = max(apply(X, 1, function(z) sqrt(mean(z^2)))) + 10^(-3)
  g = as.vector(Z[,d+1]); 
  
  min_g = min(g); max_g = 10
  g = g - min(g); g = g * ((max_g-1)/max(g)) + 1
  
  n_all = nrow(X); J = ceiling(max_g)
  n_train = ceiling(n_all * (1-test_ratio)); n_test = n_all - n_train

  for(instance_id in 1:num_instances){
    set.seed(instance_id)
    ind = sort(sample(1:n_all)[1:n_train])
    train = list(X=X[ind,], g=g[ind], n=n_train, d=d, J=J) # for training
    test = list(X=X[-ind,], g=g[-ind], n=n_test, d=d, J=J) # for test
    save(file=paste0(DESTINATION_TO_SAVE,"/instance_id=",instance_id,".RData"), 
         train, test, ind, n_all, n_train, n_test, d, J, X, g)
  }
}