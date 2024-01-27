data{
  int<lower=1> L; //data Length
  int<lower=1> N; //number of players
  int<lower=1> M; //number of rators
  int idX[L];     //player ID index
  int idY[L];     //rator ID index
  real Y[L];      // scores
}

parameters{
  real theta[N];
  real<lower=0> sig_theta[N];
  real<lower=0> sig[M];
}

model{
  for(l in 1:L){
    Y[l] ~ normal(theta[idX[l]],sig[idY[l]]);
  }
  //prior
  theta ~ normal(0, sig_theta);
  sig_theta ~ cauchy(0,5);
  sig ~ cauchy(0,5);
}
