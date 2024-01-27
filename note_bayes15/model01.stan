data{
  int<lower=1> L; //data Length
  int<lower=1> N; //number of players
  int idX[L];     //player ID index
  real Y[L];      // scores
}

parameters{
  real theta[N];
  real<lower=0> sig[N];
}

model{
  for(l in 1:L){
    Y[l] ~ normal(theta[idX[l]],sig[idX[l]]);
  }
}
