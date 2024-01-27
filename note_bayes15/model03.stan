data{
  int<lower=1> L; //data Length
  int<lower=1> N; //number of players
  int<lower=1> M; //number of rators
  int idX[L];     //player ID index
  int idY[L];     //rator ID index
  real Y[L];      // scores
}
 
parameters{
  vector[N] theta;
  vector[M] gamma;
  real<lower=0> sig_theta;
  real<lower=0> sig_gamma;
  real<lower=0> sig[N];
}
 
model{
  //likellihood
  for(l in 1:L){
    Y[l] ~ normal(theta[idX[l]] + gamma[idY[l]], sig[idX[l]]);
  }
  //prior
  theta ~ normal(0, sig_theta);
  gamma ~ normal(0, sig_gamma);
  sig_theta ~ cauchy(0,5);
  sig_gamma ~ cauchy(0,5);
  sig ~ cauchy(0,5);
}
