data{
  int<lower=1> L; //data Length
  int<lower=1> N; //number of players
  int<lower=1> M; //number of rators
  int<lower=1> O; //number of year
  int idX[L];     //player ID index
  int idY[L];     //rator ID index
  int idZ[L];     //year ID index
  real Y[L];      // scores
}
 
parameters{
  vector[N] theta; //manzai
  vector[M] gamma; //rator
  vector[O] zeta; //year
  real<lower=0> sig_theta;
  real<lower=0> sig_gamma;
  real<lower=0> sig_zeta;
  real<lower=0> sig_e;
}
 
model{
  //likellihood
  for(l in 1:L){
    Y[l] ~ normal(theta[idX[l]] + gamma[idY[l]] + zeta[idZ[l]], sig_e);
  }
  //prior
  theta ~ normal(0, sig_theta);
  gamma ~ normal(0, sig_gamma);
  zeta ~ normal(0, sig_zeta);
  sig_theta ~ cauchy(0,5);
  sig_gamma ~ cauchy(0,5);
  sig_zeta ~ cauchy(0,5);
  sig_e ~ cauchy(0,5);
}

generated quantities{
  real rho_theta;
  rho_theta = sig_theta^2 / (sig_theta^2 + sig_gamma^2 + sig_e^2);
}
