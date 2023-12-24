data {
  int N_obs;
  int N_cens;
  real Y_obs[N_obs];
  real L;
}

parameters {
  real mu;
  real<lower=0> s_Y;
}

model {
  for (n in 1:N_obs){
    Y_obs[n] ~ normal(mu, s_Y);
  }
  for (n in 1:N_cens){
    target += normal_lcdf(L | mu, s_Y);
  }
}
