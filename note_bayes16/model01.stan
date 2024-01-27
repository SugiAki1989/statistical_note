data {
  int<lower=1> J;
  real Y[J];
  real<lower=0> S[J];
}
parameters {
  real theta[J];
  real mu;
  real<lower=0> sigma;
}
model {
  for (j in 1:J) {
    theta[j] ~ normal(mu, sigma);
  }

  for (j in 1:J) {
    Y[j] ~ normal(theta[j], S[j]);
  }
}

generated quantities{
  vector[J] log_lik;
  
  for (j in 1:J) {
    log_lik[j] = normal_lpdf(Y[j]| theta[j], S[j]);
  }
} 
