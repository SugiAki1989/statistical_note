data {
  int N;       // Sample size
  real y[N]; // Data
}
parameters {
  real mu;             
  real<lower=0> sigma; // not sigma2
}
model {
  // vector version
  // y ~ normal(mu, sigma);
  // for-loop version
  for(i in 1:N){
    y[i] ~ normal(mu, sigma);
  }
}
generated quantities{
  // Posterior Predictive Distribution
  real pred[N];
  
  for(i in 1:N){
    pred[i] = normal_rng(mu, sigma);
  }
}
