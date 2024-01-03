data {
  int<lower=0> N;     // sample size
  int<lower=0> Y[N];  // response variable
}
parameters {
  real beta;
  real r[N];
  real<lower=0> sigma;
}
transformed parameters {
  real q[N];

  for (i in 1:N) {
    q[i] <- inv_logit(beta + r[i]);
  }
}
model {
  for (i in 1:N) {
		Y[i] ~ binomial(8, q[i]); // binom
  }
  
  for (n in 1:N){
    r[n] ~ normal(0, sigma);
  }
  
  beta ~ normal(0, 100);      // non-informative prior
  sigma ~ uniform(0, 10000);  // non-informative prior
}

// generated quantities {
//   real y_pred[N];
//   for (i in 1:N)
//     y_pred[i] = binomial_rng(8, q[i]);
// }

