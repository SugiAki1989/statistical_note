data {
  int T;
  real Y[T];
}

parameters {
  real mu[T];
  real season[T];
  real<lower=0> s_mu;
  real<lower=0> s_season;
  real<lower=0> s_Y;
}

transformed parameters {
  real y_mean[T];
  
  for (t in 1:T) {
    y_mean[t] = mu[t] + season[t];
  }
}

model {
  
  for (t in 2:T) {
    mu[t] ~ normal(mu[t-1], s_mu);
  }

  for (t in 4:T) {
    season[t] ~ normal(-sum(season[(t-3):(t-1)]), s_season);
  }
  
  for (t in 1:T) {
    Y[t] ~ normal(y_mean[t], s_Y);
  }
}


// data {
//   int T;
//   vector[T] Y;
// }
// 
// parameters {
//   vector[T] mu;
//   vector[T] season;
//   real<lower=0> s_mu;
//   real<lower=0> s_season;
//   real<lower=0> s_Y;
// }
// 
// transformed parameters {
//   vector[T] y_mean;
//   vector[T-3] sum_part_season;
//   y_mean = mu + season;
//   for(t in 4:T)
//     sum_part_season[t-3] = sum(season[(t-3):t]);
// }
// 
// model {
//   mu[2:T] ~ normal(mu[1:(T-1)], s_mu);
//   sum_part_season ~ normal(0, s_season);
//   Y ~ normal(y_mean, s_Y);
// }
