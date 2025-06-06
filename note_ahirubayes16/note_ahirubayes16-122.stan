data {
  int T;
  int T_pred;
  real Y[T];
}

parameters {
  real mu[T];
  real<lower=0> s_mu;
  real<lower=0> s_Y;
}

model {

  for (t in 2:T) {
    mu[t] ~ normal(mu[t-1], s_mu); 
  }

  for (t in 1:T) {
    Y[t] ~ normal(mu[t], s_Y);
  }
}

generated quantities {

  real mu_all[T+T_pred];
  real y_pred[T_pred];

  mu_all[1:T] = mu;

  for (t in 1:T_pred) {
    int idx = T + t;
    mu_all[idx] = normal_rng(mu_all[idx-1], s_mu);
    y_pred[t] = normal_rng(mu_all[idx], s_Y);
  }

}

// 定義でベクトルをしようしているが、ベクトル化はしないないバージョン
// data {
//   int T;
//   int T_pred;
//   vector[T] Y;
// }
// 
// parameters {
//   vector[T] mu;
//   real<lower=0> s_mu;
//   real<lower=0> s_Y;
// }
// 
// model {
// 
//   for (t in 2:T) {
//     mu[t] ~ normal(mu[t-1], s_mu); 
//   }
// 
//   for (t in 1:T) {
//     Y[t] ~ normal(mu[t], s_Y);
//   }
// }
// 
// generated quantities {
// 
//   vector[T+T_pred] mu_all;
//   vector[T_pred] y_pred;
// 
//   mu_all[1:T] = mu;
// 
//   for (t in 1:T_pred) {
//     int idx = T + t;
//     mu_all[idx] = normal_rng(mu_all[idx-1], s_mu);
//     y_pred[t] = normal_rng(mu_all[idx], s_Y);
//   }
// 
// }

// ベクトル化
// data {
//   int T;
//   int T_pred;
//   vector[T] Y;
// }
// 
// parameters {
//   vector[T] mu;
//   real<lower=0> s_mu;
//   real<lower=0> s_Y;
// }
// 
// model {
//   mu[2:T] ~ normal(mu[1:(T-1)], s_mu);
//   Y ~ normal(mu, s_Y);
// }
// 
// generated quantities {
//   vector[T+T_pred] mu_all;
//   vector[T_pred] y_pred;
//   mu_all[1:T] = mu;
//   for (t in 1:T_pred) {
//     mu_all[T+t] = normal_rng(mu_all[T+t-1], s_mu);
//     y_pred[t] = normal_rng(mu_all[T+t], s_Y);
//   }
// }
