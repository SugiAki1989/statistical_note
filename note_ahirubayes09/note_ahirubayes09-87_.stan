
data {
  int N;
  int T;
  real Time[T];
  real Y[N,T];
  int T_new;
  real Time_new[T_new];
}

parameters {
  real mu_a;
  real mu_b;
  real<lower=0> a[N];
  real<lower=0> b[N];
  real<lower=0> s_a;
  real<lower=0> s_b;
  real<lower=0> s_Y;
}

model {
  for (n in 1:N) {
    a[n] ~ gamma(mu_a^2/s_a, mu_a/s_a);
    b[n] ~ gamma(mu_b^2/s_b, mu_b/s_b);
  }
  for (n in 1:N)
    for (t in 1:T)
      Y[n,t] ~ normal(a[n]*(1 - exp(-b[n]*Time[t])), s_Y);
}

generated quantities {
  real y_new[N,T_new];
  for (n in 1:N)
    for (t in 1:T_new)
      y_new[n,t] = normal_rng(a[n]*(1 - exp(-b[n]*Time_new[t])), s_Y);
}
