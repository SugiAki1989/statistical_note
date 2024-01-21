data {
  int<lower=0> N;
  int<lower=0> cv[N];
  int<lower=0> session[N];
}
parameters {
  real<lower=0, upper=1> cvr[N];
}
model {
  cv ~ binomial(session, cvr);
}
