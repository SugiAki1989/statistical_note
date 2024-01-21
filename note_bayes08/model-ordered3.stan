data {
  int<lower=0> N;
  int<lower=0> cv[N];
  int<lower=0> session[N];
}
parameters {
  ordered[N] tmp;
}
transformed parameters {
  real<lower=0, upper=1> cvr[N];
  for(i in 1:N) {
    cvr[i] <- inv_logit(tmp[i]);
  }
}
model {
  cv ~ binomial(session, cvr);
}
