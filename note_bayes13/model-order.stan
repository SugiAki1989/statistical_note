data {
  int N;  // num of players
  int G;  // num of games
  int<lower=1, upper=N> LW[G,5];  // loser and winner of each game
}

parameters {
  ordered[5] performance[G];
  vector[N] mu;
  real<lower=0> s_mu;
  vector<lower=0>[N] s_pf;
}

model {
  mu ~ normal(0, s_mu);
  s_mu ~ gamma(10, 10);
  s_pf ~ gamma(10, 10);

  for (g in 1:G){
    for (i in 1:5){
      performance[g,i] ~ normal(mu[LW[g,i]], s_pf[LW[g,i]]);
    }
  }
}

