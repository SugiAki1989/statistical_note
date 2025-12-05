

data {
  int<lower=1> n_players;
  int<lower=1> n_rounds;
  array[n_rounds,4] int<lower=1,upper=n_players> player_ranks;
}
parameters {
  vector[n_players] skills;
  real<lower=0> skills_sigma;
}
model {
  vector[4] round_skills;

  skills ~ normal(0, skills_sigma);
  skills_sigma ~ cauchy(0,1);

  for (round_i in 1:n_rounds) {
    round_skills = skills[player_ranks[round_i]];
    target += categorical_logit_lpmf(1 | round_skills[1:4]);
    target += categorical_logit_lpmf(1 | round_skills[2:4]);
    target += categorical_logit_lpmf(1 | round_skills[3:4]);
  }
}

