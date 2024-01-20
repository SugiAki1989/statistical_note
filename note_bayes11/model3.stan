// ---------------------------------------------------------
// JAGS
// ---------------------------------------------------------
// model {
// for(i in 1:n_games) {
//   HomeGoals[i] ~ dpois(lambda_home[Season[i], HomeTeam[i],AwayTeam[i]])
//   AwayGoals[i] ~ dpois(lambda_away[Season[i], HomeTeam[i],AwayTeam[i]])
// }
// for(season_i in 1:n_seasons) {
//   for(home_i in 1:n_teams) {
//     for(away_i in 1:n_teams) {
//       lambda_home[season_i, home_i, away_i] <- exp( home_baseline[season_i] + skill[season_i, home_i] - skill[season_i, away_i])
//       lambda_away[season_i, home_i, away_i] <- exp( away_baseline[season_i] + skill[season_i, away_i] - skill[season_i, home_i])
//     }
//   }
// }
// skill[1, 1] <- 0
// for(j in 2:n_teams) {
//   skill[1, j] ~ dnorm(group_skill, group_tau)
// }
// group_skill ~ dnorm(0, 0.0625)
// group_tau <- 1/pow(group_sigma, 2)
// group_sigma ~ dunif(0, 3)
// home_baseline[1] ~ dnorm(0, 0.0625)
// away_baseline[1] ~ dnorm(0, 0.0625)
// for(season_i in 2:n_seasons) {
//   skill[season_i, 1] <- 0
//   for(j in 2:n_teams) {
//     skill[season_i, j] ~ dnorm(skill[season_i - 1, j], season_tau)
//   }
//   home_baseline[season_i] ~ dnorm(home_baseline[season_i - 1], season_tau)
//   away_baseline[season_i] ~ dnorm(away_baseline[season_i - 1], season_tau)
// }
// season_tau <- 1/pow(season_sigma, 2)
// season_sigma ~ dunif(0, 3)
// }
// ---------------------------------------------------------
// Stan
// ---------------------------------------------------------
// data {
//   int<lower=1> n_games;
//   int<lower=1> n_seasons;
//   int<lower=1> n_teams;
//   int<lower=1> Season[n_games];
//   int<lower=1> HomeTeam[n_games];
//   int<lower=1> AwayTeam[n_games];
//   int<lower=0> HomeGoals[n_games];
//   int<lower=0> AwayGoals[n_games];
// }
// parameters {
//   vector[n_seasons] home_baseline;
//   vector[n_seasons] away_baseline;
//   matrix[n_teams, n_seasons] skill;
//   real<lower=0> group_sigma;
//   real<lower=0> group_skill;
//   real<lower=0> season_sigma;
// }
// transformed parameters {
//   vector[n_games] lambda_home;
//   vector[n_games] lambda_away;
//   
//   for (i in 1:n_games) {
//     lambda_home[i] = exp(home_baseline[Season[i]] + skill[HomeTeam[i], Season[i]] - skill[AwayTeam[i], Season[i]]);
//     lambda_away[i] = exp(away_baseline[Season[i]] + skill[AwayTeam[i], Season[i]] - skill[HomeTeam[i], Season[i]]);
//   }
// }
// model {
//   group_sigma ~ uniform(0, 1000);
//   group_skill ~ normal(0, 4);
//   home_baseline[1] ~ normal(0, 4);
//   away_baseline[1] ~ normal(0, 4);
//   
//   for (j in 1:n_teams) {
//     if (j == 1) {
//       skill[j, 1] ~ normal(0, 0.5);
//     } else {
//       skill[j, 1] ~ normal(group_skill, group_sigma);
//     }
//   }
//   
//   for (season_i in 2:n_seasons) {
//     for (j in 1:n_teams) {
//       if (j == 1) {
//         skill[j, season_i] ~ normal(0, 0.5);
//       } else {
//         skill[j, season_i] ~ normal(skill[j, season_i-1], season_sigma);
//       }
//     }
//     home_baseline[season_i] ~ normal(home_baseline[season_i-1], season_sigma);
//     away_baseline[season_i] ~ normal(away_baseline[season_i-1], season_sigma);
//   }
//   
//   for (i in 1:n_games) {
//     HomeGoals[i] ~ poisson(lambda_home[i]);
//     AwayGoals[i] ~ poisson(lambda_away[i]);
//   }
// }

data {
  int<lower=1> n_games;
  int<lower=1> n_seasons;
  int<lower=1> n_teams;
  int<lower=1> Season[n_games];
  int<lower=1> HomeTeam[n_games];
  int<lower=1> AwayTeam[n_games];
  int<lower=0> HomeGoals[n_games];
  int<lower=0> AwayGoals[n_games];
}
parameters {
  vector[n_seasons] home_baseline;
  vector[n_seasons] away_baseline;
  matrix[n_teams, n_seasons] skill;
  real<lower=0> group_sigma;
  // real<lower=0> group_skill;
  real<lower=0> season_sigma;
}
transformed parameters {
  vector[n_games] lambda_home;
  vector[n_games] lambda_away;

  for (i in 1:n_games) {
    lambda_home[i] = exp(home_baseline[Season[i]] + skill[HomeTeam[i], Season[i]] - skill[AwayTeam[i], Season[i]]);
    lambda_away[i] = exp(away_baseline[Season[i]] + skill[AwayTeam[i], Season[i]] - skill[HomeTeam[i], Season[i]]);
  }
}
model {
  group_sigma ~ uniform(0, 1000);
  // group_skill ~ normal(0, 4);
  home_baseline[1] ~ normal(0, 4);
  away_baseline[1] ~ normal(0, 4);

  for (j in 1:n_teams) {
    skill[j, 1] ~ normal(0, group_sigma);
  }

  for (season_i in 2:n_seasons) {
    for (j in 1:n_teams) {
      skill[j, season_i] ~ normal(skill[j, season_i-1], season_sigma);
      }
    home_baseline[season_i] ~ normal(home_baseline[season_i-1], season_sigma);
    away_baseline[season_i] ~ normal(away_baseline[season_i-1], season_sigma);
  }

  for (i in 1:n_games) {
    HomeGoals[i] ~ poisson(lambda_home[i]);
    AwayGoals[i] ~ poisson(lambda_away[i]);
  }
}
