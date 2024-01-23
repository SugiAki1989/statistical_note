// model {
// for(i in 1:n_games) {
//   HomeGoals[i] ~ dpois(lambda_home[HomeTeam[i],AwayTeam[i]])
//   AwayGoals[i] ~ dpois(lambda_away[HomeTeam[i],AwayTeam[i]])
// }
// 
// for(home_i in 1:n_teams) {
//   for(away_i in 1:n_teams) {
//     lambda_home[home_i, away_i] <- exp( home_baseline + skill[home_i] - skill[away_i])
//     lambda_away[home_i, away_i] <- exp( away_baseline + skill[away_i] - skill[home_i])
//   }
// }
// 
// skill[1] <- 0 
// for(j in 2:n_teams) {
//   skill[j] ~ dnorm(group_skill, group_tau)
// }
// 
// group_skill ~ dnorm(0, 0.0625)
// group_tau <- 1/pow(group_sigma, 2)
// group_sigma ~ dunif(0, 3)
// 
// home_baseline ~ dnorm(0, 0.0625)
// away_baseline ~ dnorm(0, 0.0625)
// }


data {
  int<lower=1> n_games;                          // ゲームの総数
  int<lower=1> n_teams;                          // チームの総数
  int<lower=1, upper=n_teams> HomeTeam[n_games]; // 各ゲームのホームチームのインデックス
  int<lower=1, upper=n_teams> AwayTeam[n_games]; // 各ゲームのアウェイチームのインデックス
  int<lower=0> HomeGoals[n_games];               // 各ゲームのホームチームの得点
  int<lower=0> AwayGoals[n_games];               // 各ゲームのアウェイチームの得点
}

parameters {
  real home_baseline;        // ホームチームのベースラインのパラメータ
  real away_baseline;        // アウェイチームのベースラインのパラメータ
  real skill[n_teams];       // 各チームのスキル
  real<lower=0> group_sigma; // グループの標準偏差
  // real group_skill;          // グループのスキル
}

transformed parameters {
  matrix[n_teams, n_teams] lambda_home; // ホームチームの得点率行列
  matrix[n_teams, n_teams] lambda_away; // アウェイチームの得点率行列
  

  for (home_i in 1:n_teams) {
    for (away_i in 1:n_teams) {
      lambda_home[home_i, away_i] = exp(home_baseline + skill[home_i] - skill[away_i]);
      lambda_away[home_i, away_i] = exp(away_baseline + skill[away_i] - skill[home_i]);
    }
  }
}

model {
  group_sigma ~ uniform(0, 3);
  // group_skill ~ normal(0, 4);
  home_baseline ~ normal(0, 4);
  away_baseline ~ normal(0, 4);

  for (n in 1:n_teams) {
    skill[n] ~ normal(0, group_sigma);
  }
  
  for (i in 1:n_games) {
    HomeGoals[i] ~ poisson(lambda_home[HomeTeam[i], AwayTeam[i]]);
    AwayGoals[i] ~ poisson(lambda_away[HomeTeam[i], AwayTeam[i]]);
  }
}
