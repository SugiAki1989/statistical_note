data {
  int T;               // データ取得期間の長さ
  int len_obs;         // 観測値が得られた個数
  int y[len_obs];      // 観測値
  int obs_no[len_obs]; // 観測値が得られた時点
}

parameters {
  real mu[T];          // 状態の推定値
  real<lower=0> s_mu;  // 過程誤差の標準偏差
}

model {
  s_mu ~ student_t(3, 0, 10);
  
  // 状態方程式に従い、状態が遷移する
  for(t in 2:T) {
    mu[t] ~ normal(mu[t-1], s_mu);
  }
  
  // 観測方程式に従い、観測値が得られるが「観測値が得られた時点」でのみ実行
  for(t in 1:len_obs) {
    y[t] ~ bernoulli_logit(mu[obs_no[t]]);
  }
}

generated quantities{
  real probs[T];       // 推定された勝率
  probs = inv_logit(mu);
}
