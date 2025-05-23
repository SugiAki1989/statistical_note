data {
    int N;
    int G;
    int Y;
    int<lower=1> GY[G];
    int<lower=1, upper=N> LW[G, 2];
}

parameters {
    ordered[2] performance[G];
    matrix<lower=0>[N, Y] mu;
    matrix<lower=0>[N, Y] s_mu;
    matrix<lower=0>[N, Y] s_pf;
}

model {
    for (g in 1:G)
      for (i in 1:2)
        performance[g, i] ~ normal(mu[ LW[g, i], GY[g] ], s_pf[ LW[g, i], GY[g] ]);

    // 各選手の初年度の強さは平均0、標準偏差$\sigma_{\mu}[n][1]$の半正規分布に従う
    for (n in 1:N)
      mu[n, 1] ~ normal(0, s_mu[n, 1]);

    for (n in 1:N)
      for (y in 2:Y)
        mu[n, y] ~ normal(mu[n, y-1], s_mu[n, y]);
    
    // 各選手の年度別の強さの変化具合$\sigma_\mu[n][y]$は半正規分布に従う
    for (n in 1:N)
      s_mu[n] ~ normal(0, 1);

    for (n in 1:N)
      s_pf[n] ~ gamma(10, 10);
}
