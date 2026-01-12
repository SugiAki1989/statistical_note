functions {
  real partial_log_lik(
    array[] int race_slice,
    int start,
    int end,

    int N_teiban,
    array[,] int player_id,
    array[,] int lane_id,
    array[,] int rank,

    vector alpha,
    vector beta,
    vector sigma_u,
    matrix L_Omega,
    matrix z_u
  ) {
    real lp = 0;
    // reduce_sum により割り当てられた「race_slice の start:end 区間」だけを処理する
    for (i in start:end) {
      // レースの行インデックス（ここでは race_index = 1..N_races を渡しているので、単なる r 番目レース）
      int r = race_slice[i];
    
      vector[N_teiban] eta_all;
      row_vector[N_teiban] u_r_row;

      // レース r に固有の「レーン別ランダム効果」 u_r（非中心化：diag(sigma_u) * L_Omega * z_r）
      // u_r は N_teiban 次元で、レーン間に相関（L_Omega）とスケール（sigma_u）を持つ
      u_r_row = (diag_pre_multiply(sigma_u, L_Omega) * (z_u[r]'))';

      // 各出走スロット boat (=艇/枠) の線形予測子 eta_all[boat] を計算
      // eta_all[boat] = 選手効果 alpha[player_id] + レーン固定効果 beta[lane_id] + レース固有レーン効果 u_r[lane_id]
      for (boat in 1:N_teiban) {
        int pid = player_id[r, boat];
        int lid = lane_id[r, boat];
        eta_all[boat] = alpha[pid] + beta[lid] + u_r_row[lid];
      }

      {
        vector[N_teiban] eta;
        // 観測された順位 rank[r, k] に従って、線形予測子を「1着,2着,...」の順に並べ替える
        for (k in 1:N_teiban) {
          int boat = rank[r, k];
          eta[k] = eta_all[boat];
        }
        // Plackett–Luce の対数尤度
        for (k in 1:N_teiban) {
          lp += eta[k] - log_sum_exp(eta[k:N_teiban]);
        }
      }
    }
    return lp;
  }
}


data {
  int<lower=1> N_races;
  int<lower=1> N_teiban;
  int<lower=1> N_players;

  array[N_races, N_teiban] int player_id;
  array[N_races, N_teiban] int lane_id;
  array[N_races, N_teiban] int rank;
}

parameters {
  vector[N_players] alpha_raw;
  vector[N_teiban] beta_raw;

  real<lower=1e-6> sigma_alpha;
  real<lower=1e-6> sigma_beta;

  vector<lower=1e-6>[N_teiban] sigma_u;
  cholesky_factor_corr[N_teiban] L_Omega;
  matrix[N_races, N_teiban] z_u;
}

transformed parameters {
  // 位置の不識別性を避けるため、alpha と beta を平均0に中心化（和=0制約に相当）
  vector[N_players] alpha = alpha_raw - mean(alpha_raw);
  vector[N_teiban] beta = beta_raw - mean(beta_raw);
}

model {
  // priors
  sigma_alpha ~ exponential(1);
  sigma_beta  ~ exponential(1);
  sigma_u     ~ exponential(1);

  alpha_raw ~ normal(0, sigma_alpha);
  beta_raw  ~ normal(0, sigma_beta);
  L_Omega   ~ lkj_corr_cholesky(4);
  to_vector(z_u) ~ std_normal();

  // race index (1..N_races)
  {
    array[N_races] int race_index;
    for (i in 1:N_races) race_index[i] = i;

    target += reduce_sum(
      partial_log_lik,
      race_index,
      50,
      N_teiban,
      player_id,
      lane_id,
      rank,
      alpha,
      beta,
      sigma_u,
      L_Omega,
      z_u
    );
  }
}

generated quantities {
  // コレスキー因子から相関行列を復元
  corr_matrix[N_teiban] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
