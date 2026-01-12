functions {
  real partial_log_lik(
    int[] race_slice,          // reduce_sum が渡す race index
    int start, int end,

    int N_teiban,
    array[,] int player_id,
    array[,] int lane_id,
    array[,] int rank,

    vector alpha,
    vector beta,
    vector sigma_u,
    cholesky_factor_corr[6] L_Omega,
    matrix z_u
  ) {
    real lp = 0;

    for (i in start:end) {
      int r = race_slice[i];

      vector[N_teiban] eta_all;
      row_vector[6] u_r_row;

      u_r_row =
        (diag_pre_multiply(sigma_u, L_Omega) * (z_u[r]'))';

      for (boat in 1:N_teiban) {
        int pid = player_id[r, boat];
        int lid = lane_id[r, boat];
        eta_all[boat] = alpha[pid] + beta[lid] + u_r_row[lid];
      }

      {
        vector[N_teiban] eta;
        for (k in 1:N_teiban) {
          int boat = rank[r, k];
          eta[k] = eta_all[boat];
        }
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
  vector[6] beta_raw;

  real<lower=1e-6> sigma_alpha;
  real<lower=1e-6> sigma_beta;

  vector<lower=1e-6>[6] sigma_u;
  cholesky_factor_corr[6] L_Omega;
  matrix[N_races, 6] z_u;
}

transformed parameters {
  vector[N_players] alpha = alpha_raw - mean(alpha_raw);
  vector[6] beta = beta_raw - mean(beta_raw);
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
    int race_index[N_races];
    for (i in 1:N_races) race_index[i] = i;

    target += reduce_sum(
      partial_log_lik,
      race_index,
      50,                 // grain size（重要）
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
  corr_matrix[6] Omega =
    multiply_lower_tri_self_transpose(L_Omega);
}
