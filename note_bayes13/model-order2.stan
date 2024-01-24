data {
  int N;
  int G[2];
  int I[2];
  int LW[sum(G), 5];
}

parameters {
  ordered[5] performance5[G[1]];
  ordered[3] performance3[G[2]];
  vector[N] mu;
  real<lower=0> s_mu;
  vector<lower=0>[N] s_pf;
}

model {
  mu ~ normal(0, s_mu);
  s_mu ~ gamma(10, 10);
  s_pf ~ gamma(10, 10);

  for (r in 1:2){
    for (g in 1:G[r]){
      for (i in 1:I[r]){
        if (r==1){
          performance5[g,i] ~ normal(mu[LW[g,i]], s_pf[LW[g,i]]);
        } else {
          performance3[g,i] ~ normal(mu[LW[g,i]], s_pf[LW[g,i]]);
        }
      }
    }
  }
}



// data {
//   int N;      // num of horses
//   int G[18];  // num of races
//   int HorseID[sum(G),18,18];
// }
// 
// parameters {
//   ordered[17] performance17[G[17]];
//   ordered[18] performance18[G[18]];
//   
//   vector[N] mu_h;
//   real<lower=0> s_mu_h;
//   vector<lower=0>[N] s_pf_h;
// }
// 
// model {
//   for (r in 17:18){
//     for (g in 1:G[r]){
//       for (i in 1:r){
//         if (r==17)
//           performance17[g,i] ~ normal(mu_h[HorseID[g,i,r]], 0);
//         else if (r==18)
//           performance18[g,i] ~ normal(mu_h[HorseID[g,i,r]], 0);
//       }
//     }
//   }
// 
//   mu_h ~ normal(0, s_mu_h);
//   s_pf_h ~ gamma(10, 10);
// }
