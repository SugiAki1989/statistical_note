data {
  int<lower=1> J;
  real Y[J];
  real<lower=0> S[J];
}
parameters {
  real theta[J];
  real mu;
  real<lower=0> sigma;
}
model {
  // mu ~ uniform(-100,100);
  // sigma ~ normal(0, 50);

  for (j in 1:J) {
    theta[j] ~ normal(mu, sigma);
  }

  for (j in 1:J) {
    Y[j] ~ normal(theta[j], S[j]);
  }
}

// reparameterization version
// https://note.com/hanaori/n/n5e64896c0a30
// 
// data {
//     int<lower=1> J;            // num of schools
//     real Y[J];                 // estimated treatment effects
//     real<lower=0> S[J];    // se of effect estimates
// }
// parameters {
//     real mu;                   // population mean
//     real<lower=0> tau;         // population sd
//     real eta[J];               // school-level errors
// }
// transformed parameters {
//     real theta[J];             // school effects
//     for (j in 1:J) {
//         theta[j] <- mu + tau * eta[j];
//     }
// }
// model {
//   for (j in 1:J) {
//     eta[j] ~ normal(0, 1);
//   }
// 
//   for (j in 1:J) {
//     Y[j] ~ normal(theta[j], S[j]);
//   }
// }