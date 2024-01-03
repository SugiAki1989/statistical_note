data {
  int N;
  int week[N];
  int arrest[N];
}

parameters {
  real shape;
  real scale;
}

model {
  for(n in 1:N){
    if(arrest[n] == 0){
      target += weibull_lccdf(week[n]| shape, scale);
    }else{
      target += weibull_lpdf(week[n]| shape, scale);
    }
  }
}

generated quantities {
  real pred_Y[N];

  for(n in 1:N){
    pred_Y[n] = (1 - weibull_cdf(week[n], shape, scale));
  }
}
