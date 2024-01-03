data {
  int N ;
  int week[N] ;
  int arrest[N] ;
  int fin[N] ;
}

parameters {
  real shape ;
  real beta[2] ;
}

model {
  for(n in 1:N){
    if(arrest[n] == 0){
      target += weibull_lccdf(week[n]| shape, exp(- (beta[1] + fin[n] * beta[2]) / shape)) ;
    }else{
      target += weibull_lpdf(week[n]| shape, exp(- (beta[1] + fin[n] * beta[2]) / shape)) ;
    }
  }
}

generated quantities {
  real pred_Y1[N];
  real pred_Y2[N];

  for(n in 1:N){
    pred_Y1[n] = (1 - weibull_cdf(week[n], shape, exp(- (beta[1] + beta[2]) / shape)));
    pred_Y2[n] = (1 - weibull_cdf(week[n], shape, exp(- (beta[1]) / shape)));
  }
}
