data {
  int N;
  int G;
  simplex [N] point[G];
}

parameters {
  vector <lower=0> [N] theta;
  }

model {
  for(t in 1:G){
    point[t,] ~ dirichlet(theta);
    }
}
