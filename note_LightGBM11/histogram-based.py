# histogram-based
import numpy as np
from sklearn.datasets import make_blobs
from multiprocessing.pool import Pool

# Create a training data set.
x, y = make_blobs(n_samples=300, n_features=2, centers=[[0., 0.], [0.5, 0.3]], cluster_std=0.15, center_box=(-1., 1.))

def find_local_split_point(f, s_point, g, h, G, H, r, p_score):
  GL = HL = 0.0
  l_bound = -np.inf   # lower left bound
  max_gain = -np. inf # initialize max_gain

  for j in s_point:
    # split the parent node into the left and right nodes.
    left  = np.where(np.logical_and(f > l_bound, f <= j))[0]
    right = np.where(f > j)[0]

    # After splitting the parent node, calculate the scores of its children.
    GL += g[left].sum()
    HL += (h[left] * (1. - h[left])).sum()
    GR = G - GL
    HR = H - HL
    # Calculate the gain for this split
    gain = (GL ** 2)/ (HL + r) + (GR ** 2)/(HR + r) - p_score
    
    # Find the maximum gain.
    if gain > max_gain: 
      max_gain = gain
      b_point = j # local best split point
    l_bound = j
  return b_point, max_gain

if __name__ == '__main__':
    y0 = np.ones(shape=y.shape) * 0.5 # initial prediction
    g = -(y - y0)                     # negative residual.
    h = y0 * (1.0 - y0)               # Hessian.
    # Find the best split point of each feature
    G = g.sum()
    H = h.sum()
    r = 0.0
    gamma = 0.0
    p_score = (G ** 2) / (H + r) # parent's score before splitting the node
    # Create a histogram of the parent node for each feature
    n_bin = 30 # the number of bins
    g0_parent, f0_bin = np.histogram(x[:, 0], n_bin, weights=g) # feature 0
    g1_parent, f1_bin = np.histogram(x[:, 1], n_bin, weights=g) # feature 1



    # Find global best split point through parallel processing
    # vertical partitioning method is used.
    mp = Pool(2)
    args = [[x[:, 0], f0_bin], [x[:, 1], f1_bin]]
    ret = mp.starmap_async(find_local_split_point, [(x[:, 0], f0_bin, g, h, G, H, r, p_score), (x[:, 1], f1_bin, g, h, G, H, r, p_score)])


    mp.close()
    mp.join()

    results = ret.get()
    p1 = results [0][0]
    p2 = results [1][0]
    gain1 = results[0][1]
    gain2 = results [1][1]

    if gain1 > gain2:
        b_fid = 0
        b_point = p1
    else:
        b_fid = 1
        b_point = p2

    print('\nbest feature id =', b_fid)
    print( 'best split point =', b_point.round(3))

    # best feature id = 0
    # best split point = 0.298