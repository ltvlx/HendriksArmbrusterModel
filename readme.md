The current version of the model is simplified in the following ways:
  - only single product considered (K == 1)

  - transportation costs V are calculated via a continuous function (not as an integer part)
    to implements this, an additional integer variable u_i,j (number of trucks with capacity V) needs to be introduced:
      x_i,j <= V * u_i,j
      and in the minimized f(x) linear part should include u_i,j * c_i,j instead of V * [x_i,j] * c_i,j

  - timeframe for warehouse reaction is equal to 1 (tau_w == 1)
    this is probably the easiest point to change
  
  - prediction horizont Omega == 1