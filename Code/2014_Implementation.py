import environment as env
import metropolis_hasting as mh
import probabilistic_guidance_algorithm as pga
import plot_figs as plt_fig
import time

ts = time.time()

## Hyperparameters
# Define Environment 
m_row = 5 # Number of row
m_column = 5 # Number of column
N1 = 320 # Number of agents
N2 = 200 # Number of agents

# Desired bins for different groups of agents
ita0_N1_ind = [1,2,6,11,12,16,21,22]
ita0_N2_ind = [3,8,13,18,23]

# Probabilistic Guidance Algorithm
res_dist_ite = 1 #int(N_time/10)
Time_to_print = 5
alpha = 0.99 # For Alpha-Min Acceptance Matrix

# Plot Figures
N_time = 10
N_of_figures= 10
Fig_size_scale = 1

## Functions
N_bins, N_agent, x_init, ita0, A_a, img = env.grid_env(m_row, m_column, N1, N2, ita0_N1_ind, ita0_N2_ind)
res_dist, total_variation, counter, Delta_error = pga.pga(N_bins, x_init, N_time, m_row, m_column, res_dist_ite, Time_to_print, alpha, N1, N2, N_agent, ita0, A_a)
plt_fig.plt_fcn(N_time, N_of_figures, counter, img, Fig_size_scale, res_dist, total_variation, Delta_error, N1, ita0_N1_ind)

print("Total Time to Solve: {}".format(time.time() - ts))