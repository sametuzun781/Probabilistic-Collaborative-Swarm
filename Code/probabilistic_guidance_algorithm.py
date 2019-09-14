import time
import numpy as np
import metropolis_hasting as mh

def pga(N_bins, x_init, N_time, m_row, m_column, res_dist_ite, Time_to_print, alpha, N1, N2, N_agent, ita0, A_a):
    """ 
    Probablilistic Guidance Algorithm (PGA)
    """

    curr_dist = np.random.choice(np.arange(0, N_bins), size = N_agent, p=x_init[:,0]).astype(np.int32)
    total_variation = np.zeros([N_time])
    Delta_error = np.zeros([N_time])

    res_dist = np.zeros([m_row,m_column,int(N_time/res_dist_ite)+1])

    # Initial distribution
    for k in range(N_agent):
        i = int(curr_dist[k]/m_column)
        j = int(curr_dist[k]%m_column)
        res_dist[i,j,0] += 1
        
    counter = 1
    for t in range(N_time):
        ts_step = time.time() if t%Time_to_print == 0 else 0
        print("Time: {} of {}".format(t,N_time)) if t%Time_to_print == 0 else 0

        cdf_markov, M_where, v_desired, ita_mult = mh.mh_markov(N_bins, ita0, alpha, N_agent, A_a)

        z = np.random.uniform(0,1, size=N_agent)  

        M_where_state = np.argmax(cdf_markov[:,curr_dist + np.arange(N_agent)*N_bins] > z, axis=0)
        curr_dist = M_where[M_where_state, curr_dist + np.arange(N_agent)*N_bins].astype(np.int32)

        unique, counts = np.unique(curr_dist, return_counts=True)
        count_agent = np.zeros(v_desired.size)
        unique_list = unique.tolist()
        count_agent[[int(i) for i in unique_list]] = counts
        total_variation[t] = np.sum(np.abs(count_agent/N_agent - v_desired))
        
        if t%res_dist_ite == 0:
            for k in range(N_agent):
                i = int(curr_dist[k]/m_column)
                j = int(curr_dist[k]%m_column)
                res_dist[i,j,counter] += 1
            counter += 1

        ita_mat = np.reshape(ita0, (N_agent,N_bins))
        
        ita_mat_avg = np.sum(ita_mat,axis=0)/N_bins
        ita_mat_minsq = np.square(ita_mat - ita_mat_avg)
        ita_mat_minsqavg = (np.sum(ita_mat_minsq,axis=0))/(N_bins-1)
        Delta_error_vec = np.sqrt(ita_mat_minsqavg)
        Delta_error[t] = np.linalg.norm(Delta_error_vec)

        ita0 = ita_mult.dot(ita0)
            
        print("Total Variation: {}".format(total_variation[t])) if t%Time_to_print == 0 else 0    
        print("Time to Solve: {}".format((time.time() - ts_step)*Time_to_print)) if t%Time_to_print == 0 else 0

    return res_dist, total_variation, counter, Delta_error