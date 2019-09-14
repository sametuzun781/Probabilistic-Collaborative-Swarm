import time
import numpy as np
from scipy.sparse import identity
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse import kron

def mh_markov(N_bins, ita0, alpha, N_agent, A_a):
    """ 
    Metropolis-Hasting Algorithm
    """
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Consensus Algorithm
    Lc = N_agent*identity(N_agent) - csr_matrix(np.ones((N_agent,N_agent)))
    delta = 1/(2*N_agent-2)

    X = (identity(N_agent)-delta*Lc)
    Y = identity(N_bins)
    ita_mult = kron(X,Y)
    X = []
    Y = []
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Define Proposal Matrix
    Y = np.ones((N_bins,1))
    Y = csr_matrix(A_a).dot(csr_matrix(Y))
    Y = Y.transpose()
    Y.data **= -1

    proposal_matrix = csr_matrix(A_a.transpose()).multiply(Y) 
    A_a  = []
    Y = []
    # print('Check: Sum of Proposal Matrix (K) Matrix: {}'.format(np.mean(proposal_matrix.transpose().dot(lil_matrix(np.ones((N_bins,1)))))))

    cdf_markov = np.zeros([5,N_bins*N_agent])
    M_new = np.zeros([5,N_bins*N_agent])
    M_where = np.zeros([5,N_bins*N_agent])

    for agent in range(N_agent):
        v_desired = ita0[N_bins*(agent):N_bins*(agent+1)]

        # Define Intermediary Matrix
        Y = (proposal_matrix.multiply(v_desired))
        Y.data **= -1
        intermediary_matrix = (proposal_matrix.multiply(v_desired)).transpose().multiply(Y)
        Y = []

        # Define Acceptance Matrix
        acceptance_matrix = alpha * intermediary_matrix.minimum(1)
        intermediary_matrix = []

        # Define Markov Matrix
        markov_matrix = (proposal_matrix.multiply(acceptance_matrix))

        markov_matrix = markov_matrix.toarray()

        M_ii_sum = ((1-acceptance_matrix.toarray()) * proposal_matrix.toarray()).sum(axis=0)
        acceptance_matrix = []

        N_bins_list = list(range(N_bins))
        markov_matrix[N_bins_list,N_bins_list] = (proposal_matrix.toarray()[N_bins_list,N_bins_list] + M_ii_sum[N_bins_list].transpose())
        M_ii_sum = []

    # PMF to CDF for PGA
        N_filled_row = np.sum(markov_matrix[:,:] != 0, axis=1)
        for i in range(N_bins):
            N_filled_row_i = int(N_filled_row[i])
            M_where[:N_filled_row_i,(i)+agent*N_bins] = np.nonzero(markov_matrix[:,i])[0]
            M_new[:N_filled_row_i,(i)+agent*N_bins]   = markov_matrix[np.nonzero(markov_matrix[:,i])[0],i]
    
    markov_matrix = []
    cdf_markov[0,:] = M_new[0,:]
    for i in range(5-1):
        cdf_markov[i+1,:] = cdf_markov[i,:] + M_new[i+1,:] 

    M_new = []

    return cdf_markov, M_where, v_desired, ita_mult