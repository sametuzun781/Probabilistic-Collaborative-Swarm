import numpy as np

def grid_env(m_row, m_column, N1, N2, ita0_N1_ind, ita0_N2_ind):

	N_bins = m_row*m_column # Number of bins
	N_agent = N1 + N2 # Number of agent
	img = np.zeros([m_row,m_column])
	print('Number of agents: {} and {}'.format(N1, N2))

	ksi = np.ones([N_bins*N_agent,1])*1/N_bins # Vector of probablities
	x_init = 1/(N_agent) * np.matmul((np.kron(np.ones([1,N_agent]), np.identity(N_bins))), ksi)
	print('Check: Sum of Vector of probablities: {}'.format(np.sum(x_init,axis=0)))

	ita0 =  np.zeros([N_agent*N_bins]) # Specified distribution

	ita0_N1_ind_all = []
	ita0_N2_ind_all = []

	# Desired bins for every agent
	for i in range(len(ita0_N1_ind)):
		ita0_N1_ind_all = ita0_N1_ind_all + list(range(ita0_N1_ind[i],N_bins*N1,N_bins))

	ita0_N2_ind = [x + N_bins*N1 for x in ita0_N2_ind] 

	for i in range(len(ita0_N2_ind)):
		ita0_N2_ind_all = ita0_N2_ind_all + list(range(ita0_N2_ind[i],N_bins*N2+N_bins*N1,N_bins))

	ita0_emp_ind = [ i for i in list(range(0,N_agent*N_bins)) if i not in (ita0_N1_ind_all+ita0_N2_ind_all)]

	emp_prop = 0.001
	ita0[ita0_emp_ind] = emp_prop
	ita0[ita0_N1_ind_all] = (1-emp_prop*(N_bins-len(ita0_N1_ind)))/len(ita0_N1_ind)
	ita0[ita0_N2_ind_all] = (1-emp_prop*(N_bins-len(ita0_N2_ind)))/len(ita0_N2_ind)
	print('Check: Sum of Desired Distribution: {}'.format(np.sum(ita0,axis=0)))

	# Define adjacency matrix
	A_a = np.zeros([int(N_bins),int(N_bins)]) # Allowable transitions

	for i in range(N_bins):
		# Edges
		if (0<i) and (i<m_column-1): # Up
			A_a[i+1,i] = 1
			A_a[i-1,i] = 1
			A_a[i+m_column,i] = 1
		elif (m_column*(m_row-1) < i) and (i < (m_row*m_column)-1): # Bottom
			A_a[i+1,i] = 1
			A_a[i-1,i] = 1
			A_a[i-m_column,i] = 1
		elif (i%m_column == 0) and (i != 0) and ( i != m_column*(m_row-1)): # Left
			A_a[i+1,i] = 1
			A_a[i+m_column,i] = 1
			A_a[i-m_column,i] = 1
		elif (i%m_column == m_column-1) and (i != m_column-1) and (i != (m_row*m_column)-1): # Right
			A_a[i-1,i] = 1
			A_a[i+m_column,i] = 1
			A_a[i-m_column,i] = 1        
		# Corners        
		elif i == 0:
			A_a[i+1,i] = 1
			A_a[i+m_column,i] = 1
		elif i == m_column-1:
			A_a[i-1,i] = 1
			A_a[i+m_column,i] = 1
		elif i ==  m_column*(m_row-1):
			A_a[i+1,i] = 1
			A_a[i-m_column,i] = 1
		elif i ==  (m_row*m_column)-1:
			A_a[i-1,i] = 1
			A_a[i-m_column,i] = 1
		# The rest of pixels??    
		else:
			A_a[i+1,i] = 1
			A_a[i-1,i] = 1
			A_a[i+m_column,i] = 1
			A_a[i-m_column,i] = 1

	return N_bins, N_agent, x_init, ita0, A_a, img