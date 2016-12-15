n_nodes = 12

#calcium pairs
calcium_pairs = [(0,1),(1,2),(2,3),(3,4),(3,5),(5,6)]

observed = [] #representing which calcium nodes have observations--k-hot vector

#will contain tuple of mean and var
messages = {} #

a = 1 #emission variable
b = 0 #emission variable
var_y = 1 #variance sigma_y^2, emission variable

lambs = {} #lambdas between the calcium nodes

data = [] #length of n, will include the observed y values for observed y
#and None if unobserved  

def initialize():
	for c_node in xrange(len(observed)):
		if observed[c_node]: 
			#this is the message from y_i to c_i
			#tuple of mean and variance
			messages[(_y(c_node),c_node)] = ((data[c_node] - b)/a,var_y/a**2)
	for c_pair in calcium_pairs:
		_make_lambda(c_pair[0],c_pair[1])

def send_message(source,destination,neighbors):
'''neighbors is list of source's neighbors
'''
	#getting the variance of the message
	var_m = 0
	mu_m = 0

	for i in neighbors:
		messages_to_source = messages[(i,source)]
		var_m += 1./messages_to_source[1] #update var of message
		mu_m += messages_to_source[0]/messages_to_source[1] #update mean of message

	if(observed[source]): #only if source is observed do we add the var_y
		var_m += 1./var_y
		mu_m += (data[source] - b) / (var_y * a) #data will only have source if it's observed
	
	var_m = 1./var_m
	mu_m = mu_m * var_m

	#setting the tuple
	messages[(source,destination)] = (mu_m,var_m)

#get index for obsereved variable
def _y(c):
	return c+n_nodes

def _make_lambda(c_i,c_j):
	#make the lambdas symmetric for both message directions
	lambs[(c_i,c_j)] = 1
	lambs[(c_j,c_i)] = 1

n_calcium_edges = 5 

if __name__ == '__main__':
	initialize()

	#sending all the hard-coded messages 
	# send_message()
