import numpy as np
from pdb import set_trace

class Neuron:
    def __init__(self):
        self.graph = {0:[1], 1:[0,2], 2:[1,3,4], 3:[2], 4:[2,5], 5:[4]}
        self.observed = [True,False,True,True,True,True]

        self.true_params = {'a':1, 'b':0, 'var_y':1, 'lambs':[1,2]}

        self.lambs_ids = self.assign_lambdas([[0,1,2],[2,3,4,5]])

        self.messages = {}

        self.calcium, self.data = self.gen_toy_data()

        self.root = 0

        self.params = {'a':1, 'b':0, 'var_y':1, 'lambs':[1,2]}

        

        self.estimates = [None] * len(self.graph)
        


    def assign_lambdas(self,groups):
        lambs_ids = {}
        for gi, g in enumerate(groups):
            for node in g:
                for neighbor in self.graph[node]:
                    if neighbor in g:
                        lambs_ids[(node,neighbor)] = gi
        return lambs_ids

    def gen_toy_data(self):
        calcium = [None] * len(self.graph)
        INIT_VAL = 10
        calcium[0] = INIT_VAL

        parents = {1:0, 2:1, 3:2, 4:2, 5:4}

        for child in [1, 2, 3, 4, 5]:
            #pdb.set_trace()
            calcium[child] = calcium[parents[child]] + np.random.randn() * self.true_params['lambs'][self.lambs_ids[(child, parents[child])]]

        data = [None] * len(self.graph)
        for oi, obs in enumerate(self.observed):
            if not obs:
                data[oi] = None
            else:
                data[oi] = (self.true_params['a']*calcium[oi] + self.true_params['b']) + np.random.randn() * self.true_params['var_y']

        return calcium, data


    def send_message(self,source,destination):
        #getting the variance of the message
        var_m = 0
        mu_m = 0

        for k in self.graph[source]:
            if k == destination:
                pass
            else:
                messages_to_source = self.messages[(k,source)]
                var_m += 1./messages_to_source[1] #update var of message
                mu_m += messages_to_source[0]/messages_to_source[1] #update mean of message

        if(self.observed[source]): #only if source is observed do we add the var_y
            var_m += self.params['a']**2/self.params['var_y']
            mu_m += (self.data[source] - self.params['b']) * self.params['a']/ (self.params['var_y']) #data will only have source if it's observed
        
        var_m = 1./var_m
        mu_m = mu_m * var_m

        var_update = var_m + self.params['lambs'][self.lambs_ids[(source, destination)]]

        #setting the tuple
        self.messages[(source,destination)] = (mu_m, var_update)

    def get_marginal(self,node):
        var_m = 0
        mu_m = 0

        for k in self.graph[node]:
            messages_to_node = self.messages[(k,node)]
            var_m += 1./messages_to_node[1] #update var of message
            mu_m += messages_to_node[0]/messages_to_node[1] #update mean of message

        if(self.observed[node]): #only if source is observed do we add the var_y
            var_m += self.params['a']**2/self.params['var_y']
            mu_m += (self.data[node] - self.params['b']) * self.params['a']/ (self.params['var_y']) 

        var_m = 1./var_m
        mu_m = mu_m * var_m

        return (mu_m, var_m)


    def e_step(self):
        # set_trace()
        self.message_passing()
        estimates = [self.get_marginal(i) for i in xrange(len(self.graph))]
        return estimates


    def collect(self,i, j):
        for k in self.graph[j]:
            if k == i:
                pass
            else: 
                self.collect(j, k)
        self.send_message(j, i)

    def distribute(self,i, j):
        self.send_message(i, j)
        for k in self.graph[j]:
            if k == i:
                pass
            else:
                self.distribute(j, k)

    def message_passing(self):
        for k in self.graph[self.root]:
            self.collect(self.root,k)
        for k in self.graph[self.root]:
            self.distribute(self.root,k)


#get index for obsereved variable
# def _y(c):
#   return c+n_nodes

# def _make_lambda(c_i,c_j):
#   #make the lambdas symmetric for both message directions
#   lambs[(c_i,c_j)] = 1
#   lambs[(c_j,c_i)] = 1

if __name__ == '__main__':
    neuron = Neuron()
    print neuron.e_step()
    print neuron.calcium




#a function to generate data
#passes the observations to GBP() which will contain everything
#GBP() will repeatedly call the E step (message passing) and M step
#E step will spit out the marginals (estimates of Cs) and lambs
#pass these marginals into M step
#calculate ELBO
#after loop is done, infer missing data 


