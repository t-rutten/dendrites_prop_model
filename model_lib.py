import numpy as np
from itertools import combinations

from pudb import set_trace


class Neuron:
    def __init__(self):
        self.graph = {0: [1], 1: [0, 2], 2: [1, 3, 4],
                      3: [2], 4: [2, 5], 5: [4]}
        self.observed = [True, False, True, True, True, True]
        self.true_params = {'a': 1, 'b': 0, 'var_y': .1, 'lambs': [.1, .2]}
        self.lambs_ids = self.assign_lambdas([[0, 1, 2], [2, 3, 4, 5]])
        self.calcium, self.data = self.gen_toy_data()

        self.messages = {}
        self.root = 0
        self.params = {'a': 1, 'b': 0, 'var_y': .1, 'lambs': [.1, .2]}
        # self.estimates = [None] * len(self.graph)
        self.estimates = self.calcium

    def assign_lambdas(self, groups):
        lambs_ids = {}
        for gi, g in enumerate(groups):
            for node in g:
                for neighbor in self.graph[node]:
                    if neighbor in g:
                        lambs_ids[(node, neighbor)] = gi
        return lambs_ids

    def gen_toy_data(self):
        calcium = [None] * len(self.graph)
        INIT_VAL = 10
        calcium[0] = INIT_VAL

        parents = {1: 0, 2: 1, 3: 2, 4: 2, 5: 4}

        for child in [1, 2, 3, 4, 5]:
            calcium[child] = calcium[parents[child]] + np.random.randn() \
                * self.true_params['lambs'][
                    self.lambs_ids[(child, parents[child])]]

        data = [None] * len(self.graph)
        for oi, obs in enumerate(self.observed):
            if not obs:
                data[oi] = np.nan
            else:
                data[oi] = (self.true_params['a'] * calcium[oi] +
                            self.true_params['b']) + np.random.randn() \
                    * self.true_params['var_y']

        return calcium, data

    def send_message(self, source, destination):
        # getting the variance of the message
        var_m = 0
        mu_m = 0

        for k in self.graph[source]:
            if k == destination:
                pass
            else:
                messages_to_source = self.messages[(k, source)]
                # update var and mean of message
                var_m += 1. / messages_to_source[1]
                mu_m += messages_to_source[0] / messages_to_source[1]

        if(self.observed[source]):
            # only if source is observed do we add the var_y
            var_m += self.params['a']**2 / self.params['var_y']
            # data will only have source if it's observed
            mu_m += (self.data[source] - self.params['b']) * self.params['a'] \
                / (self.params['var_y'])

        var_m = 1. / var_m
        mu_m = mu_m * var_m

        var_update = var_m + self.params['lambs'][
            self.lambs_ids[(source, destination)]]

        # setting the tuple
        self.messages[(source, destination)] = (mu_m, var_update)

    def collect(self, i, j):
        for k in self.graph[j]:
            if k == i:
                pass
            else:
                self.collect(j, k)
        self.send_message(j, i)

    def distribute(self, i, j):
        self.send_message(i, j)
        for k in self.graph[j]:
            if k == i:
                pass
            else:
                self.distribute(j, k)

    def message_passing(self):
        for k in self.graph[self.root]:
            self.collect(self.root, k)
        for k in self.graph[self.root]:
            self.distribute(self.root, k)

    def get_marginal(self, node):
        var_m = 0
        mu_m = 0

        for k in self.graph[node]:
            messages_to_node = self.messages[(k, node)]
            # update var and mean of message
            var_m += 1. / messages_to_node[1]
            mu_m += messages_to_node[0] / messages_to_node[1]

        # only if source is observed do we add the var_y
        if(self.observed[node]):
            var_m += self.params['a']**2 / self.params['var_y']
            mu_m += (self.data[node] - self.params['b']) * self.params['a'] \
                / (self.params['var_y'])

        var_m = 1. / var_m
        mu_m = mu_m * var_m

        # return (mu_m, var_m)
        return mu_m

    def E_step(self):
        self.message_passing()
        self.estimates = [self.get_marginal(i) for
                          i in xrange(len(self.graph))]

    def M_step(self):

        # Update all parameter estimates

        # Update gain a and offset b by regression LSE
        C_mean = np.mean([e for ie, e in enumerate(self.estimates) if
                          not np.isnan(self.data[ie])])
        Y_mean = np.nanmean(self.data)

        # Exclude any nodes that were not observed
        cov = np.sum([(C_mean - c) * (Y_mean - y) for c, y in
                      zip(self.estimates, self.data) if not np.isnan(y)])
        var_est = np.sum([(C_mean - c)**2 for c, y in
                          zip(self.estimates, self.data) if not np.isnan(y)])

        self.params['a'] = cov / var_est
        self.params['b'] = Y_mean - self.params['a'] * C_mean

        # Update the observation variance
        self.params['var_y'] = np.sum([(y - (self.params['a'] * c +
                                       self.params['b']))**2 for c, y in
                                       zip(self.estimates, self.data) if
                                       not np.isnan(y)])
        self.params['var_y'] /= np.sum(self.observed)

        # Update all of our smoothing parameters
        l_updates = [0] * len(self.params['lambs'])
        # id_counts = [0] * len(self.params['lambs'])

        for (node_i, node_j) in combinations(xrange(len(self.graph)), 2):
            try:
                l_id = self.lambs_ids[(node_i, node_j)]
                l_updates[l_id] += (self.estimates[node_i] -
                                    self.estimates[node_j])**2
            except KeyError:
                # Unconnected node pairs will not have a lambda id,
                # just skip over these nodes
                pass

        # Normalize by number of pairs with that lambda_id
        # Note that count double-counts, so multiply by 2
        l_updates = [2 * l / self.lambs_ids.values().count(li) for
                     li, l in enumerate(l_updates)]

        self.params['lambs'] = l_updates
# get index for obsereved variable
# def _y(c):
#   return c+n_nodes

# def _make_lambda(c_i,c_j):
#   #make the lambdas symmetric for both message directions
#   lambs[(c_i,c_j)] = 1
#   lambs[(c_j,c_i)] = 1

if __name__ == '__main__':
    neuron = Neuron()
    # neuron.E_step()
    # print neuron.estimates
    # print neuron.calcium
    neuron.M_step()
    print neuron.params
    print neuron.true_params

# a function to generate data
# passes the observations to GBP() which will contain everything
# GBP() will repeatedly call the E step (message passing) and M step
# E step will spit out the marginals (estimates of Cs) and lambs
# pass these marginals into M step
# calculate ELBO
# after loop is done, infer missing data
