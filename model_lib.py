import numpy as np
from scipy.stats import norm
from itertools import combinations
import simulated_model_gen as mod_gen

from pudb import set_trace


class Neuron:
    def __init__(self, graph, observed, true_params, lambs_ids_groups, parents):
        self.graph = graph
        self.observed = observed
        self.true_params = true_params
        self.lambs_ids = self.assign_lambdas(lambs_ids_groups)
        self.calcium, self.data = self.gen_toy_data(parents)

        self.messages = {}
        self.root = 0
        self.params = {'a': 1, 'b': 0, 'var_y': .4, 'lambs': [.4, .4]}
        # self.params = {'a': 1, 'b': 0, 'var_y': .1, 'lambs': [.1, .2], 'a_var': 0.1}
        # self.estimates = [None] * len(self.graph)
        self.estimates = [None] * len(self.calcium)

    def assign_lambdas(self, groups):
        lambs_ids = {}
        for gi, g in enumerate(groups):
            for node in g:
                for neighbor in self.graph[node]:
                    if neighbor in g:
                        lambs_ids[(node, neighbor)] = gi
        return lambs_ids

    def gen_toy_data(self, parents):
        calcium = [None] * len(self.graph)
        INIT_VAL = 10
        calcium[0] = INIT_VAL

        for child in parents.keys():
            calcium[child] = calcium[parents[child]] + np.random.randn() \
                * np.sqrt(self.true_params['lambs'][
                    self.lambs_ids[(child, parents[child])]])

        data = [None] * len(self.graph)
        for oi, obs in enumerate(self.observed):
            if not obs:
                data[oi] = np.nan
            else:
                data[oi] = (self.true_params['a'] * calcium[oi] +
                            self.true_params['b']) + np.random.randn() \
                    * np.sqrt(self.true_params['var_y'])

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

        return (mu_m, var_m)
        # return mu_m

    def get_messages_excluding(self, node, exclude_node):
        var_m = 0
        mu_m = 0

        for k in self.graph[node]:
            if k != exclude_node:
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
        # mu_m = mu_m * var_m

        # return (mu_m, var_m)
        return var_m

    def EM(self):
        print("Ground Truth Calcium Levels")
        # print self.calcium
        print("Calcium Level Estimates")
        lastLL = None
        newLL = None
        # set_trace()
        while ((lastLL is None) or (abs(newLL - lastLL) > 1.0 * 10**(-5))):
            # run E step and M step while we have not converged
            self.E_step()
            self.M_step()
            lastLL = newLL
            newLL = self.Log_Likelihood()
            print "params: %s" % str(neuron.params)
            print "log likelihood: %f" % newLL
            print "SSE: " + str(np.sum([(c - e[0])**2 for c, e in zip(self.calcium, self.estimates)]))
        # print("Ground Truth Calcium Levels")
        # print self.calcium

    def E_step(self):
        self.message_passing()
        self.estimates = [(self.get_marginal(i)) for
                          i in xrange(len(self.graph))]

    def M_step(self):

        # Update all parameter estimates
        # Update gain a and offset b by regression LSE
        C_mean = np.mean([e[0] for ie, e in enumerate(self.estimates) if
                          not np.isnan(self.data[ie])])
        Y_mean = np.nanmean(self.data)

        # Exclude any nodes that were not observed
        cov = np.sum([(C_mean - c[0]) * (Y_mean - y) for c, y in
                      zip(self.estimates, self.data) if not np.isnan(y)])
        var_est = np.sum([(C_mean - c[0])**2 + c[1] for c, y in
                          zip(self.estimates, self.data) if not np.isnan(y)])

        # self.params['a'] = cov / var_est
        # If we want to regularize a, use the expression below
        self.params['a'] = \
            (self.true_params['a_mean'] / self.true_params['a_var'] +
                cov / self.params['var_y']) / \
            (1 / self.true_params['a_var'] + var_est / self.params['var_y'])
        self.params['b'] = Y_mean - self.params['a'] * C_mean

        # Update the observation variance
        self.params['var_y'] = np.sum([y**2 - 2 * y * (self.params['a'] *
                                       c[0] + self.params['b']) +
                                       self.params['a']**2 * (c[1] + c[0]**2) +
                                       self.params['b'] ** 2 + 2 *
                                       self.params['a'] * self.params['b'] *
                                       c[0] for c, y in zip(self.estimates,
                                       self.data) if not np.isnan(y)])

        self.params['var_y'] /= np.sum(self.observed)

        # Update all of our smoothing parameters
        l_updates = [0] * len(self.params['lambs'])
        # id_counts = [0] * len(self.params['lambs'])

        for (node_i, node_j) in combinations(xrange(len(self.graph)), 2):
            try:
                l_id = self.lambs_ids[(node_i, node_j)]
                # l_updates[l_id] += (self.estimates[node_i] -
                #                     self.estimates[node_j])**2
                Ji = self.get_messages_excluding(node_i, node_j) / \
                    (self.get_messages_excluding(node_i, node_j) +
                        self.params['var_y'])
                E_prod = self.estimates[node_j][1] * Ji + \
                    self.estimates[node_i][0] * self.estimates[node_j][0]
                l_updates[l_id] += self.estimates[node_i][1] + \
                    self.estimates[node_i][0]**2 -\
                    2 * E_prod + self.estimates[node_j][1] + \
                    self.estimates[node_j][0]**2
            except KeyError:
                # Unconnected node pairs will not have a lambda id,
                # just skip over these nodes
                pass

        # Normalize by number of pairs with that lambda_id
        # Note that count double-counts, so multiply by 2
        l_updates = [2 * l / self.lambs_ids.values().count(li) for
                     li, l in enumerate(l_updates)]

        self.params['lambs'] = l_updates

    def Log_Likelihood(self):

        ll = 0
        # Add prob of Ys P(Yi|Ci)
        for i in xrange(len(self.data)):
            if self.observed[i]:
                ll += norm.logpdf(self.data[i], loc=self.params['a'] *
                                  self.estimates[i][0] + self.params['b'],
                                  scale=np.sqrt(self.params['var_y']))

        # Add prob of Cs P(Ci, Cj)
        for (i, j) in combinations(xrange(len(self.graph)), 2):
            try:
                l_id = self.lambs_ids[(i, j)]
                ll += norm.logpdf(self.estimates[i][0] - self.estimates[j][0], loc=0,
                                  scale=np.sqrt(self.params['lambs'][l_id]))
            except KeyError:
                pass

        return ll

if __name__ == '__main__':
    # graph = {0: [1], 1: [0, 2], 2: [1, 3, 4], 3: [2], 4: [2, 5], 5: [4]}
    # observed = [True, False, True, True, True, True]
    # true_params = {'a': 1, 'b': 0, 'var_y': .01, 'lambs': [1, 1]}
    # lambs_ids_groups = [[0, 1, 2], [2, 3, 4, 5]]
    # parents = {1: 0, 2: 1, 3: 2, 4: 2, 5: 4}
    # neuron = Neuron(graph, observed, true_params, lambs_ids_groups, parents)
    # neuron.EM()

    # print("\n\n\n\n\n\n")

    # graph = {0:[1], 1:[0,2], 2:[1,3], 3:[2,4,7,9], 4:[3,5], 5:[4,6], 6:[5], 7:[3,8], 8:[7], 9:[3,10], 10:[9,11], 11:[10]}
    # observed = [True, True, False, True, False, True, True, False, True, True, False, True]
    # true_params = {'a':1, 'b':0, 'var_y':.01, 'lambs': [.1, .2]}
    # lambs_ids_groups = [[0, 1, 2, 3], [3, 4, 5, 6, 7, 8, 9, 10, 11]]
    # parents = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:3, 8:7, 9:3, 10:9, 11:10}
    # neuron = Neuron(graph, observed, true_params, lambs_ids_groups, parents)
    # neuron.EM()

    graph, parents, lamb_group_ids = mod_gen.make_graph()
    observed = [True] * len(graph)
    true_params = {'a': 1, 'b': 0, 'var_y': .1, 'lambs': [.1, .2], 'a_var': .1, 'a_mean': 1}
    neuron = Neuron(graph, observed, true_params, lamb_group_ids, parents)
    neuron.EM()
    # set_trace()

    # Infer Missing values
