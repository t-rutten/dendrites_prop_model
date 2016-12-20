import numpy as np
from scipy.stats import norm
from itertools import combinations
import simulated_model_gen as mod_gen
import pickle
import time
import matplotlib.pyplot as plt

# from pudb import set_trace


class Neuron:
    def __init__(self, graph, observed, true_params, lambs_ids_groups, parents):
        self.graph = graph
        self.observed = observed
        self.true_params = true_params
        self.lambs_ids = self.assign_lambdas(lambs_ids_groups)
        self.calcium, self.data = self.gen_toy_data(parents)

        self.messages = {}
        self.root = 0
        self.params = {'a': 1., 'b': 0., 'var_y': .4, 'lambs': [.4, .4]}
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

        return var_m

    def EM(self):
        lastLL = None
        newLL = None
        # set_trace()
        sse_trials = []
        ll_trials = []
        counter = 0

        
        best_sse_iteration = 0
        best_sse_value = np.inf

        start = time.time()

        tracker = {'a':[], 'lamb1':[], 'lamb2':[],'b':[], 'var_y':[]}

        while ((lastLL is None) or (abs(newLL - lastLL) > 1.0 * 10**(-5))):
            
            # run E step and M step while we have not converged
            self.E_step()
            self.M_step()
            lastLL = newLL
            newLL = self.Log_Likelihood()
            
            sse = np.sum([(c - e[0])**2 for c, e in zip(self.calcium, self.estimates)])
            msse = sse/len(self.calcium)

            #add history of the parameters
            a = self.params['a']
            b = self.params['b']
            lamb1 = self.params['lambs'][0]
            lamb2 = self.params['lambs'][1]
            var_y = self.params['var_y']

            tracker['a'].append(a)
            tracker['b'].append(b)
            tracker['lamb1'].append(lamb1)
            tracker['lamb2'].append(lamb2)
            tracker['var_y'].append(var_y)

            sse_trials.append(msse)
            ll_trials.append(newLL)
            
            if sse < best_sse_value:
                best_sse_value = msse
                best_sse_iteration = counter

            if counter % 100 == 0:
                print "params: %s" % str(self.params)
                print "log likelihood: %f" % newLL
                print "SSE: " + str(sse)
            
            counter += 1
        
        end = time.time()
        print "\nTime to converge: %s sec\n" % str(end - start)
        print "Amount of iterations to converge: %d\n" % counter
        print "Best SSE value: %f at iteration %d" % (best_sse_value,best_sse_iteration)
        
        return ll_trials, sse_trials, best_sse_value, tracker

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

        for (node_i, node_j) in combinations(xrange(len(self.graph)), 2):
            try:
                l_id = self.lambs_ids[(node_i, node_j)]
                # l_updates[l_id] += (self.estimates[node_i] -
                #                     self.estimates[node_j])**2
                Ji = self.get_messages_excluding(node_i, node_j) / \
                    (self.get_messages_excluding(node_i, node_j) +
                        self.params['lambs'][l_id])
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

def run_tests(big_graph = False):
    if not big_graph:
        a_size=50
        b_size=20
    else:
        a_size=100
        b_size=50

    graph, parents, lamb_group_ids = mod_gen.make_graph(apical_size=a_size, basal_size=b_size)

    observed = [True] * len(graph)
    true_params = {'a': 1, 'b': 0, 'var_y': .1, 'lambs': [.1, .2], 'a_var': .1, 'a_mean': 1}
    neuron = Neuron(graph, observed, true_params, lamb_group_ids, parents)
    ll_trials, sse_trials, best_sse_value, tracker = neuron.EM()

    if not big_graph:
        prefix = 'trials_msse-%f_' % best_sse_value
    else:
        prefix = 'trials_big_graph_msse-%f_' % best_sse_value

    plt.scatter(range(np.shape(ll_trials)[0]),ll_trials)
    plt.title('Complete log likelihood during EM')
    plt.ylabel('Complete log likelihood')
    plt.xlabel('Iteration of algorithm')
    # plt.show()
    plt.savefig('%s_ll_fig.pdf' % prefix)

    plt.clf()

    plt.plot(range(np.shape(sse_trials)[0]),sse_trials)
    plt.title('Mean squared error rates during EM')
    plt.ylabel('Mean SSE')
    plt.xlabel('Iteration of algorithm')
    # plt.show()
    plt.savefig('%s_sse_fig.pdf' % prefix)

    f = open('%s.p' % prefix,'wb')
    history = [ll_trials, sse_trials]

    plt.clf()

    iters = len(tracker['a'])

    fig = plt.figure()
    ax_a = fig.add_subplot(111)
    ax_a.plot([a for a in tracker['a']],color='b',label=r'$a$')
    
    ax_a_true = fig.add_subplot(111)
    ax_a_true.plot([true_params['a']] * iters,'--',color='b', markevery=5, label=r'True $a$')

    plt.legend()

    ax = fig.add_subplot(111)
    ax.plot([a for a in tracker['b']],color='g',label=r'$b$')
    ax = fig.add_subplot(111)
    ax.plot([true_params['b']] * iters,'--',color='g',markevery=5,label=r'True $b$ $(0)$')
    ax = fig.add_subplot(111)

    ax = fig.add_subplot(111)
    ax.plot([a for a in tracker['lamb1']],color='r',label=r'$\sigma_{c1}$')
    ax = fig.add_subplot(111)
    ax.plot([true_params['lambs'][0]] * iters,'--',color='r',markevery=5,label=r'True $\sigma_{c1}$')
    ax = fig.add_subplot(111)

    ax = fig.add_subplot(111)
    ax.plot([a for a in tracker['lamb2']],color='c',label=r'$\sigma_{c2}$')
    ax = fig.add_subplot(111)
    ax.plot([true_params['lambs'][1]] * iters,'--',color='c',markevery=5,label=r'True $\sigma_{c2}$')
    ax = fig.add_subplot(111)

    ax = fig.add_subplot(111)
    ax.plot([a for a in tracker['var_y']],color='m',label=r'$\sigma_Y$')
    ax = fig.add_subplot(111)
    ax.plot([true_params['var_y']] * iters,'--',color='m',markevery=5,label=r'True $\sigma_Y$')
    ax = fig.add_subplot(111)

    plt.legend(fontsize=10)

    ax.set_title("Model parameters during EM")
    ax.set_ylabel('Value of parameter')
    ax.set_xlabel('Iteration of algorithm')

    fig.savefig('%s_params_fig.pdf' % prefix)
    
    print "see file with \'%s\' prefix" % prefix

    pickle.dump(history,f)
    f.close()


def run_tests_gpu(big_graph = False):
    if not big_graph:
        a_size=50
        b_size=20
    else:
        a_size=300
        b_size=200

    graph, parents, lamb_group_ids = mod_gen.make_graph(apical_size=a_size, 
        basal_size=b_size)

    observed = [True] * len(graph)
    true_params = {'a': 1, 'b': 0, 'var_y': .1, 'lambs': [.1, .2], 'a_var': .1, 'a_mean': 1}
    neuron = Neuron(graph, observed, true_params, lamb_group_ids, parents)
    ll_trials, sse_trials, best_sse_value = neuron.EM()

    if not big_graph:
        prefix = 'trials_msse-%f_' % best_sse_value
    else:
        prefix = 'trials_big_graph_msse-%f_' % best_sse_value

    f = open('%s.p' % prefix,'wb')
    history = [ll_trials, sse_trials]
    
    print "see file with \'%s\' prefix" % prefix

    pickle.dump(history,f)
    f.close()

def make_plots(pick): 
    with open(pick,'rb') as handle:
        history = pickle.load(handle)
    ll_trials = history[0]
    sse_trials = history[1]

    plt.scatter(range(np.shape(ll_trials)[0]),ll_trials)
    plt.title('Complete Log likelihood during EM iterations')
    plt.ylabel('Complete log likelihood')
    plt.xlabel('Iteration')
    # plt.show()
    plt.savefig('ll_fig_from_pickle_%s.pdf' % pick[:-2])

    plt.clf()

    plt.plot(range(np.shape(sse_trials)[0]),sse_trials)
    plt.title('Error rates during during EM iterations')
    plt.ylabel('Mean SSE')
    plt.xlabel('Iteration')
    # plt.show()
    plt.savefig('sse_fig_from_pickle_%s.pdf' % pick[:-2])


if __name__ == '__main__':
    pass
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

    # graph, parents, lamb_group_ids = mod_gen.make_graph()


    # observed = [True] * len(graph)
    # true_params = {'a': 1, 'b': 0, 'var_y': .1, 'lambs': [.1, .2], 'a_var': .1, 'a_mean': 1}
    # neuron = Neuron(graph, observed, true_params, lamb_group_ids, parents)
    # neuron.EM()

    # run_tests()
    # run_tests(big_graph=True)

    # run_tests_gpu(big_graph=True)
    # make_plots('trials_best_mse-0.050438.p')

    # set_trace()

    # Infer Missing values
