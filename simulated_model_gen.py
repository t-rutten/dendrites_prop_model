import numpy.random as rnd

class Node:
	'''
	node class with name of node and var indicating whether it is 
	evidence node or not
	'''
	def __init__(self,name,is_observation=False):
		self.name =  name
		self.is_observation = is_observation


def make_apical(graph, parents, num_nodes_bounds=[3,6]):
	'''
	generates nodes in axon
	graph is list of adjacencies
	nums_to_nodes_map is mapping of node names to dict with node info
	number of nodes is between min_num_nodes and max_num_nodes
	'''
	#get number of true calcium nodes in the axon according to bounds arg
	num_apical_nodes = rnd.randint(num_nodes_bounds[0],num_nodes_bounds[1])
	
	#make initial node, add to graph and nums_to_nodes_map
	graph[0] = []
	# nums_to_nodes_map['0'] = Node('0')

	#add edges between other nodes
	for i in xrange(1,num_apical_nodes):
		#add prev node to adjacency list of current node
		graph[i] = [i-1]
		parents[i] = i-1
		#add current node to adjacency list of prev node
		graph[i-1].append(i)

	return num_apical_nodes-1
		
		#add current node to nums_to_nodes_map
		# nums_to_nodes_map[str(i)] = Node(str(i))

def make_dendritic(graph, parents, num_branches_bounds=[2,3], branch_size_bounds=[1,3]):
	'''
	generates dendritic branches
	'''
	#get soma
	last_axon_node = len(graph) - 1

	#get number of dendritic branches
	num_branches = rnd.randint(num_branches_bounds[0], num_branches_bounds[1])

	#add nodes to branches
	for b in xrange(num_branches):
		#get this branch's size
		branch_size = rnd.randint(branch_size_bounds[0],branch_size_bounds[1])
		
		#get first node in branch (name is size of graph)
		next_node_num = len(graph)

		#connect cell body to first node on branch and add first node on branch
		graph[last_axon_node].append(next_node_num)
		graph[next_node_num] = []
		graph[next_node_num].append(last_axon_node)
		parents[next_node_num] = last_axon_node
		# nums_to_nodes_map[str(next_node_num)] = Node(str(next_node_num))

		#add nodes to branches, also add them to nums_to_nodes_map
		for i in xrange(len(graph),len(graph)+branch_size):
			if i not in graph.keys():
				graph[i] = [i-1]
				parents[i] = i-1
				graph[i-1].append(i)

				# nums_to_nodes_map[str(i)] = Node(str(i))

# def make_observations(graph, nums_to_nodes_map,
# 	ratio_of_evidence_nodes=0.5):
# 	'''
# 	adds evidence nodes
# 	number of evidence nodes is according to ratio arg
# 	'''
# 	#determine which nodes will have observed calcium
# 	calcium_nodes_w_observations = rnd.choice(range(len(graph)),
# 		int(ratio_of_evidence_nodes * len(graph)),False)

# 	#connect evidence nodes to corresponding true calcium nodes
# 	#add evidence nodes to graph nodes and nums_to_nodes_map
# 	for c in calcium_nodes_w_observations:
# 		current_node_num = len(graph)
# 		graph[str(c)].append(str(current_node_num))
# 		graph[str(current_node_num)] = [str(c)]
# 		nums_to_nodes_map[str(current_node_num)] = Node(str(current_node_num),True)

def make_two_lamb_groups(soma,graph):
	return [range(0,soma+1),range(soma,len(graph))]


def make_graph():
	graph = {}
	parents = {}

	soma_node = make_apical(graph,parents)
	make_dendritic(graph,parents)
	lamb_groups = make_two_lamb_groups(soma_node,graph)


	return graph, parents, lamb_groups



	
if __name__=='__main__':
	#this example creates one random graph; we can easily create many graphs
	# graph = {}
	# nums_to_nodes = {}

	# make_apical(graph,nums_to_nodes)
	# make_dendritic(graph,nums_to_nodes)
	# make_observations(graph,nums_to_nodes)

	graph, parents, lamb_group_ids = make_graph()


	for k in graph:
		print 'node: ' + str(k) + ', connected to: ' + str(graph[k])
		# print '\t' + 'node name in nums_to_nodes map (should match above): ' + nums_to_nodes[k].name
		# print '\t' + 'is evidence node: ' + str(nums_to_nodes[k].is_observation)
	print '\n'
	print 'parents:'
	for p in parents:
		print "%d, %d" % (p,parents[p])
	print '\n'
	print "lamb groups: %s\n" % str(lamb_group_ids)

