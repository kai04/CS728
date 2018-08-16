import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import networkx as nx
import pylab

directed = True
alpha = 0.1

G = nx.DiGraph() if directed else nx.Graph()
maxnode = 5
Nodes=range(maxnode)
G.add_nodes_from(Nodes)
Edges=[(0,1), (1,2), (2,0), (4,3), (3,0) ]
G.add_edges_from(Edges)
nodepos = nx.spring_layout(G)

adj = np.zeros((maxnode, maxnode))
for (fr, to) in Edges:
    adj[fr, to] = 1

if not directed:
    adj = np.maximum(adj, np.transpose(adj))
print adj

rowsums = np.sum(adj, axis=1)
transitions = adj / rowsums[:, np.newaxis]
print transitions

teleport = np.ones(maxnode) / maxnode
pres = [0, 0, 0, 0, 1]

fig = pylab.figure()
pylab.ion()

for iter in xrange(40):
    nx.draw(G, nodepos, node_color=pres, cmap=plt.get_cmap('Greys'),
            vmin=0, vmax=1,
            edge_color='k', with_labels=True)
    fig.canvas.draw()
    print 'time', iter, pres
    time.sleep(.1)
    pres = alpha * np.dot(pres, transitions) + (1. - alpha) * teleport

pylab.ioff()
plt.show()
