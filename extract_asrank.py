import networkx as nx
import matplotlib.pyplot as plt
import sys
import pdir

g = nx.Graph()
as_relationships = open(sys.argv[1], 'r')
as_relationships = as_relationships.read().splitlines()
c = 0
nodes = set()
for line in as_relationships:
    line = line.split('|')
    if line[0][0] != '#':
        # if line[0] == '1' or line[0] == '174':
        edge = [int(x) for x in line]
        g.add_edge(edge[0], edge[1], weight = edge[2])
        nodes.add(edge[0])
        nodes.add(edge[1])
        # c += 1
        # if c > 30:
        #     break
print len(nodes)
print len(g.nodes())
print len(g.node.values())
# print pdir(g.node)
# print pdir(g.node)
# print g.nodes()

plt.figure(figsize=(5,5),dpi=2000)
plt.subplot(111)
rels = nx.get_edge_attributes(g,'weight')
# print rels
# print g.edges()
# print g.nodes()
c = 0
nodelist = list()
for i in range(0, 104):
    for j in range(0, 104):
        if c < len(g.nodes()):
            g.node.values()[c]['pos'] = (i*6, j*6)
            c += 1
            # print i, j

edge_col = ['red' if rels[e] < 0 else 'blue' for e in g.edges()]

pos = nx.get_node_attributes(g,'pos')
# pos = nx.shell_layout(G)
# pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.random_layout(G)
print type(pos)
print len(list(g))
print len(pos.values())
nx.draw(g, pos, nodelist = list(g), with_labels=True, node_color='blue', node_size = 5, font_size=2)
nx.draw_networkx_edges(g, pos, edge_color=edge_col, width = 2)
# plt.axis('off')
plt.savefig('graph3.eps', format='eps', dpi=1000)
# plt.show()
