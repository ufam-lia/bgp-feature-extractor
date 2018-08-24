import networkx as nx
import matplotlib.pyplot as plt
import sys

G = nx.Graph()
as_relationships = open(sys.argv[1], 'r')
as_relationships = as_relationships.read().splitlines()
c = 0
for line in as_relationships:
    line = line.split('|')
    if line[0][0] != '#':
        # if line[0] == '1' or line[0] == '174':
        edge = [int(x) for x in line]
        G.add_edge(edge[0], edge[1], weight = edge[2])
        # c += 1
        # if c > 30:
        #     break

plt.subplot(111)
rels = nx.get_edge_attributes(G,'weight')
# print rels
# print G.edges()
# print G.nodes()

edge_col = ['red' if rels[e] < 0 else 'blue' for e in G.edges()]
# pos = nx.shell_layout(G)
# pos = nx.spring_layout(G)
pos = nx.spectral_layout(G)
# pos = nx.random_layout(G)
nx.draw(G, pos, with_labels=True, node_color='blue')
nx.draw_networkx_edges(G, pos, edge_color=edge_col)
# plt.axis('off')
plt.savefig('graph3.eps', format='eps', dpi=1000)
# plt.show()
