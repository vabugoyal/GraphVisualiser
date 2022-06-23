import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import collections


pd.set_option("display.max_rows", None, "display.max_columns", None)

VISITED = "grey"
ACTIVE = "green"
CURRENT = "red"
PLOTNO = 0


########################################## DFS ###############################################

def dfsGO(node, G, nodeColors, edgeColors, idx, vis, par):
    vis[node] = 1
    nodeColors[node] = CURRENT
    myplot(G, nodeColors, edgeColors)

    for neigh in G[node].keys():
        if (vis[neigh] == 0):
            nodeColors[node] = ACTIVE
            dfsGO(neigh, G, nodeColors, edgeColors, idx, vis, node)
        edgeColors[idx[(node, neigh)]] = VISITED
        nodeColors[node] = CURRENT
        myplot(G, nodeColors, edgeColors)

    nodeColors[node] = VISITED


def startDFS(G):
    nodes = G.number_of_nodes()
    edges = list(G.edges())
    nodeColors = (nodes + 1) * ["black"]
    edgeColors = (len(edges)) * ["black"]

    idx = {tuple(edges[i]): i for i in range(len(edges))}
    vis = (nodes + 1) * [0]

    # displaying the graph for first time
    myplot(G, nodeColors, edgeColors)

    for node in range(1, nodes + 1):
        if (vis[node]): continue
        dfsGO(node, G, nodeColors, edgeColors, idx, vis, -1)

    myplot(G, nodeColors, edgeColors)







########################################## BFS ###############################################

def bfsGO(node, G, nodeColors, edgeColors, idx, vis):
    Q = collections.deque()
    Q.append(node)
    vis[node] = 1
    while(len(Q)):
        N = Q[0]
        Q.popleft()
        nodeColors[N] = CURRENT
        myplot(G, nodeColors, edgeColors)
        for neigh in G[N].keys():
            if (vis[neigh] == 0):
                vis[neigh] = 1
                nodeColors[neigh] = ACTIVE
                Q.append(neigh)
            edgeColors[idx[(N, neigh)]] = VISITED
            myplot(G, nodeColors, edgeColors)

        nodeColors[N] = VISITED


def startBFS(G):
    nodes = G.number_of_nodes()
    edges = list(G.edges())
    nodeColors = (nodes + 1) * ["black"]
    edgeColors = (len(edges)) * ["black"]

    idx = {tuple(edges[i]): i for i in range(len(edges))}
    vis = (nodes + 1) * [0]

    # displaying the graph for first time
    myplot(G, nodeColors, edgeColors)

    for node in range(1, nodes + 1):
        if (vis[node]): continue
        bfsGO(node, G, nodeColors, edgeColors, idx, vis)

    myplot(G, nodeColors, edgeColors)









################################ Main where some algo starts #############################

def start(g, nodes):
    """
    This will make the images at various instants of time
    and store them in output folder and will return the number of
    images formed.
    """
    global PLOTNO

    G = nx.DiGraph()

    for i in range(1, nodes + 1):
        G.add_node(i)

    G.add_edges_from(g)
    PLOTNO = 0

    startBFS(G)
    return PLOTNO




def myplot(G, nodeColors, edgeColors):
    """This will create an image of the graph at particular point."""
    global PLOTNO
    PLOTNO += 1
    nodeColors = nodeColors[1:]
    # this will set the positions of the nodes
    # pos = nx.spring_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=50, weight='weight', seed=42)
    # pos = nx.planar_layout(G)

    #     rcParams['figure.figsize'] = 5, 5
    #     pos = nx.spring_layout(G, scale=20, k=3 / np.sqrt(G.order()), seed=42)
    #     pos = nx.planar_layout(G, scale=3 / np.sqrt(G.order()))
    #     pos = nx.spring_layout(G, k=0.15, iterations=20)
    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    for row, data in nx.shortest_path_length(G):
        for col, dist in data.items():
            df.loc[row, col] = dist

    df = df.fillna(df.max().max())
    pos = nx.kamada_kawai_layout(G, dist=df.to_dict())

    options = {
        "font_size": 10,  # 36
        "font_color": "black",
        "node_size": 1000,  # 3000
        "node_color": "white",
        "edgecolors": nodeColors,
        "edge_color": edgeColors,
        "linewidths": 2,
        "width": 2,
    }
    nx.draw_networkx(G, pos, **options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.savefig(f"static/output/{PLOTNO}.png")
    plt.figure()