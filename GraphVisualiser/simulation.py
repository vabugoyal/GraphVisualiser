import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import collections
import dataframe_image as dfi
import heapq

pd.set_option("display.max_rows", None, "display.max_columns", None)

VISITED = "grey"
ACTIVE = "green"
CURRENT = "red"
PLOTNO = 0


########################################## DFS ###############################################

def dfsGO(node, G, nodeColors, edgeColors, idx, vis, par):
    vis[node] = 1
    nodeColors[node] = CURRENT
    captureGraph(G, nodeColors, edgeColors, 0)

    for neigh in G[node].keys():
        if (vis[neigh] == 0):
            nodeColors[node] = ACTIVE
            edgeColors[idx[(node, neigh)]] = ACTIVE
            dfsGO(neigh, G, nodeColors, edgeColors, idx, vis, node)
        edgeColors[idx[(node, neigh)]] = VISITED
        nodeColors[node] = CURRENT
        captureGraph(G, nodeColors, edgeColors, 0)

    nodeColors[node] = VISITED


def startDFS(g, nodes):
    """
    This will generate images for dfs simulation.
    The images will belong to only one component i.e
    Graph.
    """
    global PLOTNO
    G = nx.DiGraph()
    for i in range(1, nodes + 1):
        G.add_node(i)
    G.add_edges_from(g)
    PLOTNO = 0
    nodes = G.number_of_nodes()
    edges = list(G.edges())
    nodeColors = (nodes + 1) * ["black"]
    edgeColors = (len(edges)) * ["black"]

    idx = {tuple(edges[i]): i for i in range(len(edges))}
    vis = (nodes + 1) * [0]

    # displaying the graph for first time
    captureGraph(G, nodeColors, edgeColors, 0)

    for node in range(1, nodes + 1):
        if (vis[node]): continue
        dfsGO(node, G, nodeColors, edgeColors, idx, vis, -1)

    captureGraph(G, nodeColors, edgeColors, 0)







########################################## BFS ###############################################

def bfsGO(node, G, nodeColors, edgeColors, idx, vis):
    Q = collections.deque()
    Q.append(node)
    vis[node] = 1
    while(len(Q)):
        N = Q[0]
        Q.popleft()
        nodeColors[N] = CURRENT
        captureGraph(G, nodeColors, edgeColors, 0)
        for neigh in G[N].keys():
            if (vis[neigh] == 0):
                vis[neigh] = 1
                nodeColors[neigh] = ACTIVE
                Q.append(neigh)
            edgeColors[idx[(N, neigh)]] = VISITED
            captureGraph(G, nodeColors, edgeColors, 0)

        nodeColors[N] = VISITED


def startBFS(g, nodes):
    """
    This will generate images for bfs simulation.
    The images will be related to only one component i.e
    Graph
    """
    global PLOTNO
    PLOTNO = 0
    G = nx.DiGraph()
    for i in range(1, nodes + 1):
        G.add_node(i)
    G.add_edges_from(g)
    edges = list(G.edges())
    nodeColors = (nodes + 1) * ["black"]
    edgeColors = (len(edges)) * ["black"]

    idx = {tuple(edges[i]): i for i in range(len(edges))}
    vis = (nodes + 1) * [0]

    # displaying the graph for first time
    captureGraph(G, nodeColors, edgeColors, 0)

    for node in range(1, nodes + 1):
        if (vis[node]): continue
        bfsGO(node, G, nodeColors, edgeColors, idx, vis)

    captureGraph(G, nodeColors, edgeColors, 0)



########################### Meri pyari pyari dijkstra ################################

def startDijkstra(g, nodes):
    """
    This will generate images which will help me
    create a simulation for dijkstra.
    There will be two components of the simulation.
    1. Graph
    2. Priority Queue (distance array, vis array, etc.)
    """
    START_NODE = 1
    global PLOTNO
    PLOTNO = 0
    G = nx.DiGraph()
    for node in range(1, nodes + 1):
        G.add_node(node)
    for edge in g:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    edges = list(G.edges())
    nodeColors = (nodes + 1) * ["black"]
    edgeColors = (len(edges)) * ["black"]
    idx = {tuple(edges[i]): i for i in range(len(edges))}


    data = pd.DataFrame()
    data["Node"] = [i for i in range(1, nodes + 1)]
    data["Visited"] = [False for _ in range(nodes)]
    data["Distance"] = ["inf" for _ in range(nodes)]
    data = data.set_index("Node")
    captureGraph(G, nodeColors, edgeColors, 1)
    captureDataFrame(data)

    n = nodes
    dist = [float("inf") for _ in range(n + 1)]
    dist[START_NODE] = 0
    visited = [False for _ in range(n + 1)]
    pq = [(0, START_NODE)]
    data.at[START_NODE, "Distance"] = 0
    while len(pq) > 0:
        _, u = heapq.heappop(pq)
        if visited[u]:
            continue
        nodeColors[u] = CURRENT
        captureGraph(G, nodeColors, edgeColors, 1)
        captureDataFrame(data)
        visited[u] = True
        data.at[u, "Visited"] = True
        for v in G[u].keys():
            l = G[u][v]['weight']
            if dist[u] + l < dist[v]:
                nodeColors[v] = ACTIVE
                dist[v] = dist[u] + l
                data.at[v, "Distance"] = dist[v]
                heapq.heappush(pq, (dist[v], v))
            edgeColors[idx[(u, v)]] = VISITED
            captureGraph(G, nodeColors, edgeColors, 1)
            captureDataFrame(data)
        nodeColors[u] = VISITED

    captureGraph(G, nodeColors, edgeColors, 1)
    captureDataFrame(data)



############################################# Kruskal Algorithm ##########################################

class UndirectedGraph:
    START_NODE = 1
    def __init__(self, G, nodeColors, edgeColors, idx):
        self.G = G
        self.nodeColors = nodeColors
        self.edgeColors = edgeColors
        self.idx = idx

    def search(self, parent, i):
        if parent[i] == i:
            return i
        return self.search(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.search(parent, x)
        yroot = self.search(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal(self):
        edges = list(self.G.edges())
        edges = sorted(edges, key=lambda edge: self.G[edge[0]][edge[1]]['weight'])
        parent = []
        rank = []
        MSTWeight = 0
        for node in range(self.G.number_of_nodes() + 1):
            parent.append(node)
            rank.append(0)
        for u, v in edges:
            w = self.G[u][v]['weight']
            x = self.search(parent, u)
            y = self.search(parent, v)
            if x != y:
                self.edgeColors[self.idx[(u, v)]] = CURRENT
                MSTWeight += w
                self.apply_union(parent, rank, x, y)
            else:
                self.edgeColors[self.idx[(u, v)]] = VISITED
            captureGraph(self.G, self.nodeColors, self.edgeColors, 1)

    def prims(self):
        n = self.G.number_of_nodes()
        visited = [False for _ in range(n + 1)]
        pq = []
        MSTWeight = 0
        for neigh in self.G[self.START_NODE]:
            pq.append((self.G[self.START_NODE][neigh]['weight'], neigh, self.START_NODE))
        visited[self.START_NODE] = True
        while len(pq) > 0:
            _, u, v = heapq.heappop(pq)
            if visited[u]:
                self.edgeColors[self.idx[(u, v)]] = VISITED
                captureGraph(self.G, self.nodeColors, self.edgeColors, 1)
                continue
            MSTWeight += _
            self.edgeColors[self.idx[(u, v)]] = CURRENT
            captureGraph(self.G, self.nodeColors, self.edgeColors, 1)
            visited[u] = True
            for v in self.G[u].keys():
                if (visited[v]): continue
                l = self.G[u][v]['weight']
                heapq.heappush(pq, (l, v, u))



def startKruskal(g, nodes):
    START_NODE = 1
    global PLOTNO
    PLOTNO = 0
    G = nx.Graph()
    for node in range(1, nodes + 1):
        G.add_node(node)
    for edge in g:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    edges = list(G.edges())
    nodeColors = (nodes + 1) * ["black"]
    edgeColors = (len(edges)) * ["black"]
    idx = dict()
    for i in range(len(edges)):
        u, v = edges[i]
        idx[(u, v)] = i
        idx[(v, u)] = i

    captureGraph(G, nodeColors, edgeColors, 1)
    mstG = UndirectedGraph(G, nodeColors, edgeColors, idx)
    mstG.kruskal()


def startPrims(g, nodes):
    START_NODE = 1
    global PLOTNO
    PLOTNO = 0
    G = nx.Graph()
    for node in range(1, nodes + 1):
        G.add_node(node)
    for edge in g:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    edges = list(G.edges())
    nodeColors = (nodes + 1) * ["black"]
    edgeColors = (len(edges)) * ["black"]
    idx = dict()
    for i in range(len(edges)):
        u, v = edges[i]
        idx[(u, v)] = i
        idx[(v, u)] = i

    captureGraph(G, nodeColors, edgeColors, 1)
    mstG = UndirectedGraph(G, nodeColors, edgeColors, idx)
    mstG.prims()





################################ Main where some algo starts #############################

def start(g, nodes, algo):
    """
    This will channel graph and nodes to different functions
    based on the algo used and then generate the images which
    will help me create the animation.
    """
    if (algo == "dfs"): startDFS(g, nodes)
    elif (algo == "bfs"): startBFS(g, nodes)
    elif (algo == "dijkstra"): startDijkstra(g, nodes)
    elif (algo == "kruskal"): startKruskal(g, nodes)
    else : startPrims(g, nodes)

    return PLOTNO



####################### Capturing Graphs and DATA ############################################

def captureGraph(G, nodeColors, edgeColors, weighted):
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
    nodesize = 1000
    if (G.number_of_nodes() >= 7) : nodesize = 800
    if (G.number_of_nodes() >= 10): nodesize = 500
    options = {
        "font_size": 8,  # 36
        "font_color": "black",
        "node_size": nodesize,  # 3000
        "node_color": "white",
        "edgecolors": nodeColors,
        "edge_color": edgeColors,
        "linewidths": 2,
        "width": 2,
    }
    nx.draw_networkx(G, pos, **options)
    if weighted:
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.savefig(f"static/output/graph{PLOTNO}.png")
    plt.figure()


def captureDataFrame(df):
    dfi.export(df, f"static/output/data{PLOTNO}.png")