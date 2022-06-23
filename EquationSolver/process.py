# created by me
from . import simulation
import os


def simulateGraph(request):
    """
    This will return a tuple
    1. valid graph
    2. message
    3. number of stages in the graph
    """
    ALGOS = ["dfs", "bfs", "dijkstra"]
    nodes = request.GET.get('nodes', '')
    algo = request.GET.get('algo', '').lower()
    try:
        nodes = int(nodes)
    except:
        return False, "Invalid number of nodes", -1

    if (algo not in ALGOS): return False, "Invalid Algo", -1

    if (nodes == 0):
        return False, "Nodes can't be zero.", -1

    givenEdgeString = request.GET.get('edges', '').strip()
    givenEdgeStrings = givenEdgeString.split("\r\n")
    G = []
    for R in givenEdgeStrings:
        try:
            x = list(map(int, R.split()))
            if (len(x) != 2 and algo in ["dfs", "bfs"]) or x[0] not in range(1, nodes + 1) or x[1] not in range(1, nodes + 1):
                return False, "Invalid Edge", -1
            G.append(x)
        except:
            return False, "Invalid Edge", -1

    # clearing all the files inside output before running the function
    dir = 'static/output'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    return True, None, simulation.start(G, nodes, algo.lower())
