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

    nodes = request.GET.get('rows', '')
    try:
        nodes = int(nodes)
    except:
        return False, "Invalid number of nodes", -1

    if (nodes == 0):
        return False, "Nodes can't be zero.", -1

    givenEdgeString = request.GET.get('givenMatrix', '').strip()
    givenEdgeStrings = givenEdgeString.split("\r\n")
    G = []
    for R in givenEdgeStrings:
        try:
            x = list(map(int, R.split()))
            if (len(x)): G.append(x)
        except:
            return "Invalid Edges"

    # clearing all the files inside output before running the function
    dir = 'static/output'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    return True, None, simulation.start(G, nodes)
