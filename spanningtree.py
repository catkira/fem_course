import mesh as m
import numpy as np

class spanningtree:
    def __init__(self):
        m.computeEdges3d()
        self.graph = np.copy(m.mesh['pe'])
        self.edges = self.graph[0]
        self.edges = np.expand_dims(self.edges, axis=0)
        self.newEdges = self.edges
        generation = 0
        while len(self.newEdges) != 0:
            self.growTree()
            print(f'Generation {generation:d}: added {len(self.newEdges):d} edges')
            generation = generation + 1
        numNodes = len(np.unique(self.edges.ravel()))
        print(f"tree with {len(self.edges)} edges and {numNodes} nodes completed")

    def getConnectedNodes(self, node):
        if True:
            matches = self.graph[(self.graph[:,0] == node) | (self.graph[:,1] == node)]
            self.graph = np.delete(self.graph, (self.graph[:,0] == node) | (self.graph[:,1] == node), axis=0)
            swap = matches[:,1] == node
            matches = np.append(matches[swap,0], matches[np.invert(swap),1])
        else:
            matches = m.mesh['pe'][m.mesh['pe'][:,0] == node]
        return matches

    def growTree(self):
        # add edges that dont create circles
            # get all candidate nodes
            # filter out circles
            # add candidates
        nodesInTree = self.edges.ravel()
        nodesInTree = np.unique(nodesInTree)
        newEdges2 = np.empty((0,2))
        for node in self.newEdges:
            candidates = self.getConnectedNodes(node[1])
            if candidates == []:
                continue
            # circles = np.in1d(candidates, nodesInTree)
            # filteredCands = np.delete(candidates, circles)
            filteredCands = np.setdiff1d(candidates, nodesInTree)
            newEdges2 = np.append(newEdges2, np.column_stack([np.ones(len(filteredCands))*node[1], filteredCands]))
        self.newEdges = newEdges2.reshape(int(len(newEdges2)/2),2)
        self.edges = np.append(self.edges, self.newEdges, axis=0)

    def write(self, filename):
        if self.edges == []:
            print("Warning: tree has no edges!")
        txt = str()
        with open(filename, 'w') as file:
            txt += """View \"spantree\" {
                    TIME{0};"""
            for edge in self.edges:
                p1 = m.mesh['xp'][int(edge[0])]
                p2 = m.mesh['xp'][int(edge[1])] - p1
                txt += f"SL({p1[0]},{p1[1]},{p1[2]},{p2[0]},{p2[1]},{p2[2]})"
                txt += "{1,1};\n"
            txt += """INTERPOLATION_SCHEME
                    {
                    {0.5,-0.5},
                    {0.5,0.5}
                    }
                    {
                    {0,0,0},
                    {1,0,0}
                    }
                    {
                    {0.5,-0.5},
                    {0.5,0.5}
                    }
                    {
                    {0,0,0},
                    {1,0,0}
                    };

                    };
                    """
            file.write(txt)