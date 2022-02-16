import mesh as m
import numpy as np

# TODO add parameter to start growing the tree in a specified regions first
# so that no closed loops are created when DirichletBCs are applied in those regions
class spanningtree:
    def __init__(self):
        m.computeEdges3d()
        self.graph = np.copy(m.mesh['pe']).astype(np.int)
        self.isNodeInTree = np.zeros(m.mesh['xp'].shape[0], dtype=np.bool)

        # add first node to tree
        self.edges = self.graph[0]
        self.edges = np.expand_dims(self.edges, axis=0)
        self.isNodeInTree[self.edges[0,0]] = True
        self.isNodeInTree[self.edges[0,1]] = True

        self.newEdges = self.edges
        self.branches = np.empty((0,2))
        generation = 0
        while len(self.newEdges) != 0:
            self.growTree()
            print(f'Generation {generation:d}: added {len(self.newEdges):d} edges')
            generation = generation + 1
        numNodes = len(np.unique(self.edges.ravel()))
        print(f"tree with {len(self.edges)} edges, {len(self.branches)} branches and {numNodes} nodes completed")
        print(f"len(edges) + len(branches) = {len(self.branches)+len(self.edges)}")

    def getConnectedNodes(self, node):
        indices1 = m.mesh['pe'][m.mesh['pe'][:,0] == node][:,1]
        indices2 = m.mesh['pe'][m.mesh['pe'][:,1] == node][:,0]
        idx = np.zeros(m.mesh['xp'].shape[0], dtype=np.bool)
        idx[np.append(indices1, indices2).astype(np.int)] = True
        return idx

    def growTree(self):
        # add edges that dont create circles
            # get all candidate nodesu8 mk 
            # filter out circles
            # add candidates
        newEdges2 = np.empty((0,2))
        for node in self.newEdges:
            indices = self.getConnectedNodes(node[1])
            if np.count_nonzero(indices) == 0:
                self.branches = np.row_stack((self.branches, node))
                continue
            filteredCands = (indices & np.invert(self.isNodeInTree)).astype(np.bool)
            if np.count_nonzero(filteredCands) == 0:
                self.branches = np.row_stack((self.branches, node))
            self.isNodeInTree[filteredCands] = True
            newEdges2 = np.row_stack((newEdges2, 
                np.column_stack([np.ones(np.count_nonzero(filteredCands))*node[1], np.arange(m.mesh['xp'].shape[0])[filteredCands]])))
        self.newEdges = newEdges2
        self.edges = np.row_stack((self.edges, self.newEdges))

    def write(self, filename):
        if self.edges == []:
            print("Warning: tree has no edges!")
        txt = str()
        with open(filename, 'w') as file:
            txt += """View \"spantree\" {
TIME{0};"""
            for edge in self.edges:
                p1 = m.mesh['xp'][int(edge[0])]
                p2 = m.mesh['xp'][int(edge[1])] #- p1
                if m.mesh['problemDimension'] == 2:
                    txt += f"SL({p1[0]},{p1[1]},{0},{p2[0]},{p2[1]},{0})"
                else:
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