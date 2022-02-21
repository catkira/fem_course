import mesh as m
import numpy as np
import sys
import time

# TODO add parameter to start growing the tree in a specified regions first
# so that no closed loops are created when DirichletBCs are applied in those regions
class spanningtree:
    def __init__(self, excludedRegions=[]):
        m.computeEdges3d()
        start = time.time()
        self.graph = np.copy(m.mesh['pe'])
        self.graphSorted2 = np.column_stack([m.mesh['pe'], np.arange(m.mesh['pe'].shape[0])])[m.mesh['pe'][:, 1].argsort()]
        self.isNodeInTree = np.zeros(m.mesh['xp'].shape[0], dtype=np.bool)

        # exclude excluded Regions by setting the nodes of those regions in isNodeInTree to true
        for region in excludedRegions:
            for dim in range(len(m.getMesh()['physical'])):
                if region in m.getMesh()['physical'][dim]:
                    if dim == 1:
                        self.isNodeInTree[m.getMesh()['pt'][np.where(m.getMesh()['physical'][dim] == region)].ravel()] = True                    
                    elif dim == 2:
                        m.getMesh()['ptt'][np.where(m.getMesh()['physical'][dim] == region)]

        # add first edge to tree
        # make sure its not in an excluded region
        self.edges = []
        for edge in self.graph:
            if self.isNodeInTree[edge[0]] == False and self.isNodeInTree[edge[1]] == False:
                self.edges = edge        
                self.edges = np.expand_dims(self.edges, axis=0)
                self.isNodeInTree[self.edges[0,0]] = True
                self.isNodeInTree[self.edges[0,1]] = True
                break
        if self.edges == []:
            print('Error: no start edge for tree found!')
            sys.exit()
        
        self.newEdges = self.edges
        self.branches = np.empty(0, dtype=np.int)
        generation = 0
        while len(self.newEdges) != 0:
            self.growTree()
            print(f'Generation {generation:d}: added {len(self.newEdges):d} edges')
            generation = generation + 1
        numNodes = len(np.unique(self.edges.ravel()))
        duration = time.time() - start
        print(f"tree with {len(self.edges)} edges, {len(self.branches)} branches and {numNodes} nodes completed in {duration:.4f}s")
        #print(f"len(edges) + len(branches) = {len(self.branches)+len(self.edges)}")

    def getConnectedNodes(self, node):
        idx = np.zeros(m.mesh['xp'].shape[0], dtype=np.bool)
        if False:
            idx[self.graph[self.graph[:,0] == node][:,1]] = True
            idx[self.graph[self.graph[:,1] == node][:,0]] = True
        if False: # this is a bit faster
            idx[self.graph[np.where(self.graph[:,0] == node)][:,1]] = True
            idx[self.graph[np.where(self.graph[:,1] == node)][:,0]] = True
        if True: # this is a bit more faster
            firstMatch = np.searchsorted(self.graph[:,0], node)
            while ((firstMatch < self.graph.shape[0]) and self.graph[firstMatch,0] == node):
                idx[self.graph[firstMatch,1]] = True
                firstMatch += 1
            firstMatch = np.searchsorted(self.graphSorted2[:,1], node)
            while ((firstMatch < self.graph.shape[0]) and self.graphSorted2[firstMatch,1] == node):
                idx[self.graph[self.graphSorted2[firstMatch,2],0]] = True
                firstMatch += 1
        return idx

    def addBranch(self, nodes):
        if nodes[0] > nodes[1]:
            nodes[0],nodes[1] = nodes[1], nodes[0]
        self.branches = np.append(self.branches, np.where((m.mesh['pe'] == [nodes]).all(axis=1)))

    def growTree(self):
        # add edges that dont create circles
            # get all candidate nodesu8 mk 
            # filter out circles
            # add candidates
        newEdges2 = np.empty((0,2))
        a = 0
        b = 0
        for node in self.newEdges:
            start = time.time()
            indices = self.getConnectedNodes(node[1]) # this function needs to be optimized
            a += time.time() - start
            start = time.time()
            if np.count_nonzero(indices) == 0:
                self.addBranch(node)
                continue
            filteredCands = indices & np.invert(self.isNodeInTree)
            if np.count_nonzero(filteredCands) == 0:
                self.addBranch(node)
                continue
            self.isNodeInTree[filteredCands] = True
            newEdges2 = np.row_stack((newEdges2, 
                np.column_stack([np.ones(np.count_nonzero(filteredCands))*node[1], np.arange(m.mesh['xp'].shape[0])[filteredCands]])))
            b += time.time() - start
        print(f"a={a:.4f} b={b:.4f}")
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