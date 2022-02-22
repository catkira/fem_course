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
        self.idx = np.zeros(m.mesh['xp'].shape[0], dtype=np.bool)

        # exclude excluded Regions by setting the nodes of those regions in isNodeInTree to true
        for region in excludedRegions:
            for dim in range(len(m.getMesh()['physical'])):
                if region in m.getMesh()['physical'][dim]:
                    if dim == 1:
                        self.isNodeInTree[m.getMesh()['pt'][np.where(m.getMesh()['physical'][dim] == region)].ravel()] = True                    
                    elif dim == 2:
                        m.getMesh()['ptt'][np.where(m.getMesh()['physical'][dim] == region)]

        # precaclulate connected nodes
        self.connectedNodes = np.empty(np.max(self.graph.ravel())+1, dtype=object)
        for node in range(self.connectedNodes.shape[0]):
            self.connectedNodes[node] = np.arange(self.idx.shape[0])[self.getConnectedNodes(node)]

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

        self.branches = np.empty(0, dtype=np.int)

        if False: 
            # the recursive version is about 2x slower than the iterative version
            sys.setrecursionlimit(1000000)
            self.growTreeRecursive(edge)
        else:
            self.newEdges = self.edges
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
        self.idx[:] = False
        if False:
            self.idx[self.graph[self.graph[:,0] == node][:,1]] = True
            self.idx[self.graph[self.graph[:,1] == node][:,0]] = True
        if False: # this is a bit faster
            self.idx[self.graph[np.where(self.graph[:,0] == node)][:,1]] = True
            self.idx[self.graph[np.where(self.graph[:,1] == node)][:,0]] = True
        if False: # this is a bit more faster
            firstMatch = np.searchsorted(self.graph[:,0], node)
            while ((firstMatch < self.graph.shape[0]) and self.graph[firstMatch,0] == node):
                self.idx[self.graph[firstMatch,1]] = True
                firstMatch += 1
            firstMatch = np.searchsorted(self.graphSorted2[:,1], node)
            while ((firstMatch < self.graph.shape[0]) and self.graphSorted2[firstMatch,1] == node):
                self.idx[self.graph[self.graphSorted2[firstMatch,2],0]] = True
                firstMatch += 1
        if True: # this is slower than above
            left = np.searchsorted(self.graph[:,0], node, 'left')
            right = np.searchsorted(self.graph[:,0], node, 'right')
            self.idx[self.graph[left:right,1]] = True
            left = np.searchsorted(self.graphSorted2[:,1], node, 'left')
            right = np.searchsorted(self.graphSorted2[:,1], node, 'right')
            self.idx[self.graph[self.graphSorted2[left:right,2],0]] = True
        return self.idx

    def addBranch(self, edge):
        if edge[0] > edge[1]:
            edge[0],edge[1] = edge[1], edge[0]
        idx = m.getMesh()['pe'][:,0].searchsorted(edge[0], 'left')
        while idx < len(m.getMesh()['pe']) and m.getMesh()['pe'][idx,1] != edge[1]:
            idx += 1
        self.branches = np.append(self.branches, idx)

    def growTree(self):
        # add edges that dont create circles
            # get all candidate nodes
            # filter out circles
            # add candidates
        newEdges2 = np.empty((0,2), dtype=np.int64)
        for edge in self.newEdges:
            connectedNodes = self.connectedNodes[edge[1]]
            if len(connectedNodes) == 0:
                self.addBranch(edge)
                continue
            connectedNodes = connectedNodes[np.invert(self.isNodeInTree[connectedNodes])]
            if len(connectedNodes) == 0:
                self.addBranch(edge)
                continue
            self.isNodeInTree[connectedNodes] = True
            newEdges2 = np.row_stack((newEdges2, 
                np.column_stack([np.ones(len(connectedNodes), dtype=np.int64)*edge[1], connectedNodes])))
        self.newEdges = newEdges2
        self.edges = np.row_stack((self.edges, self.newEdges))

    # recursive version of growTree()
    def growTreeRecursive(self, edge):
        connectedNodes = self.connectedNodes[edge[1]]
        if len(connectedNodes) == 0:
            self.addBranch(edge)
            return
        self.idx = np.repeat(False, self.idx.shape[0])
        self.idx[connectedNodes] = True
        filteredCands = self.idx & np.invert(self.isNodeInTree)
        if np.count_nonzero(filteredCands) == 0:
            self.addBranch(edge)
            return
        self.isNodeInTree[filteredCands] = True
        newEdges = np.column_stack([np.ones(np.count_nonzero(filteredCands), dtype=np.int64)*edge[1], np.arange(m.mesh['xp'].shape[0])[filteredCands]])
        self.edges = np.row_stack((self.edges, newEdges))        
        for newEdge in newEdges:
            self.growTreeRecursive(newEdge)
    
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