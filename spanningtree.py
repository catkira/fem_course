import mesh as m
import numpy as np
import sys
import time

class spanningtree:
    def __init__(self, excludedRegions=[], verbose=False, verify=False):
        m.computeEdges3d()
        start = time.time()
        self.edgePool = np.copy(m.mesh['pe'])
        self.numNodes = m.mesh['xp'].shape[0]
        self.graphSorted2 = np.column_stack([self.edgePool, np.arange(self.edgePool.shape[0])])[self.edgePool[:, 1].argsort()]
        self.isNodeInTree = np.zeros(self.numNodes, dtype=np.bool)

        excludedNodes = np.empty(0, dtype = np.int64)
        for region in excludedRegions:
            for dim in range(len(m.getMesh()['physical'])):
                if region in m.getMesh()['physical'][dim]:
                    if dim == 1:
                        excludedNodes = np.append(excludedNodes, m.getMesh()['pt'][np.where(m.getMesh()['physical'][dim] == region)[0]].ravel())
                    elif dim == 2:
                        excludedNodes = np.append(excludedNodes, m.getMesh()['ptt'][np.where(m.getMesh()['physical'][dim] == region)[0]].ravel())
        excludedNodesMask = np.repeat(True, self.numNodes)
        excludedNodesMask[excludedNodes] = False

        # precaclulate connected nodes
        self.connectedNodes = np.empty(np.max(self.edgePool.ravel())+1, dtype=object)
        self.idx = np.zeros(self.numNodes, dtype=np.bool)
        for node in range(self.connectedNodes.shape[0]):
            if excludedNodesMask[node]:
                self.connectedNodes[node] = np.arange(self.numNodes)[self.getConnectedNodes(node) & excludedNodesMask]

        # verify connected nodes
        if verify:
            for node1, cn in enumerate(self.connectedNodes):
                if cn is not None:
                    for node2 in cn:
                        if (node1 in self.connectedNodes[node2]) == False:
                            print(f'nodes {node1} {node2}')
                            sys.exit()

        # add first edge to tree
        # make sure its not in an excluded region
        self.edges = []
        for edge in self.edgePool:
            if excludedNodesMask[edge[0]] & excludedNodesMask[edge[1]]:
                self.edges = edge        
                self.edges = np.expand_dims(self.edges, axis=0)
                self.isNodeInTree[self.edges[0,0]] = True
                self.isNodeInTree[self.edges[0,1]] = True
                break
        if self.edges == []:
            print('Error: no start edge for tree found!')
            sys.exit()

        self.branches = np.empty(0, dtype=np.int)

        if True: 
            # the recursive version is about 2x slower than the iterative version
            sys.setrecursionlimit(1000000)
            self.growTreeRecursive(edge)
            self.growTreeRecursive([edge[1],edge[0]])
        else:
            print("not working!")
            sys.exit()
            self.newEdges = self.edges
            generation = 0
            while len(self.newEdges) != 0:
                self.growTree()
                if verbose:
                    print(f'Generation {generation:d}: added {len(self.newEdges):d} edges')
                generation = generation + 1

        self.branches = np.unique(self.branches)
        numNodes = len(np.unique(self.edges.ravel()))
        duration = time.time() - start
        print(f"tree with {len(self.edges)} edges, {len(self.branches)} branches and {numNodes} nodes completed in {duration:.4f}s")
        #print(f"len(edges) + len(branches) = {len(self.branches)+len(self.edges)}")

    def getConnectedNodes(self, node):
        self.idx[:] = False
        if False:
            self.idx[self.edgePool[self.edgePool[:,0] == node][:,1]] = True
            self.idx[self.edgePool[self.edgePool[:,1] == node][:,0]] = True
        if False: # this is a bit faster
            self.idx[self.edgePool[np.where(self.edgePool[:,0] == node)][:,1]] = True
            self.idx[self.edgePool[np.where(self.edgePool[:,1] == node)][:,0]] = True
        if False: # this is a bit more faster
            firstMatch = np.searchsorted(self.edgePool[:,0], node)
            while ((firstMatch < self.edgePool.shape[0]) and self.edgePool[firstMatch,0] == node):
                self.idx[self.edgePool[firstMatch,1]] = True
                firstMatch += 1
            firstMatch = np.searchsorted(self.graphSorted2[:,1], node)
            while ((firstMatch < self.edgePool.shape[0]) and self.graphSorted2[firstMatch,1] == node):
                self.idx[self.edgePool[self.graphSorted2[firstMatch,2],0]] = True
                firstMatch += 1
        if True: # this is a bit more faster
            left = np.searchsorted(self.edgePool[:,0], node, 'left')
            right = np.searchsorted(self.edgePool[:,0], node, 'right')
            self.idx[self.edgePool[left:right,1]] = True
            left = np.searchsorted(self.graphSorted2[:,1], node, 'left')
            right = np.searchsorted(self.graphSorted2[:,1], node, 'right')
            self.idx[self.edgePool[self.graphSorted2[left:right,2],0]] = True
        return self.idx

    def addBranch(self, edge):
        if edge[0] > edge[1]:
            edge = [edge[1], edge[0]]
        startIdx = m.getMesh()['pe'][:,0].searchsorted(edge[0], 'left')
        while (startIdx < len(m.getMesh()['pe'])) and (m.getMesh()['pe'][startIdx,1] != edge[1]):
            startIdx += 1
        #print(f'add branch {edge[0]},{edge[1]} with id {startIdx}')
        self.branches = np.append(self.branches, startIdx)

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
            # tree needs to grow from both nodes of the edges!
            newEdges2 = np.row_stack((newEdges2, 
                np.column_stack([connectedNodes, np.ones(len(connectedNodes), dtype=np.int64)*edge[1]])))
        self.newEdges = newEdges2
        self.edges = np.row_stack((self.edges, self.newEdges))

    # recursive version of growTree()
    def growTreeRecursive(self, edge, rec=False):
        connectedNodes = self.connectedNodes[edge[1]]
        if len(connectedNodes) == 0:
            self.addBranch(edge)
            return
        connectedNodesIdx = np.repeat(False, self.numNodes)
        connectedNodesIdx[connectedNodes] = True
        filteredCands = connectedNodesIdx & np.invert(self.isNodeInTree)
        if np.count_nonzero(filteredCands) == 0:
            self.addBranch(edge)
            return
        if rec: # only need to go one step back
            return
        self.isNodeInTree[filteredCands] = True
        newEdges = np.column_stack([np.ones(np.count_nonzero(filteredCands), dtype=np.int64)*edge[1], np.arange(m.mesh['xp'].shape[0])[filteredCands]])
        self.edges = np.row_stack((self.edges, newEdges))        
        for newEdge in newEdges:
            self.growTreeRecursive(newEdge)
            # need to grow tree in both directions, because some edges of newEdges can become branches
            # in this for loop
            self.growTreeRecursive([newEdge[1],newEdge[0]], True)
    
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