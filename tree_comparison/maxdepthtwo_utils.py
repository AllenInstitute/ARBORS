# equivalent to the code in allLevel2Costs.m called in linearAssignment_matchingNodes.m when maxDepth == 2. 

import os
import lap
import numpy as np
import scipy.io as sio


def postOrderTraversalWithOptions(nodeNumber,tree,currentDepth,maxDepth):
    #get all descendents of this node in the tree from currentDepth down to maxDepth 
    """ 
    Get all descnedents of a node from the current depth down to the max depth. 

    :param nodeNumber: node id to find descendents from 
    :param tree: tree with nodes to search 
    :param currentDepth: starting depth
    :param maxDepth: ending depth
    :return: list of descendent nodes
    """
    nodeList = []
    if currentDepth < maxDepth: 
        ndChildren = tree.get_children(nodeNumber)
        for kk in range(len(ndChildren)):
            nodeList.extend(postOrderTraversalWithOptions(ndChildren[kk],tree, currentDepth+1,maxDepth))
    nodeList.append(nodeNumber['id'])
    return nodeList

def getValidSetCardinality(validSetDir, tree, node, node_children):

    """
    Get children node matching info using (pre-geneated) validSet files. 
    """

    #same as above but when loading matlab validSet file, simplifly_cells = False
    # validSetDir = r'\\allen\programs\celltypes\workgroups\mousecelltypes\SarahWB\uygar_tree_comparison\matt_version\uygar_tree_comparison\treeComparison_bitbucket\saveSomeValidSetsResults_6_1_2023'
    # tree = tree1
    # node = node1
    # node_children = node1_children

    #get the validset file to load 
    fileName = ''
    maxMaximalSetCardinality = -1
    for child in range(len(node_children)):
        thisChild = node_children[child]
        thisGrandchild = tree.get_children(thisChild)
        maxMaximalSetCardinality = maxMaximalSetCardinality + max(1, len(thisGrandchild))
        fileName = fileName + '_' + str(len(thisGrandchild))
    fileName = os.path.join(validSetDir, fileName + '.mat')

    print(fileName)

    # Get node matching info 
    if os.path.isfile(fileName):
        validSet = sio.loadmat(fileName)
        pOT1 = validSet['pOT1'][:,0]
        vs = validSet['vs'][0]
        pOT2 = postOrderTraversalWithOptions(node,tree,0,2)
        for kk in range(len(vs)): #for every combination of leavesChoosekk
            for jj in range(len(vs[kk][0][0])): #for every valid set of kk nodes
                for ii in range(len(vs[kk][0][0][jj])): #for every node in this set
                    loc = np.where(pOT1 == vs[kk][0][0][jj][ii])[0]
                    if loc.size > 0: #this node is in pOT1
                        vs[kk][0][0][jj][ii] = pOT2[loc[0]]
        minMaximalSetCardinality = len(node_children)-1
    else: 
        print('File not found: {}'.format(fileName))
        return None, None, None

    return minMaximalSetCardinality, maxMaximalSetCardinality, vs


def getMatchingChildren(maxMaximalSetCardinality1, minMaximalSetCardinality1, vs1, 
                       maxMaximalSetCardinality2, minMaximalSetCardinality2, vs2, 
                       agreement, node_id_index_dict1, node_id_index_dict2):
  
  """
  Find match of tree 1 and tree 2 children that has the highest similarity score. 
  """

  sim = -1e-10
  if minMaximalSetCardinality2 < minMaximalSetCardinality1:
    print('one')
    for kk in range(minMaximalSetCardinality2,min(minMaximalSetCardinality1,maxMaximalSetCardinality2)+1):
      for kk3 in range(0,len(vs1[minMaximalSetCardinality1][0][0])): #for num sets in vs1 Choose minMaximalSetCardinality1
        for kk4 in range(0,len(vs2[kk][0][0])):                      #for num sets in vs2 Choose kk 
          if vs1[minMaximalSetCardinality1][0][1][kk3][0] or vs2[kk][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap
            lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[minMaximalSetCardinality1][0][0][kk3]],
                                                  [node_id_index_dict2[x] for x in vs2[kk][0][0][kk4]])]
            mat_shape = lap_submat.shape
            if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0])
            thisSim, _, rowsol = lap.lapjv(-lap_submat, extend_cost=True)
            thisSim = thisSim * -1
            print('ran lap one')
            print('thisSim: ', thisSim)
            print('rowsol: ', rowsol)
            if thisSim > sim:
              if minMaximalSetCardinality1 > kk:
                sim = thisSim
                matchingChildren1 = [n for n in vs1[minMaximalSetCardinality1][0][0][kk3][rowsol]]
                matchingChildren2 = [n for n in vs2[kk][0][0][kk4]]
              else:
                sim = thisSim
                matchingChildren1 = [n for n in vs1[minMaximalSetCardinality1][0][0][kk3]]
                matchingChildren2 = [n for n in vs2[kk][0][0][kk4][rowsol]]
  else:
    print('two')
    for kk in range(minMaximalSetCardinality1,min(minMaximalSetCardinality2,maxMaximalSetCardinality1)+1):
      print('kk: ', kk)
      for kk3 in range(0,len(vs1[kk][0][0])):
        print('kk3: ', kk3)
        for kk4 in range(0,len(vs2[minMaximalSetCardinality2][0][0])):      
          print('kk4: ', kk4)
          if vs1[kk][0][1][kk3][0] or vs2[minMaximalSetCardinality2][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap
            lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[kk][0][0][kk3]],
                                                  [node_id_index_dict2[x] for x in vs2[minMaximalSetCardinality2][0][0][kk4]])]
            mat_shape = lap_submat.shape
            if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0])
            thisSim, _, rowsol = lap.lapjv(-lap_submat, extend_cost=True)
            thisSim = thisSim * -1
            print('ran lap two')
            print('thisSim: ', thisSim)
            print('rowsol: ', rowsol)
            if thisSim > sim:
              if minMaximalSetCardinality2 < kk:
                sim = thisSim
                matchingChildren1 = [n for n in vs1[kk][0][0][kk3][rowsol]]
                matchingChildren2 = [n for n in vs2[minMaximalSetCardinality2][0][0][kk4]]
              else:
                sim = thisSim
                matchingChildren1 = [n for n in vs1[kk][0][0][kk3]]
                matchingChildren2 = [n for n in vs2[minMaximalSetCardinality2][0][0][kk4][rowsol]]

  for kk in range(max(minMaximalSetCardinality1,minMaximalSetCardinality2)+1, min(maxMaximalSetCardinality1,maxMaximalSetCardinality2)):
    for kk3 in range(0, len(vs1[kk][0][0])): 
      for kk4 in range(0, len(vs2[kk][0][0])):
        if vs1[kk][0][1][kk3][0] or vs2[kk][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap
          lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[kk][0][0][kk3]],
                                                [node_id_index_dict2[x] for x in vs2[kk][0][0][kk4]])]
          mat_shape = lap_submat.shape
          if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0])
          thisSim, _, rowsol = lap.lapjv(-lap_submat, extend_cost=True)
          thisSim = thisSim * -1
          print('ran lap down here')
          if thisSim > sim: 
            sim = thisSim 
            matchingChildren1 = [n for n in vs1[kk][0][0][kk3]]
            matchingChildren2 = [n for n in vs1[kk][0][0][kk4][rowsol]]

  return matchingChildren1,matchingChildren2,sim