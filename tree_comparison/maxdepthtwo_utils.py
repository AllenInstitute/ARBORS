# equivalent to the code in allLevel2Costs.m called in linearAssignment_matchingNodes.m when maxDepth == 2. 

import os
from scipy.optimize import linear_sum_assignment
import numpy as np
import scipy.io as sio
import json 
from importlib.resources import files
import copy

def postOrderTraversalWithOptions(node,tree,currentDepth,maxDepth):
    #get all descendents of this node in the tree from currentDepth down to maxDepth 
    """ 
    Get all descnedents of a node from the current depth down to the max depth. 

    :param nodeNumber: node to find descendents from 
    :param tree: tree with nodes to search 
    :param currentDepth: starting depth
    :param maxDepth: ending depth
    :return: list of descendent nodes including input node (returns the node id)
    """
    nodeList = []
    if currentDepth < maxDepth: 
        ndChildren = tree.get_children(node)
        for kk in range(len(ndChildren)):
            nodeList.extend(postOrderTraversalWithOptions(ndChildren[kk],tree, currentDepth+1,maxDepth))
    nodeList.append(node['id'])
    return nodeList

def getValidSetCardinality(validSetDir, tree, node, node_children):

    """
    Get children node matching info using (pre-geneated) validSet files. 
    allLevel2Costs.m part 1
    """
    
    #get the validset file to load 
    fileName = ''
    maxMaximalSetCardinality = -1
    for child in range(len(node_children)):
        thisChild = node_children[child]
        thisGrandchild = tree.get_children(thisChild)
        maxMaximalSetCardinality = maxMaximalSetCardinality + max(1, len(thisGrandchild))
        fileName = fileName + '_' + str(len(thisGrandchild))
    fileName = os.path.join(validSetDir, fileName + '.mat')

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
  allLevel2Costs.m part 2
  """

  sim = -1e-10
  if minMaximalSetCardinality2 < minMaximalSetCardinality1:
    for kk in range(minMaximalSetCardinality2,min(minMaximalSetCardinality1,maxMaximalSetCardinality2)+1):
      for kk3 in range(0,len(vs1[minMaximalSetCardinality1][0][0])): #for num sets in vs1 Choose minMaximalSetCardinality1
        for kk4 in range(0,len(vs2[kk][0][0])):                      #for num sets in vs2 Choose kk 
          if vs1[minMaximalSetCardinality1][0][1][kk3][0] or vs2[kk][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap
            lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[minMaximalSetCardinality1][0][0][kk3]],
                                                  [node_id_index_dict2[x] for x in vs2[kk][0][0][kk4]])]
            mat_shape = lap_submat.shape
            if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0])

            x, rowsol = linear_sum_assignment(-lap_submat)
            thisSim = lap_submat[x, rowsol].sum()

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
    for kk in range(minMaximalSetCardinality1,min(minMaximalSetCardinality2,maxMaximalSetCardinality1)+1):
      for kk3 in range(0,len(vs1[kk][0][0])):
        for kk4 in range(0,len(vs2[minMaximalSetCardinality2][0][0])):      
          if vs1[kk][0][1][kk3][0] or vs2[minMaximalSetCardinality2][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap
            lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[kk][0][0][kk3]],
                                                  [node_id_index_dict2[x] for x in vs2[minMaximalSetCardinality2][0][0][kk4]])]
            mat_shape = lap_submat.shape
            if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0]) 

            x, rowsol = linear_sum_assignment(-lap_submat)
            thisSim = lap_submat[x, rowsol].sum()

            if thisSim > sim:
              if minMaximalSetCardinality2 < kk:
                sim = thisSim
                matchingChildren1 = [n for n in vs1[kk][0][0][kk3][rowsol]]
                matchingChildren2 = [n for n in vs2[minMaximalSetCardinality2][0][0][kk4]]
              else:
                sim = thisSim
                matchingChildren1 = [n for n in vs1[kk][0][0][kk3]]
                matchingChildren2 = [n for n in vs2[minMaximalSetCardinality2][0][0][kk4][rowsol]]

  for kk in range(max(minMaximalSetCardinality1,minMaximalSetCardinality2)+1, min(maxMaximalSetCardinality1,maxMaximalSetCardinality2)+1):
    for kk3 in range(0, len(vs1[kk][0][0])): 
      for kk4 in range(0, len(vs2[kk][0][0])):
        if vs1[kk][0][1][kk3][0] or vs2[kk][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap          
          lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[kk][0][0][kk3]],
                                                [node_id_index_dict2[x] for x in vs2[kk][0][0][kk4]])]
          mat_shape = lap_submat.shape
          if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0])

          x, rowsol = linear_sum_assignment(-lap_submat)
          thisSim = lap_submat[x, rowsol].sum()

          if thisSim > sim: 
            sim = thisSim 
            matchingChildren1 = [n for n in vs1[kk][0][0][kk3]]
            matchingChildren2 = [n for n in vs2[kk][0][0][kk4][rowsol]]

  matchingChildren1 = [node_id_index_dict1[c] for c in matchingChildren1]
  matchingChildren2 = [node_id_index_dict2[c] for c in matchingChildren2]
  return matchingChildren1,matchingChildren2,sim

# HARDCODED VALID SETS
def getValidSetCardinality_hardcode(valid_set_dict, tree, node, node_children):

    """
    Get children node matching info using (pre-geneated) validSet files. 
    allLevel2Costs.m part 1
    """
    #get the validset file to load 
    fileName = ''
    maxMaximalSetCardinality = -1
    for child in range(len(node_children)):
        thisChild = node_children[child]
        thisGrandchild = tree.get_children(thisChild)
        maxMaximalSetCardinality = maxMaximalSetCardinality + max(1, len(thisGrandchild))
        fileName = fileName + '_' + str(len(thisGrandchild))

    # Get node matching info 
    if fileName in valid_set_dict.keys():
        pOT1 = np.array(valid_set_dict[fileName]['pOT1'])
        vs_temp = valid_set_dict[fileName]['vs']
        vs = copy.deepcopy(vs_temp) #need to deep copy so valid_set_dict isn't altered
        pOT2 = postOrderTraversalWithOptions(node,tree,0,2)
        for kk in range(len(vs)): #for every combination of leavesChoosekk
            for jj in range(len(vs[kk][0][0])): #for every valid set of kk nodes
                for ii in range(len(vs[kk][0][0][jj])): #for every node in this set
                    loc = np.where(pOT1 == vs[kk][0][0][jj][ii])[0]
                    if loc.size > 0: #this node is in pOT1
                        vs[kk][0][0][jj][ii] = pOT2[loc[0]]
        minMaximalSetCardinality = len(node_children)-1
    else: 
        print('Branching pattern not in hardcoded valid sets: {}'.format(fileName))
        return None, None, None

    return minMaximalSetCardinality, maxMaximalSetCardinality, vs

def getMatchingChildren_hardcode(maxMaximalSetCardinality1, minMaximalSetCardinality1, vs1, 
                       maxMaximalSetCardinality2, minMaximalSetCardinality2, vs2, 
                       agreement, node_id_index_dict1, node_id_index_dict2):
  
  """
  Find match of tree 1 and tree 2 children that has the highest similarity score. 
  allLevel2Costs.m part 2
  """

  sim = -1e-10
  if minMaximalSetCardinality2 < minMaximalSetCardinality1:
      for kk in range(minMaximalSetCardinality2,min(minMaximalSetCardinality1,maxMaximalSetCardinality2)+1):
        for kk3 in range(0,len(vs1[minMaximalSetCardinality1][0][0])): #for num sets in vs1 Choose minMaximalSetCardinality1
          for kk4 in range(0,len(vs2[kk][0][0])):                      #for num sets in vs2 Choose kk 
            if vs1[minMaximalSetCardinality1][0][1][kk3][0] or vs2[kk][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap
              lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[minMaximalSetCardinality1][0][0][kk3]],
                                                    [node_id_index_dict2[x] for x in vs2[kk][0][0][kk4]])]
              mat_shape = lap_submat.shape
              if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0])

              x, rowsol = linear_sum_assignment(-lap_submat)
              thisSim = lap_submat[x, rowsol].sum()

              if thisSim > sim:
                if minMaximalSetCardinality1 > kk:
                  sim = thisSim
                  matchingChildren1 = [vs1[minMaximalSetCardinality1][0][0][kk3][r] for r in rowsol.tolist()]
                  matchingChildren2 = [n for n in vs2[kk][0][0][kk4]]
                else:
                  sim = thisSim
                  matchingChildren1 = [n for n in vs1[minMaximalSetCardinality1][0][0][kk3]]
                  matchingChildren2 = [vs2[kk][0][0][kk4][r] for r in rowsol.tolist()]
  else:
      for kk in range(minMaximalSetCardinality1,min(minMaximalSetCardinality2,maxMaximalSetCardinality1)+1):
        for kk3 in range(0,len(vs1[kk][0][0])):
          for kk4 in range(0,len(vs2[minMaximalSetCardinality2][0][0])): 
            if vs1[kk][0][1][kk3][0] or vs2[minMaximalSetCardinality2][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap
              lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[kk][0][0][kk3]],
                                                    [node_id_index_dict2[x] for x in vs2[minMaximalSetCardinality2][0][0][kk4]])]
              mat_shape = lap_submat.shape
              if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0]) 

              x, rowsol = linear_sum_assignment(-lap_submat)
              thisSim = lap_submat[x, rowsol].sum()

              if thisSim > sim:
                if minMaximalSetCardinality2 < kk:
                  sim = thisSim
                  matchingChildren1 = [vs1[kk][0][0][kk3][r] for r in rowsol.tolist()]
                  matchingChildren2 = [n for n in vs2[minMaximalSetCardinality2][0][0][kk4]]
                else:
                  sim = thisSim
                  matchingChildren1 = [n for n in vs1[kk][0][0][kk3]]
                  matchingChildren2 = [vs2[minMaximalSetCardinality2][0][0][kk4][r] for r in rowsol.tolist()]

  for kk in range(max(minMaximalSetCardinality1,minMaximalSetCardinality2)+1, min(maxMaximalSetCardinality1,maxMaximalSetCardinality2)+1):
      for kk3 in range(0, len(vs1[kk][0][0])): 
        for kk4 in range(0, len(vs2[kk][0][0])):
          if vs1[kk][0][1][kk3][0] or vs2[kk][0][1][kk4][0]: #if either set pulls nodes from all main subtrees, run lap          
            lap_submat = agreement['pAgrM'][np.ix_([node_id_index_dict1[x] for x in vs1[kk][0][0][kk3]],
                                                  [node_id_index_dict2[x] for x in vs2[kk][0][0][kk4]])]
            mat_shape = lap_submat.shape
            if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0])

            x, rowsol = linear_sum_assignment(-lap_submat)
            thisSim = lap_submat[x, rowsol].sum()

            if thisSim > sim: 
              sim = thisSim 
              matchingChildren1 = [n for n in vs1[kk][0][0][kk3]]
              matchingChildren2 = [vs2[kk][0][0][kk4][r] for r in rowsol.tolist()]

  matchingChildren1 = [node_id_index_dict1[c] for c in matchingChildren1]
  matchingChildren2 = [node_id_index_dict2[c] for c in matchingChildren2]
  return matchingChildren1,matchingChildren2,sim

# MAKE HARDCODED VALID SETS JSON 
def generate_combinations(n, child_options=['0', '2'], prefix='', combos=None):
    #generate combos of n children where the options for how many children they each have (grandchildren of main node) are child_options
    if combos is None:
        combos = []
    if n == 0:
        combos.append(prefix)
    else:
        for option in child_options:
            generate_combinations(n-1, child_options, prefix + '_' + option, combos)
    return combos

def _convert_numpy(obj):
    # Convert NumPy array to list, which is a JSON-serializable format
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to Python list
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def save_valid_sets(validSetDir, 
                    valid_set_dict_path, 
                    combos):
    
    # Hardcode the valid sets. 
    # Load the valid sets from pre-made matlab valid set files, 
    # convert them to python dicts and save as a json. 

    #assemble valid sets dict
    validSet_dict = {}  
    for combo in combos: 
        filepath = os.path.join(validSetDir, combo+'.mat')
        if os.path.isfile(filepath):
            validSet = sio.loadmat(filepath)
            pOT1 = validSet['pOT1'][:,0]
            vs = validSet['vs'][0]

            validSet_dict[combo] = {'pOT1' : pOT1,
                                    'vs' : vs}

    # Write dict to JSON file
    with open(valid_set_dict_path, 'w') as f:
        json.dump(validSet_dict, f, default=_convert_numpy)

def load_valid_sets(valid_set_dict_path):
    #Load a dictionary of hardcoded valid sets
    with open(valid_set_dict_path) as f:
        valid_set_dict = json.load(f)
    return valid_set_dict
