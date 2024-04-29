import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from neuron_morphology.transforms.affine_transform import AffineTransform, rotation_from_angle, affine_from_transform
from morph_utils.graph_traversal import bfs_tree, dfs_tree, get_path_to_root, get_path_and_path_dist_between_two_nodes
from tree_comparison.maxdepthtwo_utils import getValidSetCardinality, getMatchingChildren
from convexsimfunc_utils import edges_between
from more_itertools import flatten
from tree_comparison.cpp.quantized_convex_matching import quantized_convex_matching

def rotate_morphology(morphology, angle, axis=1):
    """
    Rotate morphology around an axis.

    :param morph: neuron_morphology Morphology object
    :param angle: angle to rotate in Radians. clockwise direction when viewed from the positive y-axis towards the positive x-axis (right-hand rule convention)
    :param axis: axis to rotate around (0=x, 1=y, 2=z)
    :return: rotated neuron_morphology Morphology object 
    """
    rotation_affine = AffineTransform(affine_from_transform(rotation_from_angle(angle, axis)))
    rotated_morphology = rotation_affine.transform_morphology(morphology) # if you need the original object to remain unchanged do morph.clone()

    return rotated_morphology

def ensure_flat_list(item):
    if isinstance(item, list):
        if len(item) == 1 and isinstance(item[0], list): return item[0]  # Extract the inner list
        else: return item  # Already in the desired form
    elif isinstance(item, int): return [item] # a single int, return as list 
    else: return item  # Not a list or int, return as is

def compute_nDistance_matrix(raw_morphology):
    """
    Will get the path distance between all irreducible nodes of a subtree.

    """

    irreducible_nodes = [n for n in raw_morphology.nodes() if
                         (len(raw_morphology.get_children(n)) > 1) or (len(raw_morphology.get_children(n)) == 0) or (
                                     n['parent'] == -1)]
    soma = raw_morphology.get_soma()
    if not soma:
        soma_list = [n for n in raw_morphology.nodes() if n['parent'] == -1]
        if len(soma_list) != 1:
            print("Invalid Number of somas (0 or >1)")
            print("Noneeeeee")
        else:
            soma = soma_list[0]
    if soma not in irreducible_nodes and soma:
        irreducible_nodes.append(raw_morphology.get_soma())

    num_irr_nodes = len(irreducible_nodes)
    ndist_matrix = np.zeros((num_irr_nodes, num_irr_nodes))
    node_id_index_dict = {no['id']: ct for ct, no in enumerate(irreducible_nodes)}
    leaf_nodes = raw_morphology.get_leaf_nodes()

    for leaf_no in leaf_nodes:
        seg_up = get_path_to_root(leaf_no, raw_morphology)
        irreducible_seg_up = [n for n in seg_up if n in irreducible_nodes]

        assert leaf_no in seg_up

        for seg_walker, seg_no_i in enumerate(irreducible_seg_up):
            no_i_id = seg_no_i['id']
            no_i_idx = node_id_index_dict[no_i_id]

            all_other_nodes_in_seg = irreducible_seg_up[seg_walker + 1:]
            for seg_no_j in all_other_nodes_in_seg:
                no_j_id = seg_no_j['id']
                no_j_idx = node_id_index_dict[no_j_id]

                _, dist = get_path_and_path_dist_between_two_nodes(upper_node=seg_no_j,
                                                               lower_node=seg_no_i,
                                                               morphology=raw_morphology)

                ndist_matrix[no_i_idx, no_j_idx] = 2 * dist
                ndist_matrix[no_j_idx, no_i_idx] = 2 * dist

    return ndist_matrix, node_id_index_dict

def preOrderTraversal(morphology, st_node):
    """
    Remember we are returning node ids where as Uygars code returns node indicies
    """
    visited = []
    num_downstream_nodes = []
    queue = deque([st_node])
    while len(queue) > 0:
        current_node = queue.popleft()

        _, num_nodes_down_tree = bfs_tree(current_node, morphology)
        num_nodes_down_tree = num_nodes_down_tree - 1  # since num_nodes_down_tree includes current_node in count
        num_downstream_nodes.append(num_nodes_down_tree)
        visited.append(current_node)

        for ch_no in morphology.get_children(current_node):
            queue.appendleft(ch_no)

    visited_ids = [n['id'] for n in visited]
    res = np.array([visited_ids, num_downstream_nodes]).T
    return res

def find_leaves(morphology, st_node):
    "Get the leaves/tip nodes of a morphology"
    nodes_down,_ = dfs_tree(morphology, st_node)
    leaves = [n for n in nodes_down if morphology.get_children(n)==[]]
    return leaves

def linearAssignment_matchingNodes(agreement,
                                   node1,
                                   node1_children,
                                   node1_parent_id_matrix_idx,
                                   ndDistanceMatrix1,
                                   preON1,
                                   tree1,
                                   node_id_index_dict1,
                                   node2,
                                   node2_children,
                                   node2_parent_id_matrix_idx,
                                   ndDistanceMatrix2,
                                   preON2,
                                   tree2,
                                   node_id_index_dict2,
                                   tree1_paths, 
                                   tree2_paths,
                                   maxDepth,
                                   simFunc,
                                   validSetDir):

    node1_children_array = np.array(node1_children)
    node2_children_array = np.array(node2_children)

    node1_children_matrix_idx = [node_id_index_dict1[n['id']] for n in node1_children]
    node2_children_matrix_idx = [node_id_index_dict2[n['id']] for n in node2_children]

    node1_matrix_idx = node_id_index_dict1[node1['id']]
    node2_matrix_idx = node_id_index_dict2[node2['id']]

    sub_mat_1 = agreement['agrM'][node1_matrix_idx, node2_children_matrix_idx]
    sub_mat_2 = agreement['agrM'][node1_children_matrix_idx, node2_matrix_idx]

    maxChild2 = np.argmax(sub_mat_1)
    maxSimilarity = sub_mat_1[maxChild2]

    maxChild1 = np.argmax(sub_mat_2)
    maxSimilarity2 = sub_mat_2[maxChild1]

    if maxSimilarity2 > maxSimilarity:
        maxSimilarity = maxSimilarity2
        maxMatch1 = node1_children[maxChild1]
        maxMatch2 = node2
    else:
        maxMatch1 = node1
        maxMatch2 = node2_children[maxChild2]

    if maxDepth == 1:
        lap_submat = agreement['pAgrM'][np.ix_(node1_children_matrix_idx, node2_children_matrix_idx)]
        mat_shape = lap_submat.shape
        if len(mat_shape) != 2: lap_submat = lap_submat.reshape(1, mat_shape[0])

        x, rowsol = linear_sum_assignment(-lap_submat)
        sim = lap_submat[x, rowsol].sum()

        if len(node2_children) < len(node1_children):
            matchingChildren1 = node1_children_array[rowsol]
            matchingChildren2 = node2_children
        else:
            matchingChildren1 = node1_children
            matchingChildren2 = node2_children_array[rowsol]

        matchingChildren1 = [node_id_index_dict1[c["id"]] for c in matchingChildren1]
        matchingChildren2 = [node_id_index_dict2[c["id"]] for c in matchingChildren2]

    elif maxDepth == 2:
        minMaximalSetCardinality1, maxMaximalSetCardinality1, vs1 = getValidSetCardinality(validSetDir, tree1, node1, node1_children)
        minMaximalSetCardinality2, maxMaximalSetCardinality2, vs2 = getValidSetCardinality(validSetDir, tree2, node2, node2_children)
        matchingChildren1, matchingChildren2, sim = getMatchingChildren(maxMaximalSetCardinality1, minMaximalSetCardinality1, vs1, 
                                                                        maxMaximalSetCardinality2, minMaximalSetCardinality2, vs2,
                                                                        agreement, node_id_index_dict1, node_id_index_dict2) 

    if sim > maxSimilarity:
        agreement['agrM'][node1_matrix_idx, node2_matrix_idx] = sim
        agreement['agrTypeM'][node1_matrix_idx, node2_matrix_idx] = True

        for i in range(len(matchingChildren1)):
            agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][0] = \
            ensure_flat_list(agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][0]) + ensure_flat_list([agreement['pAgrNodes'][matchingChildren1[i], matchingChildren2[i]][0]])
            agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][1] = \
            ensure_flat_list(agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][1]) + ensure_flat_list([agreement['pAgrNodes'][matchingChildren1[i], matchingChildren2[i]][1]])
        
        agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][0] = \
        agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][0] + [node1['id']]
        agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][1] = \
        agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][1] + [node2['id']]

    else:
        agreement['agrM'][node1_matrix_idx, node2_matrix_idx] = maxSimilarity
        agreement['agrTypeM'][node1_matrix_idx, node2_matrix_idx] = False

        maxMatch1_matrix_idx = node_id_index_dict1[maxMatch1['id']]
        maxMatch2_matrix_idx = node_id_index_dict2[maxMatch2['id']]

        agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][0] = \
        agreement['agrNodes'][maxMatch1_matrix_idx, maxMatch2_matrix_idx][0]
        agreement['agrNodes'][node1_matrix_idx, node2_matrix_idx][1] = \
        agreement['agrNodes'][maxMatch1_matrix_idx, maxMatch2_matrix_idx][1]

    #PLUS SIMILARITY CALCULATIONS
    if (node1['parent'] != -1) & (node2['parent'] != -1):

        maxPlusSimilarity = -1e-10
        for n1 in preON1[:, 0]:
            n1_parent_id = tree1.node_by_id(n1)['parent']
            n1_parent_id_idx = node_id_index_dict1[n1_parent_id]

            for kk2 in range(len(preON2[:, 0])):
                n2 = preON2[kk2, 0]
                n2_parent_id = tree2.node_by_id(n2)['parent']
                n2_parent_id_idx = node_id_index_dict2[n2_parent_id]

                n1_idx = node_id_index_dict1[n1]
                n2_idx = node_id_index_dict2[n2]
                
                if agreement['agrTypeM'][n1_idx, n2_idx]:

                    min_val = min(ndDistanceMatrix1[n1_idx, node1_parent_id_matrix_idx],
                                  ndDistanceMatrix2[n2_idx, node2_parent_id_matrix_idx])
                    max_val = max(ndDistanceMatrix1[n1_parent_id_idx, node1_parent_id_matrix_idx],
                                  ndDistanceMatrix2[n2_parent_id_idx, node2_parent_id_matrix_idx])
                    simpleBound = agreement['agrM'][n1_idx, n2_idx] + min_val
                    simpleBound2 = agreement['pAgrM'][n1_idx, n2_idx] + max_val

                    if (n1 == node1['id'] & n2 == node2['id']) or (maxPlusSimilarity < (min(simpleBound, simpleBound2) * 0.9999)):
                        if simFunc != "length": #convex
                            edge_lengths1, edge_orientations1, edge_areas1, pillars1 = edges_between(tree1.node_by_id(n1), tree1.node_by_id(node1['parent']), tree1, tree1_paths)
                            edge_lengths2, edge_orientations2, edge_areas2, pillars2 = edges_between(tree2.node_by_id(n2), tree2.node_by_id(node2['parent']), tree2, tree2_paths)
                            cvxSim = quantized_convex_matching(edge_lengths1, edge_orientations1, edge_areas1, pillars1, edge_lengths2, edge_orientations2, edge_areas2, pillars2)
                            thisPlusSimilarity = agreement['agrM'][n1_idx, n2_idx] + cvxSim

                        else: #length 
                            thisPlusSimilarity = simpleBound
                        if (thisPlusSimilarity > maxPlusSimilarity): 
                            maxPlusSimilarity = thisPlusSimilarity
                            maxndd1_idx = node_id_index_dict1[n1]
                            maxndd2_idx = node_id_index_dict2[n2]
                    else:
                        kk2 = kk2 + preON2[kk2, 1]
        agreement['pAgrM'][node1_matrix_idx, node2_matrix_idx] = maxPlusSimilarity
        agreement['pAgrNodes'][node1_matrix_idx, node2_matrix_idx][0] = agreement['agrNodes'][maxndd1_idx, maxndd2_idx][0]
        agreement['pAgrNodes'][node1_matrix_idx, node2_matrix_idx][1] = agreement['agrNodes'][maxndd1_idx, maxndd2_idx][1]    
    return agreement
