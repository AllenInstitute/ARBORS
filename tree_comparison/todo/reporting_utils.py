from math import radians
from neuron_morphology.swc_io import morphology_from_swc
from morph_utils.modifications import generate_irreducible_morph
from tree_comparison.utils import compute_nDistance_matrix, rotate_morphology
from tree_comparison.convexsimfunc_utils import edges_between, get_tree_paths
from tree_comparison.cpp.quantized_convex_matching import quantized_convex_matching


def get_matched_parents(irreducible_tree, matched_node_ids):
    """
    Given a set of matched node ids in a tree, find the parents 
    of these matched nodes on the other end of the matched branch. 

    :param irreducible_tree: a Morphology object with only the irreducible nodes.
    :param matched_node_ids: a list of node ids.
    :return: a list of the parent node ids of the the input nodes. 
    """
    matched_node_parent_ids = []
    for n in matched_node_ids:
        node = irreducible_tree.node_by_id(n)

        while True:
            p = node['parent']
            if p == -1: break 
            if p in matched_node_ids: break 
            node = irreducible_tree.node_by_id(p)

        matched_node_parent_ids.append(p)
    
    return matched_node_parent_ids


def get_edge_similarity_length(swc_file_1, swc_file_2, nrn1_matched_nodes, nrn2_matched_nodes): 

    """
    Return a length-based similarity score for each matched edge. 
    Matched edges are:
        edge1_i = nrn1_matched_nodes[i] to nrn1_matched_nodes_parents[i] in Tree1 
        edge2_i = nrn2_matched_nodes[i] to nrn2_matched_nodes_parents[i] in Tree2

    Similarity is the fraction of overlap in the lengths of the edges: 
        sim = shorter_edge_length/longer_edge_length 

    Similarity of 1 means the matched edges are the same length. 
    Similarity of 0 means the matched edges have no length overlap (one of the edges is the soma). 
    Similarity of 0.5 means one edge is half the length of the other. 
    Note: it does not currently tell us which edge is longer. 
    
    :param swc_file_1: path to swc file
    :param swc_file_2: path to swc file
    :param nrn1_matched_nodes: list of ordered matched node ids from tree1_raw
    :param nrn2_matched_nodes: list of ordered matched node ids from tree2_raw
    :return: list of similarity between edge matches between trees (nrns)
    """
    tree1_raw = morphology_from_swc(swc_file_1)
    tree2_raw = morphology_from_swc(swc_file_2)

    tree1 = generate_irreducible_morph(tree1_raw)
    tree2 = generate_irreducible_morph(tree2_raw)

    nrn1_matched_nodes_parents = get_matched_parents(tree1, nrn1_matched_nodes)
    nrn2_matched_nodes_parents = get_matched_parents(tree2, nrn2_matched_nodes)

    ndDistanceMatrix1, node_id_index_dict1 = compute_nDistance_matrix(tree1_raw)
    ndDistanceMatrix2, node_id_index_dict2 = compute_nDistance_matrix(tree2_raw)

    edge_similarity_score = []
    for i, n1 in enumerate(nrn1_matched_nodes):
        p1 = nrn1_matched_nodes_parents[i]
        n2 = nrn2_matched_nodes[i]
        p2 = nrn2_matched_nodes_parents[i]

        if p1 == -1: edge_length1 = 0
        else: edge_length1 = ndDistanceMatrix1[node_id_index_dict1[n1]][node_id_index_dict1[p1]]

        if p2 == -1: edge_length2 = 0
        else: edge_length2 = ndDistanceMatrix2[node_id_index_dict2[n2]][node_id_index_dict2[p2]]

        if edge_length1 == edge_length2 == 0:
            sim = 1.0 #matching soma nodes, they have no parent edges. 
        else: 
            sim = min(edge_length1, edge_length2)/max(edge_length1, edge_length2)
            sim = round(sim, 4)

        edge_similarity_score.append(sim)

    return edge_similarity_score, nrn1_matched_nodes_parents, nrn2_matched_nodes_parents

def downsample_lists(lst, downsample_factor):
    downsampled_lst = lst[::downsample_factor]
    if lst[-1] not in downsampled_lst:
        downsampled_lst.append(lst[-1])
    return downsampled_lst

def downsample_flattened_lists(lst, downsample_factor, 
                               repeat_length=3 #num elements before repeating (x,y,z = 3) --> [x1,y1,z1, x2,y2,z2, ... xn,yn,zn]
                               ):
    downsampled_lst = lst[::repeat_length * downsample_factor]
    last_triplet_start = len(lst) - repeat_length
    last_triplet = tuple(lst[last_triplet_start:])  # Convert last triplet to tuple
    if last_triplet not in (tuple(downsampled_lst[i:i+repeat_length]) for i in range(0, len(downsampled_lst), repeat_length)):
        downsampled_lst.extend(lst[last_triplet_start:])
    return downsampled_lst


def get_edge_similarity_convex(swc_file_1, swc_file_2, nrn1_matched_nodes, nrn2_matched_nodes, orientation=0.0, partition_length=1/2000):
    """
    Compute the quantized convex matching similarity score for each matched edge. 
    Matched edges are:
        edge1_i = nrn1_matched_nodes[i] to nrn1_matched_nodes_parents[i] in tree1 
        edge2_i = nrn2_matched_nodes[i] to nrn2_matched_nodes_parents[i] in tree2

    :param swc_file_1: path to swc file
    :param swc_file_2: path to swc file
    :param nrn1_matched_nodes: list of ordered matched node ids from nrn1
    :param nrn2_matched_nodes: list of ordered matched node ids from nrn2
    :return: list of similarity between edge matches between trees (nrns)
    """
    tree1_raw = morphology_from_swc(swc_file_1)
    tree2_raw = morphology_from_swc(swc_file_2)

    if not orientation == 0: 
        tree2_raw = rotate_morphology(tree2_raw, radians(orientation))

    tree1 = generate_irreducible_morph(tree1_raw)
    tree2 = generate_irreducible_morph(tree2_raw)

    nrn1_matched_nodes_parents = get_matched_parents(tree1, nrn1_matched_nodes)
    nrn2_matched_nodes_parents = get_matched_parents(tree2, nrn2_matched_nodes)

    edge_similarity_score = []
    for i, n1 in enumerate(nrn1_matched_nodes):
        p1 = nrn1_matched_nodes_parents[i]
        n2 = nrn2_matched_nodes[i]
        p2 = nrn2_matched_nodes_parents[i]

        #if either are soma, similarity = 0... just don't plot the soma node
        if p1 == -1 or p2 == -1:
            sim = 0
        else:
            # #
            # print('i: {}'.format(i))
            # print('n1: {}, n2: {}'.format(n1, n2))
            # print('p1: {}, p2: {}'.format(p1, p2))

            # edge_lengths1, edge_orientations1, edge_areas1, pillars1 = edges_between(tree1.node_by_id(n1), tree1.node_by_id(p1), tree1, tree1_paths)
            # edge_lengths2, edge_orientations2, edge_areas2, pillars2 = edges_between(tree2.node_by_id(n2), tree2.node_by_id(p2), tree2, tree2_paths)

            # #TODO downsample here
            # #if lengths are > thresh: downsample by factor 
            # downsample_thresh = 200 
            # downsample_factor = 100
            # if len(edge_lengths1) > downsample_thresh: 
            #     edge_lengths1 = downsample_lists(edge_lengths1, downsample_factor)
            #     edge_areas1 = downsample_lists(edge_areas1, downsample_factor)
            #     pillars1 = downsample_lists(pillars1, downsample_factor)
            #     edge_orientations1 = downsample_flattened_lists(edge_orientations1, downsample_factor) 
            # if len(edge_lengths2) > downsample_thresh: 
            #     edge_lengths2 = downsample_lists(edge_lengths2, downsample_factor)
            #     edge_areas2 = downsample_lists(edge_areas2, downsample_factor)
            #     pillars2 = downsample_lists(pillars2, downsample_factor)
            #     edge_orientations2 = downsample_flattened_lists(edge_orientations2, downsample_factor) 


            # # print('edge_lengths1: {}'.format(edge_lengths1))
            # # print('edge_orientations1: {}'.format(edge_orientations1))
            # # print('edge_areas1: {}'.format(edge_areas1))
            # # print('pillars1: {}'.format(pillars1))
            # # print('edge_lengths2: {}'.format(edge_lengths2))
            # # print('edge_orientations2: {}'.format(edge_orientations2))
            # # print('edge_areas2: {}'.format(edge_areas2))
            # # print('pillars2: {}'.format(pillars2))

            # print('edge_lengths1: {}'.format(len(edge_lengths1)))
            # print('edge_orientations1: {}'.format(len(edge_orientations1)))
            # print('edge_areas1: {}'.format(len(edge_areas1)))
            # print('pillars1: {}'.format(len(pillars1)))
            # print('edge_lengths2: {}'.format(len(edge_lengths2)))
            # print('edge_orientations2: {}'.format(len(edge_orientations2)))
            # print('edge_areas2: {}'.format(len(edge_areas2)))
            # print('pillars2: {}'.format(len(pillars2)))

            # sim = quantized_convex_matching(edge_lengths1, edge_orientations1, edge_areas1, pillars1, edge_lengths2, edge_orientations2, edge_areas2, pillars2)

            # ###
            sim = -1
            sim = round(sim, 4)

        edge_similarity_score.append(sim)

    return edge_similarity_score, nrn1_matched_nodes_parents, nrn2_matched_nodes_parents