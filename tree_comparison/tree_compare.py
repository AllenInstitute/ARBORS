import os
import json
import ntpath
import argschema as ags
import numpy as np
# import concurrent.futures
from multiprocessing import Pool
from importlib.resources import files
from math import pi, radians 
from tqdm import tqdm
from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.constants import AXON, BASAL_DENDRITE, APICAL_DENDRITE
from neuron_morphology.morphology import Morphology
from morph_utils.modifications import generate_irreducible_morph
from morph_utils.graph_traversal import dfs_tree, get_path_to_root
from morph_utils.measurements import tree_length
from tree_comparison.utils import compute_nDistance_matrix, find_leaves, preOrderTraversal, linearAssignment_matchingNodes, rotate_morphology, remove_duplicate_nodes, flatten
from tree_comparison.maxdepthtwo_utils import load_valid_sets
from tree_comparison.convexsimfunc_utils import get_tree_paths, edges_between
from tree_comparison.cpp.quantized_convex_matching import quantized_convex_matching
from tree_comparison.reporting_utils import get_edge_similarity_length, get_edge_similarity_convex

class IO_Schema(ags.ArgSchema):
    swc_1_path = ags.fields.InputFile(dump_default=None, metadata={'description' : "path to tree1 swc"}, allow_none=True)
    swc_2_path = ags.fields.InputFile(dump_default=None, metadata={'description' : "path to tree2 swc"}, allow_none=True)

    compartments = ags.fields.List(ags.fields.Int, cli_as_single_argument=False, dump_default=[AXON, BASAL_DENDRITE, APICAL_DENDRITE], metadata={'description' : "compartment types to compare."})

    output_dir = ags.fields.OutputDir(metadata={'description' : "dir to output jsons"}, dump_default=None, allow_none=True)
    
    similarity_function = ags.fields.String(metadata={'description' : "Similarity function to use. Options: 'length or convex'"}, dump_default='length')
    max_depth = ags.fields.Int(metadata={'description' : "Max depth to use for algorithm"}, dump_default=1)
    orientation = ags.fields.Float(metadata={'description' : "Rotation in degrees for Tree2 (around y axis)"}, dump_default=1)
    
    valid_set_dir = ags.fields.InputDir(metadata={'description' : "Directory with valid set files"}, dump_default=str(files('tree_comparison') / "data"))
    valid_set_dict = ags.fields.InputFile(metadata={'description' : "JSON file with hardcoded valid sets"}, dump_default=os.path.join(str(files('tree_comparison') / "data"), 'validSet_mouseSST.json'))
   
    partition_length = ags.fields.Float(metadata={'description' : "Partition length for downsampling tree branch"}, dump_default=1/2000)
    angle_threshold = ags.fields.Float(metadata={'description' : "Angle threshold for downsampling tree branch"}, dump_default=pi/9)
    segment_threshold = ags.fields.Float(metadata={'description' : "Segment threshold for downsampling tree branch"}, dump_default=1/200)


def compare_two_trees(swc_file_1, swc_file_2, compartments, simFunc, maxDepth, valid_set_dict, 
                      partition_length, angle_threshold, segment_threshold, orientation, valid_set_dir=None):

    """
    Will generate a similarirty score for two input swc files.

    :param swc_file_1: str, path to swc file 1
    :param swc_file_2: str, path to swc file 2
    :param simFunc: str, either 'length' or 'convex'
    :param maxDepth: int, 1 or 2
    :valid_set_dir: str, path to directory with valid set files
    :param angle_threshold: if angle between nodes is less than this, drop nodes here(?) currently unused. (for resampling a branch)
    :param partition_length: length between resampled nodes (for resampling a branch)
    :param segment_threshold: microns if segment is longer than this...? (for resampling a branch)
    :param orientation: angles (degrees) to rotate tree2 (swc_file_2) before comparison 
    :return: float, cell similarity
    """
    #load hardcoded valid set dict if given for max depth = 2
    if (maxDepth == 2) and (valid_set_dict is not None): 
        # Load hardcoded valid sets
        valid_set_dict = load_valid_sets(valid_set_dict_path = valid_set_dict) 
        hardcoded_vs = True
    else:
        valid_set_dict = {}
        hardcoded_vs = False

    tree1_raw = morphology_from_swc(swc_file_1)
    tree2_raw = morphology_from_swc(swc_file_2)

    #only keep nodes compartment type (if a tree doesn't have nodes of this type, maxAgreement = 0 and skip comp)
    tree1_comp_leaves = tree1_raw.get_leaf_nodes(node_types=compartments)
    tree1_comp_leaves_to_root = [get_path_to_root(leaf, tree1_raw) for leaf in tree1_comp_leaves]
    tree1_raw = remove_duplicate_nodes(flatten(tree1_comp_leaves_to_root))
    tree1_raw = Morphology(tree1_raw, node_id_cb=lambda node: node["id"], parent_id_cb=lambda node: node["parent"])

    tree2_comp_leaves = tree2_raw.get_leaf_nodes(node_types=compartments)
    tree2_comp_leaves_to_root = [get_path_to_root(leaf, tree2_raw) for leaf in tree2_comp_leaves]
    tree2_raw = remove_duplicate_nodes(flatten(tree2_comp_leaves_to_root))
    tree2_raw = Morphology(tree2_raw, node_id_cb=lambda node: node["id"], parent_id_cb=lambda node: node["parent"])

    if tree1_raw and tree2_raw:
        #both trees have nodes of the specified compartment types, compare them.  

        #rotate tree 2 around y axis by 'orientation' degrees before comparing to tree 1
        if not orientation == 0: 
            tree2_raw = rotate_morphology(tree2_raw, radians(orientation))

        tree1_length = round(tree_length(tree1_raw), 4)
        tree2_length = round(tree_length(tree2_raw), 4)

        tree1 = generate_irreducible_morph(tree1_raw)
        tree2 = generate_irreducible_morph(tree2_raw)

        root1, root2 = tree1.get_soma(), tree2.get_soma()

        if not root1: root1 = [n for n in tree1.nodes() if n['parent'] == -1][0]
        if not root2: root2 = [n for n in tree2.nodes() if n['parent'] == -1][0]

        postOrderNodes1,_ = dfs_tree(tree1, root1)
        postOrderNodes1.reverse()
        postOrderNodes2,_ = dfs_tree(tree2, root2)
        postOrderNodes2.reverse()

        preOrderNodes1 = preOrderTraversal(tree1, root1)
        preOrderNodes2 = preOrderTraversal(tree2, root2)

        partition_length_tree1 = partition_length*tree1_length 
        segment_threshold_tree1 = segment_threshold*tree1_length
        partition_length_tree2 = partition_length*tree2_length 
        segment_threshold_tree2 = segment_threshold*tree2_length

        tree1_paths = get_tree_paths(tree1_raw, partition_length_tree1)
        tree2_paths = get_tree_paths(tree2_raw, partition_length_tree2)

        ndDistanceMatrix1, node_id_index_dict1 = compute_nDistance_matrix(tree1_raw)
        ndDistanceMatrix2, node_id_index_dict2 = compute_nDistance_matrix(tree2_raw)

        max_matching_node_1_idx=[]
        max_matching_node_2_idx=[]

        agreement = {}
        agreement['agrM'] = np.zeros((len(tree1), len(tree2)))
        agreement['pAgrM'] = np.zeros((len(tree1), len(tree2))) 
        agreement['agrTypeM'] = np.zeros((len(tree1), len(tree2)), dtype=bool)
        agreement['agrNodes'] = np.empty((len(tree1), len(tree2)), dtype=list)
        agreement['pAgrNodes'] = np.empty((len(tree1), len(tree2)), dtype=list)

        for i in range(len(tree1)):
            for j in range(len(tree2)):
                agreement['agrNodes'][i, j] = [[], []]
                agreement['pAgrNodes'][i, j] = [[], []]

        maxAgreement = 0
        for i_1 in tqdm(range(len(tree1.nodes()))):
        # for i_1 in range(len(tree1.nodes())): #Uncomment to allow print statements - diagnostic. TODO remove when commit 
            node1 = postOrderNodes1[i_1]
            node1_children = tree1.get_children(node1)
            node_1_matrix_index = node_id_index_dict1[node1['id']]

            for node2 in postOrderNodes2:
                node2_children = tree2.get_children(node2)
                node_2_matrix_index = node_id_index_dict2[node2['id']]
            
                if node1 != root1:
                    node1_parent_id = node1['parent']
                    node1_parent_id_matrix_idx = node_id_index_dict1[node1_parent_id]
                else:
                    node1_parent_id_matrix_idx = None

                if node2 != root2:
                    node2_parent_id = node2['parent']
                    node2_parent_id_matrix_idx = node_id_index_dict2[node2_parent_id]
                else:
                    node2_parent_id_matrix_idx = None

                if (node1_children != []) and (node2_children != []):

                    subtreeRootPos1 = np.where(preOrderNodes1[:, 0] == node1['id'])[0][0]
                    subtreeRootPos2 = np.where(preOrderNodes2[:, 0] == node2['id'])[0][0]

                    # plus one to get all members below (matlab v python)
                    stop_index_1 = subtreeRootPos1 + preOrderNodes1[subtreeRootPos1, 1] + 1
                    stop_index_2 = subtreeRootPos2 + preOrderNodes2[subtreeRootPos2, 1] + 1

                    preON1 = preOrderNodes1[subtreeRootPos1: stop_index_1, :]
                    preON2 = preOrderNodes2[subtreeRootPos2: stop_index_2, :]

                    agreement = linearAssignment_matchingNodes(agreement,
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
                                                                maxDepth=maxDepth,
                                                                simFunc=simFunc,
                                                                valid_set_dict=valid_set_dict,
                                                                valid_set_dir=valid_set_dir,
                                                                hardcoded_vs = hardcoded_vs)
                    #Handle case where valid set for a branch pattern is not found. 
                    #TODO in the future we could make it, add it to the hardcoded valid sets file, then continue with the tree comparison. 
                    if agreement == -1: 
                        #branching pattern in tree1 not found in hardcoded (or .mat files if not using hardcoded mode) so skip this tree comparison 
                        #skip all the rotation comparisons, b/c branching pattern issue will be the same for all rots 
                        return  -1, -1, [], [], []
                    if agreement == -2: 
                        #branching pattern in tree2 not found in hardcoded (or .mat files if not using hardcoded mode) so skip this tree comparison 
                        #skip all the rotation comparisons, b/c branching pattern issue will be the same for all rots 
                        return -2, -2, [], [], []

                else:
                    agreement['agrM'][node_1_matrix_index, node_2_matrix_index] = 0
                    if (node1 != root1) and (node2 != root2):

                        leaves1 = find_leaves(tree1, node1)
                        leaves1_matrix_indices = [node_id_index_dict1[leaf['id']] for leaf in leaves1]

                        leaves2 = find_leaves(tree2, node2)
                        leaves2_matrix_indices = [node_id_index_dict2[leaf['id']] for leaf in leaves2]

                        if simFunc == "length":
                            sub_arr_1 = ndDistanceMatrix1[leaves1_matrix_indices, node1_parent_id_matrix_idx]
                            sub_arr_2 = ndDistanceMatrix2[leaves2_matrix_indices, node2_parent_id_matrix_idx]

                            maxHeight1, pos1 = np.max(sub_arr_1), np.argmax(sub_arr_1)
                            maxHeight2, pos2 = np.max(sub_arr_2), np.argmax(sub_arr_2)

                            lowerNode1 = leaves1[pos1]
                            lowerNode2 = leaves2[pos2]

                            min_dist = min([maxHeight1, maxHeight2])
                            agreement['pAgrM'][node_1_matrix_index, node_2_matrix_index] = min_dist


                        else: #convex
                            cvxSim = -1e-10
                            for leaf1 in leaves1:
                                sub_arr_1 = ndDistanceMatrix1[node_id_index_dict1[leaf1['id']], node1_parent_id_matrix_idx]
                                
                                for leaf2 in leaves2:
                                    sub_arr_2 = ndDistanceMatrix2[node_id_index_dict2[leaf2['id']], node2_parent_id_matrix_idx]
                                    
                                    if min(sub_arr_1, sub_arr_2) > cvxSim:
                                        edge_lengths1, edge_orientations1, edge_areas1, pillars1 = edges_between(leaf1, tree1.node_by_id(node1['parent']), tree1, tree1_paths)
                                        edge_lengths2, edge_orientations2, edge_areas2, pillars2 = edges_between(leaf2, tree2.node_by_id(node2['parent']), tree2, tree2_paths)
                                        thisCvxSim = quantized_convex_matching(edge_lengths1, edge_orientations1, edge_areas1, pillars1, edge_lengths2, edge_orientations2, edge_areas2, pillars2)

                                        if thisCvxSim > cvxSim:
                                            cvxSim = thisCvxSim
                                            lowerNode1 = leaf1
                                            lowerNode2 = leaf2

                            agreement['pAgrM'][node_1_matrix_index, node_2_matrix_index] = cvxSim
                        
                        agreement['pAgrNodes'][node_1_matrix_index, node_2_matrix_index][0] = [lowerNode1['id']]
                        agreement['pAgrNodes'][node_1_matrix_index, node_2_matrix_index][1] = [lowerNode2['id']]
                        agreement['agrNodes'][node_1_matrix_index, node_2_matrix_index][0] = [lowerNode1['id']]
                        agreement['agrNodes'][node_1_matrix_index, node_2_matrix_index][1] = [lowerNode2['id']]            
                        del lowerNode1
                        del lowerNode2
                        agreement['agrTypeM'][node_1_matrix_index, node_2_matrix_index] = ((node1 in leaves1) and (node2 in leaves2))

                    else:
                        agreement['pAgrM'][node_1_matrix_index, node_2_matrix_index] = 0
                        agreement['pAgrNodes'][node_1_matrix_index, node_2_matrix_index][0] = [node1['id']]
                        agreement['pAgrNodes'][node_1_matrix_index, node_2_matrix_index][1] = [node2['id']]
                        agreement['agrNodes'][node_1_matrix_index, node_2_matrix_index][0] = [node1['id']]
                        agreement['agrNodes'][node_1_matrix_index, node_2_matrix_index][1] = [node2['id']]    
                        agreement['agrTypeM'][node_1_matrix_index, node_2_matrix_index] = False

                if agreement['agrM'][node_1_matrix_index, node_2_matrix_index] > maxAgreement:
                    maxAgreement = agreement['agrM'][node_1_matrix_index, node_2_matrix_index]
                    max_matching_node_1_idx = node_1_matrix_index
                    max_matching_node_2_idx = node_2_matrix_index

        maxAgreement = round(maxAgreement, 4)

        matched_nodes_tree1 = agreement['agrNodes'][max_matching_node_1_idx][max_matching_node_2_idx][0]
        matched_nodes_tree2 = agreement['agrNodes'][max_matching_node_1_idx][max_matching_node_2_idx][1] 

        #save the plus agreement sim score btw all matched nodes... I think there's a reason why this is NOT correct, need to remember why that is. 
        matched_nodes_sim = [agreement['pAgrM'][node_id_index_dict1[matched_nodes_tree1[i]], node_id_index_dict2[matched_nodes_tree2[i]]] for i in range(len(matched_nodes_tree1))]

        #calculate the distance score 
        distance = tree1_length + tree2_length - maxAgreement
        distance = round(distance, 4)

        #Normalize the distance score
        norm_distance = distance / (distance + maxAgreement/2)
        norm_distance = round(norm_distance, 4)
    else: 
        #at least one tree doesn't have nodes of the specified compartment types, don't compare them. 
        maxAgreement = 0
        matched_nodes_tree1 = []
        matched_nodes_tree2 = []
        matched_nodes_sim = []
        distance = -3
        norm_distance = -3

    return distance, norm_distance, matched_nodes_tree1, matched_nodes_tree2, matched_nodes_sim

def compare_compartment(args):
    return compare_two_trees(*args)

def main(args):

    #save dict of comparison results 
    result = {
        "file1": args['swc_1_path'],
        "file2": args['swc_2_path'],
        "similarity_function": args['similarity_function'],
        "max_depth": args['max_depth'],
        "orientation": args['orientation'],
        "partition_length": args['partition_length'],
        "angle_threshold": args['angle_threshold'],
        "segment_threshold": args['segment_threshold']
    }

    #paralellize compartment comparisons 
    compartments = args['compartments'] 
    with Pool() as pool: #TODO is this actually paralellizing?
        results = pool.map(compare_compartment, [(args['swc_1_path'], 
                                                   args['swc_2_path'], 
                                                   [compartment], 
                                                   args['similarity_function'], 
                                                   args['max_depth'], 
                                                   args['valid_set_dict'], 
                                                   args['partition_length'], 
                                                   args['angle_threshold'], 
                                                   args['segment_threshold'], 
                                                   args['orientation'], 
                                                   args['valid_set_dir']) for compartment in compartments])

    #unpack results from each compartment comparison 
    for compartment, (distance, norm_distance, matched_nodes_tree1, matched_nodes_tree2, matched_nodes_similarity_pagrm) in zip(compartments, results):

        # #get matched nodes of this result #TODO only do this for the best result later on? 
        # if args['similarity_function'] == 'length':
        #     matched_nodes_similarity, matched_node_parents_tree1, matched_node_parents_tree2 = \
        #     get_edge_similarity_length(args['swc_1_path'], args['swc_2_path'], matched_nodes_tree1, matched_nodes_tree2)
        # else: # convex
        #     matched_nodes_similarity, matched_node_parents_tree1, matched_node_parents_tree2 = \
        #     get_edge_similarity_convex(args['swc_1_path'], args['swc_2_path'], matched_nodes_tree1, matched_nodes_tree2, 
        #                             args['orientation'], args['partition_length'])

        #save this compartment result
        result[f"distance_score_{compartment}"] = distance
        result[f"distance_score_normalized_{compartment}"] = norm_distance
        result[f"matched_nodes_tree1_{compartment}"] = matched_nodes_tree1
        result[f"matched_nodes_tree2_{compartment}"] = matched_nodes_tree2
        # result[f"matched_node_parents_tree1_{compartment}"] = matched_node_parents_tree1
        # result[f"matched_node_parents_tree2_{compartment}"] = matched_node_parents_tree2
        result[f"matched_node_edge_similarity_{compartment}"] = matched_nodes_similarity_pagrm


    #save comparison result (all compartment comparisons) as json
    json_path = os.path.join(args['output_dir'], f"{ntpath.basename(args['swc_1_path']).rsplit('.',1)[0]}_{ntpath.basename(args['swc_2_path']).rsplit('.',1)[0]}_rotate{args['orientation']}.json")
    with open(json_path, "w") as json_file:
        json.dump(result, json_file, indent=4)


def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)

