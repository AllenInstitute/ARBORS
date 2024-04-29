import os
import numpy as np
import pandas as pd
import ntpath
from math import pi, radians
import argschema as ags
from tqdm import tqdm
import itertools
from importlib.resources import files
from neuron_morphology.swc_io import morphology_from_swc
from morph_utils.modifications import generate_irreducible_morph
from morph_utils.graph_traversal import dfs_tree
from morph_utils.measurements import tree_length
from tree_comparison.utils import compute_nDistance_matrix, find_leaves, preOrderTraversal, linearAssignment_matchingNodes, rotate_morphology
from tree_comparison.reporting_utils import get_edge_similarity_length, get_edge_similarity_convex
from convexsimfunc_utils import get_tree_paths, edges_between
from tree_comparison.cpp.quantized_convex_matching import quantized_convex_matching

class IO_Schema(ags.ArgSchema):
    input_swc_file_1 = ags.fields.InputFile(
        default=None,
        description="1st swc file to load",
        allow_none=True,
    )
    input_swc_file_2 = ags.fields.InputFile(
        default=None,
        description="2nd swc file to load",
        allow_none=True,
    )
    input_swc_dir = ags.fields.InputDir(
        default=None,
        description="directory of swc files to make nXn comparison matrix for",
        allow_none=True
    )
    input_ref_dir = ags.fields.InputDir(
        default=None,
        description="directory of swc files to use as reference cells to compare all swcs in input_swc_dir -OR- input_swc_file_1 to",
        allow_none=True
    )
    output_file = ags.fields.OutputFile(description="Path to output csv", default="TreeCompareOutput.csv")
    similarity_function = ags.fields.String(description = "Similarity function to use. Options: 'length or convex'", default='length')
    max_depth = ags.fields.Int(description="Max depth to use for algorithm", default=1)

    valid_set_dir = ags.fields.InputDir(description="Directory with valid set files", default=str(files('tree_comparison') / "data"))

    partition_length = ags.fields.Float(description="Partition length for downsampling tree branch", default=1/2000)
    angle_threshold = ags.fields.Float(description="Angle threshold for downsampling tree branch", default=pi/9)
    segment_threshold = ags.fields.Float(description="Segment threshold for downsampling tree branch", default=1/200)

    number_of_rotations = max_depth = ags.fields.Int(description="Number of evenly sampled Tree2 rotations (around y axis) for comparison", default=1)

# def compare_two_trees(swc_file_1, swc_file_2, simFunc, maxDepth, valid_set_dir, partition_length, angle_threshold, segment_threshold, orientation):
def compare_two_trees(tree1_raw, tree2_raw, simFunc, maxDepth, valid_set_dir, partition_length, angle_threshold, segment_threshold, orientation):

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
    # tree1_raw = morphology_from_swc(swc_file_1)
    # tree2_raw = morphology_from_swc(swc_file_2)

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
                                                            validSetDir=valid_set_dir)


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
    distance = tree1_length + tree2_length - maxAgreement
    distance = round(distance, 4)

    matched_nodes_tree1 = agreement['agrNodes'][max_matching_node_1_idx][max_matching_node_2_idx][0]
    matched_nodes_tree2 = agreement['agrNodes'][max_matching_node_1_idx][max_matching_node_2_idx][1] 

    # return distance, max_matching_node_1_idx, max_matching_node_2_idx, agreement, node_id_index_dict1, node_id_index_dict2 
    return distance, matched_nodes_tree1, matched_nodes_tree2


def main(args):

    #check input file paths exist 
    input_file_1 = args['input_swc_file_1']
    input_file_2 = args['input_swc_file_2']
    input_swc_dir = args['input_swc_dir']
    input_ref_dir = args['input_ref_dir']

    if all([input_obj is None for input_obj in [input_file_1, input_file_2, input_swc_dir]]):
        msg = "No input swc files provided for comparison, no input directory provided. Nothing to do"
        raise ValueError(msg)

    if all([input_obj is not None for input_obj in [input_file_1, input_file_2, input_swc_dir]]):
        msg = "Input swc files provided and input directory provided. Either provide input swc paths OR provide directory with swc files, not both."
        raise ValueError(msg)
    
    #get number of rotations for tree2 (only relevant when using convex simfunc)
    if args['similarity_function'] == 'length': num_rotations == 1 #rotation won't change sim score when using length function
    else: num_rotations = args['number_of_rotations']
    if num_rotations < 1: num_rotations = 1
    orientations = [i * (360 / num_rotations) for i in range(num_rotations)]
    # orientations = [(i * (360 / num_rotations) + 45) for i in range(num_rotations)] #TODO delete this and use line above, special case for running 4 more rotations in test dataset


    #compare to reference dir of swcs 
    if input_ref_dir is not None:

        ref_files = [os.path.join(input_ref_dir,f) for f in os.listdir(input_ref_dir) if f.endswith(".swc")]
        if len(ref_files) < 1:
            msg = "At least one reference swc file required in the input directory."
            raise ValueError(msg)

        # compare each swc in the input swc dir to all the reference swcs
        if input_swc_dir is not None:

            swc_files = [os.path.join(input_swc_dir,f) for f in os.listdir(input_swc_dir) if f.endswith(".swc")]
            if len(swc_files) < 1:
                msg = "At least one swc file required in the input directory."
                raise ValueError(msg)
            
            for swc_file in swc_files:
                tree_raw = morphology_from_swc(swc_file)

                swc_filename = ntpath.basename(swc_file).rsplit('.',1)[0]
                csv_out_path = args['output_file'].replace('.csv', '_'+swc_filename+'.csv')

                results = []
                for ref_file in ref_files:
                    ref_tree_raw = morphology_from_swc(ref_file)

                    for i, orientation in enumerate(orientations): 
                        #TODO do the rotation here???

                        this_distance, this_matched_nodes_tree1, this_matched_nodes_tree2  =  compare_two_trees(ref_tree_raw.clone(),
                                                                                                                tree_raw.clone(), #rotate the comparison tree, not the reference tree (nice for viz later?)
                                                                                                                simFunc=args['similarity_function'],
                                                                                                                maxDepth=args['max_depth'],
                                                                                                                valid_set_dir=args['valid_set_dir'],
                                                                                                                partition_length=args['partition_length'],
                                                                                                                angle_threshold=args['angle_threshold'],
                                                                                                                segment_threshold=args['segment_threshold'],
                                                                                                                orientation=orientation)
                        if i == 0: 
                            best_distance = this_distance
                            best_orientation = orientation
                            best_matched_nodes_tree1 = this_matched_nodes_tree1
                            best_matched_nodes_tree2 = this_matched_nodes_tree2

                        elif this_distance < best_distance:
                            best_distance = this_distance
                            best_orientation = orientation 
                            best_matched_nodes_tree1 = this_matched_nodes_tree1
                            best_matched_nodes_tree2 = this_matched_nodes_tree2

                    #save this tree pair best results 
                    if args['similarity_function'] == 'length':
                        best_matched_nodes_similarity, best_matched_node_parents_tree1, best_matched_node_parents_tree2 = get_edge_similarity_length(ref_file, swc_file, best_matched_nodes_tree1, best_matched_nodes_tree2)
                    else:
                        best_matched_nodes_similarity, best_matched_node_parents_tree1, best_matched_node_parents_tree2 = get_edge_similarity_convex(ref_file, swc_file, best_matched_nodes_tree1, best_matched_nodes_tree2, best_orientation, args['partition_length'])

                    res_dict = {
                        "file1": ref_file,
                        "file2": swc_file,
                        "similarity_function": args['similarity_function'],
                        "max_depth": args['max_depth'],
                        "distance_score":best_distance,
                        "orientation": best_orientation,
                        "number_of_rotations": args['number_of_rotations'],
                        "partition_length":args['partition_length'],
                        "angle_threshold":args['angle_threshold'],
                        "segment_threshold":args['segment_threshold'],
                        "matched_nodes_tree1": best_matched_nodes_tree1,
                        "matched_nodes_tree2": best_matched_nodes_tree2,
                        "matched_node_parents_tree1": best_matched_node_parents_tree1,
                        "matched_node_parents_tree2": best_matched_node_parents_tree2,
                        "matched_node_edge_similarity": best_matched_nodes_similarity
                    }
                    results.append(res_dict)

                #save all tree pair best results to each of the reference swcs 
                pd.DataFrame.from_records(results).to_csv(csv_out_path, index=False)

        # compare each swc in the input swc dir to all the reference swcs
        elif input_file_1 is not None:
            if not all([os.path.exists(p) for p in [input_file_1]]):
                msg = "Input SWC Paths Do Not Exist. Please check the path provided: \n{} \n{}".format(input_file_1)
                raise ValueError(msg)
            
            tree1_raw = morphology_from_swc(input_file_1)
            
            results = []
            for ref_file in ref_files:
                ref_tree_raw = morphology_from_swc(ref_file)
                for i, orientation in enumerate(orientations): 

                    this_distance, this_matched_nodes_tree1, this_matched_nodes_tree2  =  compare_two_trees(ref_tree_raw.clone(),
                                                                                                            tree1_raw.clone(), #rotate the comparison tree not the reference trees noce for visualizing later I think
                                                                                                            simFunc=args['similarity_function'],
                                                                                                            maxDepth=args['max_depth'],
                                                                                                            valid_set_dir=args['valid_set_dir'],
                                                                                                            partition_length=args['partition_length'],
                                                                                                            angle_threshold=args['angle_threshold'],
                                                                                                            segment_threshold=args['segment_threshold'],
                                                                                                            orientation=orientation)
                    
                    if i == 0: 
                        best_distance = this_distance
                        best_orientation = orientation
                        best_matched_nodes_tree1 = this_matched_nodes_tree1
                        best_matched_nodes_tree2 = this_matched_nodes_tree2

                    elif this_distance < best_distance:
                        best_distance = this_distance
                        best_orientation = orientation 
                        best_matched_nodes_tree1 = this_matched_nodes_tree1
                        best_matched_nodes_tree2 = this_matched_nodes_tree2


            #save this tree pair best results 
            if args['similarity_function'] == 'length':
                best_matched_nodes_similarity, best_matched_node_parents_tree1, best_matched_node_parents_tree2 = get_edge_similarity_length(ref_file, input_file_1, best_matched_nodes_tree1, best_matched_nodes_tree2)
            else:
                best_matched_nodes_similarity, best_matched_node_parents_tree1, best_matched_node_parents_tree2 = get_edge_similarity_convex(ref_file, input_file_1, best_matched_nodes_tree1, best_matched_nodes_tree2, best_orientation, args['partition_length'])

            results = [{
                "file1": ref_file,
                "file2": input_file_1,
                "similarity_function": args['similarity_function'],
                "max_depth": args['max_depth'],
                "distance_score":best_distance,
                "orientation": best_orientation,
                "number_of_rotations": args['number_of_rotations'],
                "partition_length":args['partition_length'],
                "angle_threshold":args['angle_threshold'],
                "segment_threshold":args['segment_threshold'],
                "matched_nodes_tree1": best_matched_nodes_tree1,
                "matched_nodes_tree2": best_matched_nodes_tree2,
                "matched_node_parents_tree1": best_matched_node_parents_tree1,
                "matched_node_parents_tree2": best_matched_node_parents_tree2,
                "matched_node_edge_similarity": best_matched_nodes_similarity
            }]

            pd.DataFrame.from_records(results).to_csv(args['output_file'], index=False)

        else: 
            msg = "A directory of swcs or a single swc must be given to compare the the reference dir files."
            raise ValueError(msg)


    #compare cells within swc_dir to each other (no reference dir)
    elif input_swc_dir is not None:
        swc_files = [os.path.join(input_swc_dir,f) for f in os.listdir(input_swc_dir) if f.endswith(".swc")]

        if len(swc_files) < 2:
            msg = "At least two swc files required in the input directory. Only {} found".format(len(input_swc_dir))
            raise ValueError(msg)

        results = []
        pairs = list(itertools.combinations(swc_files, 2))
        for fn_pair in pairs:
            swc_fn_1 = fn_pair[0]
            swc_fn_2 = fn_pair[1]

            morph_1 = morphology_from_swc(swc_fn_1)
            morph_2 = morphology_from_swc(swc_fn_2)
            for i, orientation in enumerate(orientations): 

                this_distance, this_matched_nodes_tree1, this_matched_nodes_tree2 = compare_two_trees(morph_1.clone(),
                                                                                                      morph_2.clone(),
                                                                                                      simFunc=args['similarity_function'],
                                                                                                      maxDepth=args['max_depth'],
                                                                                                      valid_set_dir=args['valid_set_dir'],
                                                                                                      partition_length=args['partition_length'],
                                                                                                      angle_threshold=args['angle_threshold'],
                                                                                                      segment_threshold=args['segment_threshold'],
                                                                                                      orientation=orientation)
                if i == 0: 
                    best_distance = this_distance
                    best_orientation = orientation
                    best_matched_nodes_tree1 = this_matched_nodes_tree1
                    best_matched_nodes_tree2 = this_matched_nodes_tree2

                elif this_distance < best_distance:
                    best_distance = this_distance
                    best_orientation = orientation 
                    best_matched_nodes_tree1 = this_matched_nodes_tree1
                    best_matched_nodes_tree2 = this_matched_nodes_tree2

            #save this tree pair best results 
            if args['similarity_function'] == 'length':
                best_matched_nodes_similarity, best_matched_node_parents_tree1, best_matched_node_parents_tree2 = get_edge_similarity_length(swc_fn_1, swc_fn_2, best_matched_nodes_tree1, best_matched_nodes_tree2)
            else:
                best_matched_nodes_similarity, best_matched_node_parents_tree1, best_matched_node_parents_tree2 = get_edge_similarity_convex(swc_fn_1, swc_fn_2, best_matched_nodes_tree1, best_matched_nodes_tree2, best_orientation, args['partition_length'])

            res_dict = {
                "file1": swc_fn_1,
                "file2": swc_fn_2,
                "similarity_function": args['similarity_function'],
                "max_depth": args['max_depth'],
                "distance_score":best_distance,
                "orientation": best_orientation,
                "number_of_rotations": args['number_of_rotations'],
                "partition_length":args['partition_length'],
                "angle_threshold":args['angle_threshold'],
                "segment_threshold":args['segment_threshold'],
                "matched_nodes_tree1": best_matched_nodes_tree1,
                "matched_nodes_tree2": best_matched_nodes_tree2,
                "matched_node_parents_tree1": best_matched_node_parents_tree1,
                "matched_node_parents_tree2": best_matched_node_parents_tree2,
                "matched_node_edge_similarity": best_matched_nodes_similarity
            }
            results.append(res_dict)

        #save all tree pair best results 
        pd.DataFrame.from_records(results).to_csv(args['output_file'], index=False)


    #compare file1 to file2
    else:

        if not all([os.path.exists(p) for p in [input_file_1, input_file_2]]):
            msg = "Input SWC Paths Do Not Exist. Please check the paths provided: \n{} \n{}".format(input_file_1, input_file_2)
            raise ValueError(msg)
        
        morph_1 = morphology_from_swc(input_file_1)
        morph_2 = morphology_from_swc(input_file_2)

        for i, orientation in enumerate(orientations): 

            this_distance, this_matched_nodes_tree1, this_matched_nodes_tree2  =  compare_two_trees(morph_1.clone(),
                                                                                                    morph_2.clone(),
                                                                                                    simFunc=args['similarity_function'],
                                                                                                    maxDepth=args['max_depth'],
                                                                                                    valid_set_dir=args['valid_set_dir'],
                                                                                                    partition_length=args['partition_length'],
                                                                                                    angle_threshold=args['angle_threshold'],
                                                                                                    segment_threshold=args['segment_threshold'],
                                                                                                    orientation=orientation)
             
            if i == 0: 
                best_distance = this_distance
                best_orientation = orientation
                best_matched_nodes_tree1 = this_matched_nodes_tree1
                best_matched_nodes_tree2 = this_matched_nodes_tree2

            elif this_distance < best_distance:
                best_distance = this_distance
                best_orientation = orientation 
                best_matched_nodes_tree1 = this_matched_nodes_tree1
                best_matched_nodes_tree2 = this_matched_nodes_tree2


        #save this tree pair best results 
        if args['similarity_function'] == 'length':
            best_matched_nodes_similarity, best_matched_node_parents_tree1, best_matched_node_parents_tree2 = get_edge_similarity_length(input_file_1, input_file_2, best_matched_nodes_tree1, best_matched_nodes_tree2)
        else:
            best_matched_nodes_similarity, best_matched_node_parents_tree1, best_matched_node_parents_tree2 = get_edge_similarity_convex(input_file_1, input_file_2, best_matched_nodes_tree1, best_matched_nodes_tree2, best_orientation, args['partition_length'])

        results = [{
            "file1": input_file_1,
            "file2": input_file_2,
            "similarity_function": args['similarity_function'],
            "max_depth": args['max_depth'],
            "distance_score":best_distance,
            "orientation": best_orientation,
            "number_of_rotations": args['number_of_rotations'],
            "partition_length":args['partition_length'],
            "angle_threshold":args['angle_threshold'],
            "segment_threshold":args['segment_threshold'],
            "matched_nodes_tree1": best_matched_nodes_tree1,
            "matched_nodes_tree2": best_matched_nodes_tree2,
            "matched_node_parents_tree1": best_matched_node_parents_tree1,
            "matched_node_parents_tree2": best_matched_node_parents_tree2,
            "matched_node_edge_similarity": best_matched_nodes_similarity
        }]

        pd.DataFrame.from_records(results).to_csv(args['output_file'], index=False)

def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
