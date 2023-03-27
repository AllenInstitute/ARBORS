import pandas as pd
from neuron_morphology.swc_io import morphology_from_swc
from morph_utils.modifications import generate_irreducible_morph
from morph_utils.graph_traversal import dfs_tree
from morph_utils.measurements import tree_length
import os
from tqdm import tqdm
import numpy as np
import argschema as ags
import itertools
from tree_comparison.utils import compute_nDistance_matrix, find_leaves, preOrderTraversal, linearAssignment_matchingNodes

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
    output_file = ags.fields.OutputFile(description="Path to output csv", default="TreeCompareOutput.csv")
    similarity_functions = ags.fields.String(description = "Similarity function to use. Options: 'length or convex'",
                                             default='length')
    max_depth = ags.fields.Int(description="Max depth to use for algorithm",
                               default=1)


def compare_two_trees(swc_file_1, swc_file_2, simFunc = "length", maxDepth = 1):
    """
    Will generate a similarirty score for two input swc files.

    :param swc_file_1: str, path to swc file 1
    :param swc_file_2: str, path to swc file 2
    :param simFunc: str, either 'length' or 'convex'
    :param maxDepth: int, 1 or 2
    :return: float, cell similarity
    """

    tree1_raw = morphology_from_swc(swc_file_1)
    tree2_raw = morphology_from_swc(swc_file_2)

    tree1_length = round(tree_length(tree1_raw), 4)
    tree2_length = round(tree_length(tree2_raw), 4)

    tree1 = generate_irreducible_morph(tree1_raw)
    tree2 = generate_irreducible_morph(tree2_raw)

    root1, root2 = tree1.get_soma(), tree2.get_soma()

    if not root1:
        root1 = [n for n in tree1.nodes() if n['parent'] == -1][0]
    if not root2:
        root2 = [n for n in tree2.nodes() if n['parent'] == -1][0]

    postOrderNodes2,_ = dfs_tree(tree2, root2)
    postOrderNodes2.reverse()
    postOrderNodes1,_ = dfs_tree(tree1, root1)
    postOrderNodes1.reverse()

    preOrderNodes1 = preOrderTraversal(tree1, root1)
    preOrderNodes2 = preOrderTraversal(tree2, root2)

    ndDistanceMatrix1, node_id_index_dict1 = compute_nDistance_matrix(tree1_raw)
    ndDistanceMatrix2, node_id_index_dict2 = compute_nDistance_matrix(tree2_raw)

    agreement = {}
    agreement['agrM'] = np.zeros((len(tree1), len(tree2)))
    agreement['pAgrM'] = np.zeros((len(tree1),
                                   len(tree2)))  # question uygar has  agreement.pAgrM = agreement.agrM; in matlab does that link them??
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
            #         print("----------")
            #         print(node_1_matrix_index,node_2_matrix_index)
            #         print()

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
                #             print("~~")
                #             print(subtreeRootPos1,subtreeRootPos2)
                #             print("~~")
                # plus one to get all members below (matlab v python)
                stop_index_1 = subtreeRootPos1 + preOrderNodes1[subtreeRootPos1, 1] + 1
                stop_index_2 = subtreeRootPos2 + preOrderNodes2[subtreeRootPos2, 1] + 1

                preON1 = preOrderNodes1[subtreeRootPos1: stop_index_1, :]
                preON2 = preOrderNodes2[subtreeRootPos2: stop_index_2, :]

                # print("Calling lap at {},{}".format(node_1_matrix_index,node_2_matrix_index))
                #             print(agreement['agrTypeM'])
                linearAssignment_matchingNodes(agreement,
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
                                               maxDepth=maxDepth,
                                               simFunc='length')


            else:
                agreement['agrM'][node_1_matrix_index, node_2_matrix_index] = 0
                if (node1 != root1) and (node2 != root2):

                    leaves1 = find_leaves(tree1, node1)
                    leaves1_matrix_indices = [node_id_index_dict1[leaf['id']] for leaf in leaves1]

                    leaves2 = find_leaves(tree2, node2)
                    leaves2_matrix_indices = [node_id_index_dict2[leaf['id']] for leaf in leaves2]

                    if simFunc == "length":

                        #                     agreement['agrM'][node_1_matrix_index,node_2_matrix_index] = 0
                        sub_arr_1 = ndDistanceMatrix1[leaves1_matrix_indices, node1_parent_id_matrix_idx]
                        sub_arr_2 = ndDistanceMatrix2[leaves2_matrix_indices, node2_parent_id_matrix_idx]

                        maxHeight1, pos1 = np.max(sub_arr_1), np.argmax(sub_arr_1)
                        maxHeight2, pos2 = np.max(sub_arr_2), np.argmax(sub_arr_2)

                        lowerNode1 = leaves1[pos1]
                        lowerNode2 = leaves2[pos2]

                        min_dist = min([maxHeight1, maxHeight2])
                        #                     print("MinDist:")
                        #                     print(min_dist)
                        # print("Updating pAgrM via min_dist at {},{} to: {}".format(node_1_matrix_index,node_2_matrix_index,min_dist))
                        agreement['pAgrM'][node_1_matrix_index, node_2_matrix_index] = min_dist


                    else:
                        print("TO DO: OTHER SIM FUNC")
                        return None

                    #                 agreement.pAgrNodes{node1}{node2}{1} = lowerNode1; agreement.pAgrNodes{node1}{node2}{2} = lowerNode2;
                    #                 agreement.agrNodes{node1}{node2}{1}=lowerNode1; agreement.agrNodes{node1}{node2}{2}=lowerNode2;
                    del lowerNode1;
                    del lowerNode2;
                    agreement['agrTypeM'][node_1_matrix_index, node_2_matrix_index] = (
                            (node1 in leaves1) and (node2 in leaves2))
                else:
                    # print("Updating pAgrM via line 99 at {},{} to: 0".format(node_1_matrix_index,node_2_matrix_index))
                    agreement['pAgrM'][node_1_matrix_index, node_2_matrix_index] = 0
                    #                 agreement.agrNodes{node1}{node2}{1}=node1; agreement.agrNodes{node1}{node2}{2}=node2;
                    #                 agreement.pAgrNodes{node1}{node2}{1}=node1; agreement.pAgrNodes{node1}{node2}{2}=node2;
                    agreement['agrTypeM'][node_1_matrix_index, node_2_matrix_index] = False

            if agreement['agrM'][node_1_matrix_index, node_2_matrix_index] > maxAgreement:
                maxAgreement = agreement['agrM'][node_1_matrix_index, node_2_matrix_index]
                maxMatchingNode1 = node1;
                maxMatchingNode2 = node2;

    maxAgreement = round(maxAgreement, 4)
    distance = tree1_length + tree2_length - maxAgreement
    distance = round(distance, 4)

    return distance


def main(args):

    input_file_1 = args['input_swc_file_1']
    input_file_2 = args['input_swc_file_2']
    input_swc_dir = args['input_swc_dir']

    if all([input_obj is None for input_obj in [input_file_1, input_file_2, input_swc_dir]]):
        msg = "No input swc files provided for comparison, no input directory provided. Nothing to do"
        raise ValueError(msg)

    if all([input_obj is not None for input_obj in [input_file_1, input_file_2, input_swc_dir]]):
        msg = "Input swc files provided and input directory provided. Either provide input swc paths OR provide directory with swc files, not both."
        raise ValueError(msg)

    if input_swc_dir is not None:
        swc_files = [os.path.join(input_swc_dir,f) for f in os.listdir(input_swc_dir) if f.endswith(".swc")]

        if len(swc_files) < 2:
            msg = "At least two swc files required in the input directory. Only {} found".format(len(input_swc_dir))
            raise ValueError(msg)

        results = []
        pairs = list(itertools.combinations(swc_files, 2))
        for fn_pair in pairs:
            swc_fn_1 = fn_pair[0]
            swc_fn_2 = fn_pair[1]

            distance = compare_two_trees(swc_fn_1,
                                         swc_fn_2,
                                         simFunc=args['similarity_functions'],
                                         maxDepth=args['max_depth'])
            res_dict = {
                "file1": swc_fn_1,
                "file2": swc_fn_2,
                "similarity": distance
            }
            results.append(res_dict)

        pd.DataFrame.from_records(results).to_csv(args['output_file'])

    else:

        if not all([os.path.exists(p) for p in [input_file_1, input_file_2]]):
            msg = "Input SWC Paths Do Not Exist. Please check the paths provided: \n{} \n{}".format(input_file_1, input_file_2)
            raise ValueError(msg)

        distance = compare_two_trees(input_file_1,
                                      input_file_2,
                                      simFunc=args['similarity_functions'],
                                      maxDepth=args['max_depth'])

        results = [{"file1":input_file_1, "file2":input_file_2, "similarity":distance }]

        pd.DataFrame.from_records(results).to_csv(args['output_file'])

def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
