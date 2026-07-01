from collections import deque

import numpy as np
from morph_utils.graph_traversal import (
    bfs_tree,
    dfs_tree,
    get_path_and_path_dist_between_two_nodes,
    get_path_to_root,
)
from neuron_morphology.transforms.affine_transform import (
    AffineTransform,
    affine_from_transform,
    rotation_from_angle,
)
from scipy.optimize import linear_sum_assignment

from src.tree_comparison.convexsimfunc_utils import edges_between
from src.tree_comparison.maxdepthtwo_utils import (
    getMatchingChildren,
    getMatchingChildren_hardcode,
    getValidSetCardinality,
    getValidSetCardinality_hardcode,
)
from tree_comparison.cpp.quantized_convex_matching import quantized_convex_matching


def rotate_morphology(morphology, angle, axis=1):
    """Rotate a morphology object around a specified axis.

    Parameters:
        morph (Morphology): Morphology object to rotate.
        angle (float): Rotation angle in radians.
                       Rotation follows the right-hand rule: clockwise when viewed
                       from the positive y-axis towards the positive x-axis.
        axis (int): Axis to rotate around (0 = x, 1 = y, 2 = z).

    Returns:
        Morphology: Rotated Morphology object.

    """
    rotation_affine = AffineTransform(affine_from_transform(rotation_from_angle(angle, axis)))
    rotated_morphology = rotation_affine.transform_morphology(
        morphology
    )  # if you need the original object to remain unchanged do morph.clone()

    return rotated_morphology


def ensure_flat_list(item):
    """Ensure the input is a flat list.

    Parameters:
        item: Input to process, can be a list, integer, or other type.

    Returns:
        list or original input:
        - A single nested list is flattened.
        - A single integer is wrapped in a list.
        - Other types are returned as-is.

    """
    if isinstance(item, list):
        if len(item) == 1 and isinstance(item[0], list):
            return item[0]  # Extract the inner list
        else:
            return item  # Already in the desired form
    elif isinstance(item, int):
        return [item]  # a single int, return as list
    else:
        return item  # Not a list or int, return as is


def compute_nDistance_matrix(raw_morphology):
    """Compute the path distance matrix between all irreducible nodes of a subtree.

    Parameters:
        raw_morphology (Morphology): A morphology object containing the structure of the neuron.

    Returns:
        tuple:
            - ndist_matrix (numpy.ndarray): A square matrix where each entry (i, j) represents
            the path distance between two irreducible nodes. The distance is doubled for symmetry.
            - node_id_index_dict (dict): A mapping from node IDs to their corresponding indices in the distance matrix.

    """
    irreducible_nodes = [
        n
        for n in raw_morphology.nodes()
        if (len(raw_morphology.get_children(n)) > 1)
        or (len(raw_morphology.get_children(n)) == 0)
        or (n["parent"] == -1)
    ]
    soma = raw_morphology.get_soma()
    if not soma:
        soma_list = [n for n in raw_morphology.nodes() if n["parent"] == -1]
        if len(soma_list) != 1:
            print("Invalid Number of somas (0 or >1)")
        else:
            soma = soma_list[0]
    if soma not in irreducible_nodes and soma:
        irreducible_nodes.append(raw_morphology.get_soma())

    num_irr_nodes = len(irreducible_nodes)
    ndist_matrix = np.zeros((num_irr_nodes, num_irr_nodes))
    node_id_index_dict = {no["id"]: ct for ct, no in enumerate(irreducible_nodes)}
    leaf_nodes = raw_morphology.get_leaf_nodes()

    for leaf_no in leaf_nodes:
        seg_up = get_path_to_root(leaf_no, raw_morphology)
        irreducible_seg_up = [n for n in seg_up if n in irreducible_nodes]

        assert leaf_no in seg_up

        for seg_walker, seg_no_i in enumerate(irreducible_seg_up):
            no_i_id = seg_no_i["id"]
            no_i_idx = node_id_index_dict[no_i_id]

            all_other_nodes_in_seg = irreducible_seg_up[seg_walker + 1 :]
            for seg_no_j in all_other_nodes_in_seg:
                no_j_id = seg_no_j["id"]
                no_j_idx = node_id_index_dict[no_j_id]

                _, dist = get_path_and_path_dist_between_two_nodes(
                    upper_node=seg_no_j, lower_node=seg_no_i, morphology=raw_morphology
                )

                ndist_matrix[no_i_idx, no_j_idx] = 2 * dist
                ndist_matrix[no_j_idx, no_i_idx] = 2 * dist

    return ndist_matrix, node_id_index_dict


def preOrderTraversal(morphology, st_node):
    """Perform a pre-order traversal of the morphology starting from the specified node.

    Parameters:
        morphology (Morphology): A Morphology object.
        st_node (dict): The starting node for the traversal.

    Returns:
        numpy.ndarray: A 2D array where each row contains:
            - Node ID (int): The ID of the visited node.
            - Number of downstream nodes (int): The count of nodes downstream from the visited node, excluding itself.

    """
    visited = []
    num_downstream_nodes = []
    queue = deque([st_node])
    while len(queue) > 0:
        current_node = queue.popleft()

        _, num_nodes_down_tree = bfs_tree(current_node, morphology)
        num_nodes_down_tree = (
            num_nodes_down_tree - 1
        )  # since num_nodes_down_tree includes current_node in count
        num_downstream_nodes.append(num_nodes_down_tree)
        visited.append(current_node)

        for ch_no in morphology.get_children(current_node):
            queue.appendleft(ch_no)

    visited_ids = [n["id"] for n in visited]
    res = np.array([visited_ids, num_downstream_nodes]).T
    return res


def find_leaves(morphology, st_node):
    """Return leaf nodes of a Morphology object.

    Parameters:
        morphology (Morphology): A Morphology object.
        st_node (dict): The starting node for the traversal.

    Returns:
        Returns:
        list: A list of leaf nodes.

    """
    nodes_down, _ = dfs_tree(morphology, st_node)
    leaves = [n for n in nodes_down if morphology.get_children(n) == []]
    return leaves


def linearAssignment_matchingNodes(
    agreement,
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
    valid_set_dict,
    valid_set_dir,
    hardcoded_vs,
    relative_branch_check,
    relative_branch_threshold,
):
    """Matches nodes from two trees based on a given agreement matrix and other tree-related data. The function performs
    linear assignment to match nodes between two trees, calculates similarity based on different criteria, and updates
    the agreement matrix accordingly.

    Parameters:
        agreement (dict): Dictionary containing agreement matrices and other related data.
        node1 (dict): The node to be compared from tree1 - match it's children with the children of node2.
        node1_children (list): List of node1 children - nodes to be matched with node2_children.
        node1_parent_id_matrix_idx (int): Index of node1's parent in the distance matrix.
        ndDistanceMatrix1 (numpy.ndarray): Tree1 distance matrix.
        preON1 (numpy.ndarray): Preorder traversal of tree1 subtree (node1 & node1_children).
        tree1 (Morphology): Tree1 Morphology object.
        node_id_index_dict1 (dict): Mapping of Morphology node IDs to distance matrix indices for tree1.
        node2 (dict): The node to be compared from tree2 - match it's children with the children of node1.
        node2_children (list): List of node2 children - nodes to be matched with node1_children.
        node2_parent_id_matrix_idx (int): Index of node2's parent in the distance matrix.
        ndDistanceMatrix2 (numpy.ndarray): Tree2 distance matrix.
        preON2 (numpy.ndarray): Preorder traversal of tree2 subtree (node2 & node2_children).
        tree2 (Morphology): Tree2 Morphology object.
        node_id_index_dict2 (dict): Mapping of Morphology node IDs to distance matrix indices for tree2.
        tree1_paths (str): Path data from node1 to node1's irreducible parent.
        tree2_paths (str): Path data from node2 to node2's irreducible parent.
        maxDepth (int): Maximum depth for matching (1 or 2). Affects the matching strategy.
        simFunc (str): The similarity function to use ('length' or 'convex').
        valid_set_dict (dict): Predefined valid sets for matching (if hardcoded_vs == True).
        valid_set_dir (str): Directory containing valid sets for matching (if hardcoded_vs == False).
        hardcoded_vs (bool): Flag indicating whether to use hardcoded valid sets.

    Returns:
        dict: Updated agreement dictionary with the modified agreement matrices and node matches.

    Notes:
    - The function uses linear sum assignment to find the best match between child nodes based on agreement matrices.
    - The final similarity is calculated using different methods depending on the `maxDepth` and `simFunc` parameters.
    - The agreement matrix is updated based on the node matches and similarity scores.

    """
    node1_children_array = np.array(node1_children)
    node2_children_array = np.array(node2_children)

    node1_children_matrix_idx = [node_id_index_dict1[n["id"]] for n in node1_children]
    node2_children_matrix_idx = [node_id_index_dict2[n["id"]] for n in node2_children]

    node1_matrix_idx = node_id_index_dict1[node1["id"]]
    node2_matrix_idx = node_id_index_dict2[node2["id"]]

    sub_mat_1 = agreement["agrM"][node1_matrix_idx, node2_children_matrix_idx]
    sub_mat_2 = agreement["agrM"][node1_children_matrix_idx, node2_matrix_idx]

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
        # Find the best matching between two sets of child nodes based on some measure of agreement (stored in agreement['pAgrM']).
        # Use a linear sum assignment algorithm to maximize the agreement between matched pairs of nodes and then returns the
        # corresponding matched nodes.

        # extract the agreement matrix for these subtrees.
        lap_submat = agreement["pAgrM"][
            np.ix_(node1_children_matrix_idx, node2_children_matrix_idx)
        ]

        # check that extracted submatrix is not 2d (i.e., one of the nodes only has one child.)
        # if not, reshape lap_submat to be 2D, with one row and as many columns as there are node1 children.
        mat_shape = lap_submat.shape
        if len(mat_shape) != 2:
            lap_submat = lap_submat.reshape(1, mat_shape[0])

        # find the node match that maximizes the agreement btw the subtrees
        x, rowsol = linear_sum_assignment(-lap_submat)
        sim = lap_submat[
            x, rowsol
        ].sum()  # similarity of the best match - sum of agreement between matched nodes

        # get node matches of the optimal match
        if len(node2_children) < len(node1_children):
            matchingChildren1 = node1_children_array[rowsol]
            matchingChildren2 = node2_children
        else:
            matchingChildren1 = node1_children
            matchingChildren2 = node2_children_array[rowsol]

        matchingChildren1 = [node_id_index_dict1[c["id"]] for c in matchingChildren1]
        matchingChildren2 = [node_id_index_dict2[c["id"]] for c in matchingChildren2]

    elif maxDepth == 2:
        if not hardcoded_vs:
            minMaximalSetCardinality1, maxMaximalSetCardinality1, vs1 = getValidSetCardinality(
                valid_set_dir, tree1, node1, node1_children
            )
            minMaximalSetCardinality2, maxMaximalSetCardinality2, vs2 = getValidSetCardinality(
                valid_set_dir, tree2, node2, node2_children
            )
            matchingChildren1, matchingChildren2, sim = getMatchingChildren(
                maxMaximalSetCardinality1,
                minMaximalSetCardinality1,
                vs1,
                maxMaximalSetCardinality2,
                minMaximalSetCardinality2,
                vs2,
                agreement,
                node_id_index_dict1,
                node_id_index_dict2,
            )
        else:
            minMaximalSetCardinality1, maxMaximalSetCardinality1, vs1 = (
                getValidSetCardinality_hardcode(valid_set_dict, tree1, node1, node1_children)
            )
            minMaximalSetCardinality2, maxMaximalSetCardinality2, vs2 = (
                getValidSetCardinality_hardcode(valid_set_dict, tree2, node2, node2_children)
            )

            # Check if either didn't return a validset (branching pattern not found) and if so skip this tree comparison, set sim to low
            if any(
                value is None
                for value in (minMaximalSetCardinality1, maxMaximalSetCardinality1, vs1)
            ):
                print("skipping tree1 comparison, branching pattern not found.")
                return -1
            if any(
                value is None
                for value in (minMaximalSetCardinality2, maxMaximalSetCardinality2, vs2)
            ):
                print("skipping tree2 comparison, branching pattern not found.")
                return -2

            matchingChildren1, matchingChildren2, sim = getMatchingChildren_hardcode(
                maxMaximalSetCardinality1,
                minMaximalSetCardinality1,
                vs1,
                maxMaximalSetCardinality2,
                minMaximalSetCardinality2,
                vs2,
                agreement,
                node_id_index_dict1,
                node_id_index_dict2,
            )

    if sim > maxSimilarity:
        agreement["agrM"][node1_matrix_idx, node2_matrix_idx] = sim
        agreement["agrTypeM"][node1_matrix_idx, node2_matrix_idx] = True

        for i in range(len(matchingChildren1)):
            agreement["agrNodes"][node1_matrix_idx, node2_matrix_idx][0] = ensure_flat_list(
                agreement["agrNodes"][node1_matrix_idx, node2_matrix_idx][0]
            ) + ensure_flat_list(
                [agreement["pAgrNodes"][matchingChildren1[i], matchingChildren2[i]][0]]
            )
            agreement["agrNodes"][node1_matrix_idx, node2_matrix_idx][1] = ensure_flat_list(
                agreement["agrNodes"][node1_matrix_idx, node2_matrix_idx][1]
            ) + ensure_flat_list(
                [agreement["pAgrNodes"][matchingChildren1[i], matchingChildren2[i]][1]]
            )

        agreement["agrNodes"][node1_matrix_idx, node2_matrix_idx][0] = agreement["agrNodes"][
            node1_matrix_idx, node2_matrix_idx
        ][0] + [node1["id"]]
        agreement["agrNodes"][node1_matrix_idx, node2_matrix_idx][1] = agreement["agrNodes"][
            node1_matrix_idx, node2_matrix_idx
        ][1] + [node2["id"]]

    else:
        agreement["agrM"][node1_matrix_idx, node2_matrix_idx] = maxSimilarity
        agreement["agrTypeM"][node1_matrix_idx, node2_matrix_idx] = False

        maxMatch1_matrix_idx = node_id_index_dict1[maxMatch1["id"]]
        maxMatch2_matrix_idx = node_id_index_dict2[maxMatch2["id"]]

        agreement["agrNodes"][node1_matrix_idx, node2_matrix_idx][0] = agreement["agrNodes"][
            maxMatch1_matrix_idx, maxMatch2_matrix_idx
        ][0]
        agreement["agrNodes"][node1_matrix_idx, node2_matrix_idx][1] = agreement["agrNodes"][
            maxMatch1_matrix_idx, maxMatch2_matrix_idx
        ][1]

    # PLUS SIMILARITY CALCULATIONS
    if (node1["parent"] != -1) & (node2["parent"] != -1):
        maxPlusSimilarity = -1e-10
        for n1 in preON1[:, 0]:
            n1_parent_id = tree1.node_by_id(n1)["parent"]
            n1_parent_id_idx = node_id_index_dict1[n1_parent_id]

            for kk2 in range(len(preON2[:, 0])):
                n2 = preON2[kk2, 0]
                n2_parent_id = tree2.node_by_id(n2)["parent"]
                n2_parent_id_idx = node_id_index_dict2[n2_parent_id]

                n1_idx = node_id_index_dict1[n1]
                n2_idx = node_id_index_dict2[n2]

                if agreement["agrTypeM"][n1_idx, n2_idx]:
                    min_val = min(
                        ndDistanceMatrix1[n1_idx, node1_parent_id_matrix_idx],
                        ndDistanceMatrix2[n2_idx, node2_parent_id_matrix_idx],
                    )
                    max_val = max(
                        ndDistanceMatrix1[n1_parent_id_idx, node1_parent_id_matrix_idx],
                        ndDistanceMatrix2[n2_parent_id_idx, node2_parent_id_matrix_idx],
                    )
                    simpleBound = agreement["agrM"][n1_idx, n2_idx] + min_val
                    simpleBound2 = agreement["pAgrM"][n1_idx, n2_idx] + max_val

                    if (n1 == node1["id"] & n2 == node2["id"]) or (
                        maxPlusSimilarity < (min(simpleBound, simpleBound2) * 0.9999)
                    ):
                        # If branch lengths between n1-Node1Parent and between n2-Node2Parent are *too* different, don't compare them
                        branch_max_val = max(
                            ndDistanceMatrix1[n1_idx, node1_parent_id_matrix_idx],
                            ndDistanceMatrix2[n2_idx, node2_parent_id_matrix_idx],
                        )
                        if relative_branch_check and (
                            min_val / branch_max_val < relative_branch_threshold
                        ):
                            thisPlusSimilarity = 0

                        elif simFunc != "length":  # convex
                            edge_lengths1, edge_orientations1, edge_areas1, pillars1 = (
                                edges_between(
                                    tree1.node_by_id(n1),
                                    tree1.node_by_id(node1["parent"]),
                                    tree1,
                                    tree1_paths,
                                )
                            )
                            edge_lengths2, edge_orientations2, edge_areas2, pillars2 = (
                                edges_between(
                                    tree2.node_by_id(n2),
                                    tree2.node_by_id(node2["parent"]),
                                    tree2,
                                    tree2_paths,
                                )
                            )

                            cvxSim = quantized_convex_matching(
                                edge_lengths1,
                                edge_orientations1,
                                edge_areas1,
                                pillars1,
                                edge_lengths2,
                                edge_orientations2,
                                edge_areas2,
                                pillars2,
                            )
                            thisPlusSimilarity = agreement["agrM"][n1_idx, n2_idx] + cvxSim

                        else:  # length
                            thisPlusSimilarity = simpleBound
                        if thisPlusSimilarity > maxPlusSimilarity:
                            maxPlusSimilarity = thisPlusSimilarity
                            maxndd1_idx = node_id_index_dict1[n1]
                            maxndd2_idx = node_id_index_dict2[n2]
                    else:
                        kk2 = kk2 + preON2[kk2, 1]
        agreement["pAgrM"][node1_matrix_idx, node2_matrix_idx] = maxPlusSimilarity
        agreement["pAgrNodes"][node1_matrix_idx, node2_matrix_idx][0] = agreement["agrNodes"][
            maxndd1_idx, maxndd2_idx
        ][0]
        agreement["pAgrNodes"][node1_matrix_idx, node2_matrix_idx][1] = agreement["agrNodes"][
            maxndd1_idx, maxndd2_idx
        ][1]
    return agreement


def remove_duplicate_nodes(nodes):
    """Remove duplicate node ids from a tree.

    Parameters:
        nodes (list): A list of node dictionaries, where each dictionary contains an 'id' key.

    Returns:
        list: A list of nodes with duplicates removed, preserving the node order of the input list.

    """
    seen_ids = set()
    unique_nodes = []

    for node in nodes:
        if node["id"] not in seen_ids:
            unique_nodes.append(node)
            seen_ids.add(node["id"])

    return unique_nodes


def flatten(nested_list):
    """Recursively flatten a nested list.

    Paremeters:
        nested_list (list): A list (potentially nested).

    Returns:
        list: A flat list.

    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list
