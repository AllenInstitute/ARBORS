import numpy as np
import pandas as pd
from morph_utils.graph_traversal import get_path_to_root

def _get_segment_path(critical_nodes, new_areas, partition_length, downsampling_factor):
    orientations = np.array([[0], [0], [0]])
    lengths = np.array([])
    areas_result = np.array([])
    pillars = np.array([0])
    current_pillar = 0

    # for kk in range(0, critical_nodes.shape[0] - 1, downsampling_factor):
    #     print('kk: {}'.format(kk))
        # vec = critical_nodes[min(kk + downsampling_factor, critical_nodes.shape[0] - 1), :] - critical_nodes[kk, :]

    for kk in range(critical_nodes.shape[0] - 1):
        vec = critical_nodes[kk + 1, :] - critical_nodes[kk, :]
        vec_norm = np.linalg.norm(vec)
        unit_vec = vec / vec_norm if vec_norm != 0 else np.array([np.nan, np.nan, np.nan])
        this_edge_count = int(np.ceil(vec_norm / partition_length))

        ##### No further subdividing between critical nodes #####
        this_edge_count = 1 

        lengths = np.append(lengths, vec_norm)
        orientations = np.hstack([orientations, vec_norm * np.kron(np.arange(1, this_edge_count), unit_vec.reshape(3,1)), vec.reshape(3,1)])
        areas_result = np.append(areas_result, new_areas[kk] * np.ones(this_edge_count))
        pillars = np.append(pillars, current_pillar * np.ones(this_edge_count))
        ###################################################################################

        # ###### The Original Method with partition length sections between nodes ############

        # if this_edge_count > 0:
        #     lengths = np.append(lengths, np.ones(this_edge_count - 1) * partition_length)
        #     lengths = np.append(lengths, np.remainder(vec_norm, partition_length))
        #     orientations = np.hstack([orientations, partition_length * np.kron(np.arange(1, this_edge_count), unit_vec.reshape(3,1)), vec.reshape(3,1)])
        #     areas_result = np.append(areas_result, new_areas[kk] * np.ones(this_edge_count))
        #     pillars = np.append(pillars, current_pillar * np.ones(this_edge_count))

        #     current_pillar += this_edge_count

        # ####################################################################################

    if len(areas_result) == 0: 
        critical_nodes = np.vstack([critical_nodes, critical_nodes])
        lengths = np.array([0])
        orientations = np.array([[0, 0, 0], [0, 0, 0]])
        areas_result = np.array([1])
        pillars = np.array([0, 0])

    return lengths, orientations, areas_result, pillars

def get_tree_paths(raw_morphology, partition_length, downsampling_factor=1):

    irreducible_nodes = [n for n in raw_morphology.nodes() if
                            (len(raw_morphology.get_children(n)) > 1) or (len(raw_morphology.get_children(n)) == 0) or (
                                        n['parent'] == -1)]
    soma = raw_morphology.get_soma()
    if not soma:
        soma_list = [n for n in raw_morphology.nodes() if n['parent'] == -1]
        if len(soma_list) != 1:
            print("Invalid Number of somas (0 or >1)")
        else:
            soma = soma_list[0]
    if soma not in irreducible_nodes and soma:
        irreducible_nodes.append(raw_morphology.get_soma())


    tree_paths = {}
    for node in raw_morphology.nodes():
        if node in irreducible_nodes:
            #get raw path to irreducible parent
            node_to_root = get_path_to_root(node, raw_morphology)
            raw_path_to_irreducible_parent = [node]
            for ancestor in node_to_root[1:]:
                raw_path_to_irreducible_parent.append(ancestor)
                if ancestor in irreducible_nodes: break 

            # #TODO downsample 
            # print('len(raw_path_to_irreducible_parent): {}'.format(len(raw_path_to_irreducible_parent)))
            # print('raw_path_to_irreducible_parent: {}'.format(raw_path_to_irreducible_parent))

            # downsample_thresh = 10
            # downsampling_factor = 2
            # if len(raw_path_to_irreducible_parent) > downsample_thresh:
            #     raw_path_to_irreducible_parent = raw_path_to_irreducible_parent[::downsampling_factor]
            #     print('DOWNSAMPLE')
            #     print('len(raw_path_to_irreducible_parent): {}'.format(len(raw_path_to_irreducible_parent)))
            #     print('raw_path_to_irreducible_parent: {}'.format(raw_path_to_irreducible_parent))

            #get length and orientation info for this path to irreduicble parent 
            num_edges = len(raw_path_to_irreducible_parent)-1
            path_area = np.ones(num_edges) #TODO add option to model edges as cylindars using node radii 

            critical_nodes = pd.DataFrame(raw_path_to_irreducible_parent)[['x', 'y', 'z']].to_numpy()
            # print('len(critical_nodes): {}'.format(len(critical_nodes)))
            
            lengths, orientations, areas, pillars = _get_segment_path(critical_nodes, path_area, partition_length, downsampling_factor)


            tree_path_node = {}
            tree_path_node['critical_nodes'] = critical_nodes
            tree_path_node['lengths'] = lengths
            tree_path_node['total_length'] = np.sum(lengths)
            tree_path_node['orientations'] = orientations
            tree_path_node['areas'] = areas
            tree_path_node['pillars'] = pillars
            tree_paths[node['id']] = tree_path_node


    return tree_paths

def edges_between(descendent, ancestor, tree, tree_paths):
    """ 
    Direct translation of edges_between.m 

    Accumulates the edges between a descendent and ancestor in a tree. 

        # tree{node}{4}{1} = critical_nodes 
        # tree{node}{4}{2} = lengths
        # tree{node}{4}{3} = orientations
        # tree{node}{4}{4} = areas
        # tree{node}{4}{5} = pillars

    """
    edge_lengths = np.array([])
    edge_orientations_x = np.array([])
    edge_orientations_y = np.array([])
    edge_orientations_z = np.array([])
    edge_areas = np.array([])
    pillars = np.array([])
    while descendent != ancestor:
        if len(edge_lengths) > 0:
            pillars = np.hstack((pillars, tree_paths[descendent['id']]['pillars'][1:].astype(int) + len(edge_lengths)))
            edge_orientations_x = np.append(edge_orientations_x, tree_paths[descendent['id']]['orientations'][0,1:])
            edge_orientations_y = np.append(edge_orientations_y, tree_paths[descendent['id']]['orientations'][1,1:])
            edge_orientations_z = np.append(edge_orientations_z, tree_paths[descendent['id']]['orientations'][2,1:])
        else:
            pillars = tree_paths[descendent['id']]['pillars'].astype(int)
            edge_orientations_x = np.append(edge_orientations_x, tree_paths[descendent['id']]['orientations'][0])
            edge_orientations_y = np.append(edge_orientations_y, tree_paths[descendent['id']]['orientations'][1])
            edge_orientations_z = np.append(edge_orientations_z, tree_paths[descendent['id']]['orientations'][2])

        edge_lengths = np.hstack((edge_lengths, tree_paths[descendent['id']]['lengths']))
        edge_areas = np.hstack((edge_areas, tree_paths[descendent['id']]['areas']))
        descendent = tree.node_by_id(descendent['parent'])

    edge_orientations = np.stack((edge_orientations_x, edge_orientations_y, edge_orientations_z))
    edge_orientations = [list(l) for l in edge_orientations]
    edge_orientations = [coordinate for point in zip(*edge_orientations) for coordinate in point] #flatten orientations as: [x1,y1,z1,x2,y2,z2,x3,y3,z3,...,xn,yn,zn] -- this is important for quantized convex matching, but we could do this ouside edges_between fn. 

    return list(edge_lengths), edge_orientations, list(edge_areas), list(pillars)

