"""
Functions for converting Protein Structure Graphs to standard Data object.
"""
import networkx as nx
import numpy as np
import torch
import torch_geometric


graph_features = ['phi', 'psi', 'rsa', 'asa', 'ss', 'expasy']
map_secondary_structure = {'-': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
map_edge_types = {
    'peptide_bond': 0,
    'sequence_distance_2': 1,
    'sequence_distance_3': 2,
    'distance_threshold': 3,
    'delaunay': 4,
    'hbond': 5,
    'k_nn': 6
}


def convert_nx_to_pyg(nx_graph: nx.Graph) -> torch_geometric.data.Data:
    """
    Converting Graphein Networks from `networkx.Graph` (nx) format to `torch_geometric.data.Data` (pytorch geometric,
    PyG) format.
    """
    # Initialise dict used to construct Data object
    data_dict = {"node_id": list(nx_graph.nodes())}
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)

    # Construct Edge Index
    edge_index = torch.LongTensor(list(nx_graph.edges)).t().contiguous()

    # Add node features
    for i, (_, feat_dict) in enumerate(nx_graph.nodes(data=True)):
        for key, value in feat_dict.items():
            data_dict[str(key)] = [value] if i == 0 else data_dict[str(key)] + [value]

    # Add edge features
    for i, (_, _, feat_dict) in enumerate(nx_graph.edges(data=True)):
        for key, value in feat_dict.items():
            if key == 'distance':
                data_dict[str(key)] = [value] if i == 0 else data_dict[str(key)] + [value]
            else:
                data_dict[str(key)] = [list(value)] if i == 0 else data_dict[str(key)] + [list(value)]

    # Add graph-level features
    for feat_name in nx_graph.graph:
        data_dict[str(feat_name)] = [nx_graph.graph[feat_name]]

    data_dict["edge_index"] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data_dict)
    data.num_nodes = nx_graph.number_of_nodes()

    # remove useless intermediate data and add features for deep learning models
    reformat_data = torch_geometric.data.Data(
        edge_index=data.edge_index,
        num_nodes=len(data.node_id),
        node_id=data.node_id,
        name=data.name[0],
        sequence=getattr(data, f"sequence_{data.chain_id[0]}"),
        distance_matrix=data.dist_mat,
        distance=data.distance,
        coordinates=torch.FloatTensor(np.array(data.coords[0]))
    )

    x = np.array([np.argmax(data.amino_acid_one_hot, axis=1)]).reshape(-1, 1)
    for feat in graph_features:
        if feat == "ss":
            feature = np.array([[map_secondary_structure.get(feat_node, 0)] for feat_node in data[feat]])
        else:
            feature = np.array(data[feat])
            if len(feature.shape) == 1:
                feature = feature.reshape(-1, 1)
        x = np.concatenate((x, feature), axis=1)
    reformat_data.x = torch.FloatTensor(x)
    reformat_data.edge_type = torch.LongTensor([map_edge_types[kind[0]] for kind in data.kind])

    return reformat_data
