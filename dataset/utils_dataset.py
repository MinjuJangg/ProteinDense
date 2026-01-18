from functools import partial

from graphein.protein.config import DSSPConfig, ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_peptide_bonds,
    add_hydrogen_bond_interactions,
    add_distance_threshold
)
from graphein.protein.features.nodes.amino_acid import (
    amino_acid_one_hot,
    meiler_embedding,
    expasy_protein_scale,
    hydrogen_bond_acceptor,
    hydrogen_bond_donor
)
from graphein.protein.features.nodes.dssp import phi, psi, asa, rsa, secondary_structure


default_graph_process_config = ProteinGraphConfig(
    **{
        "node_metadata_functions": [
            amino_acid_one_hot,
            expasy_protein_scale,
            meiler_embedding,
            hydrogen_bond_acceptor,
            hydrogen_bond_donor
        ],
        "edge_construction_functions": [
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
            partial(add_distance_threshold, long_interaction_threshold=3, threshold=10.),
        ],
        "graph_metadata_functions": [asa, phi, psi, secondary_structure, rsa],
        "dssp_config": DSSPConfig(),
    }
)
