import itertools
from collections import defaultdict
from typing import List, Dict, Any, Union
import copy
from pathlib import Path
import os

import networkx as nx
import pandas as pd
import numpy as np

from pymatgen.core import Structure, Composition
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.dimensionality import get_dimensionality_larsen, get_structure_components

from utils import get_BV, get_arbitrary_types_dict, EN_DICT, COVALENT_RADIUS_DICT, VDW_RADIUS_DICT


class CrystalGraphAnalyzer:
    """
    A class for analyzing crystal structures using graph-based methods.

    Args:
        file_name (str): The name of the CIF file.
        connectivity_calculator (VoronoiNN): An object of the VoronoiNN class for computing crystal connectivity.
        bond_property (str): The selected bond property (one of the 'R', 'SA', 'A', 'BV', 'PI') to be used as the weight of edges in the crystal graph.
        arbitrary_element_types (int): 1 or 2 two different element classifications used when classifying the interfragment contacts

    Attributes:
        structure (Structure): The crystal structure.
        connectivity_calculator (VoronoiNN): The VoronoiNN object for connectivity calculations.
        bond_property (str): The selected bond property.

    Methods:
        analyze_graph(target_periodicity: int) -> None:
            Analyze the crystal graph by iteratively removing edges based on the provided border weights.
        output_subgraphs_data(save_fragments_to_cifs: bool) -> Dict:
            Output data on the obtained subgraphs of the initial crystal graph, store the fragments as cif files.
    """

    def __init__(
        self,
        file_name: str,
        connectivity_calculator: VoronoiNN,
        bond_property: str = 'BV',
        arbitrary_element_types: int = 2,
    ) -> None:
        """
        Initialize the CrystalGraphAnalyzer.

        Args:
            file_name (str): The name of the input file.
            connectivity_calculator (VoronoiNN): An object of the VoronoiNN class for computing crystal connectivity.
            bond_property (str): The selected bond property (one of the 'R', 'SA', 'A', 'BV', 'PI') to be used as the weight of edges in the crystal graph.
        """
        self.graph_name = Path(file_name).stem
        self.bond_property = bond_property
        self.connectivity_calculator = connectivity_calculator
        self.structure = Structure.from_file(file_name, primitive=False)
        self.sg = StructureGraph.with_empty_graph(self.structure, name=f"{self.graph_name}_LQG")

        self.monitor = None
        self.sg_edited = None
        self.threshold_weight = None
        self.deleted_contacts = None
        self.target_periodicity_reached = False

        # update crystal graph edge data with selected bond property
        self._suspicious_contacts = set()
        self.arbitrary_types_dict = get_arbitrary_types_dict(mode=arbitrary_element_types)
        self._update_crystal_graph()

        self.inter_bvs = np.nan
        self.intra_bvs = np.nan
        self._edited_graph_total_bvs = np.nan
        self._restored_graph_total_bvs = np.nan
        self.interfragment_contact_atoms = dict()
        self.interfragment_contact_arbitrary_types = dict()
        self._inter_contacts_bv_estimate = dict()
        self._fragments = dict()
        self.total_bvs = sum([edge_data[2] for edge_data in self.sg.graph.edges(data='BV')])


    def _add_weight_to_graph_edges(self, site_n: int, neighbor: Dict, BV_SUSPICIOUS_THRESHOLD=8) -> None:
        """
        Add an edge weight to the crystal graph based on the information from the VoronoiNN calculation.

        Args:
            site_n (int): Index of the central site.
            neighbor (Dict): Information about the neighbour site.
        """
        central_atom, neighbour_atom = self.structure[site_n].specie.symbol, neighbor['site'].specie.symbol
        A = neighbor['poly_info']['area']
        R = neighbor['poly_info']['face_dist'] * 2
        SA = neighbor['poly_info']['solid_angle']
        # on PI >>> https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02238b
        PI = 100 * (VDW_RADIUS_DICT[central_atom] + VDW_RADIUS_DICT[neighbour_atom] - R)\
            / ( (VDW_RADIUS_DICT[central_atom]   - COVALENT_RADIUS_DICT[central_atom])\
              + (VDW_RADIUS_DICT[neighbour_atom] - COVALENT_RADIUS_DICT[neighbour_atom]) )
        BV, BV_calc_method = get_BV((R, central_atom, neighbour_atom))
        
        if BV > BV_SUSPICIOUS_THRESHOLD:
            self._suspicious_contacts.add(f'{central_atom}-{neighbour_atom} {R:.3f}A {BV:.1f} vu')

        edge_properties = {'BV': BV, 'BV_calc_method': BV_calc_method, 'R': R, 'A': A, 'SA': SA, 'PI': PI}
        weight = edge_properties[self.bond_property]

        self.sg.add_edge(
            from_index=site_n,
            from_jimage=(0, 0, 0),
            to_index=neighbor["site_index"],
            to_jimage=neighbor["image"],
            edge_properties=edge_properties,
            weight=weight,
            warn_duplicates=False,
        )

    def _update_crystal_graph(self) -> None:
        """
        Update the crystal graph nodes/edges attributes
        """
        for site_n, neighbors in enumerate(self.connectivity_calculator.get_all_nn_info(self.structure)):
            for neighbor in neighbors:
                self._add_weight_to_graph_edges(site_n, neighbor)

        # add attributes to graph nodes
        for i, node in enumerate(self.sg.graph.nodes):
            self.sg.graph.nodes[i]['element'] = self.sg.structure[i].specie.symbol
            self.sg.graph.nodes[i]['ELTYPE_ARB'] = self.arbitrary_types_dict[self.sg.structure[i].specie.symbol]
            self.sg.graph.nodes[i]['EN'] = EN_DICT[self.sg.structure[i].specie.symbol]
        
        crystal_graph_periodicity = get_dimensionality_larsen(self.sg)
        if crystal_graph_periodicity < 3:
            raise RuntimeError((f'(!!!) The crystal graph periodicity is {crystal_graph_periodicity} < 3 (!!!) '
                                f'Possible reasons - the crystal structure could be unreasonable and suspicious or '
                                f'try to decrease tol parameter in the VoronoiNN class instance '
                                f'so that more interatomic contacts are identified'))

    def _get_threshold_weights(self, merge_close_weights=False) -> List[float]:
        """
        Return a list with sorted threshold values for graph editing
        by cutting edges with weights higher/smaller than threshold
        """
        DELTA = {'R': 0.01, 'BV': 0.001, 'SA': 0.01, 'A': 0.1, 'PI': 0.1}
        DECIMALS = {'R': 2, 'BV': 3, 'SA': 2, 'A': 1, 'PI': 1}
        delta = DELTA[self.bond_property]
        decimals = DECIMALS[self.bond_property]

        if self.bond_property in ('BV', 'SA', 'A', 'PI'):
            reverse = False # higher values correspond to stronger bonding
            delta = delta
        else: # for 'R'
            reverse = True # lower values (shorter bond lengths) correspond to stronger bonding
            delta = -1 * delta

        unique_weights = sorted(set(np.round(self.sg.weight_statistics['all_weights'], decimals=decimals)), reverse=reverse)
        unique_weights = np.array([w + delta for w in unique_weights])

        # merge too close weights
        if merge_close_weights:
            diff = np.ediff1d(unique_weights)
            indices = np.where(np.abs(diff) < delta + 10**-decimals)
            unique_weights = np.array([v for v in unique_weights if v not in unique_weights[indices]])

        return unique_weights

    def analyze_graph(self, target_periodicity: int) -> None:
        """
        Edit the crystal graph by iteratively removing edges based on the threshold weight.

        Args:
            target_periodicity (int): The desired periodicity of the resulting crystal graph.
        """
        monitor = []
        deleted_contacts = [] # store contacts broken during graph editing iterations
        sg_copy = copy.copy(self.sg)

        for threshold_weight in self._get_threshold_weights():

            edges_to_remove = []
            periodicity_before = get_dimensionality_larsen(sg_copy)

            for edge in sg_copy.graph.edges(data=True):

                node1, node2, edge_data = edge

                if (
                    (self.bond_property in ('BV', 'SA', 'A', 'PI') and edge_data['weight'] < threshold_weight) or
                    (self.bond_property in ('R', )                 and edge_data['weight'] > threshold_weight)
                ):
                    broken_bond = '..'.join(
                        sorted([sg_copy.graph.nodes[node]['element'] for node in (node1, node2)]))
                    broken_bond_arb = '..'.join(
                        sorted([sg_copy.graph.nodes[node]['ELTYPE_ARB'] for node in (node1, node2)]))
                    edges_to_remove.append((node1, node2, edge_data['to_jimage']))
                    deleted_contacts.append(
                        ((node1, node2, edge_data['to_jimage']), edge_data, broken_bond, broken_bond_arb)
                    )

            for edge_to_remove in edges_to_remove:
                sg_copy.break_edge(*edge_to_remove)

            periodicity_after = get_dimensionality_larsen(sg_copy)
            bvs_total_after_editing = sum([edge_data[2] for edge_data in sg_copy.graph.edges(data='BV')])
            monitor.append(
                [threshold_weight, periodicity_before, periodicity_after, len(edges_to_remove), bvs_total_after_editing]
            )

            if periodicity_after == target_periodicity:
                self.target_periodicity = target_periodicity
                self.target_periodicity_reached = True
                self.sg_edited = sg_copy
                self.threshold_weight = threshold_weight
                self.deleted_contacts = deleted_contacts
                self.monitor = pd.DataFrame(
                    monitor,
                    columns=[f'threshold_{self.bond_property}', 'periodicity_before', 'periodicity_after', 'N_edges_to_remove', 'bvs_total_after_editing']
                )
                break
            elif periodicity_after < target_periodicity:
                break

        self.monitor = pd.DataFrame(
            monitor, columns=[f'threshold_{self.bond_property}', 'periodicity_before', 'periodicity_after', 'N_edges_to_remove', 'bvs_total_after_editing'])

        return None

    def _get_unique_fragments(self, graph_dict: Dict) -> Dict[int, Dict]:
        """
        Return filtered dict with subgraphs of the initial full crystal structure graph
        """
        # check graphs by node element and edge distances
        node_matcher = nx.algorithms.isomorphism.categorical_node_match(attr='element', default='')
        edge_matcher = nx.algorithms.isomorphism.numerical_multiedge_match(attr='R', default=1.0, atol=1e-3)

        graph_list = list(graph_dict.values())

        unique_graphs = []
        for i in range(len(graph_list)):
            is_unique = True
            for j in range(len(graph_list)):
                if i > j:
                    if nx.is_isomorphic(
                        graph_list[i]['SG'].graph,
                        graph_list[j]['SG'].graph,
                        node_match=node_matcher,
                        edge_match=edge_matcher,
                    ):
                        is_unique = False
                        break
            if is_unique:
                unique_graphs.append(graph_list[i])

        return {i: g_dict for i, g_dict in enumerate(unique_graphs, 1)}

    def _calculate_fragment_charges(self, fragments_dict: Dict, inter_contacts: set, fragment_sites_dict: Dict):
        """
        Estimate fragment charge using the EN difference of the atoms in the interfragment contact.
        Atom with higher EN belonging to the fragment contributes the partial negative charge to that
        fragment amounting to contact BV value; atom with lower EN contributes the partial positive
        charge to that fragment, respectively.
        """

        EPSILON = 1e-6
        complete_fragments_dict = {}

        if len(fragment_sites_dict) == 1:
            fragment_charges = {0: 0.0}

        else:
            fragment_charges = defaultdict(int)
            site_fragments_dict = {site: fragment\
                                   for fragment, sites in fragment_sites_dict.items()\
                                   for site in sites}
            contacts_bv = {c[0]: c[1]['BV'] for c in self.deleted_contacts}
            node_ens = {n: self.sg.graph.nodes[n]['EN'] for n in range(self.sg.graph.number_of_nodes())}

            for inter_contact in inter_contacts:
                node1, node2, _ = inter_contact
                contact_bv = contacts_bv[inter_contact]
                en_difference = node_ens[node1] - node_ens[node2]
                # print(sg.structure[node1].specie.symbol, sg.structure[node2].specie.symbol,
                #       inter_contact, contact_bv, en_difference)
                if en_difference > EPSILON:
                    fragment_charges[site_fragments_dict[node1]] -= contact_bv
                    fragment_charges[site_fragments_dict[node2]] += contact_bv
                elif en_difference < -EPSILON:
                    fragment_charges[site_fragments_dict[node1]] += contact_bv
                    fragment_charges[site_fragments_dict[node2]] -= contact_bv
                else:
                    fragment_charges[site_fragments_dict[node1]] += 0.0
                    fragment_charges[site_fragments_dict[node2]] += 0.0

        for i, g_dict in fragments_dict.items():
            g_dict['estimated_charge'] = np.round(fragment_charges[i], 3)
            complete_fragments_dict[i] = g_dict

        return complete_fragments_dict

    def _restore_fragment_graph(self, fragment_sites_dict: Dict) -> None:
        """
        Restore some intra fragment contacts
        """
        # get pairs of nodes that correspond to INTRA fragment contacts
        # pairs of nodes NOT in this set correspond to INTER fragment contacts
        intra_contacts = set(c for site_ids_list in fragment_sites_dict.values()\
                             for c in itertools.combinations(site_ids_list, 2))
        intra_contacts_to_restore = [c for c in self.deleted_contacts if c[0][:2] in intra_contacts]

        # BV sum BEFORE restoring some INTRA fragment contacts
        self._edited_graph_total_bvs = sum([edge_data[2] for edge_data in self.sg_edited.graph.edges(data='BV')])

        # iterate over broken INTRA contacts and if the periodicity does not increase
        # restore this contact in a given fragment
        for broken_edge_data in intra_contacts_to_restore:
            (n1, n2, translation), edge_data, _, _ = broken_edge_data
            test_graph = copy.copy(self.sg_edited)
            test_graph.add_edge(
                n1,
                n2,
                from_jimage=(0, 0, 0),
                to_jimage=translation,
            )

            if get_dimensionality_larsen(test_graph) <= self.target_periodicity:
                self.sg_edited.add_edge(
                    from_index=n1,
                    to_index=n2,
                    from_jimage=(0, 0, 0),
                    to_jimage=translation,
                    edge_properties={k: v for k, v in edge_data.items() if k not in ('to_jimage')},
            )
        assert get_dimensionality_larsen(self.sg_edited) == self.target_periodicity, 'target dimensionsionality is not preserved'

        # BV sum AFTER restoring some INTRA fragment contacts
        self._restored_graph_total_bvs = sum([edge_data[2] for edge_data in self.sg_edited.graph.edges(data='BV')])

    def output_subgraphs_data(self, save_fragments_to_cifs=True, save_fragments_path='./') -> Dict[str, Any]:
        """
        Output data for the obtained 1-p/2-p subgraphs (components or fragments) of the initial crystal graph.

        Returns:
            Dict[int, List[int]]: Dictionary containing information on the sites being part of the obtained fragments.
        """

        if self.target_periodicity_reached:

            fragment_dict = {}
            fragment_sites_dict = {}

            for i, component in enumerate(
                get_structure_components(self.sg_edited, inc_orientation=True, inc_site_ids=True)
            ):
                periodicity = str(component['dimensionality'])
                SG = component['structure_graph']
                sites_data = component['site_ids']
                orientation = component['orientation'] if component['orientation'] is not None else periodicity
                orientation = ''.join([str(v) for v in orientation])
                composition = component['structure_graph'].structure.composition.formula.replace(" ", "")

                fragment_dict[i] = {'composition': composition, 'periodicity': periodicity,
                                    'orientation': orientation, 'SG': SG}
                fragment_sites_dict[i] = sites_data

            # before calculating BVSs check fragment graphs and restore INTRA fragment contacts
            # broken before target_periodicity has been reached
            self._restore_fragment_graph(fragment_sites_dict)

            inter_contacts = self.sg.diff(self.sg_edited)['self']
            # estimate fragment charges using edited and restored graph
            fragment_dict = self._calculate_fragment_charges(fragment_dict, inter_contacts, fragment_sites_dict)
            interfragment_contact_atoms = [c[2] for c in self.deleted_contacts if c[0] in inter_contacts]
            interfragment_contact_arbitrary_types = [c[3] for c in self.deleted_contacts if c[0] in inter_contacts]
            inter_contacts_bv_estimate = [c[1]['BV_calc_method'] for c in self.deleted_contacts if c[0] in inter_contacts]

            self.intra_bvs = sum([e[2] for e in self.sg_edited.graph.edges(data='BV')])
            self.inter_bvs = self.total_bvs - self.intra_bvs
            self.interfragment_contact_atoms = pd.Series(interfragment_contact_atoms).value_counts().to_dict()
            self.interfragment_contact_arbitrary_types = pd.Series(interfragment_contact_arbitrary_types).value_counts().to_dict()
            self._inter_contacts_bv_estimate = pd.Series(inter_contacts_bv_estimate).value_counts(normalize=True)\
                                                                                    .apply(lambda v: round(v, 4))\
                                                                                    .to_dict()

            # filter out duplicated fragment graphs
            unique_fragments_dict = self._get_unique_fragments(fragment_dict)
            # store unique fragments into cif files if required
            for i, graph_data in unique_fragments_dict.items():
                if save_fragments_to_cifs:
                    if graph_data['periodicity'] in ('1', '2'):
                        graph_data['SG'].structure.to(
                            Path(save_fragments_path) / f"{self.graph_name}-{self.bond_property}-fragment_{i}-{graph_data['composition']}-"
                            f"{graph_data['periodicity']}periodic-{graph_data['orientation']}.cif", fmt='cif'
                        )
                del unique_fragments_dict[i]['SG']

            self._fragments = unique_fragments_dict

        result_instance = CrystalGraphAnalyzerResult(self)

        return result_instance


class CrystalGraphAnalyzerResult:
    """
    A class to store the results of crystal graph analysis.

    Args:
        analyzer_instance (CrystalGraphAnalyzer): An instance of CrystalGraphAnalyzer.

    Attributes:
        target_periodicity_reached (bool): Whether the target periodicity was reached.
        input_file_name (str): Name of the input file.
        bond_property (str): The selected bond property.
        monitor (pd.DataFrame): DataFrame with monitoring data stored at each crystal graph editing step.
        total_bvs (float): Total bond valence sum.
        fragments (dict): Dictionary of fragments.
        xbvs (float): Fraction of bond valence sum in the fragments.
        mean_inter_bv (float): Mean bond valence of contacts between fragments.
        inter_bvs_per_interface (float): Bond valence sum per interface.
        interfragment_contact_atoms (dict): Atom types involved in inter-fragment contacts.
        interfragment_contact_arbitrary_types (dict): Arbitrary atom types involved in inter-fragment contacts;
            elements are grouped according to self.arbitrary_types_dict (utils.py).
        inter_contacts_bv_estimate (dict): Estimated bond valence for inter-fragment contacts.
        intra_bvs (float): Intra-fragment bond valence sum.
        inter_bvs (float): Inter-fragment bond valence sum.
        edited_graph_total_bvs (float): Total bond valence sum of the edited graph.
        restored_graph_total_bvs (float): Total bond valence sum of the restored graph.
    """

    def __init__(self, analyzer_instance: 'CrystalGraphAnalyzer') -> None:
        """
        Initialize the CrystalGraphAnalyzerResult.

        Args:
            analyzer_instance (CrystalGraphAnalyzer): An instance of CrystalGraphAnalyzer.
        """
        self.target_periodicity_reached = analyzer_instance.target_periodicity_reached
        self.input_file_name = analyzer_instance.graph_name
        self.bond_property = analyzer_instance.bond_property
        self.monitor = analyzer_instance.monitor
        self.total_bvs = analyzer_instance.total_bvs
        self._suspicious_contacts = analyzer_instance._suspicious_contacts

        if analyzer_instance.target_periodicity_reached:
            self.fragments = analyzer_instance._fragments
            self.xbvs = (analyzer_instance.total_bvs - analyzer_instance.inter_bvs) / analyzer_instance.total_bvs
            self.mean_inter_bv = analyzer_instance.inter_bvs / sum(analyzer_instance.interfragment_contact_atoms.values())
            self.inter_bvs_per_interface = analyzer_instance.inter_bvs / len(analyzer_instance._fragments)
            self.interfragment_contact_atoms = analyzer_instance.interfragment_contact_atoms
            self.interfragment_contact_arbitrary_types = analyzer_instance.interfragment_contact_arbitrary_types
            self.inter_contacts_bv_estimate = analyzer_instance._inter_contacts_bv_estimate
            self.intra_bvs = analyzer_instance.intra_bvs
            self.inter_bvs = analyzer_instance.inter_bvs
            self.edited_graph_total_bvs = analyzer_instance._edited_graph_total_bvs
            self.restored_graph_total_bvs = analyzer_instance._restored_graph_total_bvs
        else:
            self.fragments = {}
            self.xbvs = np.nan
            self.mean_inter_bv = np.nan
            self.inter_bvs_per_interface = np.nan
            self.interfragment_contact_atoms = {}
            self.interfragment_contact_arbitrary_types = {}
            self.inter_contacts_bv_estimate = {}
            self.intra_bvs = np.nan
            self.inter_bvs = np.nan
            self.edited_graph_total_bvs = np.nan
            self.restored_graph_total_bvs = np.nan

    def results_as_string(self) -> dict:
        """
        Get the results as a formatted string.

        Returns:
            dict: Dictionary containing formatted results.
        """
        results_dict = {
            'fragments': self.fragments,
            'N_fragments': len(self.fragments),
            'total_bvs': np.round(self.total_bvs, 4),
            'inter_bvs': np.round(self.inter_bvs, 4),
            'xbvs': np.round(self.xbvs, 4),
            'mean_inter_bv': np.round(self.mean_inter_bv, 4),
            'inter_bvs_per_interface': np.round(self.inter_bvs_per_interface, 4),
            'interfragment_contact_atoms': self.interfragment_contact_atoms,
            'interfragment_contact_arbitrary_types': self.interfragment_contact_arbitrary_types,
            'inter_contacts_bv_estimate':  self.inter_contacts_bv_estimate,
            'suspicious_contacts': '|'.join(sorted(self._suspicious_contacts))
        }

        formatted_string = ""
        for key, value in results_dict.items():
            formatted_string += f"{key}: {value}\n"

        return formatted_string

    def show_BVs_data(self) -> dict:
        """
        Show the bond valence data.

        Returns:
            dict: Dictionary containing bond valence data.
        """
        return {
            'xbvs': self.xbvs,
            'mean_inter_bv': self.mean_inter_bv,
            'inter_bvs_per_interface': self.inter_bvs_per_interface,
            'intra_bvs': self.intra_bvs,
            'inter_bvs': self.inter_bvs,
            'total_bvs': self.total_bvs,
            'edited_graph_total_bvs': self.edited_graph_total_bvs,
            'restored_graph_total_bvs': self.restored_graph_total_bvs,
        }

    def monitor(self) -> pd.DataFrame:
        """
        Get the monitoring data.

        Returns:
            list: List containing monitoring data.
        """
        return self.monitor

    def periodicity_partition(self, return_df=True) -> Union[pd.DataFrame, Dict]:
        """
        Estimate how the BVS is distributed among the fragments of a given periodicity (3-, 2-, 1-periodic fragmnets).
        The algorithm identifies the contacts which correspond to specific periodicity of fragments
        and assigns the BVS to a given periodicity. After that the BVSs assigned to each periodicities
        are normalised and the returning vector reflects how many of the BVS is retained in the fragment with a given periodicity.

        !!! IMPORTANT !!!
        The TARGET_PERIODICITY parameter of the analyze_graph method in the CrystalGraphAnalyzer instance should be set to 0.
        That is, the crystal graph analysis procedure stops when maximum periodicity of the fragments reached is 0
        !!! IMPORTANT !!!

        Returns:
            pd.DataFrame: formatted dataframe for analysis of single structure
            dict: dictionary with fragment periodicity as a key and share of BVS as a value
        """
        df = self.monitor.copy()
        df['periodicity_change'] = (df['periodicity_before'] - df['periodicity_after']).astype(bool).astype(int)
        df = df[df['periodicity_change'] == 1]
        df['bv_periodicity_partition'] = -np.diff(
            df[df['periodicity_change'] == 1]['bvs_total_after_editing'], prepend=self.total_bvs
        )
        df['bv_periodicity_partition_normalised'] = df['bv_periodicity_partition'] / df['bv_periodicity_partition'].sum()
        df = df[['threshold_BV', 'periodicity_before', 'bv_periodicity_partition', 'bv_periodicity_partition_normalised']]

        if return_df:
            return df
        else:
            df_dict = df[['periodicity_before', 'bv_periodicity_partition_normalised']].set_index('periodicity_before').iloc[:, 0].to_dict()
            df_dict['bv_periodicity_partition_STD'] = round(df['bv_periodicity_partition_normalised'].std(), 3)
            df_dict['bv_periodicity_partition_RANGE'] = round(df['bv_periodicity_partition_normalised'].max() - df['bv_periodicity_partition_normalised'].min(), 3)
            df_dict['input_file_name'] = self.input_file_name

            return df_dict

    def results_as_dict(self) -> Dict:
        """
        Get the results as a dictionary.

        Returns:
            dict: Dictionary containing results.
        """
        return {
            "input_file_name": self.input_file_name,
            "target_periodicity_reached": int(self.target_periodicity_reached),
            "bond_property": self.bond_property,
            'composition': [Composition(
                Composition(v.get('composition', '')).get_integer_formula_and_factor()[0]
            ).formula for v in self.fragments.values()],
            'periodicity': [v.get('periodicity', '') for v in self.fragments.values()],
            'orientation': [v.get('orientation', '') for v in self.fragments.values()],
            'estimated_charge': [v.get('estimated_charge', '') for v in self.fragments.values()],
            "inter_bvs": self.inter_bvs,
            "intra_bvs": self.intra_bvs,
            "total_bvs": self.total_bvs,
            "xbvs": self.xbvs,
            "mean_inter_bv": self.mean_inter_bv,
            "inter_bvs_per_interface": self.inter_bvs_per_interface,
            "interfragment_contact_atoms": '|'.join(sorted(self.interfragment_contact_atoms.keys())),
            "interfragment_contact_atoms_full": str(self.interfragment_contact_atoms),
            "interfragment_contact_arbitrary_types": '|'.join(sorted(self.interfragment_contact_arbitrary_types.keys())),
            "interfragment_contact_arbitrary_types_full": str(self.interfragment_contact_arbitrary_types),
            "inter_contacts_bv_estimate_(tentative_share)": self.inter_contacts_bv_estimate.get('ML_estimated (confidence: False)', 0.0),
            "inter_contacts_bv_estimate_full": str(self.inter_contacts_bv_estimate),
            'suspicious_contacts': '|'.join(sorted(self._suspicious_contacts)),
        }
