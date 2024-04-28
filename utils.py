import copy
from pathlib import Path

import pandas as pd
import numpy as np

import mendeleev


try:
    EXCEL_TABLE_PATH = r'./BV_estimated_23-04-2024.xlsx'
    BVPARAMS = pd.read_excel(EXCEL_TABLE_PATH, index_col=0)\
                 .loc[:, ['bond', 'Atom1', 'Atom2', 'confident_prediction',
                          'Rcov_sum', 'delta', 'R0_estimated', 'R0_empirical', 'B']]
except FileNotFoundError:
    print(f'excel table with BV parameters has not been found at '
          f'{EXCEL_TABLE_PATH}')
else:
    pass


EN_DICT = {
    el.symbol: el.electronegativity_allred_rochow()
    for el in mendeleev.get_all_elements()
}
COVALENT_RADIUS_DICT = {
    el.symbol: el.covalent_radius_cordero/100 if el.covalent_radius_cordero is not None else el.covalent_radius_pyykko/100
    for el in mendeleev.get_all_elements()[:100]
}
VDW_RADIUS_DICT = {
    el.symbol: el.vdw_radius_alvarez/100 if el.vdw_radius_alvarez is not None else el.vdw_radius/100
    for el in mendeleev.get_all_elements()[:100]
}

def get_arbitrary_types_dict(mode=1) -> dict:
    """
    mode == 1:
        'ENM': 'Al Ga In Sn Tl Pb Bi Nh Fl Mc Lv',
        'EPM': 'Li Be Na Mg K Ca Rb Sr Cs Ba Fr Ra',
        'FM': 'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb  Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No',
        'MTL': 'B Si Ge As Sb Te Po',
        'NG': 'He Ne Ar Kr Xe Rn Og',
        'NM': 'H C N O F P S Cl Se Br I At Ts',
        'TM': 'Sc Ti V Cr Mn Fe Co Ni Cu Zn  Y Zr Nb Mo Tc Ru Rh Pd Ag Cd  Lu Hf Ta W Re Os Ir Pt Au Hg  Lr Rf Db Sg Bh Hs Mt Ds Rg Cn'

    mode == 2:
        See "periodic_table" dict
    
    Return: dict
    """

    if mode == 1:

        arbitrary_types = {
            'Actinides': 'FM',
            'Alkali metals': 'EPM',
            'Alkaline earth metals': 'EPM',
            'Halogens': 'NM',
            'Lanthanides': 'FM',
            'Metalloids': 'MTL',
            'Noble gases': 'NG',
            'Nonmetals': 'NM',
            'Poor metals': 'ENM',
            'Transition metals': 'TM'
        }

        EL_ARBITRARY_TYPES_DICT = {el.symbol: arbitrary_types[el.series] for el in mendeleev.get_all_elements()}

    else:

        periodic_table = {
            # Period 1
            "H": "H",  "He": "NG",
            # Period 2
            "Li": "EPM", "Be": "ENM", "B": "MTL", "C": "LNM", "N": "LNM", "O": "LNM", "F": "LNM", "Ne": "NG",
            # Period 3
            "Na": "EPM", "Mg": "EPM", "Al": "ENM", "Si": "MTL", "P": "NM", "S": "NM", "Cl": "NM", "Ar": "NG",
            # Period 4
            "K": "EPM",  "Ca": "EPM",
            "Sc": "TM", "Ti": "TM", "V": "TM", "Cr": "TM", "Mn": "TM", "Fe": "TM", "Co": "TM", "Ni": "TM", "Cu": "TM", "Zn": "TM",
            "Ga": "ENM", "Ge": "MTL", "As": "MTL", "Se": "NM", "Br": "NM",  "Kr": "NG",
            # Period 5
            "Rb": "EPM", "Sr": "EPM",
            "Y": "TM", "Zr": "TM", "Nb": "TM", "Mo": "TM", "Tc": "TM", "Ru": "TM", "Rh": "TM", "Pd": "TM", "Ag": "TM", "Cd": "TM",
            "In": "ENM", "Sn": "ENM", "Sb": "MTL", "Te": "MTL", "I": "NM", "Xe": "NG",
            # Period 6
            "Cs": "EPM", "Ba": "EPM",
            "La": "FM", "Ce": "FM", "Pr": "FM", "Nd": "FM", "Pm": "FM", "Sm": "FM", "Eu": "FM",
            "Gd": "FM", "Tb": "FM", "Dy": "FM", "Ho": "FM", "Er": "FM", "Tm": "FM", "Yb": "FM", "Lu": "FM",
            "Hf": "TM", "Ta": "TM", "W": "TM",  "Re": "TM", "Os": "TM", "Ir": "TM", "Pt": "TM", "Au": "TM", "Hg": "TM",
            "Tl": "ENM", "Pb": "ENM", "Bi": "ENM", "Po": "MTL", "At": "NM", "Rn": "NG",
            # Period 7
            "Fr": "EPM", "Ra": "EPM",
            "Ac": "FM", "Th": "FM", "Pa": "FM", "U": "FM", "Np": "FM", "Pu": "FM", "Am": "FM",
            "Cm": "FM", "Bk": "FM", "Cf": "FM", "Es": "FM", "Fm": "FM", "Md": "FM", "No": "FM", "Lr": "FM",
            "Rf": "TM", "Db": "TM", "Sg": "TM", "Bh": "TM", "Hs": "TM", "Mt": "TM", "Ds": "TM", "Rg": "TM", "Cn": "TM",
            "Nh": "ENM", "Fl": "ENM", "Mc": "ENM", "Lv": "ENM", "Ts": "NM", "Og": "NG"
        }

        EL_ARBITRARY_TYPES_DICT = {el.symbol: periodic_table[el.symbol] for el in mendeleev.get_all_elements()}

    return EL_ARBITRARY_TYPES_DICT


def get_BV(args: tuple[float, str, str]) -> tuple[float, str]:
    """
    Get bond valence (BV) for a bond
    between Atom1 (el1) and Atom2 (el2)
    residing at R angstrom from each other

    Args:

        R - interatomic distance;
        el1 - element 1 symbol;
        el2 - element 2 symbol;
    
    Return:
        
        bond_valence, data_source
    """
    R, el1, el2 = args

    empirical_bvs = BVPARAMS[
        ((BVPARAMS['Atom1'] == el1) & (BVPARAMS['Atom2'] == el2))\
      | ((BVPARAMS['Atom1'] == el2) & (BVPARAMS['Atom2'] == el1))
    ]
    
    if empirical_bvs.shape[0] == 0:
        return np.nan, 'no_estimate'

    if empirical_bvs['R0_empirical'].notna().bool():
        R0 = empirical_bvs.iat[0, 7] # use R0_empirical
        B = empirical_bvs.iat[0, 8]
        data_source = 'empirical_and_extrapolated'
    elif empirical_bvs['R0_empirical'].isna().bool():
        R0 = empirical_bvs.iat[0, 6] # use R0_estimated
        B = 0.37
        confidence = bool(empirical_bvs.iat[0, 3])
        data_source = f'ML_estimated (confidence: {confidence})'
    else:
        R0 = np.nan
        B = np.nan
        data_source = 'no_estimate'

    return np.exp((R0 - R) / B), data_source
