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

# EL_ARBITRARY_TYPES_DICT:
    # 'ENM': 'Al Ga In Sn Tl Pb Bi Nh Fl Mc Lv',
    # 'EPM': 'Li Be Na Mg K Ca Rb Sr Cs Ba Fr Ra',
    # 'FM': 'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb  Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No',
    # 'MTL': 'B Si Ge As Sb Te Po',
    # 'NG': 'He Ne Ar Kr Xe Rn Og',
    # 'NM': 'H C N O F P S Cl Se Br I At Ts',
    # 'TM': 'Sc Ti V Cr Mn Fe Co Ni Cu Zn  Y Zr Nb Mo Tc Ru Rh Pd Ag Cd  Lu Hf Ta W Re Os Ir Pt Au Hg  Lr Rf Db Sg Bh Hs Mt Ds Rg Cn'


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
