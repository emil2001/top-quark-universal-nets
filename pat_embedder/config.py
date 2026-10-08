import os
import random

import numpy as np
import pytorch_lightning as L
import torch

SEED = 2025

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_HIGH_LEVEL_BRANCHES = [
    'M_lep0_jet0',
    'M_lep0_jet1',
    'M_lep0_jet2',
    'M_lep1_jet0',
    'M_lep1_jet1',
    'M_lep1_jet2',
    'M_lep2_jet0',
    'M_lep2_jet1',
    'M_lep2_jet2',
    'M_lep3_jet0',
    'M_lep3_jet1',
    'M_lep3_jet2',
    'M_jet0_jet1',
    'M_jet0_jet2',
    'M_jet1_jet2',
    'M_l_b_minDR',
    'M_b_b_minDR',
    'M_j_j_minDR',
    'min_M_l_b',
    'max_M_b_b',
    'max_M_lep1_b',
    'max_M_lep2_b',
    'Is_max_M_lep1_b_inTop',
    'Is_max_M_lep2_b_inTop',
    'M_lep0_b_min_dr',
    'M_lep1_b_min_dr',
    'M_lep2_b_min_dr',
    'M_lep3_b_min_dr',
    'M_lep0_lep1',
    'Ht',
    'Ht_without_b',
    'DR_l1_l2',
    'minDR_b_b',
    'min2DR_L_b',
    'DPhi_l1_l2',
    'minDR_l_b',
    'maxDR_b_b',
    'max2DR_L_b',
    'maxDR_l_b',
    'min2DR_b_b',
    'max2DR_b_b',
    'S_hat',
    'maxTt_M',
    'minTt_M',
    'mT_lep1',
    'mT_lep2',
    'mT2_lep',
    'mT2_b',
    'Pt_miss',
    'MtW1',
    'MtW2',
]


def seed_basic(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def seed_torch(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed=SEED):
    seed_basic(seed)
    seed_torch(seed)
    L.seed_everything(seed)
