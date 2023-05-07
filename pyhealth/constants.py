from alt_gamenets import GAMENetNoHist, GAMENetNoProc
from drug_rec_task import (
    drug_recommendation_mimic4_no_hist,
    drug_recommendation_mimic4_no_proc,
    #drug_recommendation_mimic4_flat
)

from pyhealth.models import GAMENet, RETAIN
from pyhealth.tasks import drug_recommendation_mimic3_fn, drug_recommendation_mimic4_fn

# information for the mimic data
DEV = False
DATA_DIR = "./hiddendata/extracted/{}/"
MIMIC_TABLES = ["diagnoses_icd", "procedures_icd", "prescriptions"]
MIMIC_CODE_MAP = {"NDC": ("ATC", {"target_kwargs": {"level": 3}})}

# task ids
DRUG_REC_TN = "drug_recommendation"
NO_HIST_TN = "no_hist"
NO_PROC_TN = "no_proc"
FLAT_TN = "flat"

#ALL_TASKS = [DRUG_REC_TN, NO_HIST_TN, NO_PROC_TN, FLAT_TN]
ALL_TASKS = [DRUG_REC_TN, NO_HIST_TN, NO_PROC_TN]


# set the device to be the cuda gpu
DEVICE = "cuda"

# hyperparameters for training
EPOCHS = 20
#_LR = 0.0002
LR = 1e-3
#_DECAY_WEIGHT = 0.85
DECAY_WEIGHT = 1e-5

THRESH = 0.5

# metrics to track from training/evaluation
METRICS = [
        "jaccard_samples", "accuracy", "hamming_loss",
        "precision_samples", "recall_samples",
        "pr_auc_samples", "f1_samples"
]

#REQUIRED_DL_KEYS = {'train', 'val', 'test'}

# experiment names
GAMENET_EXP = "drug_recommendation_gamenet"
RETAIN_EXP = "drug_recommendation_retain"

DEFAULT_EXPERIMENT = "drug_recommendation"

BEST_MODEL_PATH = "output/{}/best.ckpt"

# dictionary keys for the task dictionary
TN = "name"
TTASK = "task"

# dictionary keys for the model dictionary
GN_KEY = "GAMENet"
RT_KEY = "RETAIN"

# result keys
SCORE_KEY = "scores"
DPV_KEY = "avg_dpv"
DDI_RATE_KEY = "ddi_rate"

# base ddi rate for comparison
BASE_DDI_RATE = 0.0777

# default task lists
MIMIC4_TASKS = {
        DRUG_REC_TN: drug_recommendation_mimic4_fn,
        NO_HIST_TN: drug_recommendation_mimic4_no_hist,
        NO_PROC_TN: drug_recommendation_mimic4_no_proc,
        #FLAT_TN: drug_recommendation_mimic4_flat
        }
MIMIC3_TASKS = {
        DRUG_REC_TN: drug_recommendation_mimic3_fn
        }

# which model to load for the specific task
# for example, need different gamenet variant for different tasks
MODEL_TYPES_PER_TASK = {
    DRUG_REC_TN: {GN_KEY: GAMENet, RT_KEY: RETAIN},
    #NO_HIST_TN: {GN_KEY: GAMENet, RT_KEY: RETAIN},
    NO_HIST_TN: {GN_KEY: GAMENetNoHist, RT_KEY: RETAIN},
    NO_PROC_TN: {GN_KEY: GAMENetNoProc, RT_KEY: RETAIN},
    FLAT_TN: {GN_KEY: GAMENet, RT_KEY: RETAIN}
}

# which features to use in retain for the specific task
# for example, when we don't prepare procedures we don't use that as a feature
RETAIN_DEFAULT_FEATURES = ["conditions", "procedures"]

RETAIN_FEATS_PER_TASK = {
    DRUG_REC_TN: RETAIN_DEFAULT_FEATURES,
    NO_HIST_TN: RETAIN_DEFAULT_FEATURES,
    NO_PROC_TN: ["conditions"],
    FLAT_TN: RETAIN_DEFAULT_FEATURES
}
