# import pyhealth
import pyhealth
# import mimic4 dataset and drug recommendaton task
from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset
from pyhealth.tasks import drug_recommendation_mimic4_fn, drug_recommendation_mimic3_fn
# import dataloader related functions
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader
# import gamenet model
from pyhealth.models import GAMENet
# import trainer
from pyhealth.trainer import Trainer

_DEV = False
_EPOCHS = 50
#_LR = 0.0002
_LR = 1e-3
#_DECAY_WEIGHT = 0.85
_DECAY_WEIGHT = 1e-5

_DATA_DIR = "./hiddendata/extracted/{}/"
_MIMIC_TABLES = ["diagnoses_icd", "procedures_icd", "prescriptions"]
_MIMIC_CODE_MAP = {"NDC": ("ATC", {"target_kwargs": {"level": 3}})}

_METRICS = [
        "jaccard_samples", "accuracy", "hamming_loss",
        "precision_samples", "recall_samples",
        "pr_auc_samples", "f1_samples"
]

class MIMIC3():
    def dataset():
        return MIMIC3Dataset

    def task():
        return drug_recommendation_mimic3_fn

    def root():
        return _DATA_DIR.format(MIMIC3.dataname())

    def dataname():
        return "mimic3"

class MIMIC4():
    def dataset():
        return MIMIC4Dataset

    def task():
        return drug_recommendation_mimic4_fn

    def root():
        return _DATA_DIR.format(MIMIC4.dataname())

    def dataname():
        return "mimic4"

def prepare_drug_task_data(data_source=MIMIC4):
    dataset = data_source.dataset()
    task = data_source.task()
    dataroot = data_source.root()

    #print("reading data from...{}...".format(dataroot))
    print("reading {} data...".format(data_source.dataname()))

    mimic = dataset(
            root=dataroot,
            tables=_MIMIC_TABLES,
            code_mapping=_MIMIC_CODE_MAP,
            dev=_DEV
            )

    print("stat")
    mimic.stat()
    print("info")
    mimic.info()

    mimic_sample = mimic.set_task(task)
    print(mimic_sample[0])

    return mimic_sample

def get_dataloaders(mimic_sample):
    train_dataset, val_dataset, test_dataset = split_by_patient(mimic_sample, [0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

def train_gamenet(mimic_sample, train_loader, val_loader):
    #gamenet = GAMENet(mimicvi)
    gamenet = GAMENet(mimic_sample)

    #print(gamenet.generate_ddi_adj())
    trainer = Trainer(
        model = gamenet,
        #metrics = ["jaccard_weighted", "pr_auc_micro", "pr_auc_macro"],
        #metrics = ["jaccard", "pr_auc_micro", "pr_auc_macro"],
        #metrics = [
        #    "jaccard_samples", "accuracy", "hamming_loss",
        #    "precision_samples", "recall_samples",
        #    "precision_weighted", "recall_weighted",
        #    "pr_auc_samples", "f1_samples", "f1_weighted",
        #    #"pr_auc_macro", "jaccard_macro",
        #    "pr_auc_weighted",
        #    "jaccard_weighted",
        #    #"roc_auc_samples", "roc_auc_weighted"
        #    ],
        metrics = _METRICS,
        device = "cuda",
        exp_name = "drug_recommendation"
    )

    trainer.train(
        train_dataloader = train_loader,
        val_dataloader = val_loader,
        epochs = _EPOCHS,
        #monitor = "jaccard_weighted",
        #monitor = "pr_auc_macro",
        #monitor = "jaccard_samples",
        monitor = "accuracy",
        monitor_criterion = "max",
        weight_decay = _DECAY_WEIGHT,
        optimizer_params = {"lr": _LR}
    )

    return gamenet, trainer

def evaluate_gamenet(trainer, test_loader):
    result = trainer.evaluate(test_loader)
    print(result)
    return result

if __name__ == "__main__":
    #data = prepare_drug_task_data_mimic4()
    #data = prepare_drug_task_data_mimic3()
    #data = prepare_drug_task_data()
    mimic_dataset = MIMIC4
    # load and prepare data
    data = prepare_drug_task_data(mimic_dataset)
    # create the dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(data)
    # train the model
    model, trainer = train_gamenet(data, train_loader, val_loader)
    # evaluate model performance
    result = evaluate_gamenet(trainer, test_loader)
