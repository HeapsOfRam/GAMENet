import argparse

# import pyhealth
import pyhealth
# import mimic4 dataset and drug recommendaton task
from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset
from pyhealth.tasks import drug_recommendation_mimic4_fn, drug_recommendation_mimic3_fn
# import dataloader related functions
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader
# import gamenet model
from pyhealth.models import RETAIN, GAMENet
# import trainer
from pyhealth.trainer import Trainer

# information for the mimic data
_DEV = False
_DATA_DIR = "./hiddendata/extracted/{}/"
_MIMIC_TABLES = ["diagnoses_icd", "procedures_icd", "prescriptions"]
_MIMIC_CODE_MAP = {"NDC": ("ATC", {"target_kwargs": {"level": 3}})}

# experiment names
_GAMENET_EXP = "drug_recommendation_gamenet"
_RETAIN_EXP = "drug_recommendation_retain"

# set the device to be the cuda gpu
_DEVICE = "cuda"

# hyperparameters for training
_EPOCHS = 20
#_LR = 0.0002
_LR = 1e-3
#_DECAY_WEIGHT = 0.85
_DECAY_WEIGHT = 1e-5

# metrics to track from training/evaluation
_METRICS = [
        "jaccard_samples", "accuracy", "hamming_loss",
        "precision_samples", "recall_samples",
        "pr_auc_samples", "f1_samples"
]

# define data classes
## these classes wrap some pyhealth logic that is similar between the two datasets
## enables easy switching between MIMIC4/MIMIC3 data

# dataclass for MIMIC3
class MIMIC3():
    def dataset():
        return MIMIC3Dataset

    def task():
        return drug_recommendation_mimic3_fn

    def root():
        return _DATA_DIR.format(MIMIC3.dataname())

    def dataname():
        return "mimic3"

# dataclass for MIMIC4
class MIMIC4():
    def dataset():
        return MIMIC4Dataset

    def task():
        return drug_recommendation_mimic4_fn

    def root():
        return _DATA_DIR.format(MIMIC4.dataname())

    def dataname():
        return "mimic4"

# data preparation
def prepare_drug_task_data(data_source=MIMIC4):
    dataset = data_source.dataset()
    task = data_source.task()
    dataroot = data_source.root()

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

# data loaders for training and evaluation
def get_dataloaders(mimic_sample):
    train_dataset, val_dataset, test_dataset = split_by_patient(mimic_sample, [0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

# training step for the retain baseline model
def train_retain(mimic_sample, train_loader, val_loader, epochs=_EPOCHS, decay_weight=_DECAY_WEIGHT, learning_rate=_LR, experiment=_RETAIN_EXP, device=_DEVICE):
    model = RETAIN(
            mimic_sample,
            feature_keys = ["conditions", "procedures"],
            label_key = "drugs",
            mode = "multilabel"
            )

    trainer = Trainer(
            model = model,
            metrics = _METRICS,
            device = device,
            exp_name = experiment
            )

    trainer.train(
            train_dataloader = train_loader,
            val_dataloader = val_loader,
            epochs = epochs,
            monitor = "accuracy",
            monitor_criterion = "max",
            weight_decay = decay_weight,
            optimizer_params = {"lr": learning_rate}
            )

    return model, trainer

# training step for the gamenet model
def train_gamenet(mimic_sample, train_loader, val_loader, epochs=_EPOCHS, decay_weight=_DECAY_WEIGHT, learning_rate=_LR, experiment=_GAMENET_EXP, device=_DEVICE):
    gamenet = GAMENet(mimic_sample)

    trainer = Trainer(
        model = gamenet,
        metrics = _METRICS,
        device = device,
        exp_name = experiment
    )

    trainer.train(
        train_dataloader = train_loader,
        val_dataloader = val_loader,
        epochs = epochs,
        monitor = "accuracy",
        monitor_criterion = "max",
        weight_decay = decay_weight,
        optimizer_params = {"lr": learning_rate}
    )

    return gamenet, trainer

# function to evaluate the models
def evaluate_model(trainer, test_loader):
    result = trainer.evaluate(test_loader)
    print(result)
    return result

if __name__ == "__main__":
    # first, define command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "-m", "--mimic", dest="mimicvers",
            choices=[3, 4], default=4, type=int,
            help="version of mimic to use"
            )
    parser.add_argument(
            "-s", "--skip-baseline", dest="baseline",
            action='store_false', default=True,
            help="skip the retain baseline comparison"
            )
    parser.add_argument(
            "-b", "--baseline-only", dest="gamenet",
            action="store_false", default=True,
            help="whether to only run the retain baseline"
            )
    parser.add_argument(
            "-e", "--epochs", dest="epochs",
            default=_EPOCHS, type=int,
            help="how many epochs to run the training for"
            )
    parser.add_argument(
            "-d", "--decay-weight", dest="decay",
            default=_DECAY_WEIGHT, type=float,
            help="value for the decay weight hyperparameter"
            )
    parser.add_argument(
            "-l", "--learning-rate", dest="rate",
            default=_LR, type=float,
            help="value for the learning weight hyperparameter"
            )

    args = parser.parse_args()

    # choose dataset
    if args.mimicvers == 4:
        mimic_dataset = MIMIC4
    elif args.mimicvers == 3:
        mimic_dataset = MIMIC3

    # load and prepare data
    data = prepare_drug_task_data(mimic_dataset)
    # create the dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(data)

    # train the models
    baseline_result = None
    gamenet_result = None

    # baseline
    if args.baseline:
        print("---RETAIN TRAINING---")
        retain, retain_trainer = train_retain(
                data, train_loader, val_loader,
                epochs=args.epochs,
                decay_weight=args.decay,
                learning_rate=args.rate
                )
        print("---RETAIN EVALUATION---")
        baseline_result = evaluate_model(retain_trainer, test_loader)

    # gamenet
    if args.gamenet:
        print("---GAMENET TRAINING---")
        gamenet, gn_trainer = train_gamenet(
                data, train_loader, val_loader,
                epochs=args.epochs,
                decay_weight=args.decay,
                learning_rate=args.rate
                )
        print("---GAMENET EVALUATION---")
        #gn_result = evaluate_model(gn_trainer, test_loader)
        gamenet_result = evaluate_model(gn_trainer, test_loader)

    print("---RESULTS---")
    print("---baseline---")
    print(baseline_result)
    print("---gamenet---")
    print(gamenet_result)


    #print("---GAMENET TRAINING---")
    #gamenet, gn_trainer = train_gamenet(data, train_loader, val_loader)
    #print("---RETAIN TRAINING---")
    #retain, retain_trainer = train_retain(data, train_loader, val_loader)
    ## evaluate model performance
    #print("---GAMENET EVALUATION---")
    ##gn_result = evaluate_model(gn_trainer, test_loader)
    #print("---RETAIN EVALUATION---")
    #retain_result = evaluate_model(retain_trainer, test_loader)
