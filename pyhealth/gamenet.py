import argparse
import sys

from drug_rec_task import *

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

_DRUG_REC_TN="drug_recommendation"
_NO_HIST_TN = "no_hist"

_MIMIC4_TASKS = {
        _DRUG_REC_TN: drug_recommendation_mimic4_fn,
        _NO_HIST_TN: drug_recommendation_mimic4_no_hist
        }
_MIMIC3_TASKS = {
        _DRUG_REC_TN: drug_recommendation_mimic3_fn
        }

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

    def all_tasks():
        return _MIMIC3_TASKS

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

    def all_tasks():
        return _MIMIC4_TASKS

class Task():
    def __init__(self, task, name=_DRUG_REC_TN):
        self._task = task
        self._name = name

    def name(self):
        return self._name

    def task(self):
        return self._task


class MIMICWrapper():
    def __init__(self, datasource=MIMIC4, tasks=MIMIC4.all_tasks()):
        self.data_source = datasource
        self.root = datasource.root()
        self.dataname = datasource.dataname()
        self.tasks = tasks

        self.tasklist = []
        self.prepared_data = {}
        self.tasknames = []

        for taskname,task in self.tasks.items():
            self.tasknames.append(taskname)
            self.tasklist.append(Task(task, taskname))

    def load_data(self, dev=_DEV):
        dataset = self.data_source.dataset()
        dataroot = self.data_source.root()

        if dev:
            print("-*-READING DEV DATA-*-")

        print("reading {} data...".format(self.data_source.dataname()))

        self.mimic = dataset(
            root=self.root,
            tables=_MIMIC_TABLES,
            code_mapping=_MIMIC_CODE_MAP,
            dev=dev
            )

        self.print_data_stats()

        return self.mimic

    def print_data_stats(self):
        print("---DATA STATS FOR {} DATA---".format(self.dataname))
        print("stat")
        self.mimic.stat()
        print("info")
        self.mimic.info()

    def get_task_names(self):
        return self.tasknames

    def get_tasklist(self):
        return self.tasklist

    def run_task(self, task):
        print("***run task: {}".format(task.name()))
        mimic_sample = self.mimic.set_task(task.task())
        print(mimic_sample[0])

        return mimic_sample

    def run_default_tasks(self):
        self.prepared_data = {}
        for task in self.tasklist:
            self.prepared_data[task.name()] = self.run_task(task)
        #for task in self.tasklist:
        #    self.prepared_data[task.name()] = self.run_task(task.task())

    def drug_task_data(self):
        self.run_default_tasks()
        return self.prepared_data

    def get_drug_task_results(self, name=_DRUG_REC_TN):
        return self.prepared_data[name]

    def create_dataloaders(self):
        self.dataloaders = {}

        if not(self.prepared_data):
            raise Exception("NO PREPARED DATA -- run tasks first")

        for tn,drug_task_data in self.prepared_data.items():
            train, val, test = get_dataloaders(drug_task_data)
            self.dataloaders[tn] = {
                    "train": train,
                    "val": val,
                    "test": test
                    }

        return self.dataloaders

# load data
def load_mimic_data(data_source=MIMIC4, dev=_DEV):
    dataset = data_source.dataset()
    dataroot = data_source.root()

    print("reading {} data...".format(data_source.dataname()))

    mimic = dataset(
            root=dataroot,
            tables=_MIMIC_TABLES,
            code_mapping=_MIMIC_CODE_MAP,
            dev=dev
            )

    print("stat")
    mimic.stat()
    print("info")
    mimic.info()

    return mimic

# data preparation
def prepare_drug_task_data(mimic, data_source=MIMIC4):
    print("drug task prepare")
    task = data_source.task()
    mimic_sample = mimic.set_task(task)
    print(mimic_sample[0])

    return mimic_sample

def prepare_flat_drug_data(mimic, data_source=MIMIC4):
    print("flat drug prepare")
    task = drug_recommendation_mimic4_flat

    mimic_sample = mimic.set_task(task)
    print(mimic_sample[0])
    return mimic_sample

def prepare_no_hist_drug_data(mimic, data_source=MIMIC4):
    print("no hist drug prepare")
    task = drug_recommendation_mimic4_no_hist

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

def oldmain():

    # load and prepare data
    mimic_data = load_mimic_data(mimic_dataset, dev=args.dev)
    drug_task_data = prepare_drug_task_data(mimic_data, mimic_dataset)
    drug_task_data_no_hist = prepare_no_hist_drug_data(mimic_data, mimic_dataset)


    #data = prepare_drug_task_data(mimic_dataset, dev=args.dev)
    #data = prepare_flat_drug_data(mimic_dataset, dev=args.dev)
    #sys.exit(0)

    # create the dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(drug_task_data)
    train_loader_no_hist, val_loader_no_hist, test_loader_no_hist = get_dataloaders(drug_task_data_no_hist)


    # train the models
    baseline_result = None
    baseline_result_no_hist = None
    gamenet_result = None
    gamenet_result_no_hist = None

    # baseline
    if args.baseline:
        print("---RETAIN TRAINING---")
        #retain, retain_trainer = train_retain(
        #        drug_task_data, train_loader, val_loader,
        #        epochs=args.epochs,
        #        decay_weight=args.decay,
        #        learning_rate=args.rate
        #        )
        #retain_no_hist, retain_trainer_no_hist = train_retain(
        #        drug_task_data_no_hist, train_loader_no_hist, val_loader_no_hist,
        #        epochs=args.epochs,
        #        decay_weight=args.decay,
        #        learning_rate=args.rate
        #        )
        print("---RETAIN EVALUATION---")
        baseline_result = evaluate_model(retain_trainer, test_loader)
        baseline_result_no_hist = evaluate_model(retain_trainer_no_hist, test_loader_no_hist)

    # gamenet
    if args.gamenet:
        print("---GAMENET TRAINING---")
        #gamenet, gn_trainer = train_gamenet(
        #        drug_task_data, train_loader, val_loader,
        #        epochs=args.epochs,
        #        decay_weight=args.decay,
        #        learning_rate=args.rate
        #        )
        #gamenet_no_hist, gn_trainer_no_hist = train_gamenet(
        #        drug_task_data_no_hist, train_loader_no_hist, val_loader_no_hist,
        #        epochs=args.epochs,
        #        decay_weight=args.decay,
        #        learning_rate=args.rate
        #        )
        print("---GAMENET EVALUATION---")
        #gn_result = evaluate_model(gn_trainer, test_loader)
        gamenet_result = evaluate_model(gn_trainer, test_loader)
        gamenet_result_no_hist = evaluate_model(gn_trainer_no_hist, test_loader_no_hist)

    print("---RESULTS---")
    print("---baseline---")
    print(baseline_result)
    print("---baseline no hist---")
    print(baseline_result_no_hist)
    print("---gamenet---")
    print(gamenet_result)
    print("---gamenet no hist---")
    print(gamenet_result_no_hist)


    #print("---GAMENET TRAINING---")
    #gamenet, gn_trainer = train_gamenet(data, train_loader, val_loader)
    #print("---RETAIN TRAINING---")
    #retain, retain_trainer = train_retain(data, train_loader, val_loader)
    ## evaluate model performance
    #print("---GAMENET EVALUATION---")
    ##gn_result = evaluate_model(gn_trainer, test_loader)
    #print("---RETAIN EVALUATION---")
    #retain_result = evaluate_model(retain_trainer, test_loader)

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
    parser.add_argument(
            "--dev", dest="dev",
            action="store_true", default=_DEV,
            help="read the mimic data in dev mode"
            )

    args = parser.parse_args()

    # choose dataset
    if args.mimicvers == 4:
        mimic_dataset = MIMIC4
    elif args.mimicvers == 3:
        mimic_dataset = MIMIC3

    mimic = MIMICWrapper(datasource=mimic_dataset, tasks=mimic_dataset.all_tasks())
    mimicdata = mimic.load_data(args.dev)
    #mimic.run_default_tasks()
    drug_task_data = mimic.drug_task_data()
    dataloaders = mimic.create_dataloaders()

    models = {}

    #baseline_result = None
    #gamenet_result = None

    retain = {}
    rtrainer = {}
    gamenet = {}
    gtrainer = {}

    baseline_result = {}
    gamenet_result = {}

    # baseline
    if args.baseline:
        print("---RETAIN TRAINING---")
        for taskname,dataloader in dataloaders.items():
            print("--training retain on {} data--".format(taskname))
            retain[taskname], rtrainer[taskname] = train_retain(
                drug_task_data[taskname],
                dataloader["train"], dataloader["val"],
                epochs=args.epochs,
                decay_weight=args.decay,
                learning_rate=args.rate,
                experiment="{}_task_{}".format(_RETAIN_EXP, taskname)
                )
        #retain_no_hist, retain_trainer_no_hist = train_retain(
        #        drug_task_data_no_hist, train_loader_no_hist, val_loader_no_hist,
        #        epochs=args.epochs,
        #        decay_weight=args.decay,
        #        learning_rate=args.rate
        #        )
        print("---RETAIN EVALUATION---")
        for taskname in mimic.get_task_names():
            print("--eval retain on {} data--".format(taskname))
            test_loader = dataloaders[taskname]["test"]
            baseline_result[taskname] = evaluate_model(rtrainer[taskname], test_loader)
            #baseline_result = evaluate_model(retain_trainer, test_loader)

    # gamenet
    if args.gamenet:
        print("---GAMENET TRAINING---")
        for taskname,dataloader in dataloaders.items():
            print("--training gamenet on {} data--".format(taskname))
            gamenet[taskname], gtrainer[taskname] = train_gamenet(
                drug_task_data[taskname],
                dataloader["train"], dataloader["val"],
                epochs=args.epochs,
                decay_weight=args.decay,
                learning_rate=args.rate,
                experiment="{}_task_{}".format(_GAMENET_EXP, taskname)
                )
        #gamenet_no_hist, gn_trainer_no_hist = train_gamenet(
        #        drug_task_data_no_hist, train_loader_no_hist, val_loader_no_hist,
        #        epochs=args.epochs,
        #        decay_weight=args.decay,
        #        learning_rate=args.rate
        #        )
        print("---GAMENET EVALUATION---")
        #gn_result = evaluate_model(gn_trainer, test_loader)
        for taskname in mimic.get_task_names():
            print("--eval gament on {} data--".format(taskname))
            test_loader = dataloaders[taskname]["test"]
            gamenet_result[taskname] = evaluate_model(gtrainer[taskname], test_loader)

    print("---RESULTS---")
    print("---baseline---")
    #print(baseline_result)
    for taskname in mimic.get_task_names():
        print("--result for task {}--".format(taskname))
        print(baseline_result[taskname])
    print("---gamenet---")
    for taskname in mimic.get_task_names():
        print("--result for task {}--".format(taskname))
        print(gamenet_result[taskname])

    print("EXECUTION FINISHED...")

