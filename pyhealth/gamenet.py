# import regular python libraries
import argparse
import sys

# import utilities and variables from the mimic abstractions
from mimic import *
from mimic import _DEV, _DRUG_REC_TN, _NO_HIST_TN, _ALL_TASKS

from model import ModelWrapper
from model import _DEVICE, _EPOCHS, _LR, _DECAY_WEIGHT

# pyhealth imports
import pyhealth
from pyhealth.models import RETAIN, GAMENet
from pyhealth.trainer import Trainer

# experiment names
_GAMENET_EXP = "drug_recommendation_gamenet"
_RETAIN_EXP = "drug_recommendation_retain"

def main(args):
    # choose dataset
    if args.mimicvers == 4:
        mimic_dataset = MIMIC4
    elif args.mimicvers == 3:
        mimic_dataset = MIMIC3

    # create the task list we want to run
    tasks = {}
    if args.all_tasks:
        print("---RUNNING ALL TASKS!!!---")
        tasks = mimic_dataset.all_tasks()
    else:
        for task in args.tasks:
            print(task)
            tasks[task] = mimic_dataset.all_tasks()[task]

    print("---will run tasks {}---".format(tasks.keys()))

    # create mimicwrapper object
    # then, load the data and run the desired tasks on it
    # finally, create the dataloaders we need
    #mimic = MIMICWrapper(datasource=mimic_dataset, tasks=mimic_dataset.all_tasks())
    mimic = MIMICWrapper(datasource=mimic_dataset, tasks=tasks)
    mimicdata = mimic.load_data(args.dev)
    drug_task_data = mimic.drug_task_data()
    dataloaders = mimic.create_dataloaders()

    # create dictionaries for our resulting models, trainers, and results
    models = {}

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
            retain[taskname] = ModelWrapper(
                    drug_task_data[taskname],
                    model=RETAIN,
                    experiment="{}_task_{}".format(_RETAIN_EXP, taskname)
            )

            #rtrainer[taskname] = model.train_model(
            retain[taskname].train_model(
                    dataloader["train"], dataloader["val"],
                    decay_weight=args.decay,
                    learning_rate=args.rate,
                    epochs=args.epochs
            )
            #retain[taskname], rtrainer[taskname] = train_retain(
            #    drug_task_data[taskname],
            #    dataloader["train"], dataloader["val"],
            #    epochs=args.epochs,
            #    decay_weight=args.decay,
            #    learning_rate=args.rate,
            #    experiment="{}_task_{}".format(_RETAIN_EXP, taskname)
            #    )
        print("---RETAIN EVALUATION---")
        for taskname in mimic.get_task_names():
            print("--eval retain on {} data--".format(taskname))
            test_loader = dataloaders[taskname]["test"]
            #baseline_result[taskname] = evaluate_model(rtrainer[taskname], test_loader)
            baseline_result[taskname] = retain[taskname].evaluate_model(test_loader)


    # gamenet
    if args.gamenet:
        print("---GAMENET TRAINING---")
        for taskname,dataloader in dataloaders.items():
            print("--training gamenet on {} data--".format(taskname))
            gamenet[taskname] = ModelWrapper(
                drug_task_data[taskname],
                model=GAMENet,
                experiment="{}_task_{}".format(_GAMENET_EXP, taskname)
            )
            #gtrainer[taskname] = model.train_model(
            gamenet[taskname].train_model(
                dataloader["train"], dataloader["val"],
                decay_weight = args.decay,
                learning_rate = args.rate,
                epochs=args.epochs
            )
            #gamenet[taskname], gtrainer[taskname] = train_gamenet(
            #    drug_task_data[taskname],
            #    dataloader["train"], dataloader["val"],
            #    epochs=args.epochs,
            #    decay_weight=args.decay,
            #    learning_rate=args.rate,
            #    experiment="{}_task_{}".format(_GAMENET_EXP, taskname)
            #    )
        print("---GAMENET EVALUATION---")
        for taskname in mimic.get_task_names():
            print("--eval gament on {} data--".format(taskname))
            test_loader = dataloaders[taskname]["test"]
            #gamenet_result[taskname] = evaluate_model(gtrainer[taskname], test_loader)
            gamenet_result[taskname] = gamenet[taskname].evaluate_model(test_loader)


    # print results
    print("---RESULTS---")
    print("---baseline---")
    for taskname in mimic.get_task_names():
        print("--result for task {}--".format(taskname))
        print(baseline_result[taskname])
    print("---gamenet---")
    for taskname in mimic.get_task_names():
        print("--result for task {}--".format(taskname))
        print(gamenet_result[taskname])

    print("EXECUTION FINISHED...")


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
            "-t", "--task", dest="tasks",
            action="extend", nargs="+", type=str,
            choices=_ALL_TASKS,
            default=[_DRUG_REC_TN],
            help="add a task to the list of tasks to be evaluated; options are {}".format(_ALL_TASKS)
            )
    parser.add_argument(
            "-a", "--all-tasks", dest="all_tasks",
            action="store_true", default=False,
            help="will evaluate all the possible tasks on the dataset; supersedes -t argument"
            )
    parser.add_argument(
            "--dev", dest="dev",
            action="store_true", default=_DEV,
            help="read the mimic data in dev mode"
            )

    args = parser.parse_args()
    main(args)
