# import regular python libraries
import argparse
import sys
import numpy as np

# import utilities and variables from the mimic abstractions
from mimic import MIMIC3, MIMIC4, MIMICWrapper

# import model wrapper and variables
from model import ModelWrapper

# import alternative gamenet models
from alt_gamenets import GAMENetNoProc

# import our constants
from constants import (
    DEV,
    EPOCHS, LR, DECAY_WEIGHT,
    DRUG_REC_TN, ALL_TASKS,
    GN_KEY, RT_KEY,
    MODEL_TYPES_PER_TASK, RETAIN_FEATS_PER_TASK,
    GAMENET_EXP, RETAIN_EXP,
    SCORE_KEY, DPV_KEY, DDI_RATE_KEY,
    BASE_DDI_RATE
)

# pyhealth imports
import pyhealth
from pyhealth.models import RETAIN, GAMENet
from pyhealth.trainer import Trainer

def print_ddi_results(model, result):
    exp = model.get_experiment_name()
    print("{} model recommended an average of {} drugs / visit".format(exp, result[DPV_KEY]))
    print("{} model ddi rate: {}".format(exp, result[DDI_RATE_KEY]))
    print("{} model delta ddi rate: {}".format(exp, result[DDI_RATE_KEY] - BASE_DDI_RATE))
    print("\n{}\n".format("-" * 10))

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
        tasklist = args.tasks

        if not(tasklist):
            print("--NO TASK PROVIDED, RUNNING DEFAULT!--")
            tasklist = [DRUG_REC_TN]

        for task in tasklist:
            print(task)
            tasks[task] = mimic_dataset.all_tasks()[task]


    print("---will run tasks {}---".format(tasks.keys()))

    # create mimicwrapper object
    # then, load the data and run the desired tasks on it
    # finally, create the dataloaders we need
    mimic = MIMICWrapper(datasource=mimic_dataset, tasks=tasks)
    mimicdata = mimic.load_data(args.dev)
    drug_task_data = mimic.drug_task_data()
    dataloaders = mimic.create_dataloaders()

    # create dictionaries for our resulting models, trainers, and results
    models = {}

    retain = {}
    gamenet = {}

    baseline_result = {}
    gamenet_result = {}

    ddi_mats = {}

    # create ddi matrices for calculating ddi metrics
    # also, create our runtime dicts
    for taskname in mimic.get_task_names():
        model_type = MODEL_TYPES_PER_TASK[taskname][GN_KEY]
        ddi_mats[taskname] = model_type(drug_task_data[taskname]).generate_ddi_adj()

    # baseline
    if args.baseline:
        print("---RETAIN TRAINING---")
        for taskname,dataloader in dataloaders.items():
            print("--training retain on {} data--".format(taskname))
            # create and train retain model
            retain[taskname] = ModelWrapper(
                    drug_task_data[taskname],
                    model=MODEL_TYPES_PER_TASK[taskname][RT_KEY],
                    feature_keys=RETAIN_FEATS_PER_TASK[taskname],
                    experiment="{}_task_{}".format(RETAIN_EXP, taskname)
            )
            retain[taskname].train_model(
                    dataloader["train"], dataloader["val"],
                    decay_weight=args.decay,
                    learning_rate=args.rate,
                    epochs=args.epochs
            )

        print("---RETAIN EVALUATION---")
        for taskname in mimic.get_task_names():
            print("--eval retain on {} data--".format(taskname))
            test_loader = dataloaders[taskname]["test"]
            baseline_result[taskname] = {}
            baseline_result[taskname][SCORE_KEY] = retain[taskname].evaluate_model(test_loader)
            baseline_result[taskname][DPV_KEY] = retain[taskname].calc_avg_drugs_per_visit(test_loader)
            baseline_result[taskname][DDI_RATE_KEY] = retain[taskname].calc_ddi_rate(
                test_loader, ddi_mats[taskname]
            )
    else:
        print("---SKIPPING BASELINE---")

    # gamenet
    if args.gamenet:
        print("---GAMENET TRAINING---")
        for taskname,dataloader in dataloaders.items():
            print("--training gamenet on {} data--".format(taskname))
            # create and train gamenet model
            gamenet[taskname] = ModelWrapper(
                drug_task_data[taskname],
                model=MODEL_TYPES_PER_TASK[taskname][GN_KEY],
                experiment="{}_task_{}".format(GAMENET_EXP, taskname)
            )
            gamenet[taskname].train_model(
                dataloader["train"], dataloader["val"],
                decay_weight = args.decay,
                learning_rate = args.rate,
                epochs=args.epochs
            )

        print("---GAMENET EVALUATION---")
        for taskname in mimic.get_task_names():
            print("--eval gamenet on {} data--".format(taskname))
            test_loader = dataloaders[taskname]["test"]
            gamenet_result[taskname] = {}
            gamenet_result[taskname][SCORE_KEY] = gamenet[taskname].evaluate_model(test_loader)
            gamenet_result[taskname][DPV_KEY] = gamenet[taskname].calc_avg_drugs_per_visit(test_loader)
            gamenet_result[taskname][DDI_RATE_KEY] = gamenet[taskname].calc_ddi_rate(
                test_loader, ddi_mats[taskname]
            )
    else:
        print("---SKIPPING GAMENET---")

    # print results
    print("\n---RESULTS---\n\n")
    if args.baseline:
        print("---baseline---\n")
        for taskname in mimic.get_task_names():
            print("--result for experiment {}--".format(retain[taskname].get_experiment_name()))
            print(baseline_result[taskname][SCORE_KEY])

            test_loader = dataloaders[taskname]["test"]
            print("retain training took...{} seconds".format(retain[taskname].get_train_time()))
            print_ddi_results(retain[taskname], baseline_result[taskname])
    else:
        print("...BASELINE SKIPPED, NO BASELINE RESULTS...")

    print("{}\n".format("*" * 10))

    if args.gamenet:
        print("---gamenet---\n")
        for taskname in mimic.get_task_names():
            print("--result for experiment {}--".format(gamenet[taskname].get_experiment_name()))
            print(gamenet_result[taskname][SCORE_KEY])

            test_loader = dataloaders[taskname]["test"]
            print("gamenet training took...{} seconds".format(gamenet[taskname].get_train_time()))
            print_ddi_results(gamenet[taskname], gamenet_result[taskname])
    else:
        print("...GAMENET SKIPPED, NO GAMENET RESULTS...")

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
            default=EPOCHS, type=int,
            help="how many epochs to run the training for"
            )
    parser.add_argument(
            "-d", "--decay-weight", dest="decay",
            default=DECAY_WEIGHT, type=float,
            help="value for the decay weight hyperparameter"
            )
    parser.add_argument(
            "-l", "--learning-rate", dest="rate",
            default=LR, type=float,
            help="value for the learning weight hyperparameter"
            )
    parser.add_argument(
            "-t", "--task", dest="tasks",
            action="extend", nargs="+", type=str,
            choices=ALL_TASKS,
            #default=[DRUG_REC_TN],
            default=[],
            help="add a task to the list of tasks to be evaluated; options are {}".format(ALL_TASKS)
            )
    parser.add_argument(
            "-a", "--all-tasks", dest="all_tasks",
            action="store_true", default=False,
            help="will evaluate all the possible tasks on the dataset; supersedes -t argument"
            )
    parser.add_argument(
            "--dev", dest="dev",
            action="store_true", default=DEV,
            help="read the mimic data in dev mode"
            )

    args = parser.parse_args()
    main(args)
