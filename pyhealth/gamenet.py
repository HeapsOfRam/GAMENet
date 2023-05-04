# import regular python libraries
import argparse
import sys
#import torch
import numpy as np

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

def avg_drugs_per_visit(y_hat):
    num_drugs = []
    #num_drugs = 0
    #num_visits = 0
    #print(y_hat)
    #print(y_hat.shape)

    for visit in y_hat:
        #print(visit)
        num_drugs.append(np.sum(visit))
        #print(visit)
        #num_drugs = num_drugs + np.sum(visit)
        #num_visits = num_visits + 1

    return (sum(num_drugs) * 1.0) / len(num_drugs)
    #return (num_drugs * 1.0) / (num_visits)

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
    mimic = MIMICWrapper(datasource=mimic_dataset, tasks=tasks)
    mimicdata = mimic.load_data(args.dev)
    drug_task_data = mimic.drug_task_data()
    dataloaders = mimic.create_dataloaders()

    # create dictionaries for our resulting models, trainers, and results
    models = {}

    retain = {}
    gamenet = {}
    rtrainer = {}
    gtrainer = {}
    radpv = {}
    gadpv = {}

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

            rtrainer[taskname] = retain[taskname].train_model(
                    dataloader["train"], dataloader["val"],
                    decay_weight=args.decay,
                    learning_rate=args.rate,
                    epochs=args.epochs
            )
        print("---RETAIN EVALUATION---")
        for taskname in mimic.get_task_names():
            print("--eval retain on {} data--".format(taskname))
            test_loader = dataloaders[taskname]["test"]
            baseline_result[taskname] = retain[taskname].evaluate_model(test_loader)
            #adpv = avg_drugs_per_visit(retain[taskname].inference(test_loader)[0])
            #radpv[taskname] = avg_drugs_per_visit(rtrainer[taskname].inference(test_loader)[0])
            #print("recommended an average of {} drugs / visit".format(radpv[taskname]))
    else:
        print("---SKIPPING BASELINE---")

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
            gtrainer[taskname] = gamenet[taskname].train_model(
                dataloader["train"], dataloader["val"],
                decay_weight = args.decay,
                learning_rate = args.rate,
                epochs=args.epochs
            )
        print("---GAMENET EVALUATION---")
        for taskname in mimic.get_task_names():
            print("--eval gament on {} data--".format(taskname))
            test_loader = dataloaders[taskname]["test"]
            gamenet_result[taskname] = gamenet[taskname].evaluate_model(test_loader)
            #gadpv[taskname] = avg_drugs_per_visit(gtrainer[taskname].inference(test_loader)[0])
            #print("recommended an average of {} drugs / visit".format(gadpv[taskname]))
    else:
        print("---SKIPPING GAMENET---")

    # print results
    print("---RESULTS---")
    if args.baseline:
        print("---baseline---")
        for taskname in mimic.get_task_names():
            #print("--result for task {}--".format(taskname))

            print("--result for experiment {}--".format(retain[taskname].get_experiment_name()))

            print(baseline_result[taskname])

            test_loader = dataloaders[taskname]["test"]

            ilabels, iresult, iloss = retain[taskname].inference(test_loader)
            #drug_recs = iresult[iresult >= 0.5]
            drug_recs = np.where(iresult >= 0.5, 1, 0)

            adpv = avg_drugs_per_visit(drug_recs)
            print("retain recommended an average of {} drugs / visit".format(adpv))


            #adpv = avg_drugs_per_visit(retain[taskname].inference(test_loader)[0])
            #adpv = avg_drugs_per_visit(rtrainer[taskname].inference(test_loader)[0])
            #print("recommended an average of {} drugs / visit".format(adpv))
            #print("retain recommended an average of {} drugs / visit".format(radpv[taskname]))
    else:
        print("...BASELINE SKIPPED, NO BASELINE RESULTS...")
    if args.gamenet:
        print("---gamenet---")
        for taskname in mimic.get_task_names():
            #print("--result for task {}--".format(taskname))
            print("--result for experiment {}--".format(gamenet[taskname].get_experiment_name()))
            print(gamenet_result[taskname])

            test_loader = dataloaders[taskname]["test"]

            ilabels, iresult, iloss = gamenet[taskname].inference(test_loader)
            #drug_recs = iresult[iresult >= 0.5]
            drug_recs = np.where(iresult >= 0.5, 1, 0)

            #print(drug_recs)
            #print(drug_recs.shape)
            #print(iresult.shape)

            adpv = avg_drugs_per_visit(drug_recs)
            print("gamenet recommended an average of {} drugs / visit".format(adpv))


            #adpv = avg_drugs_per_visit(gamenet[taskname].inference(test_loader)[0])
            #adpv = avg_drugs_per_visit(gtrainer[taskname].inference(test_loader)[0])
            #print("recommended an average of {} drugs / visit".format(adpv))
            #print("gamenet recommended an average of {} drugs / visit".format(gadpv[taskname]))

            #print("this should be false:")
            ##print(gtrainer[taskname] == rtrainer[taskname])
            #print(np.array_equal(gamenet[taskname].inference(test_loader)[0], retain[taskname].inference(test_loader)[0]))
            #print(np.array_equal(gamenet[taskname].inference(test_loader)[1], retain[taskname].inference(test_loader)[1]))

            #print(gamenet[taskname].inference(test_loader))

            #print(gamenet[taskname].get_model_type())
            #print(retain[taskname].get_model_type())
            #print(gamenet[taskname].get_model())
            #print(retain[taskname].get_model())
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
