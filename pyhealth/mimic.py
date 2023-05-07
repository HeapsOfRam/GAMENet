# import our constants
from constants import (
    DEV, DATA_DIR,
    MIMIC_TABLES, MIMIC_CODE_MAP,
    DRUG_REC_TN,
    TN, TTASK,
    MIMIC4_TASKS, MIMIC3_TASKS
)

# import pyhealth datasets and tasks
from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset

# import dataloader related functions
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader

# data loaders for training and evaluation
def get_dataloaders(mimic_sample):
    train_dataset, val_dataset, test_dataset = split_by_patient(mimic_sample, [0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

# define data classes
## these classes wrap some pyhealth logic that is similar between the two datasets
## enables easy switching between MIMIC4/MIMIC3 data

# dataclass for MIMIC3
class MIMIC3():
    def dataset():
        return MIMIC3Dataset

    def root():
        return DATA_DIR.format(MIMIC3.dataname())

    def dataname():
        return "mimic3"

    def all_tasks():
        return MIMIC3_TASKS

# dataclass for MIMIC4
class MIMIC4():
    def dataset():
        return MIMIC4Dataset

    def root():
        return DATA_DIR.format(MIMIC4.dataname())

    def dataname():
        return "mimic4"

    def all_tasks():
        return MIMIC4_TASKS

class MIMICWrapper():
    def __init__(self, datasource=MIMIC4, tasks=MIMIC4.all_tasks()):
        self.data_source = datasource
        self.root = datasource.root()
        self.dataname = datasource.dataname()
        self.tasks = tasks

        self.tasklist = []
        self.prepared_data = {}

        for taskname,task in self.tasks.items():
            self.tasklist.append({TTASK: task, TN: taskname})

    def load_data(self, dev=DEV):
        dataset = self.data_source.dataset()
        dataroot = self.data_source.root()

        if dev:
            print("-*-READING DEV DATA-*-")

        print("reading {} data...".format(self.data_source.dataname()))

        self.mimic = dataset(
            root=self.root,
            tables=MIMIC_TABLES,
            code_mapping=MIMIC_CODE_MAP,
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
        return self.tasks.keys()

    def get_tasklist(self):
        return self.tasklist

    def run_task(self, task):
        print("***run task: {}".format(task[TN]))
        mimic_sample = self.mimic.set_task(task[TTASK])
        print(mimic_sample[0])

        return mimic_sample

    def run_tasks_from_list(self, tasklist):
        self.prepared_data = {}
        for task in tasklist:
            self.prepared_data[task[TN]] = self.run_task(task)

    def drug_task_data(self):
        #self.run_default_tasks()
        self.run_tasks_from_list(self.tasklist)
        return self.prepared_data

    def get_drug_task_results(self, name=DRUG_REC_TN):
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
