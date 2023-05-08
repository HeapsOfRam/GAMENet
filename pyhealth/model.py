import numpy as np
import datetime

from alt_gamenets import GAMENetNoHist, GAMENetNoProc
from constants import (
    DEVICE,
    EPOCHS, LR, DECAY_WEIGHT, THRESH,
    METRICS,
    DEFAULT_EXPERIMENT, BEST_MODEL_PATH
)

from pyhealth.models import RETAIN, GAMENet
from pyhealth.trainer import Trainer

class ModelWrapper():
    def __init__(
            self,
            mimic_sample,
            model=GAMENet, experiment=DEFAULT_EXPERIMENT, device=DEVICE,
            metrics=METRICS,
            feature_keys = ["conditions", "procedures"]
        ):
        # which model variant we are using: retain or gamenet, which ablations
        self.model_type = model
        # experiment name for output
        self.experiment = experiment

        if issubclass(self.model_type, GAMENet):
            print("making gamenet model...{}".format(self.model_type))
            self.model = model(mimic_sample)
        elif issubclass(self.model_type, RETAIN):
            print("making retain model...{}".format(self.model_type))
            self.model = model(
                mimic_sample,
                feature_keys = feature_keys,
                label_key = "drugs",
                mode = "multilabel"
            )
        else:
            raise Exception("!!! ERROR: please send in either GAMENet or RETAIN class as the model !!!")

        self.trainer = Trainer(
            model = self.model,
            metrics = metrics,
            device = device,
            exp_name = self.experiment
        )

        self.train_time = None

    def get_model(self):
        return self.model

    def get_trainer(self):
        return self.trainer

    def get_model_type(self):
        return self.model_type

    def get_experiment_name(self):
        return self.experiment

    def get_train_time(self):
        return self.train_time

    def load_best_model(self):
        model_path = BEST_MODEL_PATH.format(self.experiment)
        print("loading model from path... {}".format(model_path))
        self.trainer.load_ckpt(model_path)

    def train_model(
            self,
            train_loader, val_loader,
            decay_weight=DECAY_WEIGHT, learning_rate=LR, epochs=EPOCHS
        ):
        print("training model for experiment...{}".format(self.experiment))

        # start timing so we can see how long training takes
        train_start = datetime.datetime.now()

        # train the given model
        self.trainer.train(
            train_dataloader = train_loader,
            val_dataloader = val_loader,
            epochs = epochs,
            monitor = "accuracy",
            monitor_criterion = "max",
            weight_decay = decay_weight,
            optimizer_params = {"lr": learning_rate}
        )

        # get the total training time in seconds
        self.train_time = (datetime.datetime.now() - train_start).total_seconds()

        # return the trainer in case it is needed
        return self.trainer

    def evaluate_model(self, test_loader, load=False):
        if load:
            self.load_best_model
        result = self.trainer.evaluate(test_loader)
        print(result)
        return result

    def inference(self, x, load=False):
        if load:
            self.load_best_model()
        return self.trainer.inference(x)

    def calc_avg_drugs_per_visit(self, test):
        # perform inference
        _,y_hat,_ = self.inference(test)
        # indicate positive class when above the threshold
        y_hat = np.where(y_hat >= THRESH, 1, 0)

        num_drugs = 0

        # iterate over every visit in the predicted set
        for visit in y_hat:
            # count the number of drugs in the visit
            num_drugs = num_drugs + np.sum(visit)

        # take the average over the number of visits
        return (num_drugs * 1.0) / len(y_hat)

    def calc_ddi_rate(self, test, ddi_mat):
        # perform inference
        _,y_hat,_ = self.inference(test)
        # indicate positive class when above the threshold
        y_hat = np.where(y_hat >= THRESH, 1, 0)

        all_cnt = 0
        ddi_cnt = 0

        # iterate over every visit in the predicted set
        for visit in y_hat:
            # iterate over every medication combination
            for i, med_i in enumerate(visit):
                for j, med_j in enumerate(visit):
                    # only count combinations that appear in the visit
                    # also, don't double count combinations
                    if j > i and med_i == 1 and med_j == 1:
                        # count all the drug combinations over every visit
                        all_cnt = all_cnt + 1

                        # count if the combination has an edge on the ddi graph
                        if ddi_mat[i, j] == 1 or ddi_mat[j, i] == 1:
                            ddi_cnt = ddi_cnt + 1

        # edge case if there are no visits
        if all_cnt == 0:
            return 0

        # return the count of combinations with an edge on ddi graph / all combinations
        return (ddi_cnt * 1.0) / all_cnt
