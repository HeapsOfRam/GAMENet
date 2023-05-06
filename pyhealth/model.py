import numpy as np

from alt_gamenets import GAMENetNoProc

from pyhealth.models import RETAIN, GAMENet
from pyhealth.trainer import Trainer

# set the device to be the cuda gpu
_DEVICE = "cuda"

# hyperparameters for training
_EPOCHS = 20
#_LR = 0.0002
_LR = 1e-3
#_DECAY_WEIGHT = 0.85
_DECAY_WEIGHT = 1e-5

_THRESH = 0.5

# metrics to track from training/evaluation
_METRICS = [
        "jaccard_samples", "accuracy", "hamming_loss",
        "precision_samples", "recall_samples",
        "pr_auc_samples", "f1_samples"
]

_REQUIRED_DL_KEYS = {'train', 'val', 'test'}

_DEFAULT_EXPERIMENT = "drug_recommendation"

_BEST_MODEL_PATH = "output/{}/best.ckpt"

class ModelWrapper():
    def __init__(self, mimic_sample, model=GAMENet, experiment=_DEFAULT_EXPERIMENT, device=_DEVICE, metrics=_METRICS, feature_keys = ["conditions", "procedures"]):
        self.model_type = model
        self.experiment = experiment

        if self.model_type == GAMENet:
            print("making gamenet model")
            self.model = model(mimic_sample)
        elif self.model_type == GAMENetNoProc:
            print("making gamenet model without procedures...")
            self.model = model(mimic_sample)
        elif self.model_type == RETAIN:
            print("making retain model")
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

    def get_model(self):
        return self.model

    def get_trainer(self):
        return self.trainer

    def get_model_type(self):
        return self.model_type

    def get_experiment_name(self):
        return self.experiment

    def load_best_model(self):
        model_path = _BEST_MODEL_PATH.format(self.experiment)
        print("loading model from path... {}".format(model_path))
        self.trainer.load_ckpt(model_path)

    def train_model(self, train_loader, val_loader, decay_weight=_DECAY_WEIGHT, learning_rate=_LR, epochs=_EPOCHS):
        self.trainer.train(
            train_dataloader = train_loader,
            val_dataloader = val_loader,
            epochs = epochs,
            monitor = "accuracy",
            monitor_criterion = "max",
            weight_decay = decay_weight,
            optimizer_params = {"lr": learning_rate}
        )

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
        _,y_hat,_ = self.inference(test)
        y_hat = np.where(y_hat >= _THRESH, 1, 0)

        num_drugs = 0

        for visit in y_hat:
            num_drugs = num_drugs + np.sum(visit)

        return (num_drugs * 1.0) / len(y_hat)

    def calc_ddi_rate(self, test, ddi_mat):
        _,y_hat,_ = self.inference(test)
        y_hat = np.where(y_hat >= _THRESH, 1, 0)

        all_cnt = 0
        ddi_cnt = 0

        for visit in y_hat:
            for i, med_i in enumerate(visit):
                for j, med_j in enumerate(visit):
                    if j > i and med_i == 1 and med_j == 1:
                        all_cnt = all_cnt + 1

                        if ddi_mat[i, j] == 1 or ddi_mat[j, i] == 1:
                            ddi_cnt = ddi_cnt + 1

        if all_cnt == 0:
            return 0

        return (ddi_cnt * 1.0) / all_cnt
