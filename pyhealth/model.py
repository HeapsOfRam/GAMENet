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

# metrics to track from training/evaluation
_METRICS = [
        "jaccard_samples", "accuracy", "hamming_loss",
        "precision_samples", "recall_samples",
        "pr_auc_samples", "f1_samples"
]

_REQUIRED_DL_KEYS = {'train', 'val', 'test'}

_DEFAULT_EXPERIMENT = "drug_recommendation"

class ModelWrapper():
    def __init__(self, mimic_sample, model=GAMENet, experiment=_DEFAULT_EXPERIMENT, device=_DEVICE, metrics=_METRICS):
        self.model_type = model

        if self.model_type == GAMENet:
            self.model = model(mimic_sample)
        elif self.model_type == RETAIN:
            self.model = model(
                mimic_sample,
                feature_keys = ["conditions", "procedures"],
                label_key = "drugs",
                mode = "multilabel"
            )
        else:
            raise Exception("!!! ERROR: please send in either GAMENet or RETAIN class as the model !!!")


        #if not(_REQUIRED_DL_KEYS == dataloaders.keys()):
        #    raise Exception("!!! ERROR: please make sure the dataloaders dict has train, val, and test keys !!!")


        #self.train_loader = dataloaders["train"]
        #self.val_loader = dataloaders["val"]
        #self.test_loader = dataloaders["test"]

        self.trainer = Trainer(
            model = self.model,
            metrics = metrics,
            device = device,
            exp_name = experiment
        )

    def get_model(self):
        return self.model

    def get_trainer(self):
        return self.trainer

    def get_model_type(self):
        return self.model_type

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

    def evaluate_model(self, test_loader):
        result = self.trainer.evaluate(test_loader)
        print(result)
        return result



