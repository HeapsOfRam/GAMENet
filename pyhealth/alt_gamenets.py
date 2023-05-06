import math
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.medcode import ATC
from pyhealth.models import BaseModel, GAMENetLayer, GAMENet
from pyhealth.models.utils import get_last_visit, batch_to_multihot

#class GAMENetNoProc(BaseModel):
class GAMENetNoProc(GAMENet):
    """GAMENet model.

    Paper: Junyuan Shang et al. GAMENet: Graph Augmented MEmory Networks for
    Recommending Medication Combination AAAI 2019.

    Note:
        This model is only for medication prediction which takes conditions
        as feature_keys, and drugs_all as label_key (i.e., both
        current and previous drugs). It only operates on the visit level.

    Note:
        This model only accepts ATC level 3 as medication codes.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        num_layers: the number of layers used in RNN. Default is 1.
        dropout: the dropout rate. Default is 0.5.
        **kwargs: other parameters for the GAMENet layer.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        **kwargs
    ):
        super(GAMENet, self).__init__(
            dataset=dataset,
            feature_keys=["conditions"],
            label_key="drugs_all",
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        ehr_adj = self.generate_ehr_adj()
        ddi_adj = self.generate_ddi_adj()

        self.cond_rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        #self.proc_rnn = nn.GRU(
        #    embedding_dim,
        #    hidden_dim,
        #    num_layers=num_layers,
        #    dropout=dropout if num_layers > 1 else 0,
        #    batch_first=True,
        #)
        self.query = nn.Sequential(
            nn.ReLU(),
            #nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # validate kwargs for GAMENet layer
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        if "ehr_adj" in kwargs:
            raise ValueError("ehr_adj is determined by the dataset")
        if "ddi_adj" in kwargs:
            raise ValueError("ddi_adj is determined by the dataset")
        self.gamenet = GAMENetLayer(
            hidden_size=hidden_dim,
            ehr_adj=ehr_adj,
            ddi_adj=ddi_adj,
            dropout=dropout,
            **kwargs,
        )

    def generate_ehr_adj(self) -> torch.tensor:
        """Generates the EHR graph adjacency matrix."""
        label_size = self.label_tokenizer.get_vocabulary_size()
        ehr_adj = torch.zeros((label_size, label_size))
        for sample in self.dataset:
            curr_drugs = sample["drugs_all"][-1]
            encoded_drugs = self.label_tokenizer.convert_tokens_to_indices(curr_drugs)
            for idx1, med1 in enumerate(encoded_drugs):
                for idx2, med2 in enumerate(encoded_drugs):
                    if idx1 >= idx2:
                        continue
                    ehr_adj[med1, med2] = 1
                    ehr_adj[med2, med1] = 1
        return ehr_adj

    def generate_ddi_adj(self) -> torch.tensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = torch.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        return ddi_adj

    def forward(
        self,
        conditions: List[List[List[str]]],
        drugs_all: List[List[List[str]]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            conditions: a nested list in three levels [patient, visit, condition].
            drugs_all: a nested list in three levels [patient, visit, drug].

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels] representing
                    the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels] representing
                    the ground truth of each drug.

        """
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)
        # (patient, visit, code)
        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        conditions = self.embeddings["conditions"](conditions)
        # (patient, visit, embedding_dim)
        conditions = torch.sum(conditions, dim=2)
        # (batch, visit, hidden_size)
        conditions, _ = self.cond_rnn(conditions)

        # (batch, visit, 2 * hidden_size)
        patient_representations = torch.cat([conditions], dim=-1)
        # (batch, visit, hidden_size)
        queries = self.query(patient_representations)

        label_size = self.label_tokenizer.get_vocabulary_size()
        drugs_all = self.label_tokenizer.batch_encode_3d(
            drugs_all, padding=(False, False), truncation=(True, False)
        )

        curr_drugs = [p[-1] for p in drugs_all]
        curr_drugs = batch_to_multihot(curr_drugs, label_size)
        curr_drugs = curr_drugs.to(self.device)

        prev_drugs = [p[:-1] for p in drugs_all]
        max_num_visit = max([len(p) for p in prev_drugs])
        prev_drugs = [p + [[]] * (max_num_visit - len(p)) for p in prev_drugs]
        prev_drugs = [batch_to_multihot(p, label_size) for p in prev_drugs]
        prev_drugs = torch.stack(prev_drugs, dim=0)
        prev_drugs = prev_drugs.to(self.device)

        # get mask
        mask = torch.sum(conditions, dim=2) != 0

        # process drugs
        loss, y_prob = self.gamenet(queries, prev_drugs, curr_drugs, mask)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": curr_drugs,
        }
