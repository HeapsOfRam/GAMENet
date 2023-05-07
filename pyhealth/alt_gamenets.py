import math
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn

#import numpy as np

from pyhealth.datasets import SampleDataset
from pyhealth.medcode import ATC
from pyhealth.models import BaseModel, GAMENetLayer, GAMENet
from pyhealth.models.gamenet import GCN, GCNLayer
from pyhealth.models.utils import get_last_visit, batch_to_multihot

class GAMENetLayerNoDM(GAMENetLayer):
    """GAMENet layer.
    Paper: Junyuan Shang et al. GAMENet: Graph Augmented MEmory Networks for
    Recommending Medication Combination AAAI 2019.
    This layer is used in the GAMENet model. But it can also be used as a
    standalone layer.
    Args:
        hidden_size: hidden feature size.
        ehr_adj: an adjacency tensor of shape [num_drugs, num_drugs].
        ddi_adj: an adjacency tensor of shape [num_drugs, num_drugs].
        dropout : the dropout rate. Default is 0.5.
    Examples:
        >>> from pyhealth.models import GAMENetLayer
        >>> queries = torch.randn(3, 5, 32) # [patient, visit, hidden_size]
        >>> prev_drugs = torch.randint(0, 2, (3, 4, 50)).float()
        >>> curr_drugs = torch.randint(0, 2, (3, 50)).float()
        >>> ehr_adj = torch.randint(0, 2, (50, 50)).float()
        >>> ddi_adj = torch.randint(0, 2, (50, 50)).float()
        >>> layer = GAMENetLayer(32, ehr_adj, ddi_adj)
        >>> loss, y_prob = layer(queries, prev_drugs, curr_drugs)
        >>> loss.shape
        torch.Size([])
        >>> y_prob.shape
        torch.Size([3, 50])
    """

    def __init__(
        self,
        hidden_size: int,
        ehr_adj: torch.tensor,
        ddi_adj: torch.tensor,
        dropout: float = 0.5,
    ):
        super(GAMENetLayer, self).__init__()
        self.hidden_size = hidden_size
        self.ehr_adj = ehr_adj
        self.ddi_adj = ddi_adj

        num_labels = ehr_adj.shape[0]
        self.ehr_gcn = GCN(adj=ehr_adj, hidden_size=hidden_size, dropout=dropout)
        self.ddi_gcn = GCN(adj=ddi_adj, hidden_size=hidden_size, dropout=dropout)
        self.beta = nn.Parameter(torch.FloatTensor(1))
        #self.fc = nn.Linear(3 * hidden_size, num_labels)
        self.fc = nn.Linear(2 * hidden_size, num_labels)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        queries: torch.tensor,
        #prev_drugs: torch.tensor,
        curr_drugs: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.
        Args:
            queries: query tensor of shape [patient, visit, hidden_size].
            prev_drugs: multihot tensor indicating drug usage in all previous
                visits of shape [patient, visit - 1, num_drugs].
            curr_drugs: multihot tensor indicating drug usage in the current
                visit of shape [patient, num_drugs].
            mask: an optional mask tensor of shape [patient, visit] where 1
                indicates valid visits and 0 indicates invalid visits.
        Returns:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor of shape [patient, num_labels] representing
                the probability of each drug.
        """
        if mask is None:
            mask = torch.ones_like(queries[:, :, 0])

        """I: Input memory representation"""
        query = get_last_visit(queries, mask)

        """G: Generalization"""
        # memory bank
        MB = self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.beta)

        # dynamic memory
        #DM_keys = queries[:, :-1, :]
        #DM_values = prev_drugs
        ##DM_keys = np.ones(DM_keys.shape)
        ##DM_values = np.ones(DM_values.shape)
        ##DM_keys = torch.ones_like(DM_keys)
        ##DM_values = torch.ones_like(DM_values)
        ##DM_keys = torch.zeros_like(DM_keys)
        ##DM_values = torch.zeros_like(DM_values)

        """O: Output memory representation"""
        #a_c = torch.softmax(torch.mm(query, MB.t()), dim=-1)
        a_c = torch.softmax(torch.mm(query, MB.t()), dim=-1)
        o_b = torch.mm(a_c, MB)

        #a_s = torch.softmax(torch.einsum("bd,bvd->bv", query, DM_keys), dim=1)
        ##a_s2 = torch.softmax(query, dim=1)
        ##print(a_s.shape) # 64, 0
        ##print(MB.shape) # 160, 128
        ##print(DM_keys.shape) # 64, 0, 128
        ##print(DM_values.shape) # 64, 0, 160
        #a_m = torch.einsum("bv,bvz->bz", a_s, DM_values.float())
        ##print(a_m.shape)
        #o_d = torch.mm(a_m, MB)
        ##o_d = torch.mm(a_s, MB)

        """R: Response"""
        #memory_output = torch.cat([query, o_b, o_d], dim=-1)
        memory_output = torch.cat([query, o_b], dim=-1)
        logits = self.fc(memory_output)

        loss = self.bce_loss_fn(logits, curr_drugs)
        y_prob = torch.sigmoid(logits)

        return loss, y_prob

class GAMENetNoHist(GAMENet):
    """GAMENet model.
    Paper: Junyuan Shang et al. GAMENet: Graph Augmented MEmory Networks for
    Recommending Medication Combination AAAI 2019.
    Note:
        This model is only for medication prediction which takes conditions
        and procedures as feature_keys, and drugs_all as label_key (i.e., both
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
            feature_keys=["conditions", "procedures"],
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
        self.proc_rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # validate kwargs for GAMENet layer
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        if "ehr_adj" in kwargs:
            raise ValueError("ehr_adj is determined by the dataset")
        if "ddi_adj" in kwargs:
            raise ValueError("ddi_adj is determined by the dataset")
        #self.gamenet = GAMENetLayer(
        self.gamenet = GAMENetLayerNoDM(
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
        procedures: List[List[List[str]]],
        drugs_all: List[List[List[str]]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.
        Args:
            conditions: a nested list in three levels [patient, visit, condition].
            procedures: a nested list in three levels [patient, visit, procedure].
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

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        # (patient, visit, code)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        procedures = self.embeddings["procedures"](procedures)
        # (patient, visit, embedding_dim)
        procedures = torch.sum(procedures, dim=2)
        # (batch, visit, hidden_size)
        procedures, _ = self.proc_rnn(procedures)

        # (batch, visit, 2 * hidden_size)
        patient_representations = torch.cat([conditions, procedures], dim=-1)
        # (batch, visit, hidden_size)
        queries = self.query(patient_representations)

        label_size = self.label_tokenizer.get_vocabulary_size()
        drugs_all = self.label_tokenizer.batch_encode_3d(
            drugs_all, padding=(False, False), truncation=(True, False)
        )

        curr_drugs = [p[-1] for p in drugs_all]
        curr_drugs = batch_to_multihot(curr_drugs, label_size)
        curr_drugs = curr_drugs.to(self.device)

        #prev_drugs = [p[:-1] for p in drugs_all]
        #max_num_visit = max([len(p) for p in prev_drugs])
        #prev_drugs = [p + [[]] * (max_num_visit - len(p)) for p in prev_drugs]
        #prev_drugs = [batch_to_multihot(p, label_size) for p in prev_drugs]
        #prev_drugs = torch.stack(prev_drugs, dim=0)
        #prev_drugs = prev_drugs.to(self.device)

        # get mask
        mask = torch.sum(conditions, dim=2) != 0

        # process drugs
        #loss, y_prob = self.gamenet(queries, prev_drugs, curr_drugs, mask)
        loss, y_prob = self.gamenet(queries, curr_drugs, mask)


        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": curr_drugs,
        }

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
