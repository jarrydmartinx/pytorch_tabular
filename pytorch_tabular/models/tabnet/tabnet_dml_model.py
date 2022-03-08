# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""TabNet Model"""
import logging
from tkinter import Y
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_tabular.models.tabnet.tabnet_model import TabnetBackbone
from scipy.sparse import csc_matrix
import numpy as np

from ..base_model import BaseModel

logger = logging.getLogger(__name__)

class TabNetMetricLearningModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        assert config.task in ["metric_learning"], "TabularMetricLearningModel is only implemented for metric learning."
        super().__init__(config, **kwargs)
    
    def _build_network(self):
        self.backbone = TabnetBackbone(self.hparams)
        setattr(self.backbone, "output_dim", self.hparams.output_dim)
        self.head = nn.Identity()
    
    def calculate_loss(self, output, y, tag):
        reg_terms = [k for k, v in output.items() if "regularization" in k]
        reg_loss = 0
        for t in reg_terms:
            reg_loss += output[t]
            self.log(
                f"{tag}_{t}_loss",
                output[t],
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=False,
            )

        embeddings = output["backbone_features"]
        metric_loss = self.loss(embeddings, y)
        computed_loss = metric_loss + reg_loss
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag == "valid") or (tag == "test"),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
        )

        return computed_loss 

    def compute_head(self, backbone_features):
        embeddings = self.head(backbone_features)
        y_hat = None #(TODO: make some function to output the logits/distribution)
        return self.pack_output(y_hat, embeddings)

    def pack_output(self, 
                    y_hat: torch.Tensor, 
                    backbone_features: torch.tensor,
                    masks: torch.Tensor) -> Dict[str, Any]:

        return {"logits": y_hat, 
                "backbone_features": backbone_features}

    def forward(self, x: Dict):
        embeddings = self.compute_backbone(x)
        if self.hparams.task == "ssl":
            return self.compute_ssl_head(x)
        return self.compute_head(embeddings)
    
    def predict(self, x: Dict, ret_model_output: bool = True):
        # returns both the logits and the embeddings by default
        # TODO: implement the logic for the logits
        return super().predict(x, ret_model_output)

    def explain(self, x: Dict):
        """
        Return local explanation
        Parameters
        ----------
        X : Dict: `torch.Tensor`
            Input data it has inputs and targets
        Returns
        -------
        M_explain : matrix
            Importance per sample, per columns.
        masks : matrix
            Sparse matrix showing attention masks used by network.
        """

        res_explain = []
        M_explain, masks = self.backbone.forward_masks(x)
        for key, value in masks.items():
            masks[key] = csc_matrix.dot(
                value.numpy(), self.reducing_matrix
            )

        res_explain.append(
            csc_matrix.dot(M_explain.numpy(), self.reducing_matrix)
        )
        res_explain = np.vstack(res_explain)

        return res_explain, masks

        


    # Note that calculate_metrics in train/val/test step may 