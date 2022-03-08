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
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabular.models.tabnet.tabnet_model import TabnetBackbone
from pytorch_tabular.models.ft_transformer.ft_transformer import FTTransformerBackbone

from ..base_model import BaseModel

logger = logging.getLogger(__name__)

class TabularMetricLearningModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        assert config.task in ["metric_learning"], "TabularMetricLearningModel is only implemented for metric learning."
        super().__init__(config, **kwargs)
    
    def _build_network(self):
        self.backbone = self.hparams.backbone(self.hparams)
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
                "backbone_features": backbone_features, 
                "mask": masks}

    def forward(self, x: Dict):
        embeddings = self.compute_backbone(x)
        if self.hparams.task == "ssl":
            return self.compute_ssl_head(x)
        return self.compute_head(embeddings)
    
    def predict(self, x: Dict, ret_model_output: bool = True):
        # returns both the logits and the embeddings by default
        # TODO: implement the logic for the logits
        return super().predict(x, ret_model_output)

    # Note that calculate_metrics in train/val/test step may 
    







class TabNetBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        self.tabnet = TabNet(
            input_dim=self.hparams.continuous_dim + self.hparams.categorical_dim,
            output_dim=self.hparams.output_dim,
            n_d=self.hparams.n_d,
            n_a=self.hparams.n_a,
            n_steps=self.hparams.n_steps,
            gamma=self.hparams.gamma,
            cat_idxs=[i for i in range(self.hparams.categorical_dim)],
            cat_dims=[cardinality for cardinality, _ in self.hparams.embedding_dims],
            cat_emb_dim=[embed_dim for _, embed_dim in self.hparams.embedding_dims],
            n_independent=self.hparams.n_independent,
            n_shared=self.hparams.n_shared,
            epsilon=1e-15,
            virtual_batch_size=self.hparams.virtual_batch_size,
            momentum=0.02,
            mask_type=self.hparams.mask_type,
        )

    def unpack_input(self, x: Dict):
        # unpacking into a tuple
        x = x["categorical"], x["continuous"]
        # eliminating None in case there is no categorical or continuous columns
        x = (item for item in x if len(item) > 0)
        x = torch.cat(tuple(x), dim=1)
        return x

    def forward(self, x: Dict):
        # unpacking into a tuple
        x = self.unpack_input(x)
        # Returns output and Masked Loss. We only need the output
        x, _ = self.tabnet(x)
        return x
    
    def forward_mask(self, x: Dict):
        x = self.unpack_input(x)

