"""Normality model of DFKDE."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from anomalib.models.components.classification import (
    FeatureScalingMethod,
    KDEClassifier,
)

from transformers import AutoProcessor, ASTModel

from .data_ast import get_device

logger = logging.getLogger(__name__)


class DfkdeAstModel(nn.Module):
    """Normality Model for the DFKDE algorithm.

    Args:
        backbone (str): Pre-trained model backbone.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        n_comps (int, optional): Number of PCA components. Defaults to 16.
        pre_processing (str, optional): Preprocess features before passing to KDE.
            Options are between `norm` and `scale`. Defaults to "scale".
        filter_count (int, optional): Number of training points to fit the KDE model. Defaults to 40000.
        threshold_steepness (float, optional): Controls how quickly the value saturates around zero. Defaults to 0.05.
        threshold_offset (float, optional): Offset of the density function from 0. Defaults to 12.0.
    """

    def __init__(
            self,
            sampling_rate: int = 16000,
            skip_encoding: bool = True,
            n_pca_components: int = 16,
            feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
            max_training_points: int = 40000,
            layers=None,
            backbone=None,
            pre_trained=None,
    ) -> None:
        super().__init__()

        self.sampling_rate = sampling_rate
        self.device = get_device()
        self.skip_encoding = skip_encoding
        if not self.skip_encoding:
            self.load_model()

        self.classifier = KDEClassifier(
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )

    def load_model(self):
        self.skip_encoding = False
        self.processor_ = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").eval()
        self.feature_extractor = self.feature_extractor.to(self.device)

    def get_features(self, batch: Tensor) -> Tensor:
        """Extract features from the pretrained network.

        Args:
            batch (Tensor): Image batch.

        Returns:
            Tensor: Tensor containing extracted features.
        """
        if self.feature_extractor.device != self.device:
            self.feature_extractor.to(self.device)

        self.feature_extractor.eval()
        inputs = [self.processor_(data.detach().cpu(), sampling_rate=self.sampling_rate, return_tensors="pt") for data in batch]
        inputs = torch.stack([input["input_values"][0] for input in inputs]).to(self.device)
        with torch.no_grad():
            outputs = self.feature_extractor(inputs)
        encodings_batch = outputs["pooler_output"]
        return encodings_batch

        # layer_outputs = self.feature_extractor(batch)
        # for layer in layer_outputs:
        #     batch_size = len(layer_outputs[layer])
        #     layer_outputs[layer] = F.adaptive_avg_pool2d(input=layer_outputs[layer], output_size=(1, 1))
        #     layer_outputs[layer] = layer_outputs[layer].view(batch_size, -1)
        # layer_outputs = torch.cat(list(layer_outputs.values())).detach()
        # return layer_outputs

    def forward(self, batch: Tensor) -> Tensor:
        """Prediction by normality model.

        Args:
            batch (Tensor): Input images.

        Returns:
            Tensor: Predictions
        """

        # 1. apply feature extraction
        if self.skip_encoding:
            features = batch
        else:
            features = self.get_features(batch)

        if self.training:
            return features

        # 2. apply density estimation
        scores = self.classifier(features)
        return scores
