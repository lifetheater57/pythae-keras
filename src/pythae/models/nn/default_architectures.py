from typing import List

import numpy as np
import torch
import torch.nn as nn

import keras
from keras import layers, ops

from . import BaseDecoder, BaseEncoder, BaseDecoder_PT, BaseEncoder_PT, BaseDiscriminator, BaseMetric
from ..base.base_config import BaseAEConfig

from ..base.base_utils import ModelOutput


class Encoder_AE_MLP(BaseEncoder_PT):
    def __init__(self, args: dict):
        BaseEncoder_PT.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)

        return output


class Encoder_VAE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        super().__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        model = keras.Sequential([keras.Input(shape=self.input_dim)])

        model.add(layers.Dense(512, activation="relu"))

        self.layers = model
        self.depth = len(model.layers)

        self.embedding = layers.Dense(self.latent_dim)
        self.log_var = layers.Dense(self.latent_dim)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = layers.Flatten()(x)

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_covariance"] = self.log_var(out)

        return output


class Encoder_VAE_MLP_PT(BaseEncoder_PT):
    def __init__(self, args: dict):
        BaseEncoder_PT.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_var = nn.Linear(512, self.latent_dim)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_covariance"] = self.log_var(out)

        return output


class Encoder_SVAE_MLP(BaseEncoder_PT):
    def __init__(self, args: dict):
        BaseEncoder_PT.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_concentration = nn.Linear(512, 1)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_concentration"] = self.log_concentration(out)

        return output


@keras.saving.register_keras_serializable()
class Decoder_AE_MLP(BaseDecoder):
    def __init__(self, model_config: BaseAEConfig):
        super().__init__(model_config)

        self.input_dim = model_config.input_dim

        model = keras.Sequential([keras.Input(shape=model_config.latent_dim)])

        model.add(layers.Dense(512, activation='relu'))

        model.add(layers.Dense(int(ops.prod(model_config.input_dim)), activation="sigmoid"))

        self.model = model
        self.depth = len(model.layers)

    #TODO: add back type hinting
    def forward(self, z, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.model[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["reconstruction"] = ops.reshape(out, (z.shape[0],) + self.input_dim)

        return output


class Decoder_AE_MLP_PT(BaseDecoder_PT):
    def __init__(self, args: dict):
        BaseDecoder_PT.__init__(self)

        self.input_dim = args.input_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 512), nn.ReLU()))

        layers.append(
            nn.Sequential(nn.Linear(512, int(np.prod(args.input_dim))), nn.Sigmoid())
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output


class Metric_MLP(BaseMetric):
    def __init__(self, args: dict):
        BaseMetric.__init__(self)

        if args.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of ModelConfig instance must be set to 'data_shape' where "
                "the shape of the data is [mini_batch x data_shape]. Unable to build metric "
                "automatically"
            )

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(nn.Linear(np.prod(args.input_dim), 400), nn.ReLU())
        self.diag = nn.Linear(400, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(400, k)

    def forward(self, x):
        h1 = self.layers(x.reshape(-1, np.prod(self.input_dim)))
        h21, h22 = self.diag(h1), self.lower(h1)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
        indices = torch.tril_indices(
            row=self.latent_dim, col=self.latent_dim, offset=-1
        )

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h22

        # add diagonal coefficients
        L = L + torch.diag_embed(h21.exp())

        output = ModelOutput(L=L)

        return output


class Discriminator_MLP(BaseDiscriminator):
    def __init__(self, args: dict):
        BaseDiscriminator.__init__(self)

        self.discriminator_input_dim = args.discriminator_input_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(np.prod(args.discriminator_input_dim), 256), nn.ReLU()
            )
        )

        layers.append(nn.Sequential(nn.Linear(256, 1), nn.Sigmoid()))

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z.reshape(z.shape[0], -1)

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = out

        return output
