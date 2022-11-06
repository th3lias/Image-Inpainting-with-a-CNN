# -*- coding: utf-8 -*-
"""
    Author: Elias Mindlberger
    The functions below describe the network architecture of the used CNN.
    The Network is customizable wrt. the number of layers and the number of filters per layer.
"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self,
                 n_input_channels: int = 4,
                 n_conv_layers: int = 4,
                 n_kernels: int = 64,
                 kernel_size: int = 5):
        """CNN, consisting of "n_hidden_layers" linear layers, using relu
        activation function in the hidden CNN layers.

        Parameters
        ----------
        n_input_channels: int
            Number of features channels in input tensor
        n_conv_layers: int
            Number of conv. layers
        n_kernels: int
            Number of kernels in each layer
        kernel_size: int
            Number of features in output tensor
        """
        super().__init__()

        layers = []
        n_concat_channels = n_input_channels
        for i in range(n_conv_layers):
            # Add a CNN layer
            layer = nn.Conv2d(
                in_channels=n_concat_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=int(kernel_size / 2)
            )
            layers.append(layer)
            self.add_module(f"conv_{i:0{len(str(n_conv_layers))}d}", layer)
            # Prepare for concatenated input
            n_concat_channels = n_kernels + n_input_channels
            n_input_channels = n_kernels

        self.layers = layers

        self.down_cnv = nn.Conv2d(
            in_channels=n_kernels,
            out_channels=3,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2)
        )

    def forward(self, x: torch.Tensor):
        """Apply CNN to "x"

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (n_samples, n_input_channels, x, y)

        Returns
        ----------
        torch.Tensor
            Output tensor of shape (n_samples, n_output_channels, u, v)
        """
        skip_connection = None
        output = None

        # Apply layers module
        for layer in self.layers:
            # If previous output and skip_connection exist, concatenate
            # them and store previous output as new skip_connection. Otherwise,
            # use x as input and store it as skip_connection.
            if skip_connection is not None:
                assert output is not None
                inp = torch.cat([output, skip_connection], dim=1)
                skip_connection = output
            else:
                inp = x
                skip_connection = x
            # Apply CNN layer
            output = torch.relu_(layer(inp))

        output = torch.relu_(self.down_cnv(output))
        return output
