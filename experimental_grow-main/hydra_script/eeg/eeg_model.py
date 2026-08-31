import torch
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.linear_growing_module import LinearGrowingModule
from torch import Tensor, nn

__author__ = "Théo Rudkiewicz"


class EEGModel(SequentialGrowingModel):
    """
    Represents a CNN with growing linear module as end layer.
    """

    def __init__(
        self,
        in_features: int | list[int] | tuple[int, ...],
        out_features: int,
        n_channel_input: int,
        activation: nn.Module = nn.SELU(),
        use_bias: bool = True,
        flatten: bool = False,
        device: torch.device | None = None,
        initial_last_layer_size: int = 10,
    ) -> None:
        """
        Initialize the growing MLP.

        Parameters
        ----------
        in_features : int | list[int] | tuple[int, ...]
            Number of input features.
        out_features : int
            Number of output features.
        n_channel_input : int
            Number of channel .
        activation : nn.Module
            Activation function.
        use_bias : bool
            Whether to use bias in layers.
        flatten : bool
            Whether to flatten the input before passing it through the network.
        device : Optional[torch.device]
            Device to use for computation.
        initial_last_layer_size : int
            Initial size of the last layer.
            For non-growable models the recommended value is 256
        """

        if isinstance(in_features, int):
            pass
        elif isinstance(in_features, (list, tuple)):
            if flatten:
                in_features = int(torch.tensor(in_features).prod().item())
            else:
                in_features = in_features[-1]
        else:
            raise TypeError(
                f"Expected in_features to be int, list, or tuple, got {type(in_features)}"
            )
        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )

        self.use_bias = use_bias
        self.out_features = out_features
        self.activation = activation
        self.n_channel_input = n_channel_input

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(self.n_channel_input, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Dropout(0.5),
            nn.Conv2d(
                16,
                8,
                kernel_size=(1, 32),
                stride=(1, 1),
                dilation=(1, 2),
                padding=(0, 31),
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)),
            nn.Dropout(0.5),
            nn.Conv2d(
                8, 4, kernel_size=(5, 5), stride=(1, 1), dilation=(2, 2), padding=(4, 4)
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 0)),
            nn.Dropout(0.5),
            nn.Flatten(),
        ).to(self.device)

        self.classifier = nn.ModuleList()
        layer1 = LinearGrowingModule(
            88,
            initial_last_layer_size,
            post_layer_function=self.activation,
            use_bias=self.use_bias,
            name="Layer 0",
            device=self.device,
        )
        layer2 = LinearGrowingModule(
            initial_last_layer_size,
            self.out_features,
            previous_module=layer1,
            use_bias=self.use_bias,
            name="Layer 1",
            device=self.device,
        )
        self.classifier.append(layer1)
        self.classifier.append(layer2)
        self._growable_layers = [layer2]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the growing MLP.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        x = self.backbone(x)
        for layer in self.classifier:
            x = layer(x)
        return x

    def extended_forward(self, x: Tensor, mask=None) -> Tensor:
        """
        Forward pass of the growing MLP with the current modifications.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        x = self.backbone(x)
        x_ext = None
        for layer in self.classifier:
            if isinstance(layer, LinearGrowingModule):
                x, x_ext = layer.extended_forward(x, x_ext)
            else:
                x = layer(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGModel(
        in_features=(1, 8, 175),
        out_features=2,
        n_channel_input=8,
        activation=nn.ReLU(),
        use_bias=True,
        device=device,
        initial_last_layer_size=256,
    )
    print(model)
    sample_input = torch.randn(7, 1, 8, 175, device=device)
    output = model(sample_input)
    print(output.shape)
