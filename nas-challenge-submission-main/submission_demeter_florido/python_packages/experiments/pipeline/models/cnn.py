import torch
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_graph_network import GrowingGraphNetwork


class CNN(GrowingContainer):
    def __init__(
        self,
        input_shape: tuple,
        in_channels: int,
        out_channels: int,
        out_features: int,
        neurons: int,
        neuron_epochs: int,
        neuron_lrate: float,
        neuron_batch_size: int,
        loss_fn: torch.nn.Module,
        device,
        **kwargs
    ) -> None:
        growing_dag = GrowingGraphNetwork(
            in_features=in_channels,
            out_features=out_channels,
            neurons=neurons,
            neuron_epochs=neuron_epochs,
            neuron_lrate=neuron_lrate,
            neuron_batch_size=neuron_batch_size,
            loss_fn=loss_fn,
            layer_type="convolution",
            input_shape=input_shape,
            name="dag1",
        )
        input_volume = growing_dag.dag.get_node_module(growing_dag.dag.root).input_volume
        # output_volume = growing_dag.dag.get_node_module(growing_dag.dag.end).output_volume
        super().__init__(
            in_features=input_volume, out_features=out_features, device=device
        )
        self.growing_dag = growing_dag

        self.pooling = torch.nn.AvgPool2d(kernel_size=input_shape, stride=1)
        
        self.growing_dag.dag.get_node_module(
            self.growing_dag.dag.end
        ).post_merge_function = torch.nn.Sequential(
            torch.nn.SELU(),  # NOTE: SHOULD BE HERE!!
        )
        self.growing_dag.dag.get_node_module(
            self.growing_dag.dag.end
        ).reshape_function = self.pooling

        self.linear_merge = LinearMergeGrowingModule(
            in_features=out_channels,
            previous_modules=[
                self.growing_dag.dag.get_node_module(self.growing_dag.dag.end)
            ],
            name="linear_merge",
        ).to(device)
        self.mlp1 = LinearGrowingModule(
            in_features=out_channels,
            out_features=50,
            post_layer_function=torch.nn.SELU(),
            previous_module=self.linear_merge,
            name="mlp1",
            device=device,
        )
        self.mlp2 = torch.nn.Linear(
            in_features=50,
            out_features=out_features,
        ).to(device)

        self.growing_dag.dag.get_node_module(self.growing_dag.dag.end).set_next_modules(
            [self.linear_merge]
        )
        
        self.linear_merge.set_next_modules([self.mlp1])
        self.mlp1.next_module = self.mlp2

        self.set_growing_layers()

    def set_growing_layers(self):
        self._growing_layers.append(self.growing_dag)

    def forward(self, x):
        x = self.growing_dag(x)
        x = self.pooling(x)
        x = torch.nn.Flatten()(x)
        x = self.linear_merge(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x

    def extended_forward(self, x, mask={}):
        x, _ = self.growing_dag.extended_forward(x, mask=mask)
        x = self.pooling(x)
        x = torch.nn.Flatten()(x)
        x = self.linear_merge(x)
        x = self.mlp1(x)
        return self.mlp2(x)
    
class MLP(GrowingContainer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        neurons: int,
        neuron_epochs: int,
        neuron_lrate: float,
        neuron_batch_size: int,
        loss_fn: torch.nn.Module,
        device: torch.device | str | None = None,
        **kwargs
    ) -> None:
        super().__init__(in_features, out_features, device)

        self.growing_dag = GrowingGraphNetwork(
            in_features=in_features,
            out_features=out_features,
            neurons=neurons,
            neuron_epochs=neuron_epochs,
            neuron_lrate=neuron_lrate,
            neuron_batch_size=neuron_batch_size,
            loss_fn=loss_fn,
            layer_type="linear",
            name="dag1",
        )


        self.growing_dag.dag.get_node_module(
            self.growing_dag.dag.end
        ).post_merge_function = torch.nn.Sequential(
            torch.nn.SELU(),
        )

        self.set_growing_layers()
    
    def set_growing_layers(self) -> None:
        self._growing_layers.append(self.growing_dag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.growing_dag(x)

    def extended_forward(self, x, mask: dict = {}):
        x = torch.flatten(x, 1)
        return self.growing_dag.extended_forward(x, mask=mask)

if __name__ == "__main__":
    in_channels = 3
    input_shape=(32, 32)
    model = CNN(input_shape=input_shape, in_channels=in_channels, out_channels=10, out_features=10, neurons=10, neuron_epochs=100, neuron_lrate=1e-3, neuron_batch_size=256, loss_fn=torch.nn.CrossEntropyLoss(), device="cuda")
    temp = torch.zeros(1, in_channels, *input_shape, device="cuda")
    print(model(temp).shape)