import os
import re
import sys
import torch

# Make the vendored gromo (python_packages/gromo/) importable regardless of cwd or whatever else
# named 'gromo' might already be on sys.path -- see README_DEMETER.md "Local gromo patches".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_packages"))

from gromo.modules.linear_growing_module import (
    LinearGrowingModule, LinearMergeGrowingModule
)
from gromo.modules.growing_normalisation import (
    GrowingBatchNorm2d,
    GrowingGroupNorm,
    GrowingLayerNorm,
)
from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_dag import GrowingDAG
from gromo.containers.growing_graph_network import GrowingGraphNetwork


def _make_spatial_normalization(
    channels: int,
    input_shape: tuple,
    *,
    use_layer_norm: bool,
    use_batch_norm: bool,
    use_group_norm: bool,
    group_norm_num_groups: int,
    device: torch.device | str | None,
) -> torch.nn.Module:
    if use_group_norm:
        return GrowingGroupNorm(
            group_norm_num_groups,
            channels,
            affine=False,
            device=device,
        )
    if use_batch_norm:
        return GrowingBatchNorm2d(channels, affine=False, device=device)
    if use_layer_norm:
        return GrowingLayerNorm(
            [channels, *input_shape],
            elementwise_affine=False,
            device=device,
        )
    return torch.nn.Identity()


class CellArchitecture(GrowingContainer):
    def __init__(
        self,
        input_shape: tuple,
        in_channels: int,
        init_hidden_channels: int,
        hidden_features: int,
        out_features: int,
        neurons: int,
        neuron_epochs: int,
        neuron_lrate: float,
        neuron_batch_size: int,
        loss_fn: torch.nn.Module,
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
        use_group_norm: bool = False,
        group_norm_num_groups: int = 1,
        device: torch.device | str | None = None,
        **kwargs
    ) -> None:
        input_volume = input_shape[0] * input_shape[1] * in_channels
        super().__init__(in_features=input_volume, out_features=out_features, device=device)
        layer_type = "convolution"
        # layer_type = "linear"

        # DAG in 32x32 resolution
        self.dag1 = GrowingGraphNetwork(
            in_features=in_channels,
            out_features=init_hidden_channels,
            neurons=neurons,
            neuron_epochs=neuron_epochs,
            neuron_lrate=neuron_lrate,
            neuron_batch_size=neuron_batch_size,
            loss_fn=loss_fn,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            use_group_norm=use_group_norm,
            group_norm_num_groups=group_norm_num_groups,
            layer_type=layer_type,
            input_shape=input_shape,
            name="dag1",
            device=device,
        )

        # Average pooling 32x32 -> 16x16
        avg_pooling1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.dag1.dag.get_node_module(self.dag1.dag.end).post_merge_function = torch.nn.Sequential(
            _make_spatial_normalization(
                init_hidden_channels,
                input_shape,
                use_layer_norm=use_layer_norm,
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                group_norm_num_groups=group_norm_num_groups,
                device=device,
            ),
            torch.nn.SELU(),
            avg_pooling1,
        )
        in_shape = (
            (input_shape[0] - 2) // 2 + 1,
            (input_shape[1] - 2) // 2 + 1,
        )

        # DAG in 16x16 resolution
        self.dag2 = GrowingGraphNetwork(
            in_features=init_hidden_channels,
            out_features=init_hidden_channels,
            neurons=neurons,
            neuron_epochs=neuron_epochs,
            neuron_lrate=neuron_lrate,
            neuron_batch_size=neuron_batch_size,
            loss_fn=loss_fn,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            use_group_norm=use_group_norm,
            group_norm_num_groups=group_norm_num_groups,
            layer_type=layer_type,
            input_shape=in_shape,
            name="dag2",
            device=device,
        )

        # Average pooling 16x16 -> 8x8
        avg_pooling2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.dag2.dag.get_node_module(self.dag2.dag.end).post_merge_function = torch.nn.Sequential(
            _make_spatial_normalization(
                init_hidden_channels,
                in_shape,
                use_layer_norm=use_layer_norm,
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                group_norm_num_groups=group_norm_num_groups,
                device=device,
            ),
            torch.nn.SELU(),
            avg_pooling2,
        )
        in_shape = (
            (in_shape[0] - 2) // 2 + 1,
            (in_shape[1] - 2) // 2 + 1,
        )

        # DAG in 8x8 resolution
        self.dag3 = GrowingGraphNetwork(
            in_features=init_hidden_channels,
            out_features=init_hidden_channels,
            neurons=neurons,
            neuron_epochs=neuron_epochs,
            neuron_lrate=neuron_lrate,
            neuron_batch_size=neuron_batch_size,
            loss_fn=loss_fn,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            use_group_norm=use_group_norm,
            group_norm_num_groups=group_norm_num_groups,
            layer_type=layer_type,
            input_shape=in_shape,
            name="dag3",
            device=device,
        )

        # Average pooling 8x8 -> 4x4
        avg_pooling3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.dag3.dag.get_node_module(self.dag3.dag.end).post_merge_function = torch.nn.Sequential(
            _make_spatial_normalization(
                init_hidden_channels,
                in_shape,
                use_layer_norm=use_layer_norm,
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                group_norm_num_groups=group_norm_num_groups,
                device=device,
            ),
            torch.nn.SELU(),
            avg_pooling3,
        )
        in_shape = (
            (in_shape[0] - 2) // 2 + 1,
            (in_shape[1] - 2) // 2 + 1,
        )

        # DAG in 4x4 resolution
        self.dag4 = GrowingGraphNetwork(
            in_features=init_hidden_channels,
            out_features=init_hidden_channels,
            neurons=neurons,
            neuron_epochs=neuron_epochs,
            neuron_lrate=neuron_lrate,
            neuron_batch_size=neuron_batch_size,
            loss_fn=loss_fn,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            use_group_norm=use_group_norm,
            group_norm_num_groups=group_norm_num_groups,
            layer_type=layer_type,
            input_shape=in_shape,
            name="dag4",
            device=device,
        )

        # Global Average Pooling and flatenning
        self.global_pooling = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = torch.nn.Flatten()
        
        self.dag4.dag.get_node_module(self.dag4.dag.end).post_merge_function = torch.nn.Sequential(
            _make_spatial_normalization(
                init_hidden_channels,
                in_shape,
                use_layer_norm=use_layer_norm,
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                group_norm_num_groups=group_norm_num_groups,
                device=device,
            ),
            torch.nn.SELU(),
            self.global_pooling,
        )

        # Define output reshaping
        self.dag4.dag.get_node_module(self.dag4.dag.end).reshape_function = self.flatten

        # MLP classification head
        self.linear_merge = LinearMergeGrowingModule(
            in_features=init_hidden_channels,
            name="linear_merge",
            device=device,
        )
        self.mlp1 = LinearGrowingModule(
            in_features=init_hidden_channels,
            out_features=hidden_features,
            name="mlp1",
            device=device,
        )
        self.mlp_merge = LinearMergeGrowingModule(
            in_features=hidden_features,
            post_merge_function=torch.nn.SELU(),
            name="mlp_merge",
            device=device,
        )
        self.mlp2 = LinearGrowingModule(
            in_features=hidden_features,
            out_features=out_features,
            name="mlp2",
            device=device,
        )

        # Resolve connections
        end_of_dag1 = self.dag1.dag.get_node_module(self.dag1.dag.end)
        root_of_dag2 = self.dag2.dag.get_node_module(self.dag2.dag.root)
        end_of_dag2 = self.dag2.dag.get_node_module(self.dag2.dag.end)
        root_of_dag3 = self.dag3.dag.get_node_module(self.dag3.dag.root)
        end_of_dag3 = self.dag3.dag.get_node_module(self.dag3.dag.end)
        root_of_dag4 = self.dag4.dag.get_node_module(self.dag4.dag.root)
        end_of_dag4 = self.dag4.dag.get_node_module(self.dag4.dag.end)

        end_of_dag1.add_next_module(root_of_dag2)
        root_of_dag2.add_previous_module(end_of_dag1)
        end_of_dag2.add_next_module(root_of_dag3)
        root_of_dag3.add_previous_module(end_of_dag2)
        end_of_dag3.add_next_module(root_of_dag4)
        root_of_dag4.add_previous_module(end_of_dag3)
        end_of_dag4.add_next_module(self.linear_merge)
        self.linear_merge.add_previous_module(end_of_dag4)
        self.linear_merge.add_next_module(self.mlp1)
        self.mlp1.previous_module = self.linear_merge
        # self.mlp1.next_module = self.mlp2
        # self.mlp2.previous_module = self.mlp1
        self.mlp1.next_module = self.mlp_merge
        self.mlp_merge.add_previous_module(self.mlp1)
        self.mlp_merge.add_next_module(self.mlp2)
        self.mlp2.previous_module = self.mlp_merge
        
        self.set_growing_layers()
    
    def set_growing_layers(self) -> None:
        self._growing_layers.append(self.dag1)
        self._growing_layers.append(self.dag2)
        self._growing_layers.append(self.dag3)
        self._growing_layers.append(self.dag4)
        # self._growing_layers.append(self.linear_merge)
        self._growing_layers.append(self.mlp1)
        self._growing_layers.append(self.mlp_merge)
        self._growing_layers.append(self.mlp2)
    
    # def init_computation(self) -> None:
    #     super().init_computation()
    #     self.mlp_merge.init_computation()

    def update_size(self) -> None:
        self.dag1.update_size()
        self.dag2.update_size()
        self.dag3.update_size()
        self.dag4.update_size()
        self.linear_merge.update_size()
        self.mlp_merge.update_size()
    
    def recreate_model(self, dag_states: list[dict]) -> None:
        self._growing_layers.clear()
        for i, graph in enumerate([self.dag1, self.dag2, self.dag3, self.dag4]):
            graph.dag = GrowingDAG(
                DAG_parameters=dag_states[i],
                in_features=graph.in_features,
                out_features=graph.out_features,
                neurons=graph.neurons,
                use_bias=graph.use_bias,
                use_layer_norm=graph.use_layer_norm,
                use_batch_norm=graph.use_batch_norm,
                use_group_norm=graph.use_group_norm,
                group_norm_num_groups=graph.group_norm_num_groups,
                default_layer_type=graph.layer_type,
                name=graph._name,
                input_shape=graph.input_shape,
                device=graph.device,
            )
            graph.update_size()
        print()

        # Average pooling 32x32 -> 16x16
        avg_pooling1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        channels = dag_states[0]["node_attributes"]["end@dag1"]["size"]
        self.dag1.dag.get_node_module(self.dag1.dag.end).post_merge_function = torch.nn.Sequential(
            _make_spatial_normalization(
                channels,
                self.dag1.input_shape,
                use_layer_norm=self.dag1.use_layer_norm,
                use_batch_norm=self.dag1.use_batch_norm,
                use_group_norm=self.dag1.use_group_norm,
                group_norm_num_groups=self.dag1.group_norm_num_groups,
                device=self.dag1.device,
            ),  # type: ignore
            torch.nn.SELU(),
            avg_pooling1,
        )

        # Average pooling 16x16 -> 8x8
        avg_pooling2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        channels = dag_states[1]["node_attributes"]["end@dag2"]["size"]
        self.dag2.dag.get_node_module(self.dag2.dag.end).post_merge_function = torch.nn.Sequential(
            _make_spatial_normalization(
                channels,
                self.dag2.input_shape,
                use_layer_norm=self.dag2.use_layer_norm,
                use_batch_norm=self.dag2.use_batch_norm,
                use_group_norm=self.dag2.use_group_norm,
                group_norm_num_groups=self.dag2.group_norm_num_groups,
                device=self.dag2.device,
            ),  # type: ignore
            torch.nn.SELU(),
            avg_pooling2,
        )

        # Average pooling 8x8 -> 4x4
        avg_pooling3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        channels = dag_states[2]["node_attributes"]["end@dag3"]["size"]
        self.dag3.dag.get_node_module(self.dag3.dag.end).post_merge_function = torch.nn.Sequential(
            _make_spatial_normalization(
                channels,
                self.dag3.input_shape,
                use_layer_norm=self.dag3.use_layer_norm,
                use_batch_norm=self.dag3.use_batch_norm,
                use_group_norm=self.dag3.use_group_norm,
                group_norm_num_groups=self.dag3.group_norm_num_groups,
                device=self.dag3.device,
            ),  # type: ignore
            torch.nn.SELU(),
            avg_pooling3,
        )

        # Global Average Pooling and flatenning
        channels = dag_states[3]["node_attributes"]["end@dag4"]["size"]
        self.dag4.dag.get_node_module(self.dag4.dag.end).post_merge_function = torch.nn.Sequential(
            _make_spatial_normalization(
                channels,
                self.dag4.input_shape,
                use_layer_norm=self.dag4.use_layer_norm,
                use_batch_norm=self.dag4.use_batch_norm,
                use_group_norm=self.dag4.use_group_norm,
                group_norm_num_groups=self.dag4.group_norm_num_groups,
                device=self.dag4.device,
            ),  # type: ignore
            torch.nn.SELU(),
            self.global_pooling,
        )

        # Define output reshaping
        self.dag4.dag.get_node_module(self.dag4.dag.end).reshape_function = self.flatten

        # MLP classification head
        self.linear_merge = LinearMergeGrowingModule(
            in_features=self.dag4.out_features,
            name="linear_merge",
            device=self.device,
        )
        self.mlp1 = LinearGrowingModule(
            in_features=self.dag4.out_features,
            out_features=self.mlp1.out_features,
            name="mlp1",
            device=self.device,
        )

        # Resolve connections
        end_of_dag1 = self.dag1.dag.get_node_module(self.dag1.dag.end)
        root_of_dag2 = self.dag2.dag.get_node_module(self.dag2.dag.root)
        end_of_dag2 = self.dag2.dag.get_node_module(self.dag2.dag.end)
        root_of_dag3 = self.dag3.dag.get_node_module(self.dag3.dag.root)
        end_of_dag3 = self.dag3.dag.get_node_module(self.dag3.dag.end)
        root_of_dag4 = self.dag4.dag.get_node_module(self.dag4.dag.root)
        end_of_dag4 = self.dag4.dag.get_node_module(self.dag4.dag.end)

        end_of_dag1.add_next_module(root_of_dag2)
        root_of_dag2.add_previous_module(end_of_dag1)
        end_of_dag2.add_next_module(root_of_dag3)
        root_of_dag3.add_previous_module(end_of_dag2)
        end_of_dag3.add_next_module(root_of_dag4)
        root_of_dag4.add_previous_module(end_of_dag3)
        end_of_dag4.add_next_module(self.linear_merge)
        self.linear_merge.add_previous_module(end_of_dag4)
        self.linear_merge.add_next_module(self.mlp1)
        self.mlp1.previous_module = self.linear_merge
        self.mlp1.next_module = self.mlp_merge
        self.mlp_merge.set_previous_modules([self.mlp1])
        self.mlp_merge.set_next_modules([self.mlp2])
        self.mlp2.previous_module = self.mlp_merge

        self.set_growing_layers()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dag1(x)
        x = self.dag2(x)
        x = self.dag3(x)
        x = self.dag4(x)
        x = self.flatten(x)
        x = self.linear_merge(x)
        x = self.mlp1(x)
        x = self.mlp_merge(x)
        x = self.mlp2(x)
        return x
    
    def extended_forward(self, x: torch.Tensor, mask: dict = {}) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, x_ext = self.dag1.extended_forward(x, mask=mask)
        x, x_ext = self.dag2.extended_forward(x, x_ext, mask=mask)
        x, x_ext = self.dag3.extended_forward(x, x_ext, mask=mask)
        x, x_ext = self.dag4.extended_forward(x, x_ext, mask=mask)
        x = self.flatten(x)
        x = self.linear_merge(x)
        if x_ext is not None:
            x_ext = self.flatten(x_ext)
            x_ext = self.linear_merge(x_ext)
        x, x_ext = self.mlp1.extended_forward(x, x_ext, use_extended_input="linear_merge" in mask.get("nodes", []), use_extended_output="mlp_merge" in mask.get("nodes", []))
        x = self.mlp_merge(x)
        if x_ext is not None:
            x_ext = self.mlp_merge(x_ext)
        x, x_ext = self.mlp2.extended_forward(x, x_ext, use_extended_input="mlp_merge" in mask.get("nodes", []), use_extended_output=False)
        return x, x_ext

if __name__ == "__main__":
    input_shape = (32, 32)
    in_channels = 3
    model = CellArchitecture(
        input_shape=input_shape, 
        in_channels=in_channels,
        init_hidden_channels=10,
        hidden_features=50,
        out_features=10,
        neurons=10, 
        neuron_epochs=100,
        neuron_lrate=1e-3,
        neuron_batch_size=256,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device="cuda")
    print(model)
    print(f"{model.number_of_parameters()=}")
    for param, v in model.named_parameters():
        print(param, v.numel())
    x = torch.zeros(2, in_channels, *input_shape, device="cuda")
    model(x)
    model.extended_forward(x)
    
