import torch
import torch.nn as nn
from gromo.containers.resnet import init_full_resnet_structure

# 1. Build a ResNet starting with thin hidden layers (reduction_factor=0 means 0 hidden channels)
model = init_full_resnet_structure(
    input_shape=(3, 32, 32),
    out_features=10,
    number_of_blocks_per_stage=2,
    nb_stages=2,
    inplanes=16,
    reduction_factor=0.0,   # start with empty blocks (0 hidden channels)
    use_preactivation=True,
    device=torch.device("cpu"),
)

criterion = nn.CrossEntropyLoss()
x = torch.randn(8, 3, 32, 32)
y = torch.randint(0, 10, (8,))

# Pick a block to grow — e.g. first block of first stage
block = model.stages[0][0]

# 2. Enable statistics accumulation
block.init_computation()

# 3. Accumulate gradient statistics over one or more batches
model.zero_grad()
out = model(x)
loss = criterion(out, y)
loss.backward()
block.update_computation()

# 4. Compute optimal extension using GradMax (not TINY)
#    GradMax flags:  compute_delta=False, use_covariance=False,
#                    alpha_zero=True, use_projection=False, ignore_singular_values=True
block.compute_optimal_updates(
    compute_delta=False,          # GradMax: skip natural-gradient weight update
    use_covariance=False,         # GradMax: use identity instead of S matrix
    alpha_zero=True,              # GradMax: incoming weights start at zero
    use_projection=False,         # GradMax: use raw gradient direction
    ignore_singular_values=True,  # GradMax: treat singular values as 1
    maximum_added_neurons=16,
)

print("first_order_improvement:", block.first_order_improvement.item())

# 5. Sub-select to the desired number of neurons before applying
#    (extension_size passed to apply_change only affects BatchNorm growth,
#     not the weight matrices — sub_select truncates the weights to exactly 8)
extension_size = 8   # number of new hidden channels to add
block.sub_select_optimal_added_parameters(keep_neurons=extension_size)

# 6. Apply the change — actually adds the channels
block.apply_change()

# 7. Reset statistics and clean up computed optimal parameters
block.delete_update()
block.reset_computation()

print(f"Block hidden channels after growth: {block.hidden_neurons}")

# 8. Verify model still runs
with torch.no_grad():
    out = model(x)
print("Output shape:", out.shape)  # (8, 10)
