# 2-Channel CNN learned weights
# Copy-paste into Python to reconstruct the model

import numpy as np

# Layer 1: Conv2d(1 → 2, 3x3)
conv1_channel_0 = np.array([
    [0.5379045010, 0.5388053060, 0.5390484333],
    [0.5405381322, -0.3285791278, 0.5375956297],
    [0.5404155254, 0.5379188061, 0.5400795341],
])

conv1_channel_1 = np.array([
    [-0.8436394930, -0.8436390162, -0.8438786268],
    [-0.8436351418, -0.8418656588, -0.8437571526],
    [-0.8437519670, -0.8437506557, -0.8437247872],
])

conv1_bias = np.array([-0.7810028195, 2.5308468342])

# Layer 2: Conv2d(2 → 1, 1x1)
conv2_weights = np.array([-1.8710340261, -3.9949836731])
conv2_bias = 1.8602601290

# Total parameters: 23
# conv1: 2 * (3*3 + 1) = 20
# conv2: 2 * 1 + 1 = 3
