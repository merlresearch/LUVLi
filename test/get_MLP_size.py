# Copyright (c) 2017-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
from torchsummary import summary
import torch.nn as nn

model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 204))
print(model)
model = model.cuda()

summary(model,(1,2048))
