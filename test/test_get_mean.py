# Copyright (c) 2019-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
   Test get_mean_function
"""
import sys
sys.path.insert(0, './pylib')
from SpatialMean import get_spatial_mean

import torch
from torch.autograd import Variable

x = -torch.ones(2,3,2,2)
x[0,0,0] = torch.ones(2,)
x = x.cuda()
x = Variable(x)
print(x)
print(get_spatial_mean(x))
