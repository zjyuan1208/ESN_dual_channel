# -*- coding: utf-8 -*-
#
# File : echotorch/nn/__init__.py
# Description : nn init file.
# Date : 29th of October, 2019
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

# Import basis
from .Node import Node

# Import conceptor nodes
# from .conceptors.Conceptor import Conceptor
# from .conceptors.ConceptorNet import ConceptorNet

# Import feature transformation nodes
# from .features.ICACell import ICACell
# from .features.OnlinePCACell import OnlinePCACell
# from .features.PCACell import PCACell
# from .features.SFACell import SFACell

# Functional
from .functional.losses import CSTLoss

# Import reservoir nodes
# from .reservoir.BDESN import BDESN
# from .reservoir.BDESNCell import BDESNCell
# from .reservoir.BDESNPCA import BDESNPCA
# Import linear nodes
# from .linear.RRCell import RRCell

# Import utils nodes
# from .utils.Identity import Identity

# All
__all__ = [
    'Node', 'CSTLoss'
]
