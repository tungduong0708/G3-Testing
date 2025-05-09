import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from ..rff.layers import GaussianEncoding
from pyproj import Proj, Transformer

SF = 66.50336

class ProjectionRFF(nn.Module):
    def __init__(self, projection="mercator", sigma=[2**0, 2**4, 2**8]):
        super(ProjectionRFF, self).__init__()

        self.sigma = sigma
        self.num_hierarchies = len(self.sigma)
        self.projection = projection.lower()
        self.embedding_dim = [512] * self.num_hierarchies

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), GaussianEncoding(sigma=s, input_size=2, encoded_size=256))

        proj_wgs84 = Proj('epsg:4326')

        if self.projection == "mercator":
            proj_target = Proj('epsg:3857')
            self.normalizer = 20037508.3427892
        elif self.projection == "eep":
            proj_target = Proj('epsg:8857')
            self.normalizer = 180/SF 
        elif self.projection == "ecef":
            proj_target = Proj('epsg:4978')
            self.normalizer = 6378137.0  # radius of Earth, not exact for ECEF but usable
        else:
            raise ValueError(f"Unsupported projection: {self.projection}")

        self.transformer = Transformer.from_proj(proj_wgs84, proj_target, always_xy=True)

    def forward(self, input):
        lat = input[:, 0].float().detach().cpu().numpy()
        lon = input[:, 1].float().detach().cpu().numpy()
        # lon (batch), lat (batch)

        projected = self.transformer.transform(lon, lat)

        # Shape: (batch, 2) or (batch, 3) depending on projection
        if self.projection == "ecef":
            location = list(zip(*projected))  # X, Y, Z
            # location = torch.Tensor(location).to(input.device)
            location = torch.Tensor(location)
        else:
            location = [[y, x] for x, y in zip(*projected)]
            # location = torch.Tensor(location).to(input.device)
            location = torch.Tensor(location)

        location = location / self.normalizer
        # location_features = torch.zeros(location.shape[0], 512).to(input.device)
        out = [] 

        for i in range(self.num_hierarchies):
            out.append(self._modules['LocEnc' + str(i)](location))

        location_features = torch.cat(out, dim=0) # (hierarchies, batch, 512)
        return location_features