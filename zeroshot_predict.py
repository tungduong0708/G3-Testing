import faiss
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import pandas as pd
import ast
import itertools
from PIL import Image
from geopy.distance import geodesic
from transformers import CLIPImageProcessor, CLIPModel
from utils.utils import MP16Dataset, im2gps3kDataset, yfcc4kDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

def load_gps_data(csv_file):
    data = pd.read_csv(csv_file)
    lat_lon = data[['LAT', 'LON']]
    gps_tensor = torch.tensor(lat_lon.values, dtype=torch.float32)
    return gps_tensor

class ZeroShotPredictor:
    def __init__(self, model_path, index_path, image_processor, device='cuda', queue_size=4096):
        self.model = torch.load(model_path, map_location=device)
        self.model.requires_grad_(False)

        self.gps_gallery = load_gps_data("coordinates_100K.csv")
        self._initialize_gps_queue(queue_size)

        self.device = device

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def forward(self, image, location):

        image_embeds = self.model.vision_projection_else(image_embeds)
        location_embeds = self.model.location_projection_else(location_embeds.reshape(location_embeds.shape[0], -1))

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        location_embeds = location_embeds / location_embeds.norm(p=2, dim=-1, keepdim=True)

        # image with location
        logit_scale = self.model.logit_scale2.exp()
        logits_per_image = torch.matmul(image_embeds, location_embeds.t()) * logit_scale

        return logits_per_image
    

    def predict_image(self, image_path, top_k):
        self.model.eval()

        # Step 1: Load and preprocess the input image
        image = Image.open(image_path).convert("RGB")
        image = self.image_encoder.preprocess_image(image)  # Should return a tensor
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension

        # Step 2: Compute image embedding
        with torch.no_grad():
            vision_output = self.model.vision_model(image)[1]
            
            image_embed = self.model.vision_projection(vision_output)
            image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True)

            image_text_embed = self.model.vision_projection_else_1(image_embed)
            image_text_embed = image_text_embed / image_text_embed.norm(p=2, dim=-1, keepdim=True)

            image_location_embed = self.model.vision_projection_else_2(image_embed)
            image_location_embed = image_location_embed / image_location_embed.norm(p=2, dim=-1, keepdim=True)

            # Concatenate final embedding
            full_embed = torch.cat([image_embed, image_text_embed, image_location_embed], dim=1)  # Shape: [1, 768*3]

        # Step 3: Compute similarity with GPS gallery
        gps_gallery = self.gps_gallery.to(self.device)  # [N, 768*3]
        logits_per_image = self.forward(full_embed, gps_gallery)  # [1, N]

        # Step 4: Softmax to get probabilities (optional)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Step 5: Top-K predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]  # [top_k, 768*3]
        top_pred_prob = top_pred.values[0]                   # [top_k]

        return top_pred_gps, top_pred_prob
    
    def predict_dataset(self, top_k):
        self.model.eval()

        # Prepare dataset and dataloader
        dataset = im2gps3kDataset(vision_processor=self.image_encoder.preprocess_image, text_processor=None)
        dataloader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=16,
            pin_memory=True, prefetch_factor=5
        )

        gps_gallery = self.gps_gallery.to(self.device)  # [N, 768*3]
        all_topk_gps = []
        all_topk_probs = []

        print('Generating embeddings and computing top-k predictions...')
        for images, _, _, _ in tqdm(dataloader):
            images = images.to(self.device)

            with torch.no_grad():
                # Base vision model output
                vision_output = self.model.vision_model(images)[1]
                image_embed = self.model.vision_projection(vision_output)
                image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True)

                # Three-way projections
                image_text_embed = self.model.vision_projection_else_1(image_embed)
                image_text_embed = image_text_embed / image_text_embed.norm(p=2, dim=-1, keepdim=True)

                image_location_embed = self.model.vision_projection_else_2(image_embed)
                image_location_embed = image_location_embed / image_location_embed.norm(p=2, dim=-1, keepdim=True)

                # Concatenate embeddings [B, 768*3]
                full_embed = torch.cat([image_embed, image_text_embed, image_location_embed], dim=1)

                # Compute similarity (logits) and top-k
                logits_per_image = self.forward(full_embed, gps_gallery)  # [B, N]
                probs_per_image = logits_per_image.softmax(dim=-1).cpu()  # [B, N]

                top_preds = torch.topk(probs_per_image, top_k, dim=1)  # values: [B, k], indices: [B, k]

                # Store top-k GPS coordinates and probs
                batch_topk_gps = self.gps_gallery[top_preds.indices]  # [B, k, 768*3]
                batch_topk_probs = top_preds.values  # [B, k]

                all_topk_gps.append(batch_topk_gps)
                all_topk_probs.append(batch_topk_probs)

        # Concatenate results for the entire dataset
        all_topk_gps = torch.cat(all_topk_gps, dim=0)     # [Total_Images, k, 768*3]
        all_topk_probs = torch.cat(all_topk_probs, dim=0) # [Total_Images, k]

        return all_topk_gps, all_topk_probs
