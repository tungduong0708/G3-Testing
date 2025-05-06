# import faiss
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

class ZeroShotPredictor(nn.Module):
    def __init__(self, model_path, device='cuda', queue_size=4096):
        super().__init__()
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
    
    def forward(self, image_embeds, location_embeds):
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
        dataset = im2gps3kDataset(vision_processor=self.model.vision_processor, text_processor=None)
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

def evaluate_dataset_from_prediction(model, df_path, all_topk_gps, all_topk_probs, device='cuda:0'):
    """
    Evaluate predictions using geodesic error.
    
    Parameters:
        model: the trained model with location encoder
        df_path: pd.DataFrame with ground truth columns ['LAT', 'LON']
        database: pd.DataFrame containing location reference data (not directly used here)
        all_topk_gps: torch.Tensor of shape [N, top_k, 768*3]
        all_topk_probs: torch.Tensor of shape [N, top_k]
        device: CUDA device
    """
    print("Start evaluation...")

    model.eval()
    df = pd.read_csv(df_path)
    # df = pd.read_csv('./data/im2gps3k/im2gps3k_places365.csv')

    top_k = all_topk_gps.shape[1]
    bsz = all_topk_gps.shape[0]

    # Extract 2D coordinates from the 768*3 embeddings using the location encoder
    all_lat_lons = []
    for i in tqdm(range(0, bsz, 256)):
        gps_batch = all_topk_gps[i:i+256].to(device)  # [B, k, 768*3]
        b, k, d = gps_batch.shape
        gps_batch = gps_batch.reshape(b * k, d)

        with torch.no_grad():
            gps_decoded = model.location_encoder(gps_batch)  # [b*k, 2]
            gps_decoded = gps_decoded.reshape(b, k, 2)        # [b, k, 2]

        all_lat_lons.append(gps_decoded.cpu())
        print(f"Processed {i} to {i+256} images.")

    all_lat_lons = torch.cat(all_lat_lons, dim=0)  # [N, k, 2]

    # Choose the prediction with the highest probability
    max_indices = torch.argmax(all_topk_probs, dim=1)  # [N]
    final_lat_lons = []
    for i in range(bsz):
        lat, lon = all_lat_lons[i, max_indices[i]]
        # Clamp invalid predictions
        lat = lat.item()
        lon = lon.item()
        if lat < -90 or lat > 90:
            lat = 0
        if lon < -180 or lon > 180:
            lon = 0
        final_lat_lons.append((lat, lon))

    # Update dataframe
    df['LAT_pred'] = [lat for lat, _ in final_lat_lons]
    df['LON_pred'] = [lon for _, lon in final_lat_lons]

    # Compute geodesic distance in km
    df['geodesic'] = df.apply(lambda x: geodesic((x['LAT'], x['LON']), (x['LAT_pred'], x['LON_pred'])).km, axis=1)

    # Save and print results
    print(df.head())

    print('2500km level: ', (df['geodesic'] < 2500).mean())
    print('750km level: ', (df['geodesic'] < 750).mean())
    print('200km level: ', (df['geodesic'] < 200).mean())
    print('25km level: ', (df['geodesic'] < 25).mean())
    print('1km level: ', (df['geodesic'] < 1).mean())

    return df


def main():
    # Initialize predictor
    predictor = ZeroShotPredictor(
        model_path='g3_9_.pth',  # Path to your model
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Set parameters
    top_k = 5  # Number of top predictions to return
    im2gps3k_path = './data/im2gps3k/im2gps3k_places365.csv'  # Path to im2gps3k dataset

    print("Starting prediction on im2gps3k dataset...")
    # Get predictions for the entire dataset
    all_topk_gps, all_topk_probs = predictor.predict_dataset(top_k)

    print("Starting evaluation...")
    # Evaluate predictions
    results_df = evaluate_dataset_from_prediction(
        model=predictor.model,
        df_path=im2gps3k_path,
        all_topk_gps=all_topk_gps,
        all_topk_probs=all_topk_probs,
        device=predictor.device
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'results_im2gps3k_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()