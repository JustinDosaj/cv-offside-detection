from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


# Generate batches from sequence
# @Params: sequence<Iterable[V]>, batch_size<int>
# @Return: Generator<List[V], none, none>
def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:

    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch

# Use Siglip pretrained model to extract player features for grouping
class TeamClassifier:
    
    # Intiliaze classifier
    # @Params: device<str> = 'cpu' || 'cuda, batch_size<int>
    # @Return: none
    def __init__(self, device: str = 'cpu', batch_size: int = 32):

        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

        #Set UMAP to reduce to 3 dimensions
        self.reducer = umap.UMAP(n_components=3)

        # Set cluser equal to total teams --> may need more clusers for refere/goalkeepers
        self.cluster_model = KMeans(n_clusters=2)

    # Get features from list of cropped images
    # @Params: crops<List[np.ndarray]>
    # @Return: np.ndarray
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:

        # Convert crops from openCV to pillow format
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []

        # Loop through each batch created
        # Use embedding processor to pre/post process data
        # Execute the SIGLIP model on each batch
        # Get embeddings from model outputs
        # Average out embeddings on dim=1 (2 dimensions) --> creates single vector for each image 
        # last_hidden_state: # of images in batch, # of tokens, # of dimensions (embedding dimension)
        # Append embeddings to data array
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        
        # Return Concatenate all embeddings into one array
        # Data contains total cropped images containing players and a 768 dimensional vector of semantic information to group players
        return np.concatenate(data)

    # Reduce data to lower dimension and fit to clustering model
    # @Params: crops<List[np.ndarray]>
    # @Return: none
    def fit(self, crops: List[np.ndarray]) -> None:

        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    # Predict cluser labels for each image inside crops
    # @Params: crops<List[np.ndarray]>
    # @Return: np.ndarray
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)