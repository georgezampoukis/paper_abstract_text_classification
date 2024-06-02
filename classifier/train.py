from datasets import load_dataset, DatasetDict
import pandas as pd
from torch.utils.data import DataLoader
import time
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataloader import DriveDataset
from .model import BertClassifier
from .trainfuncs import train_classifier




if __name__ == '__main__':
    # Set Configuration Variables
    BATCH_SIZE: int = 32
    HIDDEN_SIZE: int = 1024
    EPOCHS: int = 1000
    LEARNING_RATE: float = 2e-5
    DROPOUT: float = 0.4
    NUM_WORKERS: int = 8
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TIMESTAMP: int = int(time.time())
    MODEL_NAME :str = f'Classifier_{BATCH_SIZE}BS_{HIDDEN_SIZE}HS_{LEARNING_RATE}LR'
    PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', f'{MODEL_NAME}_{TIMESTAMP}')
    TRAIN_SPLIT: float = 0.9
    AMP: bool = True

    # load dataset
    dataset: DatasetDict = load_dataset("owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH", cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, 'dataset'))

    # Convert dataset to pandas DataFrame
    pandas_dataset: pd.DataFrame = dataset.data["train"].to_pandas()

    # Split dataset into train and validation
    train_dataset: pd.DataFrame = pandas_dataset.iloc[:int(pandas_dataset.shape[0] * TRAIN_SPLIT)]
    validation_dataset: pd.DataFrame = pandas_dataset.iloc[int(pandas_dataset.shape[0] * TRAIN_SPLIT):]

    # Create dataset and dataloader
    train_set: DriveDataset = DriveDataset(dataset=train_dataset)
    validation_set: DriveDataset = DriveDataset(dataset=validation_dataset)

    train_loader: DataLoader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True if NUM_WORKERS > 0 else False)
    validation_loader: DataLoader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True if NUM_WORKERS > 0 else False)

    # Initialize model
    model = BertClassifier(hidden_size=HIDDEN_SIZE, dropout=DROPOUT, output_size=14, activation=False).to(DEVICE)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)

    # Initialize Model Checkpoint Path
    os.mkdir(PATH)

    # Train model
    train_classifier(classifier=model, train_loader=train_loader, valid_loader=validation_loader, epochs=EPOCHS, optimizer=optimizer, device=DEVICE, model_name=MODEL_NAME, sceduler=scheduler, save_path=PATH, amp=AMP)