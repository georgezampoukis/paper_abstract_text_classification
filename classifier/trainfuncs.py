import time
import os
import numpy as np
import colorama
from colorama import Fore
import torch
from torch.nn import Module, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import sigmoid
from torch import GradScaler
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex
import matplotlib.pyplot as plt





def train_classifier(classifier: Module, train_loader: DataLoader, valid_loader: DataLoader, epochs: int, optimizer: Optimizer, device: torch.device, model_name: str, sceduler: LRScheduler, save_path: str, amp: bool):
    colorama.init(autoreset=True)

    # Initialize Best Validation Loss
    best_valid_loss = float('inf')

    # Define Loss and Metrics
    bce_loss = BCEWithLogitsLoss().to(device)
    accuracy = BinaryAccuracy().to(device)
    jaccard = BinaryJaccardIndex().to(device)

    # Store Per Epoch Metrics
    train_losses = np.array([], dtype=np.float32)
    val_losses = np.array([], dtype=np.float32)
    train_accs = np.array([], dtype=np.float32)
    val_accs = np.array([], dtype=np.float32)
    train_jaccards = np.array([], dtype=np.float32)
    val_jaccards = np.array([], dtype=np.float32)
    
    # Define Gradient Scaler for Stability in Mixed Precision
    scaler = GradScaler()

    print(Fore.GREEN + "\n------------------------ Training Begins ------------------------\n")

    for epoch in range(epochs):
        start_time = time.time()

        # Store Current Epoch Metrics
        train_epoch_loss = np.array([], dtype=np.float32)
        train_epoch_acc = np.array([], dtype=np.float32)
        train_epoch_jaccard = np.array([], dtype=np.float32)

        valid_epoch_loss = np.array([], dtype=np.float32)
        valid_epoch_acc = np.array([], dtype=np.float32)
        valid_epoch_jaccard = np.array([], dtype=np.float32)
        
        classifier.train()

        # TRAINING
        for index, batch in enumerate(train_loader):
            loop_start = time.time()

            train_input_ids, train_attention_mask, train_token_type_ids, train_label = batch
            train_input_ids = train_input_ids.to(device)
            train_attention_mask = train_attention_mask.to(device)
            train_token_type_ids = train_token_type_ids.to(device)
            train_label = train_label.to(device)

            classifier.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                train_pred_labels = classifier(train_input_ids, train_attention_mask, train_token_type_ids)
                
                train_loss = bce_loss(train_pred_labels, train_label)

                train_acc = accuracy(sigmoid(train_pred_labels.detach()), train_label.detach())
                train_jacc = jaccard(sigmoid(train_pred_labels.detach()), train_label.detach())
                
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_epoch_loss = np.append(train_epoch_loss, train_loss.cpu().item())
            train_epoch_acc = np.append(train_epoch_acc, train_acc.cpu())
            train_epoch_jaccard = np.append(train_epoch_jaccard, train_jacc.cpu())

            loop_end = time.time()

            display_fps = 1 / (loop_end - loop_start) if loop_end - loop_start > 0 else 0

            print(f"Epoch: {epoch + 1} | Step: {index + 1} / {len(train_loader)} [{display_fps:.2f} it/s] | Loss: {train_epoch_loss.mean():.5f} | Acc: {train_epoch_acc.mean():.5f} | Jaccard: {train_epoch_jaccard.mean():.5f}  ", end='\r')

        print("\n")

        # EVALUATION
        classifier.eval()
        for index, batch in enumerate(valid_loader):
            loop_start = time.time()
            
            valid_input_ids, valid_attention_mask, valid_token_type_ids, valid_label = batch
            valid_input_ids = valid_input_ids.to(device)
            valid_attention_mask = valid_attention_mask.to(device)
            valid_token_type_ids = valid_token_type_ids.to(device)
            valid_label = valid_label.to(device)

            with torch.no_grad():
                valid_pred_labels = classifier(valid_input_ids, valid_attention_mask, valid_token_type_ids)

                valid_loss = bce_loss(valid_pred_labels, valid_label)
                
                valid_acc = accuracy(sigmoid(valid_pred_labels.detach()), valid_label.detach())
                valid_jacc = jaccard(sigmoid(valid_pred_labels.detach()), valid_label.detach())

            valid_epoch_loss = np.append(valid_epoch_loss, valid_loss.cpu().item())
            valid_epoch_acc = np.append(valid_epoch_acc, valid_acc.cpu())
            valid_epoch_jaccard = np.append(valid_epoch_jaccard, valid_jacc.cpu())

            loop_end = time.time()

            display_fps = 1 / (loop_end - loop_start) if loop_end - loop_start > 0 else 0

            print(f"{Fore.MAGENTA}Validation: {epoch + 1} | Step: {index + 1} / {len(valid_loader)} [{display_fps:.2f} it/s] | Loss: {valid_epoch_loss.mean():.5f} | Acc: {valid_epoch_acc.mean():.5f} | Jaccard: {valid_epoch_jaccard.mean():.5f}  ", end='\r')

        print("\n")

        # Append Epoch Results
        train_losses = np.append(train_losses, train_epoch_loss.mean())
        val_losses = np.append(val_losses, valid_epoch_loss.mean())
        train_accs = np.append(train_accs, train_epoch_acc.mean())
        val_accs = np.append(val_accs, valid_epoch_acc.mean())
        train_jaccards = np.append(train_jaccards, train_epoch_jaccard.mean())
        val_jaccards = np.append(val_jaccards, valid_epoch_jaccard.mean())

        # Print Classifier Epoch Results
        print(Fore.GREEN + f"--------- classifier ---------\n")
        print(Fore.GREEN + f"\t[Train] Loss: {train_epoch_loss.mean():.5f}\n")
        print(Fore.GREEN + f"\t[Train] Acc: {train_epoch_acc.mean():.5f}\n")
        print(Fore.GREEN + f"\t[Train] Jaccard: {train_epoch_jaccard.mean():.5f}\n")
        print(Fore.GREEN + f"\t[Validation] Loss: {valid_epoch_loss.mean():.5f}\n")
        print(Fore.GREEN + f"\t[Validation] Acc: {valid_epoch_acc.mean():.5f}\n")
        print(Fore.GREEN + f"\t[Validation] Jaccard: {valid_epoch_jaccard.mean():.5f}\n")
        print(Fore.GREEN + f"--------------------------------\n")
        
        # Schedule Learning Rate
        sceduler.step(valid_epoch_loss.mean())

        # Calculate Epoch Time
        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)
        print(Fore.YELLOW + f"\tElapsed Time: {mins} min, {secs} sec\n")

        # Plot Metrics
        plot_metric(train_losses, val_losses, 'Loss', save_path)
        plot_metric(train_accs, val_accs, 'Accuracy', save_path)
        plot_metric(train_jaccards, val_jaccards, 'Jaccard', save_path)

        # Model Checkpoint
        checkpoint_loss = valid_epoch_loss.mean()

        if checkpoint_loss < best_valid_loss:
            print(f"{Fore.CYAN}[MODELCHECKPOINT]: Validation Loss Improved from {best_valid_loss:.5f} to {checkpoint_loss:.5f}. Saving CheckPoint: {model_name}\n")
            best_valid_loss = checkpoint_loss
            save_model(classifier, model_name, save_path)
            save_optimizer(optimizer, model_name, save_path)





def epoch_time(start_time: float, end_time: float) -> tuple[int, int]:
    elapsed_time = end_time - start_time
    mins = int(elapsed_time / 60)
    secs = int(elapsed_time - (mins * 60))
    return mins, secs




def save_model(model: Module, model_name: str, path: str):
    torch.save(model.state_dict(), f'{path}/{model_name}.pt')




def save_optimizer(optimizer: torch.optim.Optimizer, model_name: str, path: str):
    torch.save(optimizer.state_dict(), f'{path}/{model_name}_Optimizer.pt')




def load_model(path: str, model: Module) -> Module:
    model.load_state_dict(torch.load(path, map_location=next(model.parameters()).device), strict=True)
    return model




def load_optimizer(path: str, optimizer: Optimizer, lr: float = None) -> Optimizer:
    optimizer.load_state_dict(torch.load(path))
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return optimizer




def plot_metric(train_metric: list | np.ndarray, valid_metric: list | np.ndarray, label: str, path: str) -> None:
    if len(train_metric) < 2:
        return None
    
    x = np.arange(1, len(train_metric) + 1)

    if isinstance(train_metric, list):
        train_metric = np.array(train_metric)
    if isinstance(valid_metric, list):
        valid_metric = np.array(valid_metric)

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(x, train_metric, color=(1, 0.35, 0), label='Train')
    plt.plot(x, valid_metric, color=(0, 0.35, 1), label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(label)
    plt.savefig(os.path.join(path, f'{label}.png'))
    plt.close()