import os
import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(model, 
                          train_loader, 
                          test_loader, 
                          criterion, 
                          optimizer, 
                          scheduler, 
                          device, 
                          num_epochs, 
                          checkpoint_path):

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Found checkpoint {checkpoint_path}. Loading...")
        if device == "cuda":
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        train_losses = checkpoint["train_losses"]
        train_accuracies = checkpoint["train_accuracies"]
        test_losses = checkpoint["test_losses"]
        test_accuracies = checkpoint["test_accuracies"]
        
        loaded_epoch = checkpoint["epoch"]
        start_epoch = loaded_epoch + 1
        
        if loaded_epoch == num_epochs:
            print(f"[INFO] Model was fully trained up to epoch {loaded_epoch}.")
            print("       Skipping training and returning existing logs.")
            print("       Last Accuracy: {:.2f}%\n".format(test_accuracies[-1]))
            return train_losses, train_accuracies, test_losses, test_accuracies
        else:
            print(f"[INFO] Resuming training from epoch {start_epoch}...\n")
    else:
        print(f"[INFO] No checkpoint found at {checkpoint_path}. Starting fresh training.\n")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%\n")
        
        scheduler.step()

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies
        }, checkpoint_path)
    
    return train_losses, train_accuracies, test_losses, test_accuracies
