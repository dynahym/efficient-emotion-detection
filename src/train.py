import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm


def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create directory if it doesn't exist
    model_dir = f"models/{model.__class__.__name__.lower()}"
    os.makedirs(model_dir, exist_ok=True)
    
    best_acc = 0.0
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {model.__class__.__name__}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Using tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Training", ncols=100)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        val_pbar = tqdm(test_loader, desc=f"Validation", ncols=100)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(test_loader)
        
        # Print epoch summary
        print(f"\n{'─'*60}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"{'─'*60}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f"{model_dir}/best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved! Accuracy: {best_acc:.2f}%")
        
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved at: {model_dir}/best_model.pth")
    print(f"{'='*60}\n")
    
    return model