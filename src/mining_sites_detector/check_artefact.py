#%%
import pickle
import sys
import utils
import torch

#%%
sys.modules['__main__'] = utils
    
#%%
ckpt = "/home/lin/codebase/mining_sites_detector/src/mining_sites_detector/[32]_9999.pkl"


with open(ckpt, "rb") as f:
    archive_1l = pickle.load(f)

# %%
archive_1l['configs']


# %%
archive_1l.keys()
# %%
run_history = archive_1l["run_history"]
# %%
log = run_history[5]["log"]

#%%

log.plot_epochs()

# %%
run_history[5]["model"].encoder#.keys()
# %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#%%

datasets.STL10(root="./data", split="train", download=True)

# %%
probe_test_set = datasets.STL10(root="./data", split="test", download=True)

#%%

clean_transform = transforms.Compose([transforms.ToTensor()])

# %%
probe_train_set = datasets.STL10(root="./data", split="train", download=True, transform=clean_transform)
probe_train_loader = DataLoader(probe_train_set, batch_size=32, shuffle=False)
probe_test_set = datasets.STL10(root="./data", split="test", download=True, transform=clean_transform)
probe_test_loader = DataLoader(probe_test_set, batch_size=32, shuffle=False)



#%%

probe_train_set[1000]#[0].shape



# %%
encoder = run_history[5]["model"].encoder


#%%

run_history[5].keys()
# %%
encoder = encoder.eval()
# %%
for param in encoder.parameters():
    param.requires_grad = False
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.no_grad():
    sample_batch, _ = next(iter(probe_train_loader))
    sample_batch = sample_batch.to(device)
    latent_features = encoder(sample_batch)
    
# %%
latent_features.size()
# %%
latent_features.view(latent_features.size(0), -1).size()
#[0].shape
# %%

def probe_dataloader(batch_size=64):
    clean_transform = transforms.Compose([transforms.ToTensor()])
    
    probe_train_set = datasets.STL10(root="./data", split="train", download=True, transform=clean_transform)
    probe_test_set = datasets.STL10(root="./data", split="test", download=True, transform=clean_transform)
    
    train_loader = DataLoader(probe_train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(probe_test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader



def evaluate_encoder(encoder, train_loader, test_loader, device="cuda"):
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
        
    with torch.no_grad():
        sample_batch, _ = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        latent_features = encoder(sample_batch)
        dim = latent_features.view(latent_features.size(0), -1).shape[1]
        
    classifier = torch.nn.Linear(dim, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        run_loss = 0.0
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder(images)
                features = features.view(features.size(0), -1)
                
            pred = classifier(features)
            loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()
            
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            features = features.view(features.size(0), -1)
            pred = classifier(features)
            _, predicted = pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_accuracy = 100.0 * (correct / total)
    return test_accuracy


#%%

def evaluate_multi_seed(checkpoint_paths, batch_size=64, device="cuda",
                        research_seeds = [42, 101, 223, 456, 789, 1111, 2024, 5555, 7777, 9999]
                        ):

    with open(checkpoint_paths, "rb") as f:
        model_chpt = pickle.load(f)
        
    configs = model_chpt['configs']
    
    layer_arch = f"prober_layers_{configs['layer_channels']}"
    stride = f"stride_{configs['stride']}"
    pool = f"pool_{configs['pool_type']}"
    use_bn = f"use_bn_{configs['use_bn']}"
    decoder_use_bn = f"decoder_use_bn_{configs['decoder_use_bn']}"
    batch_size = f"batch_size_{configs['batch_size']}"
    
    checkpoint_path = f"{filename}.pkl"
        
    run_history = model_chpt["run_history"]
    test_acc = []
    probe_history = {}
    seeds = []
    
    for seed_key, model_obj in run_history.items():
        current_seed = research_seeds[seed_key-1]
        filename = f"{layer_arch}_{stride}_{pool}_{use_bn}_{decoder_use_bn}_{batch_size}_seed_{current_seed}"
        encoder = model_obj["model"].encoder
        train_loader, test_loader = probe_dataloader(batch_size=batch_size)
        test_accuracy = evaluate_encoder(encoder, train_loader, test_loader, device=device)
        print(f"Seed {seed_key}: Test Accuracy: {test_accuracy:.2f}%")
        test_acc.append(test_accuracy)
        seeds.append(current_seed)
        
        probe_history[seed_key] = {"model": trained_model,
                                    "log": log,
                                    "history": history,
                                    "model_summary": model_summary,
                                    "seed": current_seed
                                    }
        
        master_archive = {"configs": saved_configs,
                          "run_history": run_history
                          }
        


#%%

train_loader, test_loader = probe_dataloader()
# %%
test_accuracy = evaluate_encoder(encoder, train_loader, test_loader)
# %%
print(f"Test Accuracy: {test_accuracy:.2f}%")
# %%
