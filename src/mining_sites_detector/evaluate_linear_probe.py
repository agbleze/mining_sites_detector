import pickle
import sys
import utils
import torch
from copy import deepcopy
import numpy as np
import scipy.stats as stats
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from glob import glob
import json
from utils import setup_data_pipeline, validate_batch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import gc

sys.modules['__main__'] = utils


def evaluate_unsupervised_clustering(encoder, test_loader, 
                                     device="cuda"
                                     ):
        
    encoder.to(device)
    encoder.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            features = encoder(images)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    y_pred = kmeans.fit_predict(all_features)
    
    nmi_score = normalized_mutual_info_score(all_labels, y_pred)
    ari_score = adjusted_rand_score(all_labels, y_pred)
    return nmi_score, ari_score


def evaluate_clustering_multi_seed(checkpoint_path, batch_size=32, device="cuda",
                                    research_seeds = [42, 101, 223, 456, 789, 1111, 2024, 5555, 7777, 9999],
                                    save_dir = "/home/lin/codebase/data_store/linear_probe"
                                    ):
    with open(checkpoint_path, "rb") as f:
        model_chpt = pickle.load(f)
        
    configs = model_chpt['configs']
    
    layer_arch = f"cluster_probe_layers_{configs['layer_channels']}"
    stride = f"stride_{configs['stride']}"
    pool = f"pool_{configs['pool_type']}"
    use_bn = f"use_bn_{configs['use_bn']}"
    decoder_use_bn = f"decoder_use_bn_{configs['decoder_use_bn']}"
    _batch_size = f"batch_size_{configs['batch_size']}"
    
    run_history = model_chpt["run_history"]
    seeds = []
    probe_checkpoint_path = f"{layer_arch}_{stride}_{pool}_{use_bn}_{decoder_use_bn}_{_batch_size}.pkl"
    probe_checkpoint_path = os.path.join(save_dir, probe_checkpoint_path)
    
    cluster_history = {}
    test_nmi_scores = []
    test_ari_scores = []
    
    train_loader, test_loader = probe_dataloader(batch_size=batch_size)
    
    for seed_key, model_obj in run_history.items():
        current_seed = research_seeds[seed_key-1]
        
        print(f"\n[INFO] Evaluating Linear Probe for Seed {current_seed}...")
        encoder = model_obj["model"].encoder
        print(f"Clustering Probe : {probe_checkpoint_path}")
        seeds.append(current_seed)
        

        
        nmi_score, ari_score = evaluate_unsupervised_clustering(encoder=encoder, 
                                                                test_loader=test_loader
                                                                )
        test_nmi_scores.append(nmi_score)
        test_ari_scores.append(ari_score)
        
        print(f" ===> Seed {current_seed}: NMI Score: {nmi_score:.5f}, ARI Score: {ari_score:.5f}")
        
        
        cluster_history[seed_key] = {"model": encoder.cpu(),
                                     "seed": current_seed,
                                     "nmi_score": nmi_score,
                                     "ari_score": ari_score
                                     }
        
        
        torch.cuda.empty_cache()
    
    _test_nmi_scores = np.array(test_nmi_scores)
    num_runs = len(_test_nmi_scores)

    mean_nmi = np.mean(test_nmi_scores)
    std_nmi = np.std(test_nmi_scores, ddof=1) if num_runs > 1 else 0.0
    
    _test_ari_scores = np.array(test_ari_scores)
    mean_ari = np.mean(test_ari_scores)
    std_ari = np.std(test_ari_scores, ddof=1) if num_runs > 1 else 0.0

    if num_runs > 1:
        df = num_runs - 1
        critical_t = stats.t.ppf(0.975, df)
        
        sem_nmi = stats.sem(_test_nmi_scores)
        ci_nmi = critical_t * sem_nmi
        
        sem_ari = stats.sem(_test_ari_scores)
        ci_ari = critical_t * sem_ari
    else:
        ci_nmi = 0.0
        ci_ari = 0.0
        
    master_archive = {"configs": configs,
                        "cluster_history": cluster_history,
                        "test_nmi_scores": test_nmi_scores,
                        "mean_test_nmi": mean_nmi,
                        "std_test_nmi": std_nmi,
                        "CI_95_test_nmi": ci_nmi,
                        
                        "test_ari_scores": test_ari_scores,
                        "mean_test_ari": mean_ari,
                        "std_test_ari": std_ari,
                        "CI_95_test_ari": ci_ari
                    }
        
    with open(probe_checkpoint_path, "wb") as f:
        pickle.dump(master_archive, f)
        
    
    print(f"\n" + "="*80)
    print(f"SUCCESS: Master Downstream Clustering Probe Suite Compiled Securely.")
    print(f"\n[COMPLETE] Clustering probe completed for all seeds. checkpoint: {probe_checkpoint_path}")
    print(f"Target Output Path : {probe_checkpoint_path}")

    print(f"Aggregate NMI Score : {mean_nmi:.4f} (± Sample Std Dev: {std_nmi:.4f})")
    print(f"All seeds MEAN TEST NMI  : {mean_nmi:.4f} (± 95% CI: {ci_nmi:.4f})  [Std Dev: {std_nmi:.4f}]")
    print(f"Aggregate ARI Score : {mean_ari:.4f} (± Sample Std Dev: {std_ari:.4f})")
    print(f"All seeds MEAN TEST ARI  : {mean_ari:.4f} (± 95% CI: {ci_ari:.4f})  [Std Dev: {std_ari:.4f}]")  
    
    print("="*80 + "\n")
    
    gc.collect()
    torch.cuda.empty_cache()





def evaluate_model(model, criterion, batch_size, 
                   device="cuda",
                   ): 
    _, test_loader, test_generator = setup_data_pipeline(seed=1, batch_size=batch_size)
    test_generator.manual_seed(1)
    
    N = len(test_loader)
    running_test_loss = 0.0
    model.eval()
    
    for ix, data in enumerate(test_loader):
        loss = validate_batch(data, model, criterion, device=device)
        running_test_loss += loss.item()
    test_loss = running_test_loss / N
    return test_loss


def probe_dataloader(batch_size=64):
    clean_transform = transforms.Compose([transforms.ToTensor()])
    
    probe_train_set = datasets.STL10(root="./data", split="train", download=True, transform=clean_transform)
    probe_test_set = datasets.STL10(root="./data", split="test", download=True, transform=clean_transform)
    
    train_loader = DataLoader(probe_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
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
        N = len(train_loader)
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder(images)
                features = features.view(features.size(0), -1)
            
            optimizer.zero_grad()
    
            pred = classifier(features)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        epoch_loss = run_loss / N
        print(f"Epoch {epoch+1:02d}/10 | Convex Optimization Cross-Entropy Loss: {epoch_loss:.4f}")
    
    classifier.eval()    
    final_classifier = deepcopy(classifier).cpu()
    
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
    test_accuracy = correct / total
    
    classifier.cpu()
    del classifier, optimizer, criterion
    return final_classifier, test_accuracy



def evaluate_multi_seed(checkpoint_path, batch_size=64, device="cuda",
                        research_seeds = [42, 101, 223, 456, 789, 1111, 2024, 5555, 7777, 9999],
                        save_dir = "/home/lin/codebase/data_store/linear_probe"
                        ):

    with open(checkpoint_path, "rb") as f:
        model_chpt = pickle.load(f)
        
    configs = model_chpt['configs']
    
    layer_arch = f"eval_linearprobe_layers_{configs['layer_channels']}"
    stride = f"stride_{configs['stride']}"
    pool = f"pool_{configs['pool_type']}"
    use_bn = f"use_bn_{configs['use_bn']}"
    decoder_use_bn = f"decoder_use_bn_{configs['decoder_use_bn']}"
    _batch_size = f"batch_size_{configs['batch_size']}"
    
    dae_criterion = torch.nn.MSELoss()
    run_history = model_chpt["run_history"]
    test_acc = []
    probe_history = {}
    seeds = []
    probe_checkpoint_path = f"{layer_arch}_{stride}_{pool}_{use_bn}_{decoder_use_bn}_{_batch_size}.pkl"
    probe_checkpoint_path = os.path.join(save_dir, probe_checkpoint_path)

    eval_history = {}
    test_losses = []
    
    cluster_history = {}
    test_nmi_scores = []
    test_ari_scores = []

    train_loader, test_loader = probe_dataloader(batch_size=batch_size)
    
    for seed_key, model_obj in run_history.items():
        current_seed = research_seeds[seed_key-1]
        
        print(f"\n[INFO] Evaluating Linear Probe for Seed {current_seed}...")
        encoder = model_obj["model"].encoder
        model = model_obj["model"].eval()
        classifier, test_accuracy = evaluate_encoder(encoder, train_loader, test_loader, device=device)
        print(f"Linear Probe : {probe_checkpoint_path}")
        print(f" ===> Seed {current_seed}: Test Accuracy: {test_accuracy:.5f}")
        test_acc.append(test_accuracy)
        seeds.append(current_seed)
        
        probe_history[seed_key] = {"model": classifier.cpu(),
                                    "seed": current_seed,
                                    "test_accuracy": test_accuracy
                                    }
        
        test_loss = evaluate_model(model=model,
                                    criterion=dae_criterion, 
                                    device=device,
                                    batch_size=batch_size,
                                    )
        test_losses.append(test_loss)

        print(f" ===> Seed {current_seed}: Test Loss: {test_loss:.5f}")
        
        model.cpu()
        # eval_history[seed_key] = {"model": model,
        #                           "seed": current_seed,
        #                           "test_loss": test_loss
        #                           }
        
        # nmi_score, ari_score = evaluate_unsupervised_clustering(encoder=encoder, 
        #                                                         test_loader=test_loader
        #                                                         )
        # test_nmi_scores.append(nmi_score)
        # test_ari_scores.append(ari_score)
        
        # print(f" ===> Seed {current_seed}: NMI Score: {nmi_score:.5f}, ARI Score: {ari_score:.5f}")
        
        
        # cluster_history[seed_key] = {"model": model,
        #                              "seed": current_seed,
        #                              "nmi_score": nmi_score,
        #                              "ari_score": ari_score
        #                              }
        
        
        torch.cuda.empty_cache()
    _test_acc = np.array(test_acc)
    num_runs = len(_test_acc)
    mean_acc = np.mean(test_acc)
    std_acc = np.std(test_acc, ddof=1) if num_runs > 1 else 0.0
    
    _test_losses = np.array(test_losses)
    mean_loss = np.mean(test_losses)
    std_loss = np.std(test_losses, ddof=1) if num_runs > 1 else 0.0
    
    # _test_nmi_scores = np.array(test_nmi_scores)
    # mean_nmi = np.mean(test_nmi_scores)
    # std_nmi = np.std(test_nmi_scores, ddof=1) if num_runs > 1 else 0.0
    
    # _test_ari_scores = np.array(test_ari_scores)
    # mean_ari = np.mean(test_ari_scores)
    # std_ari = np.std(test_ari_scores, ddof=1) if num_runs > 1 else 0.0

    if num_runs > 1:
        df = num_runs - 1
        critical_t = stats.t.ppf(0.975, df)
        sem_acc = stats.sem(_test_acc)
        ci_acc = critical_t * sem_acc
        
        sem_loss = stats.sem(_test_losses)
        ci_loss = critical_t * sem_loss
        
        # sem_nmi = stats.sem(_test_nmi_scores)
        # ci_nmi = critical_t * sem_nmi
        
        # sem_ari = stats.sem(_test_ari_scores)
        # ci_ari = critical_t * sem_ari
    else:
        ci_acc = 0.0
        ci_loss = 0.0
        # ci_nmi = 0.0
        # ci_ari = 0.0
        
    master_archive = {"configs": configs,
                    "probe_history": probe_history,
                    "test_accuracy": test_acc,
                    "mean_test_accuracy": mean_acc,
                    "std_test_accuracy": std_acc,
                    "CI_95_test_accuracy": ci_acc,
                    
                    "eval_history": eval_history,
                    "test_loss": test_losses,
                    "mean_test_loss": mean_loss,
                    "std_test_loss": std_loss,
                    "CI_95_test_loss": ci_loss,
                    
                    # "cluster_history": cluster_history,
                    # "test_nmi_scores": test_nmi_scores,
                    # "mean_test_nmi": mean_nmi,
                    # "std_test_nmi": std_nmi,
                    # "CI_95_test_nmi": ci_nmi,
                    # "test_ari_scores": test_ari_scores,
                    # "mean_test_ari": mean_ari,
                    # "std_test_ari": std_ari,
                    # "CI_95_test_ari": ci_ari
                    }
        
    with open(probe_checkpoint_path, "wb") as f:
        pickle.dump(master_archive, f)
        
    
    print(f"\n" + "="*80)
    print(f"SUCCESS: Master Downstream Linear Probe Suite Compiled Securely.")
    print(f"\n[COMPLETE] Linear probe completed for all seeds. checkpoint: {probe_checkpoint_path}")
    print(f"Target Output Path : {probe_checkpoint_path}")
    print(f"Aggregate Score Matrix : {mean_acc:.4f} (± Sample Std Dev: {std_acc:.4f})")
    print(f"All seeds MEAN TEST ACCURACY  : {mean_acc:.4f} (± 95% CI: {ci_acc:.4f})  [Std Dev: {std_acc:.4f}]")
    
    print(f"Aggregate Loss Matrix : {mean_loss:.4f} (± Sample Std Dev: {std_loss:.4f})")
    print(f"All seeds MEAN TEST LOSS  : {mean_loss:.4f} (± 95% CI: {ci_loss:.4f})  [Std Dev: {std_loss:.4f}]")
    
    # print(f"Aggregate NMI Score : {mean_nmi:.4f} (± Sample Std Dev: {std_nmi:.4f})")
    # print(f"All seeds MEAN TEST NMI  : {mean_nmi:.4f} (± 95% CI: {ci_nmi:.4f})  [Std Dev: {std_nmi:.4f}]")
    # print(f"Aggregate ARI Score : {mean_ari:.4f} (± Sample Std Dev: {std_ari:.4f})")
    # print(f"All seeds MEAN TEST ARI  : {mean_ari:.4f} (± 95% CI: {ci_ari:.4f})  [Std Dev: {std_ari:.4f}]")  
    
    print("="*80 + "\n")
    
    #del model_chpt, run_history, master_archive, eval_history, probe_history, cluster_history
    gc.collect()
    torch.cuda.empty_cache()



def main(clustering_probe=True):
    curr_folder = "/home/lin/codebase/mining_sites_detector/src/mining_sites_detector"
    save_dir = "/home/lin/codebase/data_store/linear_probe"
    checkpoint_file = "evaluated_model_and_linear_probe_checkpoints.json"
    if clustering_probe:
        checkpoint_file = "clustering_probe_checkpoints.json"
    completed_linear_probe_path = os.path.join(curr_folder, checkpoint_file)
    
    if os.path.exists(completed_linear_probe_path):
        with open(completed_linear_probe_path, "r") as f:
            completed_linear_probe = json.load(f)["fp"]
    else:
        completed_linear_probe = []
    
    ckpt_files = glob(f"{curr_folder}/*.pkl")
    full_checkpoint_files = [f for f in ckpt_files if "linearprobe" not in f and "9999" in f]
    
    print(f"[INFO] Found {len(full_checkpoint_files)} full checkpoint files:")
    completed_set = set(full_checkpoint_files).intersection(set(completed_linear_probe))
    
    print(f"[INFO] Found {len(completed_set)} / {len(full_checkpoint_files)} already completed linear probe evaluation:")
    
    try:
        for ckpt_file in full_checkpoint_files:
            if ckpt_file not in completed_linear_probe:
                print(f"\n[INFO] Processing checkpoint: {ckpt_file}")
                if clustering_probe:
                    evaluate_clustering_multi_seed(ckpt_file, batch_size=32, 
                                                    device="cuda",
                                                    save_dir=curr_folder, 
                                                   )
                else:
                    evaluate_multi_seed(ckpt_file, batch_size=32, 
                                        device="cuda",
                                        save_dir=curr_folder, #save_dir,
                                        )
                completed_linear_probe.append(ckpt_file)
                
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"[INFO] Skipping already completed checkpoint: {ckpt_file}")
    except Exception as e:
        print(f"Error occurred while processing checkpoint: {e}")
        print(f"[INFO] Saving completed linear probe results to {completed_linear_probe_path}")
        with open(completed_linear_probe_path, "w") as f:
            json.dump({"fp": completed_linear_probe}, f)
    finally:        
        with open(completed_linear_probe_path, "w") as f:
            json.dump({"fp": completed_linear_probe}, f)
            
            
        
if __name__ == "__main__":
    main(clustering_probe=True)