
#%%
import torch
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn
from torchinfo import summary
from copy import deepcopy
from torch_snippets import Report
import scipy.stats as stats
import matplotlib.pyplot as plt
import json
import pickle
import os
from torchvision import datasets, transforms
import random
from torch.utils.data import Dataset, DataLoader


            
def kernel_initializer(m, initializer_type="he_normal", no_grad=True):
    # Group the check using a tuple for cleaner code
    valid_layers = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d, 
                    nn.LazyConv2d, nn.LazyLinear, nn.LazyConvTranspose2d)
    
    if isinstance(m, valid_layers):
        if no_grad:
            with torch.no_grad(): # <-- CRITICAL FIX: Disables gradient tracking during init
                if initializer_type == "he_normal":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif initializer_type == "glorot_uniform":
                    nn.init.xavier_uniform_(m.weight)
                
                # Check for bias (Lazy layers might have an uninitialized bias attribute)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        else:
            if initializer_type == "he_normal":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif initializer_type == "glorot_uniform":
                nn.init.xavier_uniform_(m.weight)
            
            # Check for bias (Lazy layers might have an uninitialized bias attribute)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
            


class Encoder(nn.Module):
    def __init__(self, layers: list[int], use_bn=False, pool_type="max", stride=2,
                 kernel_size=3, pool_kernel_size=2, pool_stride=2, encoder_padding=1,
                 **kwargs
                 ):
        super().__init__()
        
        encoder_layers = []
        for out_ch in layers:
            enc = []
            enc.append(nn.LazyConv2d(out_channels=out_ch, 
                                    kernel_size=kernel_size, stride=stride, 
                                    padding=encoder_padding, 
                                    bias=True
                                    ),
                       )
            enc.append(nn.ReLU(True))
            
            if use_bn:
                enc.append(nn.LazyBatchNorm2d())
                
            if pool_type == "max":
                enc.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
            elif pool_type == "avg":
                enc.append(nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
                
            encoder_layer = nn.Sequential(*enc)
            encoder_layers.append(encoder_layer)
            
        self.encoder = nn.Sequential(*encoder_layers)
                
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, layers: list[int], decoder_kernel_size=3,
                 decoder_stride=2, decoder_use_bn=False,
                 decoder_padding=0,
                 **kwargs
                 ):
        super().__init__()
        decoder_layers = []
        decoder_layers_out_ch = layers[::-1]
        
        for out_ch in decoder_layers_out_ch:
            decoder_layer = nn.Sequential(nn.LazyConvTranspose2d(out_channels=out_ch,
                                                                 kernel_size=decoder_kernel_size, 
                                                                 stride=decoder_stride, 
                                                                 padding=decoder_padding, 
                                                                 #output_padding=1, 
                                                                 bias=True
                                                                 ),
                                        nn.LazyBatchNorm2d() if decoder_use_bn else nn.Identity(), 
                                        nn.ReLU(True)
                                        )
            decoder_layers.append(decoder_layer)
        
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    
    def forward(self, x):
        x = self.decoder(x)
        return x
    

class ReconstructTask(nn.Module):
    def __init__(self, out_channels, input_size=(28, 28),
                 last_act_func = "tanh", decoder_kernel_size=3,
                 decoder_stride=2, decoder_use_bn=False
                 ):
        super().__init__()
        self.input_size = input_size
        self.task_layer = nn.Sequential(nn.LazyConvTranspose2d(out_channels=out_channels,
                                                               kernel_size=decoder_kernel_size, 
                                                               stride=decoder_stride, 
                                                               padding=1, 
                                                              #output_padding=1, 
                                                              bias=False
                                                              ),
                                        nn.LazyBatchNorm2d() if decoder_use_bn else nn.Identity()
                                        #nn.Tanh(),
                                        #nn.Sigmoid()
                                        )
        
        if last_act_func == "tanh":
            self.last_act = nn.Tanh() 
        elif last_act_func == "sigmoid":
            self.last_act = nn.Sigmoid()
        elif last_act_func == "relu":
            self.last_act = nn.ReLU()
        elif last_act_func == "leaky_relu":
            self.last_act = nn.LeakyReLU()
        else:
            self.last_act = nn.Identity()
        #self.align = nn.AdaptiveAvgPool2d(output_size=input_size)
        
    def forward(self, x):
        x = self.task_layer(x)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        #x = self.align(x)
        x = self.last_act(x)
        return x
    
    
    
class AutoEncoder(nn.Module):
    def __init__(self, layer_channels: list[int], input_size, out_channels=3,
                 last_act_func="tanh", 
                 use_bn=False, pool_type="max", stride=2,
                 kernel_size=3, pool_kernel_size=2, pool_stride=2,
                 decoder_kernel_size=3,
                 decoder_stride=2, decoder_use_bn=False,
                 **kwargs
                 ):
        super().__init__()
        self.encoder = Encoder(layer_channels, use_bn=use_bn, pool_type=pool_type,
                               stride=stride, kernel_size=kernel_size,
                               pool_kernel_size=pool_kernel_size, pool_stride=pool_stride,
                               )
        self.decoder = Decoder(layer_channels, decoder_kernel_size=decoder_kernel_size,
                               decoder_stride=decoder_stride,
                               decoder_use_bn=decoder_use_bn,
                               )
        self.input_size = input_size
        self.reconstruct_task = ReconstructTask(out_channels, input_size, last_act_func=last_act_func,
                                                decoder_kernel_size=decoder_kernel_size,
                                                decoder_stride=decoder_stride, 
                                                decoder_use_bn=decoder_use_bn
                                                )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.reconstruct_task(x)
        return x



def make_model(layer_channels, data,
               device="cuda", 
               initializer_type="he_normal",
               no_grad=True,
               last_act_func="tanh",
               
                 use_bn=False, pool_type="max", stride=2,
                 kernel_size=3, pool_kernel_size=2, pool_stride=2,
                 decoder_kernel_size=3,
                 decoder_stride=2, decoder_use_bn=False,
                 **kwargs
               ):
    out_channels = data.shape[1]
    input_size = tuple(data.shape[2:])
    model = AutoEncoder(layer_channels=layer_channels, 
                        input_size=input_size,
                        out_channels=out_channels,
                        last_act_func=last_act_func,
                        use_bn=use_bn, pool_type=pool_type,
                        stride=stride, kernel_size=kernel_size,
                        pool_kernel_size=pool_kernel_size,
                        pool_stride=pool_stride, decoder_kernel_size=decoder_kernel_size,
                        decoder_stride=decoder_stride, 
                        decoder_use_bn=decoder_use_bn
                        ).to(device)
    data = data.to(device)
    _ = model(data)
    model.apply(lambda module: kernel_initializer(module, no_grad=no_grad,
                                                  initializer_type=initializer_type
                                                  )
                )
    return model



class GaussianNoise:
    def __init__(self, sigma=25, generator=None, clip=True):
        """
        A high-performance, deterministic Gaussian Noise injector.
        Accepts a persistent torch.Generator to preserve stochastic diversity
        without violating reproducibility.
        """
        self.sigma = sigma / 255.0
        self.clip = clip
        self.generator = generator  # Pass your master loop generator here

    def __call__(self, image):
        # 1. Use the pre-allocated persistent generator safely
        if self.generator is not None:
            # Match the generator device dynamically if it transitions to GPU
            noise = torch.randn(
                image.shape,
                generator=self.generator,
                device=image.device,
                dtype=image.dtype
            )
        else:
            noise = torch.randn_like(image, device=image.device, dtype=image.dtype)

        noisy = image + noise * self.sigma

        if self.clip:
            noisy = noisy.clamp(0.0, 1.0)

        return noisy




# 1. A clean wrapper to return the dual (noisy, clean) inputs your loops expect
class PairedDenoisingDataset(Dataset):
    def __init__(self, base_dataset, noise_transform, clean_transform):
        self.base_dataset = base_dataset
        self.noise_transform = noise_transform
        self.clean_transform = clean_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # STL-10 returns a PIL image and a label
        img, _ = self.base_dataset[idx] 
        
        # Create the matching pair
        clean_img = self.clean_transform(img)
        noisy_img = self.noise_transform(clean_img)
        
        return noisy_img, clean_img

def get_stl10_loaders(seed_value=42, batch_size=32):
    # Set up your persistent, deterministic generators
    g_loader = torch.Generator()
    g_loader.manual_seed(seed_value)
    
    g_noise = torch.Generator()
    g_noise.manual_seed(seed_value)

    # Core transforms
    clean_transform = transforms.Compose([transforms.ToTensor()])
    noise_transform = GaussianNoise(sigma=25, generator=g_noise, clip=True)

    # Pull the 100,000 real-world unlabeled images for your DAE pretext task
    raw_stl10 = datasets.STL10(root="./data", split="unlabeled", download=True)
    
    paired_dataset = PairedDenoisingDataset(raw_stl10, noise_transform, clean_transform)
    
    # Deterministic Dataloader worker function
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        paired_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g_loader
    )
    
    return loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_data_pipeline(seed, batch_size, sigma=25, device="cuda", **kwargs):
    g_loader = torch.Generator()
    g_loader.manual_seed(seed)
    clean_transform = transforms.Compose([transforms.ToTensor()])
    
    train_noise = GaussianNoise(sigma=sigma, generator=None, clip=True)
    raw_train = datasets.STL10(root="./data", split="unlabeled", download=True)
    train_dataset = PairedDenoisingDataset(base_dataset=raw_train, 
                                           noise_transform=train_noise,
                                           clean_transform=clean_transform
                                           )
    
    val_generator = torch.Generator(device="cpu")#device)
    val_generator.manual_seed(seed + 9999)
    val_noise = GaussianNoise(sigma=sigma, generator=val_generator, clip=True)
    
    raw_val = datasets.STL10(root="./data", split="test", download=True)
    val_dataset = PairedDenoisingDataset(base_dataset=raw_val, 
                                         noise_transform=val_noise, 
                                         clean_transform=clean_transform
                                         )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, 
                              generator=g_loader,
                              worker_init_fn=seed_worker,
                              num_workers=0,
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0
                            )
    
    return train_loader, val_loader, val_generator



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    # Insures structural operations (like convolutions) use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_batch(input, model, criterion, optimizer, device="cuda", **kwargs):
    optimizer.zero_grad()
    noisy_input, clean_input = input
    noisy_input = noisy_input.to(device)
    clean_input = clean_input.to(device)
    output = model(noisy_input)
    loss = criterion(output, clean_input)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def validate_batch(input, model, criterion, device="cuda", **kwargs):
    noisy_input, clean_input = input
    noisy_input = noisy_input.to(device)
    clean_input = clean_input.to(device)
    output = model(noisy_input)
    loss = criterion(output, clean_input)
    return loss
       


def train_model(model, criterion, optimizer, trn_dl, val_dl,
                num_epochs, scheduler, val_generator, seed, 
                device="cuda",
                **kwargs
                ):
    model.to(device)
    criterion.to(device)
    log = Report(num_epochs)
    history = {"train_epoch_loss": [], "val_epoch_loss": []}
    best_model = None
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        N = len(trn_dl)
        running_train_loss = 0.0
        model.train()
        for ix, data in enumerate(trn_dl):
            loss = train_batch(data, model, criterion, optimizer, device=device, **kwargs)
            running_train_loss += loss.item()
            log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end="\r")
        train_epoch_loss = running_train_loss / N    
        history["train_epoch_loss"].append(train_epoch_loss)
        
        N = len(val_dl)
        running_val_loss = 0.0
        model.eval()
        
        val_generator.manual_seed(seed + 9999)
        for ix, data in enumerate(val_dl):
            loss = validate_batch(data, model, criterion, device=device, **kwargs)
            running_val_loss += loss.item()
            log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end="\r")
        val_epoch_loss = running_val_loss / N
        history["val_epoch_loss"].append(val_epoch_loss)
        log.report_avgs(epoch+1)
        
        if scheduler:
            if not torch.isnan(torch.tensor(val_epoch_loss)):
                scheduler.step(val_epoch_loss)    
                    
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = deepcopy(model)
        
    log.plot_epochs(log=True)
    return best_model, log, history

def train_multi_seed(seeds, configs, checkpoint_path="ablation_results.pkl"):
    
    if os.path.exists(checkpoint_path):
        print(f"FOUND EXISTING CHECKPOINT loading past progress from {checkpoint_path}")
        
        with open(checkpoint_path, "rb") as f:
            archive = pickle.load(f)
        run_history = archive.get("run_history", {})
        saved_configs = archive.get("configs", {})
        print(f"Loaded {len(run_history)} completed seed runs")
    else:
        print("\n[NO CHECKPOINT FOUND] Initializing clean state matrix...")    
    
        run_history = {}
        saved_configs = configs
    
    for run_idx, current_seed in enumerate(seeds):
        run_key = run_idx + 1
        
        if run_key in run_history:
            print(f"Skipping Seed Trial {run_key} / {len(seeds)} Seed {current_seed} ALREADY COMPLETED")
            continue
        
        print(f"\n=== EXECUTION TRAIL {run_key}/{len(seeds)} | current seed: {current_seed} ===")
        
        set_seed(current_seed)
        train_loader, val_loader, val_generator = setup_data_pipeline(seed=current_seed, **configs)
        train_batch_sample = next(iter(train_loader))[0]
        model = make_model(data=train_batch_sample, 
                            **configs
                            )

        model_summary = summary(model, train_batch_sample.shape)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.MSELoss().to(configs.get("device"))

        print(model_summary)
        
        trained_model, log, history = train_model(model=model, 
                                                    criterion=criterion, 
                                                    optimizer=optimizer,
                                                    trn_dl=train_loader, 
                                                    val_dl=val_loader, 
                                                    num_epochs=configs.get("num_epochs"),
                                                    scheduler=scheduler,
                                                    val_generator=val_generator, 
                                                    seed=current_seed,
                                                    )
        
        run_history[run_key] = {"model": trained_model,
                                    "log": log,
                                    "history": history,
                                    "model_summary": model_summary,
                                    "seed": current_seed
                                    }
        
        master_archive = {"configs": saved_configs,
                          "run_history": run_history
                          }
        layer_arch = f"layers_{configs['layer_channels']}"
        stride = f"stride_{configs['stride']}"
        pool = f"pool_{configs['pool_type']}"
        use_bn = f"use_bn_{configs['use_bn']}"
        decoder_use_bn = f"decoder_use_bn_{configs['decoder_use_bn']}"
        batch_size = f"batch_size_{configs['batch_size']}"
        
        filename = f"{layer_arch}_{stride}_{pool}_{use_bn}_{decoder_use_bn}_{batch_size}_seed_{current_seed}"
        checkpoint_path = f"{filename}.pkl"
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(master_archive, f)
        print(f"==> Successfully committed full metadata suite for Trial {run_key} to {checkpoint_path}")
    print(f"\n[COMPLETE] All seeds successfully resolved for checkpoint: {checkpoint_path}")
    
    chart_output_path = f"{filename}_convergence_profile.png"
    plot_manuscript_curves(run_history=run_history, 
                           num_epochs=configs.get("num_epochs"),
                           chart_output_path=chart_output_path
                           )
    return run_history





def compile_manuscript_metrics(run_history, num_epochs=10):
    """
    Compiles the statistical profiles across all seed trials 
    and prints a structured evaluation matrix for the manuscript.
    """
    num_runs = len(run_history)
    trn_matrix = np.zeros((num_runs, num_epochs))
    val_matrix = np.zeros((num_runs, num_epochs))
    
    # 1. Harvest history vectors from the storage matrix
    for run_idx in range(num_runs):
        run_data = run_history[run_idx + 1]["history"]
        trn_matrix[run_idx, :] = run_data["train_epoch_loss"]
        val_matrix[run_idx, :] = run_data["val_epoch_loss"]
        
    # 2. Extract NaN-safe stats
    mean_trn = np.nanmean(trn_matrix, axis=0)
    mean_val = np.nanmean(val_matrix, axis=0)
    
    sem_trn = stats.sem(trn_matrix, axis=0, ddof=1, nan_policy='omit')
    sem_val = stats.sem(val_matrix, axis=0, ddof=1, nan_policy='omit')
    
    # Compute Student's t-value for 95% Confidence interval (df = N - 1)
    df = num_runs - 1
    critical_t = stats.t.ppf(0.975, df)
    
    ci_trn = critical_t * sem_trn
    ci_val = critical_t * sem_val
    
    print("\n" + "="*70)
    print("                MANUSCRIPT COMPILATION STATISTICS MATRIX           ")
    print("="*70)
    print(f"{'EPOCH':<8} | {'MEAN TRN LOSS':<15} (± 95% CI)   | {'MEAN VAL LOSS':<15} (± 95% CI)")
    print("-"*70)
    for e in range(num_epochs):
        # Epoch 1 validation placeholder nan handling
        val_loss_str = f"{mean_val[e]:.5f}" if not np.isnan(mean_val[e]) else "   nan  "
        val_ci_str   = f"{ci_val[e]:.5f}"   if not np.isnan(ci_val[e])   else "  nan  "
        
        print(f"Epoch {e+1:<3} | {mean_trn[e]:.5f}        (± {ci_trn[e]:.5f}) | {val_loss_str}        (± {val_ci_str})")
    print("="*70 + "\n")





def plot_manuscript_curves(run_history, num_epochs=3,
                           chart_output_path = "manuscript_convergence_profile.png"
                           ):
    """
    Generates a publication-grade, high-DPI plot displaying the Mean trajectory,
    95% Confidence Interval bands, and Standard Deviation markers.
    """
    num_runs = len(run_history)
    trn_matrix = np.zeros((num_runs, num_epochs))
    val_matrix = np.zeros((num_runs, num_epochs))
    
    # 1. Extract loss profiles from your multi-seed history matrix
    for run_idx in range(num_runs):
        run_data = run_history[run_idx + 1]["history"]
        trn_matrix[run_idx, :] = run_data["train_epoch_loss"]
        val_matrix[run_idx, :] = run_data["val_epoch_loss"]
        
    # 2. Compute foundational statistical averages and deviations
    mean_trn = np.nanmean(trn_matrix, axis=0)
    mean_val = np.nanmean(val_matrix, axis=0)
    
    std_trn = np.nanstd(trn_matrix, axis=0, ddof=1)
    std_val = np.nanstd(val_matrix, axis=0, ddof=1)
    
    # Calculate Student's t-value for 95% Confidence Intervals (df = N - 1 = 9)
    df = num_runs - 1
    critical_t = stats.t.ppf(0.975, df)
    
    sem_trn = stats.sem(trn_matrix, axis=0, ddof=1, nan_policy='omit')
    sem_val = stats.sem(val_matrix, axis=0, ddof=1, nan_policy='omit')
    
    ci_trn = critical_t * sem_trn
    ci_val = critical_t * sem_val
    
    # 3. Setup professional chart aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300) # Crisp 300 DPI layout
    epochs_range = np.arange(1, num_epochs + 1)
    
    # --- Plot Training Trajectory with Shaded 95% CI ---
    ax.plot(epochs_range, mean_trn, label='Mean Training Loss', color='#1f77b4', lw=2.5)
    ax.fill_between(epochs_range, mean_trn - ci_trn, mean_trn + ci_trn, 
                    color='#1f77b4', alpha=0.18, label='95% Confidence Interval (Train)')
    
    # --- Plot Validation Trajectory with Shaded 95% CI ---
    ax.plot(epochs_range, mean_val, label='Mean Validation Loss', color='#d62728', lw=2.5, linestyle='--')
    ax.fill_between(epochs_range, mean_val - ci_val, mean_val + ci_val, 
                    color='#d62728', alpha=0.12, label='95% Confidence Interval (Val)')
    
    # --- Add Standard Deviation Error Bars (Offset slightly to prevent overlap) ---
    ax.errorbar(epochs_range - 0.03, mean_trn, yerr=std_trn, fmt='o', color='#1f77b4', 
                capsize=5, elinewidth=1.5, markeredgewidth=1.5, alpha=0.7, label='Sample Std Dev (Train)')
    ax.errorbar(epochs_range + 0.03, mean_val, yerr=std_val, fmt='s', color='#d62728', 
                capsize=5, elinewidth=1.5, markeredgewidth=1.5, alpha=0.7, label='Sample Std Dev (Val)')
    
    # --- Academic Polish ---
    ax.set_title('Optimization Convergence Profile & Structural Stochastic Stability Matrix (N=10)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Training Epochs', fontsize=11, labelpad=8)
    ax.set_ylabel('Mean Squared Error (MSE) Loss', fontsize=11, labelpad=8)
    
    ax.set_xticks(epochs_range)
    ax.grid(True, which='both', linestyle=':', alpha=0.5, color='gray')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=9, loc='upper right')
    plt.tight_layout()
    
    
    plt.show()
    
    plt.savefig(chart_output_path, dpi=300, bbox_inches='tight')
    plt.close() # Free memory blocks immediately
    print(f"==> Exported publication-ready chart securely to: {chart_output_path}")
    

    
    
# #%%
# ckpt = "/home/lin/codebase/mining_sites_detector/src/mining_sites_detector/[32]_9999.pkl"


# with open(ckpt, "rb") as f:
#     archive_1l = pickle.load(f)

#%%
if __name__ == "__main__":
    
    
    #%% 1. Define your 10 objective, non-cherry-picked research seeds
    research_seeds = [42, 101, 223, 456, 789, 1111, 2024, 5555, 7777, 9999]
    
    # 2. Package your winning single-layer configuration parameters
    experimental_configs  = {"layer_channels": [16, 21], #[8, 16, 18],
                            "last_act_func": "sigmoid",       # Binds output to your [0.0, 1.0] ToTensor range
                            "stride": 1,                      # Handles the 2x spatial downsampling via Conv
                            "pool_type": None,                # Crucial: Disabled to prevent accidental double-downsampling
                            "kernel_size": 3,
                            "decoder_kernel_size": 3,
                            "decoder_stride": 1,              # Symmetrically handles the 2x upsampling
                            "use_bn": True,                  # Keep disabled for the pure baseline run
                            "decoder_use_bn": True,
                            "initializer_type": "he_normal", #"glorot_uniform"  # Optimal variance alignment for hidden ReLU activations
                            "no_grad": True,                   # Prevents early computation-graph optimization leakage

                            "batch_size": 32,
                            "sigma": 25,
                            "device": "cuda",
                            "num_epochs": 10,
                            }
    # 3. Trigger the full multi-seed simulation sweep
    print(f"Initiating full {len(research_seeds)}-seed rigorous evaluation sweep...")
    master_run_results = train_multi_seed(seeds=research_seeds, 
                                          configs=experimental_configs,
                                          checkpoint_path="/home/lin/codebase/mining_sites_detector/src/mining_sites_detector/layer.pkl"
                                          )
    
    # 4. Compile the final statistical table with 95% Confidence Intervals
    compile_manuscript_metrics(master_run_results, num_epochs=experimental_configs["num_epochs"])


# %%
"""
Hypothesis 1:
Spatial dimension preservation is more important that feature expansion

Under the arch setup of bn for both encoder and decoder and no pooling
Stop all downsampling by setting stride to 1 and train for [32], [32, 64] and [32, 64, 128]
allow downsampling by learning using stride = 2 only without pooling for [32], [32, 64], [32, 64, 128]



Given that the depth was accompanied by width expansion, it will be appropriate to also hold with 
constant while varying the depth. for this, 32 channels is will be used for the experiment

"""



"""



==> Exported publication-ready chart securely to: layers_[32, 64, 128]_stride_1_pool_None_use_bn_True_decoder_use_bn_True_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.01062        (± 0.00004) | 0.00899        (± 0.00007)
Epoch 2   | 0.00922        (± 0.00002) | 0.00893        (± 0.00016)
Epoch 3   | 0.00899        (± 0.00002) | 0.00872        (± 0.00011)
Epoch 4   | 0.00885        (± 0.00002) | 0.00869        (± 0.00008)
Epoch 5   | 0.00876        (± 0.00002) | 0.00863        (± 0.00006)
Epoch 6   | 0.00870        (± 0.00002) | 0.00863        (± 0.00007)
Epoch 7   | 0.00864        (± 0.00005) | 0.00851        (± 0.00007)
Epoch 8   | 0.00861        (± 0.00005) | 0.00850        (± 0.00010)
Epoch 9   | 0.00857        (± 0.00006) | 0.00845        (± 0.00008)
Epoch 10  | 0.00853        (± 0.00007) | 0.00846        (± 0.00009)
======================================================================




==> Exported publication-ready chart securely to: layers_[32, 64]_stride_1_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.00884        (± 0.00003) | 0.00720        (± 0.00010)
Epoch 2   | 0.00744        (± 0.00002) | 0.00703        (± 0.00008)
Epoch 3   | 0.00721        (± 0.00002) | 0.00690        (± 0.00007)
Epoch 4   | 0.00710        (± 0.00002) | 0.00693        (± 0.00007)
Epoch 5   | 0.00703        (± 0.00002) | 0.00685        (± 0.00008)
Epoch 6   | 0.00698        (± 0.00002) | 0.00686        (± 0.00011)
Epoch 7   | 0.00691        (± 0.00003) | 0.00678        (± 0.00010)
Epoch 8   | 0.00687        (± 0.00004) | 0.00674        (± 0.00009)
Epoch 9   | 0.00684        (± 0.00004) | 0.00675        (± 0.00009)
Epoch 10  | 0.00682        (± 0.00005) | 0.00671        (± 0.00005)
======================================================================




==> Exported publication-ready chart securely to: layers_[32]_stride_1_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.00637        (± 0.00006) | 0.00453        (± 0.00005)
Epoch 2   | 0.00487        (± 0.00002) | 0.00437        (± 0.00005)
Epoch 3   | 0.00469        (± 0.00002) | 0.00437        (± 0.00010)
Epoch 4   | 0.00458        (± 0.00002) | 0.00421        (± 0.00003)
Epoch 5   | 0.00450        (± 0.00002) | 0.00421        (± 0.00005)
Epoch 6   | 0.00444        (± 0.00002) | 0.00415        (± 0.00002)
Epoch 7   | 0.00441        (± 0.00002) | 0.00417        (± 0.00004)
Epoch 8   | 0.00438        (± 0.00001) | 0.00415        (± 0.00005)
Epoch 9   | 0.00436        (± 0.00001) | 0.00415        (± 0.00005)
Epoch 10  | 0.00435        (± 0.00002) | 0.00418        (± 0.00006)
======================================================================








==> Exported publication-ready chart securely to: layers_[32]_stride_2_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.01005        (± 0.00008) | 0.00785        (± 0.00007)
Epoch 2   | 0.00813        (± 0.00001) | 0.00769        (± 0.00012)
Epoch 3   | 0.00794        (± 0.00001) | 0.00757        (± 0.00007)
Epoch 4   | 0.00785        (± 0.00002) | 0.00748        (± 0.00006)
Epoch 5   | 0.00779        (± 0.00002) | 0.00742        (± 0.00003)
Epoch 6   | 0.00776        (± 0.00002) | 0.00743        (± 0.00006)
Epoch 7   | 0.00772        (± 0.00002) | 0.00738        (± 0.00004)
Epoch 8   | 0.00768        (± 0.00003) | 0.00739        (± 0.00006)
Epoch 9   | 0.00766        (± 0.00002) | 0.00743        (± 0.00011)
Epoch 10  | 0.00763        (± 0.00002) | 0.00736        (± 0.00005)
======================================================================



==> Exported publication-ready chart securely to: layers_[32, 64]_stride_2_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.01635        (± 0.00005) | 0.01384        (± 0.00008)
Epoch 2   | 0.01388        (± 0.00002) | 0.01338        (± 0.00006)
Epoch 3   | 0.01356        (± 0.00002) | 0.01329        (± 0.00017)
Epoch 4   | 0.01340        (± 0.00002) | 0.01332        (± 0.00027)
Epoch 5   | 0.01329        (± 0.00002) | 0.01313        (± 0.00018)
Epoch 6   | 0.01321        (± 0.00002) | 0.01307        (± 0.00013)
Epoch 7   | 0.01315        (± 0.00002) | 0.01303        (± 0.00009)
Epoch 8   | 0.01310        (± 0.00002) | 0.01292        (± 0.00008)
Epoch 9   | 0.01304        (± 0.00005) | 0.01290        (± 0.00008)
Epoch 10  | 0.01298        (± 0.00007) | 0.01285        (± 0.00009)
======================================================================



==> Exported publication-ready chart securely to: layers_[32, 64, 128]_stride_2_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.01597        (± 0.00007) | 0.01242        (± 0.00014)
Epoch 2   | 0.01238        (± 0.00003) | 0.01181        (± 0.00017)
Epoch 3   | 0.01187        (± 0.00002) | 0.01144        (± 0.00014)
Epoch 4   | 0.01160        (± 0.00002) | 0.01126        (± 0.00014)
Epoch 5   | 0.01144        (± 0.00001) | 0.01113        (± 0.00012)
Epoch 6   | 0.01133        (± 0.00002) | 0.01111        (± 0.00011)
Epoch 7   | 0.01123        (± 0.00007) | 0.01100        (± 0.00016)
Epoch 8   | 0.01114        (± 0.00010) | 0.01091        (± 0.00012)
Epoch 9   | 0.01106        (± 0.00011) | 0.01083        (± 0.00016)
Epoch 10  | 0.01103        (± 0.00012) | 0.01079        (± 0.00015)
======================================================================









==> Exported publication-ready chart securely to: layers_[32, 32, 32]_stride_2_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.01861        (± 0.00009) | 0.01408        (± 0.00015)
Epoch 2   | 0.01375        (± 0.00005) | 0.01303        (± 0.00016)
Epoch 3   | 0.01299        (± 0.00004) | 0.01250        (± 0.00019)
Epoch 4   | 0.01261        (± 0.00003) | 0.01223        (± 0.00020)
Epoch 5   | 0.01237        (± 0.00004) | 0.01211        (± 0.00018)
Epoch 6   | 0.01221        (± 0.00004) | 0.01192        (± 0.00010)
Epoch 7   | 0.01209        (± 0.00004) | 0.01181        (± 0.00014)
Epoch 8   | 0.01200        (± 0.00004) | 0.01174        (± 0.00018)
Epoch 9   | 0.01193        (± 0.00003) | 0.01172        (± 0.00014)
Epoch 10  | 0.01188        (± 0.00003) | 0.01158        (± 0.00015)
======================================================================


==> Exported publication-ready chart securely to: layers_[32, 32]_stride_2_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.01708        (± 0.00004) | 0.01425        (± 0.00007)
Epoch 2   | 0.01428        (± 0.00003) | 0.01378        (± 0.00009)
Epoch 3   | 0.01391        (± 0.00003) | 0.01371        (± 0.00025)
Epoch 4   | 0.01372        (± 0.00003) | 0.01347        (± 0.00015)
Epoch 5   | 0.01360        (± 0.00003) | 0.01336        (± 0.00010)
Epoch 6   | 0.01352        (± 0.00003) | 0.01327        (± 0.00006)
Epoch 7   | 0.01345        (± 0.00003) | 0.01328        (± 0.00008)
Epoch 8   | 0.01339        (± 0.00003) | 0.01329        (± 0.00010)
Epoch 9   | 0.01331        (± 0.00007) | 0.01314        (± 0.00013)
Epoch 10  | 0.01323        (± 0.00007) | 0.01309        (± 0.00010)
======================================================================



==> Exported publication-ready chart securely to: layers_[16, 21]_stride_2_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.01877        (± 0.00015) | 0.01520        (± 0.00017)
Epoch 2   | 0.01514        (± 0.00007) | 0.01451        (± 0.00008)
Epoch 3   | 0.01468        (± 0.00005) | 0.01429        (± 0.00011)
Epoch 4   | 0.01447        (± 0.00004) | 0.01411        (± 0.00009)
Epoch 5   | 0.01434        (± 0.00004) | 0.01401        (± 0.00009)
Epoch 6   | 0.01425        (± 0.00004) | 0.01390        (± 0.00009)
Epoch 7   | 0.01418        (± 0.00005) | 0.01393        (± 0.00011)
Epoch 8   | 0.01412        (± 0.00005) | 0.01387        (± 0.00006)
Epoch 9   | 0.01407        (± 0.00005) | 0.01392        (± 0.00018)
Epoch 10  | 0.01401        (± 0.00006) | 0.01377        (± 0.00008)
======================================================================



==> Exported publication-ready chart securely to: layers_[8, 16, 18]_stride_2_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.02396        (± 0.00018) | 0.01811        (± 0.00018)
Epoch 2   | 0.01752        (± 0.00017) | 0.01642        (± 0.00025)
Epoch 3   | 0.01647        (± 0.00017) | 0.01602        (± 0.00041)
Epoch 4   | 0.01595        (± 0.00015) | 0.01540        (± 0.00023)
Epoch 5   | 0.01562        (± 0.00015) | 0.01513        (± 0.00024)
Epoch 6   | 0.01538        (± 0.00015) | 0.01492        (± 0.00016)
Epoch 7   | 0.01521        (± 0.00015) | 0.01484        (± 0.00030)
Epoch 8   | 0.01507        (± 0.00014) | 0.01466        (± 0.00030)
Epoch 9   | 0.01497        (± 0.00014) | 0.01474        (± 0.00028)
Epoch 10  | 0.01487        (± 0.00013) | 0.01441        (± 0.00017)
======================================================================



==> Exported publication-ready chart securely to: layers_[8, 16, 18]_stride_1_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.01295        (± 0.00016) | 0.01040        (± 0.00012)
Epoch 2   | 0.01047        (± 0.00005) | 0.00986        (± 0.00011)
Epoch 3   | 0.01009        (± 0.00004) | 0.00964        (± 0.00013)
Epoch 4   | 0.00990        (± 0.00003) | 0.00954        (± 0.00010)
Epoch 5   | 0.00978        (± 0.00003) | 0.00947        (± 0.00010)
Epoch 6   | 0.00968        (± 0.00003) | 0.00936        (± 0.00008)
Epoch 7   | 0.00961        (± 0.00003) | 0.00935        (± 0.00011)
Epoch 8   | 0.00954        (± 0.00003) | 0.00928        (± 0.00012)
Epoch 9   | 0.00949        (± 0.00003) | 0.00928        (± 0.00007)
Epoch 10  | 0.00943        (± 0.00005) | 0.00923        (± 0.00010)
======================================================================



==> Exported publication-ready chart securely to: layers_[16, 21]_stride_1_pool_None_use_bn_True_decoder_use_bn_True_batch_size_32_seed_9999_convergence_profile.png

======================================================================
                MANUSCRIPT COMPILATION STATISTICS MATRIX
======================================================================
EPOCH    | MEAN TRN LOSS   (± 95% CI)   | MEAN VAL LOSS   (± 95% CI)
----------------------------------------------------------------------
Epoch 1   | 0.00985        (± 0.00009) | 0.00776        (± 0.00014)
Epoch 2   | 0.00795        (± 0.00003) | 0.00744        (± 0.00006)
Epoch 3   | 0.00769        (± 0.00001) | 0.00729        (± 0.00007)
Epoch 4   | 0.00756        (± 0.00002) | 0.00726        (± 0.00011)
Epoch 5   | 0.00747        (± 0.00002) | 0.00717        (± 0.00005)
Epoch 6   | 0.00741        (± 0.00002) | 0.00713        (± 0.00005)
Epoch 7   | 0.00737        (± 0.00002) | 0.00710        (± 0.00004)
Epoch 8   | 0.00733        (± 0.00002) | 0.00709        (± 0.00007)
Epoch 9   | 0.00730        (± 0.00002) | 0.00702        (± 0.00003)
Epoch 10  | 0.00726        (± 0.00003) | 0.00708        (± 0.00013)
======================================================================

"""

