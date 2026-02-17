# Inventario de Fórmulas Matemáticas Core (Python vs CUDA)

Este documento lista las fórmulas matemáticas identificadas en el codebase, con referencias exactas a las líneas de código.

## Archivos Python

### analyze_cuda_system.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 99 | analyze_cuda_system | `v = torch.ones(B, D, device=device) * 2.0` |
| 103 | analyze_cuda_system | `U_stack = torch.ones(num_layers * H * D, 2, device=device) * 0.001` |
| 104 | analyze_cuda_system | `W_stack = torch.ones(num_layers * H * 2, D, device=device) * 0.001` |
| 141 | analyze_cuda_system | `pyd_files = glob.glob(os.path.join(build_dir, '**', '*.pyd'), recursive=True)` |
| 142 | analyze_cuda_system | `so_files = glob.glob(os.path.join(build_dir, '**', '*.so'), recursive=True)` |

#### Fórmulas Listas para Usar (Python)
```python
# analyze_cuda_system (L99)
v = torch.ones(B, D, device=device) * 2.0
# analyze_cuda_system (L103)
U_stack = torch.ones(num_layers * H * D, 2, device=device) * 0.001
# analyze_cuda_system (L104)
W_stack = torch.ones(num_layers * H * 2, D, device=device) * 0.001
# analyze_cuda_system (L141)
pyd_files = glob.glob(os.path.join(build_dir, '**', '*.pyd'), recursive=True)
# analyze_cuda_system (L142)
so_files = glob.glob(os.path.join(build_dir, '**', '*.so'), recursive=True)
```

### debug_autograd.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 30 | debug_autograd | `v = torch.ones(B, D, device=device, requires_grad=True) * 5.0` |
| 35 | debug_autograd | `U_stack = torch.ones(num_layers * H * D, rank, device=device) * 0.001` |
| 36 | debug_autograd | `W_stack = torch.ones(num_layers * H * rank, D, device=device) * 0.001` |

#### Fórmulas Listas para Usar (Python)
```python
# debug_autograd (L30)
v = torch.ones(B, D, device=device, requires_grad=True) * 5.0
# debug_autograd (L35)
U_stack = torch.ones(num_layers * H * D, rank, device=device) * 0.001
# debug_autograd (L36)
W_stack = torch.ones(num_layers * H * rank, D, device=device) * 0.001
```

### debug_indexing.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 29 | debug_indexing_issue | `v = torch.ones(B, D, device=device, requires_grad=True) * 5.0` |
| 34 | debug_indexing_issue | `U_stack = torch.ones(num_layers * H * D, rank, device=device) * 0.001` |
| 35 | debug_indexing_issue | `W_stack = torch.ones(num_layers * H * rank, D, device=device) * 0.001` |
| 47 | debug_indexing_issue | `calculated_num_layers = U_stack.shape[0] // H` |
| 62 | debug_indexing_issue | `head_dim = D // H` |
| 63 | debug_indexing_issue | `U_reshaped = U_stack.view(num_layers, H, head_dim, -1)` |
| 64 | debug_indexing_issue | `W_reshaped = W_stack.view(num_layers, H, head_dim, -1).permute(0, 1, 3, 2)` |
| 100 | debug_indexing_issue | `dt_scales_large = torch.ones(num_layers + 5, device=device)` |

#### Fórmulas Listas para Usar (Python)
```python
# debug_indexing_issue (L29)
v = torch.ones(B, D, device=device, requires_grad=True) * 5.0
# debug_indexing_issue (L34)
U_stack = torch.ones(num_layers * H * D, rank, device=device) * 0.001
# debug_indexing_issue (L35)
W_stack = torch.ones(num_layers * H * rank, D, device=device) * 0.001
# debug_indexing_issue (L47)
calculated_num_layers = U_stack.shape[0] // H
# debug_indexing_issue (L62)
head_dim = D // H
# debug_indexing_issue (L63)
U_reshaped = U_stack.view(num_layers, H, head_dim, -1)
# debug_indexing_issue (L64)
W_reshaped = W_stack.view(num_layers, H, head_dim, -1).permute(0, 1, 3, 2)
# debug_indexing_issue (L100)
dt_scales_large = torch.ones(num_layers + 5, device=device)
```

### demos\copy_task\train_copy_task.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 38 | __init__ | `self.EOS = vocab_size + 1  # End of sequence` |
| 51 | _generate_samples | `sequence = [random.randint(0, self.vocab_size - 1) for _ in range(seq_len)]` |
| 54 | _generate_samples | `full_seq = sequence + [self.SEP] + sequence + [self.EOS]` |
| 57 | _generate_samples | `input_seq = full_seq[:-1]` |
| 78 | collate_fn | `srcs, tgts = zip(*batch)` |
| 88 | collate_fn | `pad_len = max_len - len(src)` |
| 106 | evaluate_accuracy | `preds = torch.argmax(logits, dim=-1)` |
| 111 | evaluate_accuracy | `eos_idx = (target == vocab_size + 1).nonzero(as_tuple=True)[0]` |
| 124 | train_copy_task | `parser.add_argument('--config', type=str, default='configs/demos/copy_task.yaml')` |
| 179 | train_copy_task | `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )` |
| 198 | train_copy_task | `loss = criterion(logits.view(-1, config['model']['vocab_size']), tgt.view(-1))` |
| 204 | train_copy_task | `total_loss += loss.item()` |
| 207 | train_copy_task | `cur_loss = total_loss / (i + 1)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L38)
self.EOS = vocab_size + 1  # End of sequence
# _generate_samples (L51)
sequence = [random.randint(0, self.vocab_size - 1) for _ in range(seq_len)]
# _generate_samples (L54)
full_seq = sequence + [self.SEP] + sequence + [self.EOS]
# _generate_samples (L57)
input_seq = full_seq[:-1]
# collate_fn (L78)
srcs, tgts = zip(*batch)
# collate_fn (L88)
pad_len = max_len - len(src)
# evaluate_accuracy (L106)
preds = torch.argmax(logits, dim=-1)
# evaluate_accuracy (L111)
eos_idx = (target == vocab_size + 1).nonzero(as_tuple=True)[0]
# train_copy_task (L124)
parser.add_argument('--config', type=str, default='configs/demos/copy_task.yaml')
# train_copy_task (L179)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )
# train_copy_task (L198)
loss = criterion(logits.view(-1, config['model']['vocab_size']), tgt.view(-1))
# train_copy_task (L204)
total_loss += loss.item()
# train_copy_task (L207)
cur_loss = total_loss / (i + 1)
```

### demos\fractal_recall_demo.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 30 | train_showcase | `optimizer = RiemannianAdam(model.parameters(), lr=3e-4, weight_decay=0.01)` |
| 87 | train_showcase | `mask = 2**torch.arange(coord_dim).to(device)` |
| 88 | train_showcase | `bits = (lm_targets.unsqueeze(-1) & mask) > 0` |
| 89 | train_showcase | `target_coords = bits.float() * 2 - 1 # [B, L-1, 32]` |
| 91 | train_showcase | `pred_coords_shifted = pred_coords[:, :-1, :] # [B, L-1, 32]` |
| 106 | train_showcase | `total_loss += l_g` |
| 111 | train_showcase | `total_loss += l_c` |
| 125 | train_showcase | `if christoffels: desc += f" \| Curv: {l_g.item():.4f}"` |
| 138 | __init__ | `self.mod = vocab_size - 10 # Start of special tokens` |
| 139 | __init__ | `self.START = self.mod + 0` |
| 140 | __init__ | `self.END = self.mod + 1` |
| 141 | __init__ | `self.OPEN_BRACKET = self.mod + 2` |
| 142 | __init__ | `self.CLOSE_BRACKET = self.mod + 3` |
| 143 | __init__ | `self.KEY_VAL_SEP = self.mod + 4` |
| 144 | __init__ | `self.QUERY = self.mod + 5` |
| 145 | __init__ | `self.NOISE = self.mod + 6 # "Singularity" token` |
| 181 | __getitem__ | `pad_len = self.seq_len - len(seq) - 1 # -1 for target` |
| 184 | __getitem__ | `seq = seq[:self.seq_len-1]` |
| 232 | train_showcase | `optimizer = RiemannianAdam(model.parameters(), lr=3e-4, weight_decay=0.01)` |
| 269 | train_showcase | `lm_targets = inputs[:, 1:] # [B, T-1]` |
| 273 | train_showcase | `mask = 2**torch.arange(coord_dim).to(device)` |
| 274 | train_showcase | `bits = (lm_targets.unsqueeze(-1) & mask) > 0` |
| 275 | train_showcase | `target_coords = bits.float() * 2 - 1 # [B, T-1, 32]` |
| 278 | train_showcase | `pred_coords_shifted = pred_coords[:, :-1, :] # [B, T-1, 32]` |
| 296 | train_showcase | `v_start = model.v0.norm().item()` |
| 297 | train_showcase | `v_end = layer0.last_v.norm(dim=-1).mean().item() if hasattr(layer0, 'last_v') else 0.0` |
| 298 | train_showcase | `drift = abs(v_end - v_start)` |
| 317 | train_showcase | `res_dir = PROJECT_ROOT / "tests/benchmarks/results/showcase"` |
| 342 | train_showcase | `ckpt_path = PROJECT_ROOT / "checkpoints/showcase_v1.0.pt"` |

#### Fórmulas Listas para Usar (Python)
```python
# train_showcase (L30)
optimizer = RiemannianAdam(model.parameters(), lr=3e-4, weight_decay=0.01)
# train_showcase (L87)
mask = 2**torch.arange(coord_dim).to(device)
# train_showcase (L88)
bits = (lm_targets.unsqueeze(-1) & mask) > 0
# train_showcase (L89)
target_coords = bits.float() * 2 - 1 # [B, L-1, 32]
# train_showcase (L91)
pred_coords_shifted = pred_coords[:, :-1, :] # [B, L-1, 32]
# train_showcase (L106)
total_loss += l_g
# train_showcase (L111)
total_loss += l_c
# train_showcase (L125)
if christoffels: desc += f" | Curv: {l_g.item():.4f}"
# __init__ (L138)
self.mod = vocab_size - 10 # Start of special tokens
# __init__ (L139)
self.START = self.mod + 0
# __init__ (L140)
self.END = self.mod + 1
# __init__ (L141)
self.OPEN_BRACKET = self.mod + 2
# __init__ (L142)
self.CLOSE_BRACKET = self.mod + 3
# __init__ (L143)
self.KEY_VAL_SEP = self.mod + 4
# __init__ (L144)
self.QUERY = self.mod + 5
# __init__ (L145)
self.NOISE = self.mod + 6 # "Singularity" token
# __getitem__ (L181)
pad_len = self.seq_len - len(seq) - 1 # -1 for target
# __getitem__ (L184)
seq = seq[:self.seq_len-1]
# train_showcase (L232)
optimizer = RiemannianAdam(model.parameters(), lr=3e-4, weight_decay=0.01)
# train_showcase (L269)
lm_targets = inputs[:, 1:] # [B, T-1]
# train_showcase (L273)
mask = 2**torch.arange(coord_dim).to(device)
# train_showcase (L274)
bits = (lm_targets.unsqueeze(-1) & mask) > 0
# train_showcase (L275)
target_coords = bits.float() * 2 - 1 # [B, T-1, 32]
# train_showcase (L278)
pred_coords_shifted = pred_coords[:, :-1, :] # [B, T-1, 32]
# train_showcase (L296)
v_start = model.v0.norm().item()
# train_showcase (L297)
v_end = layer0.last_v.norm(dim=-1).mean().item() if hasattr(layer0, 'last_v') else 0.0
# train_showcase (L298)
drift = abs(v_end - v_start)
# train_showcase (L317)
res_dir = PROJECT_ROOT / "tests/benchmarks/results/showcase"
# train_showcase (L342)
ckpt_path = PROJECT_ROOT / "checkpoints/showcase_v1.0.pt"
```

### demos\multimodal\multimodal_mnist.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 68 | __init__ | `layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)` |
| 75 | forward | `cls = self.cls_token.expand(B, -1, -1)` |
| 84 | log_oom | `match_gib = re.search(r"Tried to allocate ([\d\.]+) GiB", err_msg)` |
| 86 | log_oom | `match_mib = re.search(r"Tried to allocate ([\d\.]+) MiB", err_msg)` |
| 107 | benchmark_multimodal_scaling | `num_patches = (res // patch_size)**2` |
| 115 | vit_call | `x = torch.randn(1, num_patches, patch_size**2).to(device)` |
| 167 | train_multimodal_omni | `dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())` |
| 170 | train_multimodal_omni | `params = list(manifold.parameters()) + list(v_bridge.parameters()) + list(t_bridge.parameters())` |
| 171 | train_multimodal_omni | `opt = RiemannianAdam(params, lr=3e-4)` |
| 188 | train_multimodal_omni | `prompts = torch.zeros(B, 1, dtype=torch.long, device=device) + 42 # "What is this?"` |
| 199 | train_multimodal_omni | `predictions = logits[:, -1]` |
| 205 | train_multimodal_omni | `acc = (predictions.argmax(1) == labels).float().mean().item()` |
| 212 | train_multimodal_omni | `total_loss = loss_ce + loss_ham + loss_geo` |
| 225 | train_multimodal_omni | `out_dir = PROJECT_ROOT / "tests/benchmarks/results/multimodal"` |
| 237 | train_multimodal_omni | `px_counts = [(r//4)**2 for r in res]` |
| 238 | train_multimodal_omni | `ax2.plot(px_counts, mems["manifold"], 'o-', color='#3498DB', label="MANIFOLD (O(1))", linewidth=4, markersize=10)` |
| 239 | train_multimodal_omni | `ax2.plot(px_counts, mems["transformer"], '^-', color='#E74C3C', label="ViT (O(N^2))", linewidth=2, linestyle='--')` |
| 249 | train_multimodal_omni | `ax2.annotate('4K Inference on 200MB', xy=(px_counts[-1], mems["manifold"][-1]), xytext=(px_counts[-2], mems["manifold"][-1]*3), arrowprops=dict(facecolor='white', shrink=0.05))` |
| 254 | train_multimodal_omni | `plt.savefig(out_dir / "omni_scaling_final.png", dpi=200)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L68)
layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
# forward (L75)
cls = self.cls_token.expand(B, -1, -1)
# log_oom (L84)
match_gib = re.search(r"Tried to allocate ([\d\.]+) GiB", err_msg)
# log_oom (L86)
match_mib = re.search(r"Tried to allocate ([\d\.]+) MiB", err_msg)
# benchmark_multimodal_scaling (L107)
num_patches = (res // patch_size)**2
# vit_call (L115)
x = torch.randn(1, num_patches, patch_size**2).to(device)
# train_multimodal_omni (L167)
dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# train_multimodal_omni (L170)
params = list(manifold.parameters()) + list(v_bridge.parameters()) + list(t_bridge.parameters())
# train_multimodal_omni (L171)
opt = RiemannianAdam(params, lr=3e-4)
# train_multimodal_omni (L188)
prompts = torch.zeros(B, 1, dtype=torch.long, device=device) + 42 # "What is this?"
# train_multimodal_omni (L199)
predictions = logits[:, -1]
# train_multimodal_omni (L205)
acc = (predictions.argmax(1) == labels).float().mean().item()
# train_multimodal_omni (L212)
total_loss = loss_ce + loss_ham + loss_geo
# train_multimodal_omni (L225)
out_dir = PROJECT_ROOT / "tests/benchmarks/results/multimodal"
# train_multimodal_omni (L237)
px_counts = [(r//4)**2 for r in res]
# train_multimodal_omni (L238)
ax2.plot(px_counts, mems["manifold"], 'o-', color='#3498DB', label="MANIFOLD (O(1))", linewidth=4, markersize=10)
# train_multimodal_omni (L239)
ax2.plot(px_counts, mems["transformer"], '^-', color='#E74C3C', label="ViT (O(N^2))", linewidth=2, linestyle='--')
# train_multimodal_omni (L249)
ax2.annotate('4K Inference on 200MB', xy=(px_counts[-1], mems["manifold"][-1]), xytext=(px_counts[-2], mems["manifold"][-1]*3), arrowprops=dict(facecolor='white', shrink=0.05))
# train_multimodal_omni (L254)
plt.savefig(out_dir / "omni_scaling_final.png", dpi=200)
```

### demos\sorting\train_hyper_sorting.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | __init__ | `self.EOS = vocab_size + 1` |
| 22 | __init__ | `self.full_vocab = vocab_size + 2` |
| 34 | generate_batch | `src = full_seq[:, :-1]` |
| 39 | train_convergence | `def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):` |
| 40 | train_convergence | `optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)` |
| 45 | train_convergence | `pbar = tqdm(range(max_steps), desc=f"Hyper-Sorting Training")` |
| 53 | train_convergence | `logits = logits.reshape(-1, task.full_vocab)` |
| 54 | train_convergence | `y = y.reshape(-1)` |
| 63 | train_convergence | `normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val` |
| 67 | train_convergence | `preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1) # Assuming standard batch` |
| 70 | train_convergence | `start_idx = task.length + 1` |
| 72 | train_convergence | `true_sort = y.view(-1, 21)[:, start_idx:]` |
| 74 | train_convergence | `correct = (pred_sort == true_sort).all(dim=1).float().mean().item()` |
| 88 | run | `total_vocab = vocab_range + 2` |
| 111 | run | `train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L21)
self.EOS = vocab_size + 1
# __init__ (L22)
self.full_vocab = vocab_size + 2
# generate_batch (L34)
src = full_seq[:, :-1]
# train_convergence (L39)
def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
# train_convergence (L40)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# train_convergence (L45)
pbar = tqdm(range(max_steps), desc=f"Hyper-Sorting Training")
# train_convergence (L53)
logits = logits.reshape(-1, task.full_vocab)
# train_convergence (L54)
y = y.reshape(-1)
# train_convergence (L63)
normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
# train_convergence (L67)
preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1) # Assuming standard batch
# train_convergence (L70)
start_idx = task.length + 1
# train_convergence (L72)
true_sort = y.view(-1, 21)[:, start_idx:]
# train_convergence (L74)
correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
# run (L88)
total_vocab = vocab_range + 2
# run (L111)
train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)
```

### demos\sorting\train_inf_sorting.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 27 | __init__ | `self.EOS = vocab_size + 1 # Token for end` |
| 34 | __getitem__ | `vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]` |
| 39 | __getitem__ | `full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]` |
| 41 | __getitem__ | `src = torch.tensor(full_seq[:-1], dtype=torch.long)` |
| 50 | get_binary_coords | `mask = 2**torch.arange(coord_dim).to(device)` |
| 51 | get_binary_coords | `bits = (token_ids.unsqueeze(-1) & mask) > 0` |
| 56 | main | `parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')` |
| 66 | main | `real_vocab_size = vocab_range + 2` |
| 78 | main | `coord_dim = config['physics']['embedding']['coord_dim'] # 16 bits -> 65k vocab` |
| 105 | main | `optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)` |
| 107 | main | `scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=config['training']['lr'] * 10, # Aggressive peak steps_per_epoch=len(train_loader), epochs=config['training']['epochs'], pct_start=0.3 )` |
| 123 | main | `pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")` |
| 141 | main | `total_loss += loss.item()` |
| 168 | main | `pred_sorted_logits = pred[:, start_check : start_check + seq_len]` |
| 169 | main | `tgt_sorted_bits = tgt_bits[:, start_check : start_check + seq_len]` |
| 174 | main | `correct_bits += (pred_bits == tgt_sorted_bits).sum().item()` |
| 175 | main | `total_bits += tgt_sorted_bits.numel()` |
| 179 | main | `token_matches = torch.all(pred_bits == tgt_sorted_bits, dim=-1)` |
| 180 | main | `correct_tokens += token_matches.sum().item()` |
| 181 | main | `total_tokens += token_matches.numel()` |
| 185 | main | `seq_matches = torch.all(token_matches, dim=-1)` |
| 186 | main | `correct_seqs += seq_matches.sum().item()` |
| 187 | main | `total_seqs += src.size(0)` |
| 189 | main | `bit_acc = correct_bits / total_bits` |
| 190 | main | `token_acc = correct_tokens / total_tokens` |
| 191 | main | `seq_acc = correct_seqs / total_seqs` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L27)
self.EOS = vocab_size + 1 # Token for end
# __getitem__ (L34)
vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
# __getitem__ (L39)
full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]
# __getitem__ (L41)
src = torch.tensor(full_seq[:-1], dtype=torch.long)
# get_binary_coords (L50)
mask = 2**torch.arange(coord_dim).to(device)
# get_binary_coords (L51)
bits = (token_ids.unsqueeze(-1) & mask) > 0
# main (L56)
parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')
# main (L66)
real_vocab_size = vocab_range + 2
# main (L78)
coord_dim = config['physics']['embedding']['coord_dim'] # 16 bits -> 65k vocab
# main (L105)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
# main (L107)
scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=config['training']['lr'] * 10, # Aggressive peak steps_per_epoch=len(train_loader), epochs=config['training']['epochs'], pct_start=0.3 )
# main (L123)
pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
# main (L141)
total_loss += loss.item()
# main (L168)
pred_sorted_logits = pred[:, start_check : start_check + seq_len]
# main (L169)
tgt_sorted_bits = tgt_bits[:, start_check : start_check + seq_len]
# main (L174)
correct_bits += (pred_bits == tgt_sorted_bits).sum().item()
# main (L175)
total_bits += tgt_sorted_bits.numel()
# main (L179)
token_matches = torch.all(pred_bits == tgt_sorted_bits, dim=-1)
# main (L180)
correct_tokens += token_matches.sum().item()
# main (L181)
total_tokens += token_matches.numel()
# main (L185)
seq_matches = torch.all(token_matches, dim=-1)
# main (L186)
correct_seqs += seq_matches.sum().item()
# main (L187)
total_seqs += src.size(0)
# main (L189)
bit_acc = correct_bits / total_bits
# main (L190)
token_acc = correct_tokens / total_tokens
# main (L191)
seq_acc = correct_seqs / total_seqs
```

### demos\sorting\train_mom_sorting.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | __init__ | `self.EOS = vocab_size + 1` |
| 22 | __init__ | `self.full_vocab = vocab_size + 2` |
| 34 | generate_batch | `src = full_seq[:, :-1]` |
| 39 | train_convergence | `def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):` |
| 40 | train_convergence | `optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)` |
| 45 | train_convergence | `pbar = tqdm(range(max_steps), desc=f"MoM-Sorting Training")` |
| 53 | train_convergence | `logits = logits.reshape(-1, task.full_vocab)` |
| 54 | train_convergence | `y = y.reshape(-1)` |
| 63 | train_convergence | `normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val` |
| 67 | train_convergence | `preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1) # Assuming standard batch` |
| 69 | train_convergence | `start_idx = task.length + 1` |
| 71 | train_convergence | `true_sort = y.view(-1, 21)[:, start_idx:]` |
| 73 | train_convergence | `correct = (pred_sort == true_sort).all(dim=1).float().mean().item()` |
| 87 | run | `total_vocab = vocab_range + 2` |
| 119 | run | `train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L21)
self.EOS = vocab_size + 1
# __init__ (L22)
self.full_vocab = vocab_size + 2
# generate_batch (L34)
src = full_seq[:, :-1]
# train_convergence (L39)
def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
# train_convergence (L40)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# train_convergence (L45)
pbar = tqdm(range(max_steps), desc=f"MoM-Sorting Training")
# train_convergence (L53)
logits = logits.reshape(-1, task.full_vocab)
# train_convergence (L54)
y = y.reshape(-1)
# train_convergence (L63)
normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
# train_convergence (L67)
preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1) # Assuming standard batch
# train_convergence (L69)
start_idx = task.length + 1
# train_convergence (L71)
true_sort = y.view(-1, 21)[:, start_idx:]
# train_convergence (L73)
correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
# run (L87)
total_vocab = vocab_range + 2
# run (L119)
train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)
```

### demos\sorting\train_sorting.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 39 | _generate_samples | `vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]` |
| 88 | __init__ | `self.EOS = vocab_size + 1 # Token for end` |
| 97 | __getitem__ | `vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]` |
| 103 | __getitem__ | `full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]` |
| 105 | __getitem__ | `src = torch.tensor(full_seq[:-1], dtype=torch.long)` |
| 112 | train_sorting | `parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')` |
| 123 | train_sorting | `real_vocab_size = vocab_range + 2 # + SEP, EOS` |
| 148 | train_sorting | `optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)` |
| 163 | train_sorting | `loss = criterion(logits.view(-1, real_vocab_size), tgt.view(-1))` |
| 168 | train_sorting | `total_loss += loss.item()` |
| 181 | train_sorting | `preds = torch.argmax(logits, dim=-1)` |
| 190 | train_sorting | `sorted_preds = preds[:, seq_len + 1 : seq_len + 1 + seq_len]` |
| 191 | train_sorting | `sorted_tgts = tgt[:, seq_len + 1 : seq_len + 1 + seq_len]` |
| 195 | train_sorting | `correct += row_matches.sum().item()` |
| 196 | train_sorting | `total += src.size(0)` |
| 198 | train_sorting | `val_acc = correct / total` |

#### Fórmulas Listas para Usar (Python)
```python
# _generate_samples (L39)
vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
# __init__ (L88)
self.EOS = vocab_size + 1 # Token for end
# __getitem__ (L97)
vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
# __getitem__ (L103)
full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]
# __getitem__ (L105)
src = torch.tensor(full_seq[:-1], dtype=torch.long)
# train_sorting (L112)
parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')
# train_sorting (L123)
real_vocab_size = vocab_range + 2 # + SEP, EOS
# train_sorting (L148)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
# train_sorting (L163)
loss = criterion(logits.view(-1, real_vocab_size), tgt.view(-1))
# train_sorting (L168)
total_loss += loss.item()
# train_sorting (L181)
preds = torch.argmax(logits, dim=-1)
# train_sorting (L190)
sorted_preds = preds[:, seq_len + 1 : seq_len + 1 + seq_len]
# train_sorting (L191)
sorted_tgts = tgt[:, seq_len + 1 : seq_len + 1 + seq_len]
# train_sorting (L195)
correct += row_matches.sum().item()
# train_sorting (L196)
total += src.size(0)
# train_sorting (L198)
val_acc = correct / total
```

### demos\sorting\train_sorting_v2.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | __init__ | `self.EOS = vocab_size + 1` |
| 22 | __init__ | `self.full_vocab = vocab_size + 2` |
| 51 | generate_batch | `src = full_seq[:, :-1]` |
| 56 | train_until_convergence | `def train_until_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):` |
| 58 | train_until_convergence | `optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)` |
| 72 | train_until_convergence | `loss = criterion(logits.reshape(-1, task.full_vocab), y.reshape(-1))` |
| 81 | train_until_convergence | `normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val` |
| 86 | train_until_convergence | `preds = torch.argmax(logits, dim=-1)` |
| 88 | train_until_convergence | `start_idx = task.length + 1` |
| 93 | train_until_convergence | `correct = (pred_sort == true_sort).all(dim=1).float().mean().item()` |
| 114 | run_sorting_v2 | `total_vocab = vocab_range + 2` |
| 131 | run_sorting_v2 | `train_until_convergence(model, task, max_steps=5000, lr=3e-3, device=device)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L21)
self.EOS = vocab_size + 1
# __init__ (L22)
self.full_vocab = vocab_size + 2
# generate_batch (L51)
src = full_seq[:, :-1]
# train_until_convergence (L56)
def train_until_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
# train_until_convergence (L58)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# train_until_convergence (L72)
loss = criterion(logits.reshape(-1, task.full_vocab), y.reshape(-1))
# train_until_convergence (L81)
normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
# train_until_convergence (L86)
preds = torch.argmax(logits, dim=-1)
# train_until_convergence (L88)
start_idx = task.length + 1
# train_until_convergence (L93)
correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
# run_sorting_v2 (L114)
total_vocab = vocab_range + 2
# run_sorting_v2 (L131)
train_until_convergence(model, task, max_steps=5000, lr=3e-3, device=device)
```

### demos\sorting\train_transformer_sorting.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | __init__ | `self.EOS = vocab_size + 1` |
| 22 | __init__ | `self.full_vocab = vocab_size + 2` |
| 34 | generate_batch | `src = full_seq[:, :-1]` |
| 39 | train_convergence | `def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):` |
| 41 | train_convergence | `optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)` |
| 54 | train_convergence | `logits = logits.reshape(-1, task.full_vocab)` |
| 55 | train_convergence | `y = y.reshape(-1)` |
| 64 | train_convergence | `normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val` |
| 68 | train_convergence | `preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1)` |
| 70 | train_convergence | `start_idx = task.length + 1` |
| 72 | train_convergence | `true_sort = y.view(-1, 21)[:, start_idx:]` |
| 74 | train_convergence | `correct = (pred_sort == true_sort).all(dim=1).float().mean().item()` |
| 88 | run | `total_vocab = vocab_range + 2` |
| 103 | run | `train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L21)
self.EOS = vocab_size + 1
# __init__ (L22)
self.full_vocab = vocab_size + 2
# generate_batch (L34)
src = full_seq[:, :-1]
# train_convergence (L39)
def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
# train_convergence (L41)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# train_convergence (L54)
logits = logits.reshape(-1, task.full_vocab)
# train_convergence (L55)
y = y.reshape(-1)
# train_convergence (L64)
normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
# train_convergence (L68)
preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1)
# train_convergence (L70)
start_idx = task.length + 1
# train_convergence (L72)
true_sort = y.view(-1, 21)[:, start_idx:]
# train_convergence (L74)
correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
# run (L88)
total_vocab = vocab_range + 2
# run (L103)
train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)
```

### demos\tinystories\train_tinystories.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 61 | read_tokens | `dataset = load_dataset('roneneldan/TinyStories', split=self.split, trust_remote_code=True)` |
| 72 | read_tokens | `chunk = dataset[i:min(i+chunk_size, total_stories)]` |
| 74 | read_tokens | `chunk_tokens = re.findall(r'\S+', chunk_text.lower())` |
| 88 | __init__ | `self.num_samples = len(data) // seq_len` |
| 94 | __getitem__ | `start = idx * self.seq_len` |
| 95 | __getitem__ | `chunk = self.data[start : start + self.seq_len + 1]` |
| 97 | __getitem__ | `src = chunk[:-1]` |
| 102 | __getitem__ | `pad_len = self.seq_len - len(src)` |
| 110 | train_tinystories | `parser.add_argument('--config', type=str, default='configs/demos/tinystories.yaml')` |
| 136 | train_tinystories | `train_data = torch.clamp(train_data, 0, len(vocab) - 1)` |
| 137 | train_tinystories | `val_data = torch.clamp(val_data, 0, len(vocab) - 1)` |
| 166 | train_tinystories | `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )` |
| 185 | train_tinystories | `loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))` |
| 191 | train_tinystories | `total_loss += loss.item()` |
| 194 | train_tinystories | `cur_loss = total_loss / (i + 1)` |
| 204 | train_tinystories | `loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))` |
| 205 | train_tinystories | `val_loss += loss.item()` |
| 207 | train_tinystories | `val_loss /= len(val_loader)` |
| 208 | train_tinystories | `ppl = torch.exp(torch.tensor(val_loss))` |

#### Fórmulas Listas para Usar (Python)
```python
# read_tokens (L61)
dataset = load_dataset('roneneldan/TinyStories', split=self.split, trust_remote_code=True)
# read_tokens (L72)
chunk = dataset[i:min(i+chunk_size, total_stories)]
# read_tokens (L74)
chunk_tokens = re.findall(r'\S+', chunk_text.lower())
# __init__ (L88)
self.num_samples = len(data) // seq_len
# __getitem__ (L94)
start = idx * self.seq_len
# __getitem__ (L95)
chunk = self.data[start : start + self.seq_len + 1]
# __getitem__ (L97)
src = chunk[:-1]
# __getitem__ (L102)
pad_len = self.seq_len - len(src)
# train_tinystories (L110)
parser.add_argument('--config', type=str, default='configs/demos/tinystories.yaml')
# train_tinystories (L136)
train_data = torch.clamp(train_data, 0, len(vocab) - 1)
# train_tinystories (L137)
val_data = torch.clamp(val_data, 0, len(vocab) - 1)
# train_tinystories (L166)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )
# train_tinystories (L185)
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# train_tinystories (L191)
total_loss += loss.item()
# train_tinystories (L194)
cur_loss = total_loss / (i + 1)
# train_tinystories (L204)
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# train_tinystories (L205)
val_loss += loss.item()
# train_tinystories (L207)
val_loss /= len(val_loader)
# train_tinystories (L208)
ppl = torch.exp(torch.tensor(val_loss))
```

### demos\wikitext103\train_wikitext103.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 60 | read_tokens | `dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=self.split, trust_remote_code=True)` |
| 68 | read_tokens | `tokens = re.findall(r'\S+', all_text.lower())` |
| 77 | __init__ | `self.num_samples = len(data) // seq_len` |
| 83 | __getitem__ | `start = idx * self.seq_len` |
| 85 | __getitem__ | `chunk = self.data[start : start + self.seq_len + 1]` |
| 88 | __getitem__ | `src = chunk[:-1]` |
| 93 | __getitem__ | `pad_len = self.seq_len - len(src)` |
| 101 | train_wikitext103 | `parser.add_argument('--config', type=str, default='configs/demos/wikitext103.yaml')` |
| 127 | train_wikitext103 | `train_data = torch.clamp(train_data, 0, len(vocab) - 1)` |
| 128 | train_wikitext103 | `val_data = torch.clamp(val_data, 0, len(vocab) - 1)` |
| 157 | train_wikitext103 | `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )` |
| 176 | train_wikitext103 | `loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))` |
| 182 | train_wikitext103 | `total_loss += loss.item()` |
| 185 | train_wikitext103 | `cur_loss = total_loss / (i + 1)` |
| 195 | train_wikitext103 | `loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))` |
| 196 | train_wikitext103 | `val_loss += loss.item()` |
| 198 | train_wikitext103 | `val_loss /= len(val_loader)` |
| 199 | train_wikitext103 | `ppl = torch.exp(torch.tensor(val_loss))` |

#### Fórmulas Listas para Usar (Python)
```python
# read_tokens (L60)
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=self.split, trust_remote_code=True)
# read_tokens (L68)
tokens = re.findall(r'\S+', all_text.lower())
# __init__ (L77)
self.num_samples = len(data) // seq_len
# __getitem__ (L83)
start = idx * self.seq_len
# __getitem__ (L85)
chunk = self.data[start : start + self.seq_len + 1]
# __getitem__ (L88)
src = chunk[:-1]
# __getitem__ (L93)
pad_len = self.seq_len - len(src)
# train_wikitext103 (L101)
parser.add_argument('--config', type=str, default='configs/demos/wikitext103.yaml')
# train_wikitext103 (L127)
train_data = torch.clamp(train_data, 0, len(vocab) - 1)
# train_wikitext103 (L128)
val_data = torch.clamp(val_data, 0, len(vocab) - 1)
# train_wikitext103 (L157)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )
# train_wikitext103 (L176)
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# train_wikitext103 (L182)
total_loss += loss.item()
# train_wikitext103 (L185)
cur_loss = total_loss / (i + 1)
# train_wikitext103 (L195)
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# train_wikitext103 (L196)
val_loss += loss.item()
# train_wikitext103 (L198)
val_loss /= len(val_loader)
# train_wikitext103 (L199)
ppl = torch.exp(torch.tensor(val_loss))
```

### demos\wikitext\generate.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 53 | generate_text | `tokens = re.findall(r'\S+', prompt.lower())` |
| 65 | generate_text | `next_token_logits = logits[0, -1, :] / temperature` |
| 69 | generate_text | `cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)` |
| 73 | generate_text | `sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()` |
| 77 | generate_text | `next_token_logits[indices_to_remove] = float('-inf')` |
| 80 | generate_text | `probs = F.softmax(next_token_logits, dim=-1)` |
| 87 | generate_text | `if next_token.item() == vocab.stoi.get('<eos>', -1):` |
| 100 | main | `config_path = 'configs/demos/wikitext.yaml'` |
| 123 | main | `checkpoint_path = f"{config['training']['save_dir']}/checkpoint_epoch_93.pt"` |

#### Fórmulas Listas para Usar (Python)
```python
# generate_text (L53)
tokens = re.findall(r'\S+', prompt.lower())
# generate_text (L65)
next_token_logits = logits[0, -1, :] / temperature
# generate_text (L69)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
# generate_text (L73)
sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
# generate_text (L77)
next_token_logits[indices_to_remove] = float('-inf')
# generate_text (L80)
probs = F.softmax(next_token_logits, dim=-1)
# generate_text (L87)
if next_token.item() == vocab.stoi.get('<eos>', -1):
# main (L100)
config_path = 'configs/demos/wikitext.yaml'
# main (L123)
checkpoint_path = f"{config['training']['save_dir']}/checkpoint_epoch_93.pt"
```

### demos\wikitext\train_wikitext.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 46 | WikiText2Custom | `URL = 'https://s3.amazonaws.com/fast-ai-nlp/wikitext-2.tgz'` |
| 50 | __init__ | `self.dataset_path = self.root / 'wikitext-2'` |
| 56 | download | `train_file = self.dataset_path / 'train.csv'` |
| 60 | download | `archive_path = self.root / 'wikitext-2.tgz'` |
| 70 | read_tokens | `file_path = self.dataset_path / file_map[self.split]` |
| 73 | read_tokens | `with open(file_path, 'r', encoding='utf-8') as f:` |
| 79 | read_tokens | `tokens = re.findall(r'\S+', text.lower())` |
| 86 | __init__ | `self.num_samples = len(data) // seq_len` |
| 92 | __getitem__ | `start = idx * self.seq_len` |
| 94 | __getitem__ | `chunk = self.data[start : start + self.seq_len + 1]` |
| 97 | __getitem__ | `src = chunk[:-1]` |
| 102 | __getitem__ | `pad_len = self.seq_len - len(src)` |
| 110 | train_wikitext | `parser.add_argument('--config', type=str, default='configs/demos/wikitext.yaml')` |
| 136 | train_wikitext | `train_data = torch.clamp(train_data, 0, len(vocab) - 1)` |
| 137 | train_wikitext | `val_data = torch.clamp(val_data, 0, len(vocab) - 1)` |
| 166 | train_wikitext | `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )` |
| 185 | train_wikitext | `loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))` |
| 191 | train_wikitext | `total_loss += loss.item()` |
| 194 | train_wikitext | `cur_loss = total_loss / (i + 1)` |
| 204 | train_wikitext | `loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))` |
| 205 | train_wikitext | `val_loss += loss.item()` |
| 207 | train_wikitext | `val_loss /= len(val_loader)` |
| 208 | train_wikitext | `ppl = torch.exp(torch.tensor(val_loss))` |

#### Fórmulas Listas para Usar (Python)
```python
# WikiText2Custom (L46)
URL = 'https://s3.amazonaws.com/fast-ai-nlp/wikitext-2.tgz'
# __init__ (L50)
self.dataset_path = self.root / 'wikitext-2'
# download (L56)
train_file = self.dataset_path / 'train.csv'
# download (L60)
archive_path = self.root / 'wikitext-2.tgz'
# read_tokens (L70)
file_path = self.dataset_path / file_map[self.split]
# read_tokens (L73)
with open(file_path, 'r', encoding='utf-8') as f:
# read_tokens (L79)
tokens = re.findall(r'\S+', text.lower())
# __init__ (L86)
self.num_samples = len(data) // seq_len
# __getitem__ (L92)
start = idx * self.seq_len
# __getitem__ (L94)
chunk = self.data[start : start + self.seq_len + 1]
# __getitem__ (L97)
src = chunk[:-1]
# __getitem__ (L102)
pad_len = self.seq_len - len(src)
# train_wikitext (L110)
parser.add_argument('--config', type=str, default='configs/demos/wikitext.yaml')
# train_wikitext (L136)
train_data = torch.clamp(train_data, 0, len(vocab) - 1)
# train_wikitext (L137)
val_data = torch.clamp(val_data, 0, len(vocab) - 1)
# train_wikitext (L166)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )
# train_wikitext (L185)
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# train_wikitext (L191)
total_loss += loss.item()
# train_wikitext (L194)
cur_loss = total_loss / (i + 1)
# train_wikitext (L204)
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# train_wikitext (L205)
val_loss += loss.item()
# train_wikitext (L207)
val_loss /= len(val_loader)
# train_wikitext (L208)
ppl = torch.exp(torch.tensor(val_loss))
```

### examples\load_checkpoint.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 52 | load_manifold_checkpoint | `checkpoint_path = PROJECT_ROOT / "checkpoints" / "manifold_parity_superiority.pt"` |
| 83 | load_manifold_checkpoint | `print(f"   Accuracy: {(preds == expected).float().mean().item() * 100:.1f}%")` |

#### Fórmulas Listas para Usar (Python)
```python
# load_manifold_checkpoint (L52)
checkpoint_path = PROJECT_ROOT / "checkpoints" / "manifold_parity_superiority.pt"
# load_manifold_checkpoint (L83)
print(f"   Accuracy: {(preds == expected).float().mean().item() * 100:.1f}%")
```

### gfn\__init__.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 20 | Global | `optimizer = RiemannianAdam(model.parameters(), lr=1e-3)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L20)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
```

### gfn\aggregation\geodesic_attention.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 68 | euclidean_distance | `dist_sq = ((x1_exp - x2_exp) ** 2).sum(dim=-1)  # [B, L, L]` |
| 69 | euclidean_distance | `dist = torch.sqrt(dist_sq + 1e-8)  # Add epsilon for numerical stability` |
| 91 | riemannian_distance | `similarity = torch.bmm(Q, K.transpose(1, 2)) / (self.dim ** 0.5)  # [B, L, L]` |
| 94 | riemannian_distance | `dist = -similarity` |
| 122 | forward | `attn_weights = F.softmax(-dist / self.temperature, dim=-1)  # [B, L, L]` |
| 128 | forward | `x_attended = torch.bmm(attn_weights, V)  # [B, L, dim]` |
| 130 | forward | `x_agg = x_attended[:, -1]  # [B, dim]` |
| 133 | forward | `x_attended = torch.bmm(attn_weights, x_seq)  # [B, L, dim]` |
| 134 | forward | `x_agg = x_attended[:, -1]  # [B, dim]` |
| 138 | forward | `v_attended = torch.bmm(attn_weights, v_seq)  # [B, L, dim]` |
| 139 | forward | `v_agg = v_attended[:, -1]  # [B, dim]` |

#### Fórmulas Listas para Usar (Python)
```python
# euclidean_distance (L68)
dist_sq = ((x1_exp - x2_exp) ** 2).sum(dim=-1)  # [B, L, L]
# euclidean_distance (L69)
dist = torch.sqrt(dist_sq + 1e-8)  # Add epsilon for numerical stability
# riemannian_distance (L91)
similarity = torch.bmm(Q, K.transpose(1, 2)) / (self.dim ** 0.5)  # [B, L, L]
# riemannian_distance (L94)
dist = -similarity
# forward (L122)
attn_weights = F.softmax(-dist / self.temperature, dim=-1)  # [B, L, L]
# forward (L128)
x_attended = torch.bmm(attn_weights, V)  # [B, L, dim]
# forward (L130)
x_agg = x_attended[:, -1]  # [B, dim]
# forward (L133)
x_attended = torch.bmm(attn_weights, x_seq)  # [B, L, dim]
# forward (L134)
x_agg = x_attended[:, -1]  # [B, dim]
# forward (L138)
v_attended = torch.bmm(attn_weights, v_seq)  # [B, L, dim]
# forward (L139)
v_agg = v_attended[:, -1]  # [B, dim]
```

### gfn\aggregation\hamiltonian_pooling.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 25 | HamiltonianPooling | `H = K + U where:` |
| 26 | HamiltonianPooling | `- K = (1/2) v^T g v  (kinetic energy)` |
| 27 | HamiltonianPooling | `- U = (1/2) \|\|x\|\|^2  (potential energy)` |
| 56 | kinetic_energy | `Compute kinetic energy: K = 0.5 * v^T @ g @ v` |
| 66 | kinetic_energy | `metric_expanded = self.metric.view(1, 1, -1).expand(B, L, -1)  # [B, L, dim]` |
| 67 | kinetic_energy | `weighted_v = v * metric_expanded  # [B, L, dim]` |
| 68 | kinetic_energy | `K = 0.5 * (v * weighted_v).sum(dim=-1)  # [B, L]` |
| 73 | potential_energy | `Compute potential energy: U = 0.5 * \|\|x\|\|^2` |
| 83 | potential_energy | `U = 0.5 * (x ** 2).sum(dim=-1)  # [B, L]` |
| 104 | forward | `H = K + U  # [B, L] - Total Hamiltonian` |
| 107 | forward | `weights = F.softmax(H / self.temperature, dim=-1)  # [B, L]` |
| 110 | forward | `x_agg = (weights.unsqueeze(-1) * x_seq).sum(dim=1)  # [B, dim]` |
| 111 | forward | `v_agg = (weights.unsqueeze(-1) * v_seq).sum(dim=1)  # [B, dim]` |

#### Fórmulas Listas para Usar (Python)
```python
# HamiltonianPooling (L25)
H = K + U where:
# HamiltonianPooling (L26)
- K = (1/2) v^T g v  (kinetic energy)
# HamiltonianPooling (L27)
- U = (1/2) ||x||^2  (potential energy)
# kinetic_energy (L56)
Compute kinetic energy: K = 0.5 * v^T @ g @ v
# kinetic_energy (L66)
metric_expanded = self.metric.view(1, 1, -1).expand(B, L, -1)  # [B, L, dim]
# kinetic_energy (L67)
weighted_v = v * metric_expanded  # [B, L, dim]
# kinetic_energy (L68)
K = 0.5 * (v * weighted_v).sum(dim=-1)  # [B, L]
# potential_energy (L73)
Compute potential energy: U = 0.5 * ||x||^2
# potential_energy (L83)
U = 0.5 * (x ** 2).sum(dim=-1)  # [B, L]
# forward (L104)
H = K + U  # [B, L] - Total Hamiltonian
# forward (L107)
weights = F.softmax(H / self.temperature, dim=-1)  # [B, L]
# forward (L110)
x_agg = (weights.unsqueeze(-1) * x_seq).sum(dim=1)  # [B, dim]
# forward (L111)
v_agg = (weights.unsqueeze(-1) * v_seq).sum(dim=1)  # [B, dim]
```

### gfn\aggregation\momentum_accumulation.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 11 | Global | `- Total impulse = ∫ force dt determines final momentum` |
| 25 | MomentumAccumulation | `x_final = x_last + alpha * accumulated_states` |
| 58 | __init__ | `self.gate = nn.Sequential( nn.Linear(dim * 2, dim),  # Input: [x_final, v_final] concatenated nn.Tanh(), nn.Linear(dim, 1), nn.Sigmoid()  # Output: scalar gate in [0, 1] )` |
| 86 | forward | `accumulated_states = x_seq.sum(dim=1)  # [B, dim]` |
| 88 | forward | `accumulated_states = x_seq.mean(dim=1)  # [B, dim]` |
| 91 | forward | `x_last = x_seq[:, -1]  # [B, dim]` |
| 97 | forward | `gate_input = torch.cat([x_last, accumulated_states], dim=-1)  # [B, 2*dim]` |
| 99 | forward | `effective_alpha = self.alpha * gate_value` |
| 105 | forward | `x_final = x_last + effective_alpha * accumulated_states` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L11)
- Total impulse = ∫ force dt determines final momentum
# MomentumAccumulation (L25)
x_final = x_last + alpha * accumulated_states
# __init__ (L58)
self.gate = nn.Sequential( nn.Linear(dim * 2, dim),  # Input: [x_final, v_final] concatenated nn.Tanh(), nn.Linear(dim, 1), nn.Sigmoid()  # Output: scalar gate in [0, 1] )
# forward (L86)
accumulated_states = x_seq.sum(dim=1)  # [B, dim]
# forward (L88)
accumulated_states = x_seq.mean(dim=1)  # [B, dim]
# forward (L91)
x_last = x_seq[:, -1]  # [B, dim]
# forward (L97)
gate_input = torch.cat([x_last, accumulated_states], dim=-1)  # [B, 2*dim]
# forward (L99)
effective_alpha = self.alpha * gate_value
# forward (L105)
x_final = x_last + effective_alpha * accumulated_states
```

### gfn\constants.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 15 | Global | `- EPSILON_STANDARD = 1e-8 (division safety)` |
| 16 | Global | `- EPSILON_STRONG   = 1e-8 (strong division protection)` |
| 17 | Global | `- EPSILON_SMOOTH   = 1e-8 (gradient smoothing)` |
| 18 | Global | `- CLAMP_MIN_STRONG = 1e-8 (minimum denominators)` |
| 22 | Global | `- ADAM_EPSILON = 1e-7 (optimizer-specific, documented choice)` |
| 76 | Global | `FRICTION_SCALE = 0.02  # Was 5.0, then 0.5, then 0.05 - Now optimal` |
| 95 | Global | `EPSILON_STRONG = 1e-7  # Was 1e-8 - Better balance` |
| 98 | Global | `EPSILON_STANDARD = 1e-7  # Was 1e-8 - Match strong for consistency` |
| 101 | Global | `EPSILON_SMOOTH = 1e-7` |
| 104 | Global | `CLAMP_MIN_STRONG = 1e-7` |
| 107 | Global | `CLAMP_MIN_STANDARD = 1e-7` |
| 116 | Global | `LAMBDA_H_DEFAULT = 0.0  # Was 0.001 - Disabled for clean convergence` |
| 119 | Global | `LAMBDA_G_DEFAULT = 0.00005  # Was 0.0001 - Lower for better curvature preservation` |
| 125 | Global | `LAMBDA_K_DEFAULT = 0.0001  # Was 0.001 - Reduced for stability` |
| 136 | Global | `DEFAULT_LR = 1e-4  # Was 1e-3` |
| 142 | Global | `ADAM_BETA2 = 0.99  # Was 0.999 - increased for stability` |
| 145 | Global | `ADAM_EPSILON = 1e-7  # Was 1e-8` |
| 168 | Global | `GATE_BIAS_OPEN = 1.0  # sigmoid(1.0) ≈ 0.73 - Was 2.0` |
| 171 | Global | `GATE_BIAS_CLOSED = -3.0  # sigmoid(-3.0) ≈ 0.05 - Was -5.0` |
| 180 | Global | `DEFAULT_DT = 0.05  # Was 0.02 - Better exploration while maintaining stability` |
| 187 | Global | `LEAPFROG_SUBSTEPS = 3  # Was 5 - Cleaner backward pass` |
| 198 | Global | `DEFAULT_PLASTICITY = 0.02  # Was 0.01 - Better responsiveness` |
| 201 | Global | `SINGULARITY_THRESHOLD = 0.5  # Was 0.8 - Lower threshold for earlier activation` |
| 204 | Global | `BLACK_HOLE_STRENGTH = 1.5  # Was 2.0 - Reduced for stability` |
| 218 | Global | `SINGULARITY_GATE_SLOPE = 0.5  # Was 1.0 - Smoother transitions` |
| 244 | Global | `VELOCITY_SATURATION = 100.0  # Was 50.0 - Allow higher velocities` |
| 263 | Global | `HYSTERESIS_FORGET_GATE_INIT = 0.9  # sigmoid(2.0) ≈ 0.88 - gradual decay` |
| 282 | Global | `TOROIDAL_PERIOD = 6.283185307179586  # 2 * π` |
| 338 | get_stable_lr_scale | `warmup_steps = int(total_steps * warmup_ratio)` |
| 345 | get_stable_lr_scale | `decay_steps = total_steps - warmup_steps` |
| 346 | get_stable_lr_scale | `decay_progress = float(step - warmup_steps) / max(1, decay_steps)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L15)
- EPSILON_STANDARD = 1e-8 (division safety)
# Global (L16)
- EPSILON_STRONG   = 1e-8 (strong division protection)
# Global (L17)
- EPSILON_SMOOTH   = 1e-8 (gradient smoothing)
# Global (L18)
- CLAMP_MIN_STRONG = 1e-8 (minimum denominators)
# Global (L22)
- ADAM_EPSILON = 1e-7 (optimizer-specific, documented choice)
# Global (L76)
FRICTION_SCALE = 0.02  # Was 5.0, then 0.5, then 0.05 - Now optimal
# Global (L95)
EPSILON_STRONG = 1e-7  # Was 1e-8 - Better balance
# Global (L98)
EPSILON_STANDARD = 1e-7  # Was 1e-8 - Match strong for consistency
# Global (L101)
EPSILON_SMOOTH = 1e-7
# Global (L104)
CLAMP_MIN_STRONG = 1e-7
# Global (L107)
CLAMP_MIN_STANDARD = 1e-7
# Global (L116)
LAMBDA_H_DEFAULT = 0.0  # Was 0.001 - Disabled for clean convergence
# Global (L119)
LAMBDA_G_DEFAULT = 0.00005  # Was 0.0001 - Lower for better curvature preservation
# Global (L125)
LAMBDA_K_DEFAULT = 0.0001  # Was 0.001 - Reduced for stability
# Global (L136)
DEFAULT_LR = 1e-4  # Was 1e-3
# Global (L142)
ADAM_BETA2 = 0.99  # Was 0.999 - increased for stability
# Global (L145)
ADAM_EPSILON = 1e-7  # Was 1e-8
# Global (L168)
GATE_BIAS_OPEN = 1.0  # sigmoid(1.0) ≈ 0.73 - Was 2.0
# Global (L171)
GATE_BIAS_CLOSED = -3.0  # sigmoid(-3.0) ≈ 0.05 - Was -5.0
# Global (L180)
DEFAULT_DT = 0.05  # Was 0.02 - Better exploration while maintaining stability
# Global (L187)
LEAPFROG_SUBSTEPS = 3  # Was 5 - Cleaner backward pass
# Global (L198)
DEFAULT_PLASTICITY = 0.02  # Was 0.01 - Better responsiveness
# Global (L201)
SINGULARITY_THRESHOLD = 0.5  # Was 0.8 - Lower threshold for earlier activation
# Global (L204)
BLACK_HOLE_STRENGTH = 1.5  # Was 2.0 - Reduced for stability
# Global (L218)
SINGULARITY_GATE_SLOPE = 0.5  # Was 1.0 - Smoother transitions
# Global (L244)
VELOCITY_SATURATION = 100.0  # Was 50.0 - Allow higher velocities
# Global (L263)
HYSTERESIS_FORGET_GATE_INIT = 0.9  # sigmoid(2.0) ≈ 0.88 - gradual decay
# Global (L282)
TOROIDAL_PERIOD = 6.283185307179586  # 2 * π
# get_stable_lr_scale (L338)
warmup_steps = int(total_steps * warmup_ratio)
# get_stable_lr_scale (L345)
decay_steps = total_steps - warmup_steps
# get_stable_lr_scale (L346)
decay_progress = float(step - warmup_steps) / max(1, decay_steps)
```

### gfn\core\adjoint.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 31 | GeodesicODEFunc | `dv/dt = f - Γ(v, v)` |
| 32 | GeodesicODEFunc | `df/dt = 0  (Force is constant during integration step)` |
| 45 | forward | `self.dim = state.shape[-1] // 3` |
| 49 | forward | `v = state[..., dim:2*dim]` |
| 50 | forward | `f = state[..., 2*dim:]` |
| 56 | forward | `dv_dt = f - self.christoffel(v, x, force=f)` |
| 61 | forward | `return torch.cat([dx_dt, dv_dt, df_dt], dim=-1)` |
| 84 | forward | `state = torch.cat([x, v, force], dim=-1)` |
| 96 | forward | `final_state = out[-1]` |
| 99 | forward | `dt = self.integration_time / self.n_steps` |
| 104 | forward | `k2 = self.ode_func(0, curr_state + 0.5 * dt * k1)` |
| 105 | forward | `k3 = self.ode_func(0, curr_state + 0.5 * dt * k2)` |
| 106 | forward | `k4 = self.ode_func(0, curr_state + dt * k3)` |
| 107 | forward | `curr_state = curr_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)` |
| 114 | forward | `v_out = final_state[..., dim:2*dim]` |
| 134 | __init__ | `raise NotImplementedError("AdjointManifold currently only supports heads=1. " "For Multi-Head Geodesic Flows, use standard Manifold (use_adjoint=False).")` |
| 143 | __init__ | `self.layers = nn.ModuleList([ AdjointMLayer(dim, rank=rank, integration_time=integration_time/depth, n_steps=5) for _ in range(depth) ])` |
| 156 | __init__ | `self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)` |
| 157 | __init__ | `self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)` |
| 177 | forward | `x = self.x0.expand(batch_size, -1)` |
| 178 | forward | `v = self.v0.expand(batch_size, -1)` |
| 217 | forward | `mask = attention_mask.unsqueeze(-1).float()` |
| 226 | forward | `force = all_forces[:, t] * mask[:, t]` |
| 265 | sample_next | `next_logit = logits[:, -1, :] / temp` |
| 270 | sample_next | `next_logit[next_logit < v[:, [-1]]] = -float('Inf')` |
| 275 | sample_next | `cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)` |
| 277 | sample_next | `sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()` |
| 280 | sample_next | `next_logit[indices_to_remove] = -float('Inf')` |
| 284 | sample_next | `return torch.argmax(next_logit, dim=-1, keepdim=True)` |
| 286 | sample_next | `probs = torch.softmax(next_logit, dim=-1)` |

#### Fórmulas Listas para Usar (Python)
```python
# GeodesicODEFunc (L31)
dv/dt = f - Γ(v, v)
# GeodesicODEFunc (L32)
df/dt = 0  (Force is constant during integration step)
# forward (L45)
self.dim = state.shape[-1] // 3
# forward (L49)
v = state[..., dim:2*dim]
# forward (L50)
f = state[..., 2*dim:]
# forward (L56)
dv_dt = f - self.christoffel(v, x, force=f)
# forward (L61)
return torch.cat([dx_dt, dv_dt, df_dt], dim=-1)
# forward (L84)
state = torch.cat([x, v, force], dim=-1)
# forward (L96)
final_state = out[-1]
# forward (L99)
dt = self.integration_time / self.n_steps
# forward (L104)
k2 = self.ode_func(0, curr_state + 0.5 * dt * k1)
# forward (L105)
k3 = self.ode_func(0, curr_state + 0.5 * dt * k2)
# forward (L106)
k4 = self.ode_func(0, curr_state + dt * k3)
# forward (L107)
curr_state = curr_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
# forward (L114)
v_out = final_state[..., dim:2*dim]
# __init__ (L134)
raise NotImplementedError("AdjointManifold currently only supports heads=1. " "For Multi-Head Geodesic Flows, use standard Manifold (use_adjoint=False).")
# __init__ (L143)
self.layers = nn.ModuleList([ AdjointMLayer(dim, rank=rank, integration_time=integration_time/depth, n_steps=5) for _ in range(depth) ])
# __init__ (L156)
self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)
# __init__ (L157)
self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)
# forward (L177)
x = self.x0.expand(batch_size, -1)
# forward (L178)
v = self.v0.expand(batch_size, -1)
# forward (L217)
mask = attention_mask.unsqueeze(-1).float()
# forward (L226)
force = all_forces[:, t] * mask[:, t]
# sample_next (L265)
next_logit = logits[:, -1, :] / temp
# sample_next (L270)
next_logit[next_logit < v[:, [-1]]] = -float('Inf')
# sample_next (L275)
cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
# sample_next (L277)
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
# sample_next (L280)
next_logit[indices_to_remove] = -float('Inf')
# sample_next (L284)
return torch.argmax(next_logit, dim=-1, keepdim=True)
# sample_next (L286)
probs = torch.softmax(next_logit, dim=-1)
```

### gfn\core\manifold.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 107 | __init__ | `in_dim = (3 * dim) if (self.physics_config.get('topology', {}).get('type') == 'torus') else (2 * dim)` |
| 155 | __init__ | `self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)` |
| 156 | __init__ | `self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)` |
| 248 | forward | `x_scan = self.x0.expand(batch_size, seq_len, -1)` |
| 275 | forward | `mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]` |
| 354 | forward | `force = all_forces[:, t] * mask[:, t]` |
| 367 | forward | `mem_input = torch.cat([torch.sin(x), torch.cos(x), v], dim=-1)` |
| 369 | forward | `mem_input = torch.cat([x, v], dim=-1)` |
| 371 | forward | `deformation_update = torch.tanh(self.hysteresis_update(mem_input))` |
| 373 | forward | `hysteresis_state = self.hysteresis_decay * hysteresis_state + deformation_update` |
| 425 | sample_next | `next_logit = logits[:, -1, :] / temp` |
| 426 | sample_next | `probs = torch.softmax(next_logit, dim=-1)` |
| 431 | sample_next | `next_logit[next_logit < v[:, [-1]]] = -float('Inf')` |
| 436 | sample_next | `cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)` |
| 441 | sample_next | `sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()` |
| 445 | sample_next | `next_logit[indices_to_remove] = -float('Inf')` |
| 450 | sample_next | `return torch.argmax(next_logit, dim=-1, keepdim=True)` |
| 453 | sample_next | `probs = torch.softmax(next_logit, dim=-1)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L107)
in_dim = (3 * dim) if (self.physics_config.get('topology', {}).get('type') == 'torus') else (2 * dim)
# __init__ (L155)
self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)
# __init__ (L156)
self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)
# forward (L248)
x_scan = self.x0.expand(batch_size, seq_len, -1)
# forward (L275)
mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
# forward (L354)
force = all_forces[:, t] * mask[:, t]
# forward (L367)
mem_input = torch.cat([torch.sin(x), torch.cos(x), v], dim=-1)
# forward (L369)
mem_input = torch.cat([x, v], dim=-1)
# forward (L371)
deformation_update = torch.tanh(self.hysteresis_update(mem_input))
# forward (L373)
hysteresis_state = self.hysteresis_decay * hysteresis_state + deformation_update
# sample_next (L425)
next_logit = logits[:, -1, :] / temp
# sample_next (L426)
probs = torch.softmax(next_logit, dim=-1)
# sample_next (L431)
next_logit[next_logit < v[:, [-1]]] = -float('Inf')
# sample_next (L436)
cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
# sample_next (L441)
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
# sample_next (L445)
next_logit[indices_to_remove] = -float('Inf')
# sample_next (L450)
return torch.argmax(next_logit, dim=-1, keepdim=True)
# sample_next (L453)
probs = torch.softmax(next_logit, dim=-1)
```

### gfn\cuda\autograd.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 62 | record | `self._counts[name] += 1` |
| 93 | summary | `lines = ["=" * 60, "EXECUTION TIME SUMMARY", "=" * 60]` |
| 128 | wrapper | `result = func(*args, **kwargs)` |
| 129 | wrapper | `duration = time.perf_counter() - start` |
| 536 | christoffel_fused_autograd | `r: float = 1.0) -> torch.Tensor:` |
| 587 | leapfrog_fused_autograd | `hyst_enabled: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:` |
| 664 | recurrent_manifold_fused_autograd | `head_dim = D_total // num_heads` |
| 665 | recurrent_manifold_fused_autograd | `num_layers = U_stack.shape[0] // num_heads # Should be 1 for a single MLayer` |
| 725 | recurrent_manifold_fused_autograd | `U_heads = U_stack.view(num_heads, head_dim, -1)` |
| 726 | recurrent_manifold_fused_autograd | `W_heads = W_stack.view(num_heads, head_dim, -1).permute(0, 2, 1)` |
| 730 | recurrent_manifold_fused_autograd | `Wf_heads = Wf.view(num_heads, head_dim, -1)` |
| 755 | recurrent_manifold_fused_autograd | `dt_eff = (dt * dt_scales).view(1, num_heads, 1) # [1, H, 1]` |
| 756 | recurrent_manifold_fused_autograd | `h = 0.5 * dt_eff` |
| 764 | recurrent_manifold_fused_autograd | `feat_h = torch.cat([torch.sin(x_h), torch.cos(x_h)], dim=-1)` |
| 766 | recurrent_manifold_fused_autograd | `gate = torch.einsum('bhd,hdk->bhk', feat_h, Wf_heads.transpose(1, 2))` |
| 768 | recurrent_manifold_fused_autograd | `gate = gate + bf_heads.view(1, num_heads, head_dim)` |
| 770 | recurrent_manifold_fused_autograd | `gate = gate + torch.einsum('bhd,hdk->bhk', f_h, Wi_heads.transpose(1, 2))` |
| 771 | recurrent_manifold_fused_autograd | `mu_h = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE` |
| 774 | recurrent_manifold_fused_autograd | `v_norm = torch.norm(v_h, dim=-1, keepdim=True) / (head_dim**0.5 + 1e-8)` |
| 775 | recurrent_manifold_fused_autograd | `mu_h = mu_h * (1.0 + kwargs['velocity_friction_scale'] * v_norm)` |
| 780 | recurrent_manifold_fused_autograd | `fg_h = torch.einsum('bhd,hdk->bhk', m_h, h_rd_w.view(num_heads, head_dim, head_dim).transpose(1, 2))` |
| 782 | recurrent_manifold_fused_autograd | `fg_h = fg_h + h_rd_b.view(1, num_heads, head_dim)` |
| 786 | recurrent_manifold_fused_autograd | `h_proj = torch.einsum('bhd,hdr->bhr', v_h, U_heads)` |
| 787 | recurrent_manifold_fused_autograd | `energy = torch.sum(h_proj * h_proj, dim=-1, keepdim=True) / max(1, h_proj.shape[-1])` |
| 788 | recurrent_manifold_fused_autograd | `s_norm = 1.0 / (1.0 + torch.sqrt(energy) + 1e-8)` |
| 791 | recurrent_manifold_fused_autograd | `gamma_h = torch.einsum('bhr,hrd->bhd', h_proj * h_proj, W_heads) * s_norm` |
| 798 | recurrent_manifold_fused_autograd | `T = max(thermo_temp, 1e-8)` |
| 800 | recurrent_manifold_fused_autograd | `f_energy = (f_h ** 2).mean(dim=-1, keepdim=True) # [B, H, 1]` |
| 801 | recurrent_manifold_fused_autograd | `modulator = torch.exp(-thermo_alpha * f_energy / T)` |
| 802 | recurrent_manifold_fused_autograd | `gamma_h = gamma_h * modulator` |
| 812 | recurrent_manifold_fused_autograd | `v_dot_gz = (v_h * gz_h).sum(dim=-1, keepdim=True)` |
| 813 | recurrent_manifold_fused_autograd | `v_sq = (v_h * v_h).sum(dim=-1, keepdim=True)` |
| 815 | recurrent_manifold_fused_autograd | `gamma_ads = -(1.0 / (z_h + 1e-8)) * (2.0 * v_dot_gz * v_h - v_sq * gz_h)` |
| 816 | recurrent_manifold_fused_autograd | `gamma_h = gamma_h + gamma_ads` |
| 819 | recurrent_manifold_fused_autograd | `gamma_h = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_h / CudaConstants.CURVATURE_CLAMP)` |
| 823 | recurrent_manifold_fused_autograd | `v_half = (v_h + h * (f_h + fg_h - gamma_h)) / (1.0 + h * mu_h + 1e-8)` |
| 827 | recurrent_manifold_fused_autograd | `x_new_h = x_h + dt_eff * v_half` |
| 829 | recurrent_manifold_fused_autograd | `x_new_h = torch.atan2(torch.sin(x_new_h), torch.cos(x_new_h))` |
| 839 | recurrent_manifold_fused_autograd | `feat_h = torch.cat([torch.sin(x_new_h), torch.cos(x_new_h)], dim=-1)` |
| 840 | recurrent_manifold_fused_autograd | `gate = torch.einsum('bhd,hdk->bhk', feat_h, Wf_heads.transpose(1, 2))` |
| 841 | recurrent_manifold_fused_autograd | `if bf_heads is not None: gate = gate + bf_heads.view(1, num_heads, head_dim)` |
| 842 | recurrent_manifold_fused_autograd | `if Wi_heads is not None: gate = gate + torch.einsum('bhd,hdk->bhk', f_h, Wi_heads.transpose(1, 2))` |
| 843 | recurrent_manifold_fused_autograd | `mu_h = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE` |
| 845 | recurrent_manifold_fused_autograd | `h_proj = torch.einsum('bhd,hdr->bhr', v_half, U_heads)` |
| 846 | recurrent_manifold_fused_autograd | `energy = torch.sum(h_proj * h_proj, dim=-1, keepdim=True) / max(1, h_proj.shape[-1])` |
| 847 | recurrent_manifold_fused_autograd | `s_norm = 1.0 / (1.0 + torch.sqrt(energy) + 1e-8)` |
| 848 | recurrent_manifold_fused_autograd | `gamma_h = torch.einsum('bhr,hrd->bhd', h_proj * h_proj, W_heads) * s_norm` |
| 852 | recurrent_manifold_fused_autograd | `gamma_h = gamma_h * modulator # Use same modulator as it depends on force_t` |
| 855 | recurrent_manifold_fused_autograd | `gamma_ads = -(1.0 / (z_h + 1e-8)) * (2.0 * v_dot_gz * v_half - v_sq * gz_h)` |
| 856 | recurrent_manifold_fused_autograd | `gamma_h = gamma_h + gamma_ads` |
| 859 | recurrent_manifold_fused_autograd | `gamma_h = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_h / CudaConstants.CURVATURE_CLAMP)` |
| 861 | recurrent_manifold_fused_autograd | `v_new_h = (v_half + h * (f_h + fg_h - gamma_h)) / (1.0 + h * mu_h + 1e-8)` |
| 872 | recurrent_manifold_fused_autograd | `h_in = torch.cat([torch.sin(x_new_h), torch.cos(x_new_h)], dim=-1)` |
| 873 | recurrent_manifold_fused_autograd | `h_in = torch.cat([h_in, v_new_h], dim=-1)` |
| 875 | recurrent_manifold_fused_autograd | `h_gate = torch.einsum('bhd,hdk->bhk', h_in, h_up_w.view(num_heads, head_dim, -1).transpose(1, 2))` |
| 877 | recurrent_manifold_fused_autograd | `h_gate = h_gate + h_up_b.view(1, num_heads, head_dim)` |
| 879 | recurrent_manifold_fused_autograd | `h_state = h_state * h_decay + torch.tanh(h_gate.reshape(B, D_total))` |
| 956 | toroidal_leapfrog_fused_autograd | `v_half = v_curr + 0.5 * dt * force_t` |
| 959 | toroidal_leapfrog_fused_autograd | `x_new = x_curr + dt * v_half` |
| 963 | toroidal_leapfrog_fused_autograd | `x_new = torch.atan2(torch.sin(x_new), torch.cos(x_new))` |
| 970 | toroidal_leapfrog_fused_autograd | `phi = x_new[:, i + 1]` |
| 972 | toroidal_leapfrog_fused_autograd | `v_phi = v_half[:, i + 1]` |
| 975 | toroidal_leapfrog_fused_autograd | `cos_theta = torch.cos(theta)` |
| 976 | toroidal_leapfrog_fused_autograd | `denom = torch.clamp(R + r * cos_theta, min=1e-6)` |
| 979 | toroidal_leapfrog_fused_autograd | `sin_theta = torch.sin(theta)` |
| 980 | toroidal_leapfrog_fused_autograd | `gamma_theta = denom * sin_theta / (r + 1e-6) * (v_phi * v_phi)` |
| 983 | toroidal_leapfrog_fused_autograd | `gamma_phi = -(r * sin_theta) / (denom + 1e-6) * 2.0 * v_theta * v_phi` |
| 986 | toroidal_leapfrog_fused_autograd | `gamma[:, i + 1] = gamma_phi` |
| 989 | toroidal_leapfrog_fused_autograd | `gamma = torch.clamp(gamma, -10.0, 10.0)` |
| 992 | toroidal_leapfrog_fused_autograd | `v_new = v_half + 0.5 * dt * (force_t - gamma)` |

#### Fórmulas Listas para Usar (Python)
```python
# record (L62)
self._counts[name] += 1
# summary (L93)
lines = ["=" * 60, "EXECUTION TIME SUMMARY", "=" * 60]
# wrapper (L128)
result = func(*args, **kwargs)
# wrapper (L129)
duration = time.perf_counter() - start
# christoffel_fused_autograd (L536)
r: float = 1.0) -> torch.Tensor:
# leapfrog_fused_autograd (L587)
hyst_enabled: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
# recurrent_manifold_fused_autograd (L664)
head_dim = D_total // num_heads
# recurrent_manifold_fused_autograd (L665)
num_layers = U_stack.shape[0] // num_heads # Should be 1 for a single MLayer
# recurrent_manifold_fused_autograd (L725)
U_heads = U_stack.view(num_heads, head_dim, -1)
# recurrent_manifold_fused_autograd (L726)
W_heads = W_stack.view(num_heads, head_dim, -1).permute(0, 2, 1)
# recurrent_manifold_fused_autograd (L730)
Wf_heads = Wf.view(num_heads, head_dim, -1)
# recurrent_manifold_fused_autograd (L755)
dt_eff = (dt * dt_scales).view(1, num_heads, 1) # [1, H, 1]
# recurrent_manifold_fused_autograd (L756)
h = 0.5 * dt_eff
# recurrent_manifold_fused_autograd (L764)
feat_h = torch.cat([torch.sin(x_h), torch.cos(x_h)], dim=-1)
# recurrent_manifold_fused_autograd (L766)
gate = torch.einsum('bhd,hdk->bhk', feat_h, Wf_heads.transpose(1, 2))
# recurrent_manifold_fused_autograd (L768)
gate = gate + bf_heads.view(1, num_heads, head_dim)
# recurrent_manifold_fused_autograd (L770)
gate = gate + torch.einsum('bhd,hdk->bhk', f_h, Wi_heads.transpose(1, 2))
# recurrent_manifold_fused_autograd (L771)
mu_h = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
# recurrent_manifold_fused_autograd (L774)
v_norm = torch.norm(v_h, dim=-1, keepdim=True) / (head_dim**0.5 + 1e-8)
# recurrent_manifold_fused_autograd (L775)
mu_h = mu_h * (1.0 + kwargs['velocity_friction_scale'] * v_norm)
# recurrent_manifold_fused_autograd (L780)
fg_h = torch.einsum('bhd,hdk->bhk', m_h, h_rd_w.view(num_heads, head_dim, head_dim).transpose(1, 2))
# recurrent_manifold_fused_autograd (L782)
fg_h = fg_h + h_rd_b.view(1, num_heads, head_dim)
# recurrent_manifold_fused_autograd (L786)
h_proj = torch.einsum('bhd,hdr->bhr', v_h, U_heads)
# recurrent_manifold_fused_autograd (L787)
energy = torch.sum(h_proj * h_proj, dim=-1, keepdim=True) / max(1, h_proj.shape[-1])
# recurrent_manifold_fused_autograd (L788)
s_norm = 1.0 / (1.0 + torch.sqrt(energy) + 1e-8)
# recurrent_manifold_fused_autograd (L791)
gamma_h = torch.einsum('bhr,hrd->bhd', h_proj * h_proj, W_heads) * s_norm
# recurrent_manifold_fused_autograd (L798)
T = max(thermo_temp, 1e-8)
# recurrent_manifold_fused_autograd (L800)
f_energy = (f_h ** 2).mean(dim=-1, keepdim=True) # [B, H, 1]
# recurrent_manifold_fused_autograd (L801)
modulator = torch.exp(-thermo_alpha * f_energy / T)
# recurrent_manifold_fused_autograd (L802)
gamma_h = gamma_h * modulator
# recurrent_manifold_fused_autograd (L812)
v_dot_gz = (v_h * gz_h).sum(dim=-1, keepdim=True)
# recurrent_manifold_fused_autograd (L813)
v_sq = (v_h * v_h).sum(dim=-1, keepdim=True)
# recurrent_manifold_fused_autograd (L815)
gamma_ads = -(1.0 / (z_h + 1e-8)) * (2.0 * v_dot_gz * v_h - v_sq * gz_h)
# recurrent_manifold_fused_autograd (L816)
gamma_h = gamma_h + gamma_ads
# recurrent_manifold_fused_autograd (L819)
gamma_h = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_h / CudaConstants.CURVATURE_CLAMP)
# recurrent_manifold_fused_autograd (L823)
v_half = (v_h + h * (f_h + fg_h - gamma_h)) / (1.0 + h * mu_h + 1e-8)
# recurrent_manifold_fused_autograd (L827)
x_new_h = x_h + dt_eff * v_half
# recurrent_manifold_fused_autograd (L829)
x_new_h = torch.atan2(torch.sin(x_new_h), torch.cos(x_new_h))
# recurrent_manifold_fused_autograd (L839)
feat_h = torch.cat([torch.sin(x_new_h), torch.cos(x_new_h)], dim=-1)
# recurrent_manifold_fused_autograd (L840)
gate = torch.einsum('bhd,hdk->bhk', feat_h, Wf_heads.transpose(1, 2))
# recurrent_manifold_fused_autograd (L841)
if bf_heads is not None: gate = gate + bf_heads.view(1, num_heads, head_dim)
# recurrent_manifold_fused_autograd (L842)
if Wi_heads is not None: gate = gate + torch.einsum('bhd,hdk->bhk', f_h, Wi_heads.transpose(1, 2))
# recurrent_manifold_fused_autograd (L843)
mu_h = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
# recurrent_manifold_fused_autograd (L845)
h_proj = torch.einsum('bhd,hdr->bhr', v_half, U_heads)
# recurrent_manifold_fused_autograd (L846)
energy = torch.sum(h_proj * h_proj, dim=-1, keepdim=True) / max(1, h_proj.shape[-1])
# recurrent_manifold_fused_autograd (L847)
s_norm = 1.0 / (1.0 + torch.sqrt(energy) + 1e-8)
# recurrent_manifold_fused_autograd (L848)
gamma_h = torch.einsum('bhr,hrd->bhd', h_proj * h_proj, W_heads) * s_norm
# recurrent_manifold_fused_autograd (L852)
gamma_h = gamma_h * modulator # Use same modulator as it depends on force_t
# recurrent_manifold_fused_autograd (L855)
gamma_ads = -(1.0 / (z_h + 1e-8)) * (2.0 * v_dot_gz * v_half - v_sq * gz_h)
# recurrent_manifold_fused_autograd (L856)
gamma_h = gamma_h + gamma_ads
# recurrent_manifold_fused_autograd (L859)
gamma_h = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_h / CudaConstants.CURVATURE_CLAMP)
# recurrent_manifold_fused_autograd (L861)
v_new_h = (v_half + h * (f_h + fg_h - gamma_h)) / (1.0 + h * mu_h + 1e-8)
# recurrent_manifold_fused_autograd (L872)
h_in = torch.cat([torch.sin(x_new_h), torch.cos(x_new_h)], dim=-1)
# recurrent_manifold_fused_autograd (L873)
h_in = torch.cat([h_in, v_new_h], dim=-1)
# recurrent_manifold_fused_autograd (L875)
h_gate = torch.einsum('bhd,hdk->bhk', h_in, h_up_w.view(num_heads, head_dim, -1).transpose(1, 2))
# recurrent_manifold_fused_autograd (L877)
h_gate = h_gate + h_up_b.view(1, num_heads, head_dim)
# recurrent_manifold_fused_autograd (L879)
h_state = h_state * h_decay + torch.tanh(h_gate.reshape(B, D_total))
# toroidal_leapfrog_fused_autograd (L956)
v_half = v_curr + 0.5 * dt * force_t
# toroidal_leapfrog_fused_autograd (L959)
x_new = x_curr + dt * v_half
# toroidal_leapfrog_fused_autograd (L963)
x_new = torch.atan2(torch.sin(x_new), torch.cos(x_new))
# toroidal_leapfrog_fused_autograd (L970)
phi = x_new[:, i + 1]
# toroidal_leapfrog_fused_autograd (L972)
v_phi = v_half[:, i + 1]
# toroidal_leapfrog_fused_autograd (L975)
cos_theta = torch.cos(theta)
# toroidal_leapfrog_fused_autograd (L976)
denom = torch.clamp(R + r * cos_theta, min=1e-6)
# toroidal_leapfrog_fused_autograd (L979)
sin_theta = torch.sin(theta)
# toroidal_leapfrog_fused_autograd (L980)
gamma_theta = denom * sin_theta / (r + 1e-6) * (v_phi * v_phi)
# toroidal_leapfrog_fused_autograd (L983)
gamma_phi = -(r * sin_theta) / (denom + 1e-6) * 2.0 * v_theta * v_phi
# toroidal_leapfrog_fused_autograd (L986)
gamma[:, i + 1] = gamma_phi
# toroidal_leapfrog_fused_autograd (L989)
gamma = torch.clamp(gamma, -10.0, 10.0)
# toroidal_leapfrog_fused_autograd (L992)
v_new = v_half + 0.5 * dt * (force_t - gamma)
```

### gfn\cuda\core.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 72 | get_device | `def get_device(self, index: int = 0) -> torch.device:` |
| 128 | CudaConstants | `EPSILON_STANDARD = 1e-7` |
| 129 | CudaConstants | `EPSILON_STRONG = 1e-7` |
| 130 | CudaConstants | `EPSILON_SMOOTH = 1e-7` |
| 144 | CudaConstants | `TOROIDAL_PERIOD = 6.283185307179586  # 2 * π` |

#### Fórmulas Listas para Usar (Python)
```python
# get_device (L72)
def get_device(self, index: int = 0) -> torch.device:
# CudaConstants (L128)
EPSILON_STANDARD = 1e-7
# CudaConstants (L129)
EPSILON_STRONG = 1e-7
# CudaConstants (L130)
EPSILON_SMOOTH = 1e-7
# CudaConstants (L144)
TOROIDAL_PERIOD = 6.283185307179586  # 2 * π
```

### gfn\cuda\ops.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 74 | _get_load_paths | `cuda_path = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{ver}/bin")` |
| 217 | ChristoffelOperation | `Computes: Γ^k_ij = Σ_r λ_kr * (U_ir * U_jr)` |
| 245 | forward | `topology: int = 0) -> torch.Tensor:` |
| 264 | forward | `h = torch.matmul(v, U)  # [B, R]` |
| 267 | forward | `energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])` |
| 268 | forward | `scale = 1.0 / (1.0 + torch.sqrt(energy) + self.epsilon)` |
| 273 | forward | `E = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])` |
| 274 | forward | `M = 1.0 + plasticity * 0.1 * torch.tanh(E)` |
| 279 | forward | `pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)` |
| 281 | forward | `pot = torch.sum(x * V_w, dim=-1, keepdim=True)` |
| 283 | forward | `gate = torch.sigmoid(pot)` |
| 284 | forward | `soft_m = torch.sigmoid(self.singularity_gate_slope * (gate - sing_thresh))` |
| 285 | forward | `M = M * (1.0 + (sing_strength - 1.0) * soft_m)` |
| 288 | forward | `gamma = torch.matmul(h * h, W.t()) * scale * M` |
| 289 | forward | `gamma = self.curvature_clamp * torch.tanh(gamma / self.curvature_clamp)` |
| 300 | backward | `topology: int = 0) -> Tuple[torch.Tensor, ...]:` |
| 398 | launch_toroidal_leapfrog_fused | `x_final = x_out[:, -1, :]  # [batch, dim]` |
| 399 | launch_toroidal_leapfrog_fused | `v_final = v_out[:, -1, :]  # [batch, dim]` |
| 459 | forward | `velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:` |
| 464 | forward | `eff_dt = self.dt * dt_scale` |
| 465 | forward | `h = 0.5 * eff_dt` |
| 485 | forward | `feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)` |
| 486 | forward | `gate = torch.matmul(feat, Wf.t()) + bf` |
| 488 | forward | `gate = gate + torch.matmul(f, W_input.t())` |
| 489 | forward | `mu = torch.sigmoid(gate) * self.friction_scale` |
| 491 | forward | `v_norm = torch.norm(curr_v, dim=-1, keepdim=True)` |
| 492 | forward | `v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + CudaConstants.EPSILON_SMOOTH)` |
| 493 | forward | `mu = mu * (1.0 + velocity_friction_scale * v_norm)` |
| 497 | forward | `curr_v = (curr_v + h * (f - gamma)) / (1.0 + h * mu + self.epsilon)` |
| 500 | forward | `curr_x = curr_x + eff_dt * curr_v` |
| 509 | forward | `feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)` |
| 510 | forward | `gate = torch.matmul(feat, Wf.t()) + bf` |
| 512 | forward | `gate = gate + torch.matmul(f, W_input.t())` |
| 513 | forward | `mu = torch.sigmoid(gate) * self.friction_scale` |
| 515 | forward | `v_norm = torch.norm(curr_v, dim=-1, keepdim=True)` |
| 516 | forward | `v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + CudaConstants.EPSILON_SMOOTH)` |
| 517 | forward | `mu = mu * (1.0 + velocity_friction_scale * v_norm)` |
| 520 | forward | `curr_v = (curr_v + h * (f - gamma2)) / (1.0 + h * mu + self.epsilon)` |
| 536 | backward | `velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, ...]:` |
| 576 | christoffel_fused | `r: float = 1.0) -> torch.Tensor:` |
| 617 | leapfrog_fused | `velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:` |
| 657 | heun_fused | `velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:` |
| 698 | heun_fused | `eff_dt = dt * dt_scale` |
| 709 | dynamics | `if topology == 1: feat = torch.cat([torch.sin(tx), torch.cos(tx)], dim=-1)` |
| 710 | dynamics | `gate = torch.matmul(feat, W_forget.t()) + b_forget` |
| 712 | dynamics | `gate = gate + torch.matmul(f, W_input.t())` |
| 713 | dynamics | `mu = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE` |
| 717 | dynamics | `vm = torch.norm(tv, dim=-1, keepdim=True) / (tv.shape[-1]**0.5 + 1e-8)` |
| 718 | dynamics | `mu = mu * (1.0 + velocity_friction_scale * vm)` |
| 720 | dynamics | `acc = f - gamma - mu * tv` |
| 728 | dynamics | `x_pred = curr_x + eff_dt * dx1` |
| 729 | dynamics | `v_pred = curr_v + eff_dt * dv1` |
| 736 | dynamics | `curr_x = curr_x + 0.5 * eff_dt * (dx1 + dx2)` |
| 737 | dynamics | `curr_v = curr_v + 0.5 * eff_dt * (dv1 + dv2)` |
| 740 | dynamics | `curr_x = torch.atan2(torch.sin(curr_x), torch.cos(curr_x))` |
| 748 | euler_fused | `topology: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:` |
| 751 | euler_fused | `eff_dt = dt * dt_scale` |
| 758 | euler_fused | `curr_v = curr_v + eff_dt * acc` |
| 759 | euler_fused | `curr_x = curr_x + eff_dt * curr_v` |
| 766 | head_mixing_fused | `topology: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:` |
| 775 | head_mixing_fused | `x_cat = x_heads.permute(1, 0, 2).contiguous().view(batch, -1)` |
| 776 | head_mixing_fused | `v_cat = v_heads.permute(1, 0, 2).contiguous().view(batch, -1)` |
| 778 | head_mixing_fused | `v_mix = torch.tanh(v_cat / 100.0)` |
| 779 | head_mixing_fused | `mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)` |
| 780 | head_mixing_fused | `x_next = torch.matmul(mixer_in_x, W_x.t())` |
| 782 | head_mixing_fused | `x_next = torch.matmul(x_cat, W_x.t())` |
| 783 | head_mixing_fused | `v_next = torch.matmul(v_cat, W_v.t())` |
| 785 | head_mixing_fused | `x_next = torch.atan2(torch.sin(x_next), torch.cos(x_next))` |
| 786 | head_mixing_fused | `v_next = 100.0 * torch.tanh(v_next / 100.0)` |
| 799 | dynamic_gating_fused | `hidden = torch.tanh(torch.matmul(x, W1.t()) + b1)` |
| 800 | dynamic_gating_fused | `out = torch.matmul(hidden, W2.t()) + b2` |

#### Fórmulas Listas para Usar (Python)
```python
# _get_load_paths (L74)
cuda_path = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{ver}/bin")
# ChristoffelOperation (L217)
Computes: Γ^k_ij = Σ_r λ_kr * (U_ir * U_jr)
# forward (L245)
topology: int = 0) -> torch.Tensor:
# forward (L264)
h = torch.matmul(v, U)  # [B, R]
# forward (L267)
energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])
# forward (L268)
scale = 1.0 / (1.0 + torch.sqrt(energy) + self.epsilon)
# forward (L273)
E = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
# forward (L274)
M = 1.0 + plasticity * 0.1 * torch.tanh(E)
# forward (L279)
pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
# forward (L281)
pot = torch.sum(x * V_w, dim=-1, keepdim=True)
# forward (L283)
gate = torch.sigmoid(pot)
# forward (L284)
soft_m = torch.sigmoid(self.singularity_gate_slope * (gate - sing_thresh))
# forward (L285)
M = M * (1.0 + (sing_strength - 1.0) * soft_m)
# forward (L288)
gamma = torch.matmul(h * h, W.t()) * scale * M
# forward (L289)
gamma = self.curvature_clamp * torch.tanh(gamma / self.curvature_clamp)
# backward (L300)
topology: int = 0) -> Tuple[torch.Tensor, ...]:
# launch_toroidal_leapfrog_fused (L398)
x_final = x_out[:, -1, :]  # [batch, dim]
# launch_toroidal_leapfrog_fused (L399)
v_final = v_out[:, -1, :]  # [batch, dim]
# forward (L459)
velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
# forward (L464)
eff_dt = self.dt * dt_scale
# forward (L465)
h = 0.5 * eff_dt
# forward (L485)
feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
# forward (L486)
gate = torch.matmul(feat, Wf.t()) + bf
# forward (L488)
gate = gate + torch.matmul(f, W_input.t())
# forward (L489)
mu = torch.sigmoid(gate) * self.friction_scale
# forward (L491)
v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
# forward (L492)
v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + CudaConstants.EPSILON_SMOOTH)
# forward (L493)
mu = mu * (1.0 + velocity_friction_scale * v_norm)
# forward (L497)
curr_v = (curr_v + h * (f - gamma)) / (1.0 + h * mu + self.epsilon)
# forward (L500)
curr_x = curr_x + eff_dt * curr_v
# forward (L509)
feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
# forward (L510)
gate = torch.matmul(feat, Wf.t()) + bf
# forward (L512)
gate = gate + torch.matmul(f, W_input.t())
# forward (L513)
mu = torch.sigmoid(gate) * self.friction_scale
# forward (L515)
v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
# forward (L516)
v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + CudaConstants.EPSILON_SMOOTH)
# forward (L517)
mu = mu * (1.0 + velocity_friction_scale * v_norm)
# forward (L520)
curr_v = (curr_v + h * (f - gamma2)) / (1.0 + h * mu + self.epsilon)
# backward (L536)
velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, ...]:
# christoffel_fused (L576)
r: float = 1.0) -> torch.Tensor:
# leapfrog_fused (L617)
velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
# heun_fused (L657)
velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
# heun_fused (L698)
eff_dt = dt * dt_scale
# dynamics (L709)
if topology == 1: feat = torch.cat([torch.sin(tx), torch.cos(tx)], dim=-1)
# dynamics (L710)
gate = torch.matmul(feat, W_forget.t()) + b_forget
# dynamics (L712)
gate = gate + torch.matmul(f, W_input.t())
# dynamics (L713)
mu = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
# dynamics (L717)
vm = torch.norm(tv, dim=-1, keepdim=True) / (tv.shape[-1]**0.5 + 1e-8)
# dynamics (L718)
mu = mu * (1.0 + velocity_friction_scale * vm)
# dynamics (L720)
acc = f - gamma - mu * tv
# dynamics (L728)
x_pred = curr_x + eff_dt * dx1
# dynamics (L729)
v_pred = curr_v + eff_dt * dv1
# dynamics (L736)
curr_x = curr_x + 0.5 * eff_dt * (dx1 + dx2)
# dynamics (L737)
curr_v = curr_v + 0.5 * eff_dt * (dv1 + dv2)
# dynamics (L740)
curr_x = torch.atan2(torch.sin(curr_x), torch.cos(curr_x))
# euler_fused (L748)
topology: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
# euler_fused (L751)
eff_dt = dt * dt_scale
# euler_fused (L758)
curr_v = curr_v + eff_dt * acc
# euler_fused (L759)
curr_x = curr_x + eff_dt * curr_v
# head_mixing_fused (L766)
topology: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
# head_mixing_fused (L775)
x_cat = x_heads.permute(1, 0, 2).contiguous().view(batch, -1)
# head_mixing_fused (L776)
v_cat = v_heads.permute(1, 0, 2).contiguous().view(batch, -1)
# head_mixing_fused (L778)
v_mix = torch.tanh(v_cat / 100.0)
# head_mixing_fused (L779)
mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)
# head_mixing_fused (L780)
x_next = torch.matmul(mixer_in_x, W_x.t())
# head_mixing_fused (L782)
x_next = torch.matmul(x_cat, W_x.t())
# head_mixing_fused (L783)
v_next = torch.matmul(v_cat, W_v.t())
# head_mixing_fused (L785)
x_next = torch.atan2(torch.sin(x_next), torch.cos(x_next))
# head_mixing_fused (L786)
v_next = 100.0 * torch.tanh(v_next / 100.0)
# dynamic_gating_fused (L799)
hidden = torch.tanh(torch.matmul(x, W1.t()) + b1)
# dynamic_gating_fused (L800)
out = torch.matmul(hidden, W2.t()) + b2
```

### gfn\cuda\precompile_kernels.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 42 | Global | `print("\n" + "="*70)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L42)
print("\n" + "="*70)
```

### gfn\cuda\setup.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 9 | Global | `cuda_sources = [ 'cuda_kernels.cpp', 'src/geometry/lowrank_christoffel.cu', 'src/geometry/lowrank_christoffel_backward.cu', 'src/geometry/lowrank_christoffel_friction_backward.cu', 'src/integrators/symplectic/leapfrog_fused.cu', 'src/integrators/symplectic/leapfrog_backward.cu', 'src/integrators/toroidal/toroidal_christoffel_fused.cu', 'src/integrators/runge_kutta/heun_fused.cu', 'src/integrators/runge_kutta/heun_backward.cu', 'src/integrators/unified_mlayer.cu', ]` |
| 40 | Global | `cxx_flags = ["-O3"]` |
| 41 | Global | `nvcc_flags = [ "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-gencode=arch=compute_75,code=sm_75", ]` |
| 49 | Global | `cxx_flags = ["/O2", "/bigobj", "/EHsc", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN"]` |
| 50 | Global | `nvcc_flags = [ "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-Xcompiler", "/bigobj", "-Xcompiler", "/EHsc", "-Xcompiler", "/DNOMINMAX", "-Xcompiler", "/DWIN32_LEAN_AND_MEAN", ] + nvcc_flags[3:]` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L9)
cuda_sources = [ 'cuda_kernels.cpp', 'src/geometry/lowrank_christoffel.cu', 'src/geometry/lowrank_christoffel_backward.cu', 'src/geometry/lowrank_christoffel_friction_backward.cu', 'src/integrators/symplectic/leapfrog_fused.cu', 'src/integrators/symplectic/leapfrog_backward.cu', 'src/integrators/toroidal/toroidal_christoffel_fused.cu', 'src/integrators/runge_kutta/heun_fused.cu', 'src/integrators/runge_kutta/heun_backward.cu', 'src/integrators/unified_mlayer.cu', ]
# Global (L40)
cxx_flags = ["-O3"]
# Global (L41)
nvcc_flags = [ "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-gencode=arch=compute_75,code=sm_75", ]
# Global (L49)
cxx_flags = ["/O2", "/bigobj", "/EHsc", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN"]
# Global (L50)
nvcc_flags = [ "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-Xcompiler", "/bigobj", "-Xcompiler", "/EHsc", "-Xcompiler", "/DNOMINMAX", "-Xcompiler", "/DWIN32_LEAN_AND_MEAN", ] + nvcc_flags[3:]
```

### gfn\datasets\math.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 27 | __init__ | `self.char_to_id['+'] = 10` |
| 28 | __init__ | `self.char_to_id['-'] = 11` |
| 29 | __init__ | `self.char_to_id['*'] = 12` |
| 38 | _generate_problem | `ops = ['+', '-', '*']` |
| 43 | _generate_problem | `a = random.randint(0, 10**min(3, self.max_digits) - 1)` |
| 44 | _generate_problem | `b = random.randint(0, 10**min(3, self.max_digits) - 1)` |
| 47 | _generate_problem | `a = random.randint(0, 10**self.max_digits - 1)` |
| 51 | _generate_problem | `a = random.randint(0, 10**self.max_digits - 1)` |
| 52 | _generate_problem | `b = random.randint(0, 10**self.max_digits - 1)` |
| 83 | collate_fn | `pad_len = max_len - len(x)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L27)
self.char_to_id['+'] = 10
# __init__ (L28)
self.char_to_id['-'] = 11
# __init__ (L29)
self.char_to_id['*'] = 12
# _generate_problem (L38)
ops = ['+', '-', '*']
# _generate_problem (L43)
a = random.randint(0, 10**min(3, self.max_digits) - 1)
# _generate_problem (L44)
b = random.randint(0, 10**min(3, self.max_digits) - 1)
# _generate_problem (L47)
a = random.randint(0, 10**self.max_digits - 1)
# _generate_problem (L51)
a = random.randint(0, 10**self.max_digits - 1)
# _generate_problem (L52)
b = random.randint(0, 10**self.max_digits - 1)
# collate_fn (L83)
pad_len = max_len - len(x)
```

### gfn\datasets\mixed.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 26 | __init__ | `self.wiki_en = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)` |
| 30 | __init__ | `self.math_thinking = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)` |
| 58 | __iter__ | `a = random.randint(0, 10**8 - 1)` |
| 59 | __iter__ | `b = random.randint(0, 10**8 - 1)` |
| 61 | __iter__ | `text = f"Math: {a} + {b} = {c}"` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L26)
self.wiki_en = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
# __init__ (L30)
self.math_thinking = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
# __iter__ (L58)
a = random.randint(0, 10**8 - 1)
# __iter__ (L59)
b = random.randint(0, 10**8 - 1)
# __iter__ (L61)
text = f"Math: {a} + {b} = {c}"
```

### gfn\embeddings\functional.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 85 | __init__ | `self.net = nn.Sequential(*net)` |
| 90 | __init__ | `self.out_proj.weight.data *= 1.5` |
| 95 | __init__ | `if coord_dim % 2 != 0: self.coord_dim += 1` |
| 97 | __init__ | `freqs = torch.exp(torch.arange(0, self.coord_dim, 2).float() * -(np.log(10000.0) / self.coord_dim))` |
| 111 | forward | `inputs = input_ids.unsqueeze(-1).float()` |
| 115 | forward | `mask = 2**torch.arange(self.coord_dim).to(input_ids.device)` |
| 116 | forward | `bits = (input_ids.unsqueeze(-1) & mask) > 0` |
| 120 | forward | `coords = bits.float() * 2 - 1 # Map {0, 1} to {-1, 1} for SIREN` |
| 123 | forward | `args = inputs * self.freqs` |
| 124 | forward | `coords = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)` |
| 132 | forward | `out = out * self.impulse_scale` |
| 137 | forward | `active_mask = (bits.float().sum(dim=-1, keepdim=True) > 0).float()` |
| 138 | forward | `out = out * active_mask` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L85)
self.net = nn.Sequential(*net)
# __init__ (L90)
self.out_proj.weight.data *= 1.5
# __init__ (L95)
if coord_dim % 2 != 0: self.coord_dim += 1
# __init__ (L97)
freqs = torch.exp(torch.arange(0, self.coord_dim, 2).float() * -(np.log(10000.0) / self.coord_dim))
# forward (L111)
inputs = input_ids.unsqueeze(-1).float()
# forward (L115)
mask = 2**torch.arange(self.coord_dim).to(input_ids.device)
# forward (L116)
bits = (input_ids.unsqueeze(-1) & mask) > 0
# forward (L120)
coords = bits.float() * 2 - 1 # Map {0, 1} to {-1, 1} for SIREN
# forward (L123)
args = inputs * self.freqs
# forward (L124)
coords = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
# forward (L132)
out = out * self.impulse_scale
# forward (L137)
active_mask = (bits.float().sum(dim=-1, keepdim=True) > 0).float()
# forward (L138)
out = out * active_mask
```

### gfn\embeddings\implicit.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 29 | ImplicitEmbedding | `- Standard: 10k * 256 = 2.56M params` |
| 30 | ImplicitEmbedding | `- Implicit: 10k * 16 + ~50k = 210k params (~12x reduction)` |
| 83 | __init__ | `self.net = nn.Sequential(*net)` |

#### Fórmulas Listas para Usar (Python)
```python
# ImplicitEmbedding (L29)
- Standard: 10k * 256 = 2.56M params
# ImplicitEmbedding (L30)
- Implicit: 10k * 16 + ~50k = 210k params (~12x reduction)
# __init__ (L83)
self.net = nn.Sequential(*net)
```

### gfn\embeddings\siren.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 52 | init_weights | `bound = 1 / self.linear.weight.size(1)` |
| 56 | init_weights | `bound = np.sqrt(6 / self.linear.weight.size(1)) / self.omega_0` |

#### Fórmulas Listas para Usar (Python)
```python
# init_weights (L52)
bound = 1 / self.linear.weight.size(1)
# init_weights (L56)
bound = np.sqrt(6 / self.linear.weight.size(1)) / self.omega_0
```

### gfn\geometry\__init__.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 18 | __init__ | `input_dim = 2 * dim if topology == 1 else dim` |
| 19 | __init__ | `self.net = nn.Sequential( nn.Linear(input_dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1), nn.Sigmoid() )` |
| 31 | forward | `x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L18)
input_dim = 2 * dim if topology == 1 else dim
# __init__ (L19)
self.net = nn.Sequential( nn.Linear(input_dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1), nn.Sigmoid() )
# forward (L31)
x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
```

### gfn\geometry\adaptive.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 18 | __init__ | `self.U_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)` |
| 19 | __init__ | `self.W_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)` |
| 31 | forward | `def forward(self, v, x=None, **kwargs):` |
| 34 | forward | `rank_ratio = 0.1 + 0.9 * self.complexity_net(v.detach())` |
| 37 | forward | `avg_ratio = rank_ratio.mean().item()` |
| 39 | forward | `eff_rank = max(4, min(self.max_rank, int(avg_ratio * self.max_rank)))` |
| 46 | forward | `proj = torch.matmul(v, U)` |
| 47 | forward | `norm = torch.norm(proj, dim=-1, keepdim=True)` |
| 48 | forward | `scale = 1.0 / (1.0 + norm + EPSILON_STANDARD)` |
| 49 | forward | `sq = (proj * proj) * scale` |
| 50 | forward | `gamma = torch.matmul(sq, W.t())` |
| 56 | forward | `gamma = CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L18)
self.U_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)
# __init__ (L19)
self.W_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)
# forward (L31)
def forward(self, v, x=None, **kwargs):
# forward (L34)
rank_ratio = 0.1 + 0.9 * self.complexity_net(v.detach())
# forward (L37)
avg_ratio = rank_ratio.mean().item()
# forward (L39)
eff_rank = max(4, min(self.max_rank, int(avg_ratio * self.max_rank)))
# forward (L46)
proj = torch.matmul(v, U)
# forward (L47)
norm = torch.norm(proj, dim=-1, keepdim=True)
# forward (L48)
scale = 1.0 / (1.0 + norm + EPSILON_STANDARD)
# forward (L49)
sq = (proj * proj) * scale
# forward (L50)
gamma = torch.matmul(sq, W.t())
# forward (L56)
gamma = CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)
```

### gfn\geometry\analytical.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 13 | forward | `def forward(self, v, x=None, **kwargs):` |
| 24 | HyperbolicChristoffel | `Geodesic Accel: a = -Gamma(v,v)` |
| 26 | HyperbolicChristoffel | `Uses Conformal Factor lambda = 2 / (1 - \|x\|^2)` |
| 31 | __init__ | `self.curvature = -1.0` |
| 47 | forward | `x_sq = torch.sum(x*x, dim=-1, keepdim=True)` |
| 48 | forward | `v_sq = torch.sum(v*v, dim=-1, keepdim=True)` |
| 49 | forward | `xv = torch.sum(x*v, dim=-1, keepdim=True)` |
| 52 | forward | `gamma = 2 * xv * v - v_sq * x` |
| 76 | forward | `x_sq = torch.sum(x*x, dim=-1, keepdim=True)` |
| 77 | forward | `v_sq = torch.sum(v*v, dim=-1, keepdim=True)` |
| 78 | forward | `xv = torch.sum(x*v, dim=-1, keepdim=True)` |
| 82 | forward | `gamma = -(2 * xv * v - v_sq * x)` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L13)
def forward(self, v, x=None, **kwargs):
# HyperbolicChristoffel (L24)
Geodesic Accel: a = -Gamma(v,v)
# HyperbolicChristoffel (L26)
Uses Conformal Factor lambda = 2 / (1 - |x|^2)
# __init__ (L31)
self.curvature = -1.0
# forward (L47)
x_sq = torch.sum(x*x, dim=-1, keepdim=True)
# forward (L48)
v_sq = torch.sum(v*v, dim=-1, keepdim=True)
# forward (L49)
xv = torch.sum(x*v, dim=-1, keepdim=True)
# forward (L52)
gamma = 2 * xv * v - v_sq * x
# forward (L76)
x_sq = torch.sum(x*x, dim=-1, keepdim=True)
# forward (L77)
v_sq = torch.sum(v*v, dim=-1, keepdim=True)
# forward (L78)
xv = torch.sum(x*v, dim=-1, keepdim=True)
# forward (L82)
gamma = -(2 * xv * v - v_sq * x)
```

### gfn\geometry\boundaries.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 12 | apply_boundary_python | `For topology_id == 1 (torus), positions are wrapped to [0, 2π). The wrapping is periodic: x = x % (2*π) 2. VELOCITY HANDLING: Velocity vectors should NOT be wrapped! - Position: x is on the manifold, needs wrapping - Velocity: v is in the TANGENT SPACE, invariant under wrapping The wrapping of velocity would create artificial discontinuities that break the smoothness of geodesic flow. If you need to apply velocity corrections, use apply_velocity_correction(). Topology IDs: 0: Euclidean (None) - No boundary conditions 1: Toroidal (Periodic [0, 2*PI)) - Positions wrapped Args: x: Position tensor [batch, dim] topology_id: Integer topology identifier Returns: Position tensor with boundaries applied """ if topology_id == 1: PI = 3.14159265359 TWO_PI = 2.0 * PI x_wrapped = torch.atan2(torch.sin(x), torch.cos(x)) x_wrapped = torch.where(x_wrapped < 0, x_wrapped + TWO_PI, x_wrapped) return x_wrapped return x def apply_velocity_correction(v, x_old, x_new, topology_id): """ Correct velocity for toroidal boundary crossings. When position crosses the boundary (e.g., from 6.28 to 0.01), the apparent velocity is wrong. This function computes the true velocity considering boundary crossings. AUDIT FIX: This function handles velocity correction for torus. Args: v: Velocity tensor [batch, dim] x_old: Previous position [batch, dim] x_new: Current position [batch, dim] topology_id: Topology identifier Returns: Corrected velocity tensor """ if topology_id != 1: return v PI = 3.14159265359 TWO_PI = 2.0 * PI apparent_disp = x_new - x_old wrapped_disp = torch.atan2(torch.sin(apparent_disp), torch.cos(apparent_disp)) return wrapped_disp def toroidal_dist_python(x1, x2): """ Shortest angular distance on Torus. IMPORTANT (Auditoria 2026-02-06): This computes distance on a FLAT torus (product of circles). It does NOT account for the LEARNED Christoffel curvature. For tasks requiring true geodesic distance on the learned manifold, use Christoffel-based distance computation instead. Args: x1: Position tensor [batch, dim] x2: Position tensor [batch, dim] Returns: Distance tensor """ PI = 3.14159265359 diff = x1 - x2 diff = torch.atan2(torch.sin(diff), torch.cos(diff)) return torch.norm(diff, dim=-1) def resolve_topology_id(christoffel, topology_id_arg=None): """ Resolve topology ID from Christoffel geometry or argument. Args: christoffel: The Christoffel geometry object topology_id_arg: Optional override from kwargs Returns: Integer topology ID (0=Euclidean, 1=Torus) """ if topology_id_arg is not None: return topology_id_arg tid = getattr(christoffel, 'topology_id', 0) if tid == 0 and hasattr(christoffel, 'is_torus') and christoffel.is_torus: return 1 return tid def get_boundary_features(x, topology_id): """ Extract features relevant to the topology boundary. For Euclidean (0): Returns x For Toroidal (1): Returns [sin(x), cos(x)] Args: x: Position tensor [batch, dim] topology_id: Integer topology identifier Returns: Feature tensor [batch, dim] or [batch, 2*dim] """ if topology_id == 1: return torch.cat([torch.sin(x), torch.cos(x)], dim=-1) return x` |

#### Fórmulas Listas para Usar (Python)
```python
# apply_boundary_python (L12)
For topology_id == 1 (torus), positions are wrapped to [0, 2π). The wrapping is periodic: x = x % (2*π) 2. VELOCITY HANDLING: Velocity vectors should NOT be wrapped! - Position: x is on the manifold, needs wrapping - Velocity: v is in the TANGENT SPACE, invariant under wrapping The wrapping of velocity would create artificial discontinuities that break the smoothness of geodesic flow. If you need to apply velocity corrections, use apply_velocity_correction(). Topology IDs: 0: Euclidean (None) - No boundary conditions 1: Toroidal (Periodic [0, 2*PI)) - Positions wrapped Args: x: Position tensor [batch, dim] topology_id: Integer topology identifier Returns: Position tensor with boundaries applied """ if topology_id == 1: PI = 3.14159265359 TWO_PI = 2.0 * PI x_wrapped = torch.atan2(torch.sin(x), torch.cos(x)) x_wrapped = torch.where(x_wrapped < 0, x_wrapped + TWO_PI, x_wrapped) return x_wrapped return x def apply_velocity_correction(v, x_old, x_new, topology_id): """ Correct velocity for toroidal boundary crossings. When position crosses the boundary (e.g., from 6.28 to 0.01), the apparent velocity is wrong. This function computes the true velocity considering boundary crossings. AUDIT FIX: This function handles velocity correction for torus. Args: v: Velocity tensor [batch, dim] x_old: Previous position [batch, dim] x_new: Current position [batch, dim] topology_id: Topology identifier Returns: Corrected velocity tensor """ if topology_id != 1: return v PI = 3.14159265359 TWO_PI = 2.0 * PI apparent_disp = x_new - x_old wrapped_disp = torch.atan2(torch.sin(apparent_disp), torch.cos(apparent_disp)) return wrapped_disp def toroidal_dist_python(x1, x2): """ Shortest angular distance on Torus. IMPORTANT (Auditoria 2026-02-06): This computes distance on a FLAT torus (product of circles). It does NOT account for the LEARNED Christoffel curvature. For tasks requiring true geodesic distance on the learned manifold, use Christoffel-based distance computation instead. Args: x1: Position tensor [batch, dim] x2: Position tensor [batch, dim] Returns: Distance tensor """ PI = 3.14159265359 diff = x1 - x2 diff = torch.atan2(torch.sin(diff), torch.cos(diff)) return torch.norm(diff, dim=-1) def resolve_topology_id(christoffel, topology_id_arg=None): """ Resolve topology ID from Christoffel geometry or argument. Args: christoffel: The Christoffel geometry object topology_id_arg: Optional override from kwargs Returns: Integer topology ID (0=Euclidean, 1=Torus) """ if topology_id_arg is not None: return topology_id_arg tid = getattr(christoffel, 'topology_id', 0) if tid == 0 and hasattr(christoffel, 'is_torus') and christoffel.is_torus: return 1 return tid def get_boundary_features(x, topology_id): """ Extract features relevant to the topology boundary. For Euclidean (0): Returns x For Toroidal (1): Returns [sin(x), cos(x)] Args: x: Position tensor [batch, dim] topology_id: Integer topology identifier Returns: Feature tensor [batch, dim] or [batch, 2*dim] """ if topology_id == 1: return torch.cat([torch.sin(x), torch.cos(x)], dim=-1) return x
```

### gfn\geometry\confusion.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 19 | ConfusionChristoffel | `g_new = g_base + lambda * (F @ F.T)` |
| 36 | forward | `def forward(self, v, x=None, force=None, **kwargs):` |
| 39 | forward | `gamma = self.base_christoffel(v, x, force=force, **kwargs)` |
| 45 | forward | `confusion = (force ** 2).mean(dim=-1, keepdim=True)` |
| 52 | forward | `scale = 1.0 + self.sensitivity * confusion` |
| 54 | forward | `gamma = gamma * scale` |

#### Fórmulas Listas para Usar (Python)
```python
# ConfusionChristoffel (L19)
g_new = g_base + lambda * (F @ F.T)
# forward (L36)
def forward(self, v, x=None, force=None, **kwargs):
# forward (L39)
gamma = self.base_christoffel(v, x, force=force, **kwargs)
# forward (L45)
confusion = (force ** 2).mean(dim=-1, keepdim=True)
# forward (L52)
scale = 1.0 + self.sensitivity * confusion
# forward (L54)
gamma = gamma * scale
```

### gfn\geometry\gauge.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | GaugeChristoffel | `Gamma^g_uv = Gamma^R_uv + g * (D_u v - d_u v)` |
| 38 | __init__ | `self.A_net = nn.Sequential( nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, gauge_dim * dim) # Output shape: [batch, dim * gauge_dim] )` |
| 64 | compute_field_strength | `Computes the field strength tensor F_mn = d_m A_n - d_n A_m + [A_m, A_n]` |
| 104 | get_A | `F = d_A.permute(1, 0, 2) - d_A # [mu, nu, a] - [nu, mu, a]` |
| 132 | parallel_transport | `phase_shift = torch.bmm(v.unsqueeze(1), A).squeeze(1)` |
| 141 | parallel_transport | `modulation = torch.cos(phase_shift.mean(dim=-1, keepdim=True))` |
| 146 | forward | `def forward(self, v, x=None, force=None, **kwargs):` |
| 150 | forward | `Gamma^g = Gamma^base + g * (D v - d v)` |
| 152 | forward | `Since D v - d v = A v (approx), the correction represents the` |
| 156 | forward | `gamma_base = self.base_christoffel(v, x, force, **kwargs)` |
| 166 | forward | `gamma_gauge = self.gauge_coupling * (v_transported - v)` |
| 173 | gauge_invariant_loss | `L_gauge = MSE(f(x), f(g*x))` |

#### Fórmulas Listas para Usar (Python)
```python
# GaugeChristoffel (L16)
Gamma^g_uv = Gamma^R_uv + g * (D_u v - d_u v)
# __init__ (L38)
self.A_net = nn.Sequential( nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, gauge_dim * dim) # Output shape: [batch, dim * gauge_dim] )
# compute_field_strength (L64)
Computes the field strength tensor F_mn = d_m A_n - d_n A_m + [A_m, A_n]
# get_A (L104)
F = d_A.permute(1, 0, 2) - d_A # [mu, nu, a] - [nu, mu, a]
# parallel_transport (L132)
phase_shift = torch.bmm(v.unsqueeze(1), A).squeeze(1)
# parallel_transport (L141)
modulation = torch.cos(phase_shift.mean(dim=-1, keepdim=True))
# forward (L146)
def forward(self, v, x=None, force=None, **kwargs):
# forward (L150)
Gamma^g = Gamma^base + g * (D v - d v)
# forward (L152)
Since D v - d v = A v (approx), the correction represents the
# forward (L156)
gamma_base = self.base_christoffel(v, x, force, **kwargs)
# forward (L166)
gamma_gauge = self.gauge_coupling * (v_transported - v)
# gauge_invariant_loss (L173)
L_gauge = MSE(f(x), f(g*x))
```

### gfn\geometry\hierarchical.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 22 | __init__ | `self.scale_weights = nn.Parameter(torch.ones(len(ranks)) / len(ranks))` |
| 27 | forward | `def forward(self, v, x=None, force=None, **kwargs):` |
| 37 | forward | `gamma, mu = scale(v, x, force, **kwargs)` |
| 44 | forward | `weights = torch.softmax(self.scale_weights, dim=0)` |
| 46 | forward | `gamma_combined = sum(w * g for w, g in zip(weights, gammas))` |
| 48 | forward | `mu_combined = sum(w * m for w, m in zip(weights, mus))` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L22)
self.scale_weights = nn.Parameter(torch.ones(len(ranks)) / len(ranks))
# forward (L27)
def forward(self, v, x=None, force=None, **kwargs):
# forward (L37)
gamma, mu = scale(v, x, force, **kwargs)
# forward (L44)
weights = torch.softmax(self.scale_weights, dim=0)
# forward (L46)
gamma_combined = sum(w * g for w, g in zip(weights, gammas))
# forward (L48)
mu_combined = sum(w * m for w, m in zip(weights, mus))
```

### gfn\geometry\holographic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 14 | AdSCFTChristoffel | `g_ij = (1 / z(x)^2) * delta_ij` |
| 17 | AdSCFTChristoffel | `Gamma^k_ij = -1/z * (z_j delta^k_i + z_i delta^k_j - z_k delta_ij)` |
| 26 | __init__ | `self.radial_net = nn.Sequential( nn.Linear(self.dim, self.dim // 2), nn.SiLU(), nn.Linear(self.dim // 2, 1), nn.Softplus() )` |
| 41 | get_z_and_grad | `z = self.radial_net(x_req) + self.z_min` |
| 45 | get_z_and_grad | `grad_z = torch.autograd.grad( z.sum(), x_req, create_graph=self.training, retain_graph=True )[0]` |
| 58 | forward | `gamma_base = self.base_christoffel(v, x, **kwargs)` |
| 67 | forward | `v_dot_gradz = (v * grad_z).sum(dim=-1, keepdim=True) # [B, 1]` |
| 68 | forward | `v_sq = (v * v).sum(dim=-1, keepdim=True) # [B, 1]` |
| 70 | forward | `gamma_ads = -(1.0 / z) * (2.0 * v_dot_gradz * v - v_sq * grad_z)` |

#### Fórmulas Listas para Usar (Python)
```python
# AdSCFTChristoffel (L14)
g_ij = (1 / z(x)^2) * delta_ij
# AdSCFTChristoffel (L17)
Gamma^k_ij = -1/z * (z_j delta^k_i + z_i delta^k_j - z_k delta_ij)
# __init__ (L26)
self.radial_net = nn.Sequential( nn.Linear(self.dim, self.dim // 2), nn.SiLU(), nn.Linear(self.dim // 2, 1), nn.Softplus() )
# get_z_and_grad (L41)
z = self.radial_net(x_req) + self.z_min
# get_z_and_grad (L45)
grad_z = torch.autograd.grad( z.sum(), x_req, create_graph=self.training, retain_graph=True )[0]
# forward (L58)
gamma_base = self.base_christoffel(v, x, **kwargs)
# forward (L67)
v_dot_gradz = (v * grad_z).sum(dim=-1, keepdim=True) # [B, 1]
# forward (L68)
v_sq = (v * v).sum(dim=-1, keepdim=True) # [B, 1]
# forward (L70)
gamma_ads = -(1.0 / z) * (2.0 * v_dot_gradz * v - v_sq * grad_z)
```

### gfn\geometry\hyper.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 11 | HyperChristoffel | `Gamma(v, v \| x) = W(x) * (U(x)^T v)^2` |
| 14 | HyperChristoffel | `U(x) = U_static * diag(Gate_u(x))` |
| 15 | HyperChristoffel | `W(x) = W_static * diag(Gate_w(x))` |
| 35 | forward | `def forward(self, v, x=None, force=None, **kwargs):` |
| 37 | forward | `return super().forward(v, None, force=force, **kwargs)` |
| 41 | forward | `g_u = torch.sigmoid(self.gate_u(x)) * 2.0 # [batch, rank]` |
| 42 | forward | `g_w = torch.sigmoid(self.gate_w(x)) * 2.0 # [batch, rank]` |
| 59 | forward | `proj_static = torch.matmul(v, self.U) # [batch, rank]` |
| 62 | forward | `proj_dynamic = proj_static * g_u # [batch, rank]` |
| 66 | forward | `sq_dynamic = (proj_dynamic * proj_dynamic) / (1.0 + torch.abs(proj_dynamic))` |
| 69 | forward | `sq_modulated = sq_dynamic * g_w # [batch, rank]` |
| 73 | forward | `gamma = torch.matmul(sq_modulated, self.W.t()) # [batch, dim]` |
| 76 | forward | `x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)` |
| 91 | forward | `gate_activ = torch.matmul(x_in, Wf.t()) + bf` |
| 93 | forward | `gate_activ = gate_activ + torch.matmul(force, Wi.t())` |
| 97 | forward | `gate_activ = gate_activ + self.input_gate(force)` |
| 99 | forward | `mu_base = torch.sigmoid(gate_activ) * FRICTION_SCALE` |
| 100 | forward | `velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)` |
| 101 | forward | `velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + EPSILON_STRONG)` |
| 102 | forward | `mu = mu_base * (1.0 + self.velocity_friction_scale * velocity_magnitude)` |
| 108 | forward | `gamma = gamma + mu * v` |

#### Fórmulas Listas para Usar (Python)
```python
# HyperChristoffel (L11)
Gamma(v, v | x) = W(x) * (U(x)^T v)^2
# HyperChristoffel (L14)
U(x) = U_static * diag(Gate_u(x))
# HyperChristoffel (L15)
W(x) = W_static * diag(Gate_w(x))
# forward (L35)
def forward(self, v, x=None, force=None, **kwargs):
# forward (L37)
return super().forward(v, None, force=force, **kwargs)
# forward (L41)
g_u = torch.sigmoid(self.gate_u(x)) * 2.0 # [batch, rank]
# forward (L42)
g_w = torch.sigmoid(self.gate_w(x)) * 2.0 # [batch, rank]
# forward (L59)
proj_static = torch.matmul(v, self.U) # [batch, rank]
# forward (L62)
proj_dynamic = proj_static * g_u # [batch, rank]
# forward (L66)
sq_dynamic = (proj_dynamic * proj_dynamic) / (1.0 + torch.abs(proj_dynamic))
# forward (L69)
sq_modulated = sq_dynamic * g_w # [batch, rank]
# forward (L73)
gamma = torch.matmul(sq_modulated, self.W.t()) # [batch, dim]
# forward (L76)
x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# forward (L91)
gate_activ = torch.matmul(x_in, Wf.t()) + bf
# forward (L93)
gate_activ = gate_activ + torch.matmul(force, Wi.t())
# forward (L97)
gate_activ = gate_activ + self.input_gate(force)
# forward (L99)
mu_base = torch.sigmoid(gate_activ) * FRICTION_SCALE
# forward (L100)
velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
# forward (L101)
velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + EPSILON_STRONG)
# forward (L102)
mu = mu_base * (1.0 + self.velocity_friction_scale * velocity_magnitude)
# forward (L108)
gamma = gamma + mu * v
```

### gfn\geometry\hysteresis.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 9 | HysteresisChristoffel | `Γ_hyst = Γ_base + δΓ(h)` |
| 25 | __init__ | `self.U_hyst = nn.Parameter(torch.randn(dim, rank) * 0.01)` |
| 26 | __init__ | `self.W_hyst = nn.Parameter(torch.randn(dim, rank) * 0.01)` |
| 28 | forward | `def forward(self, v, x=None, memory_state=None, **kwargs):` |
| 38 | forward | `gamma = self.base_christoffel(v, x, **kwargs)` |
| 49 | forward | `delta = torch.matmul(memory_state, self.U_hyst) # [Batch, Rank]` |
| 50 | forward | `delta = torch.matmul(delta, self.W_hyst.t())    # [Batch, Dim]` |
| 55 | forward | `v_norm = torch.norm(v, dim=-1, keepdim=True)` |
| 56 | forward | `delta = delta * v_norm` |
| 58 | forward | `gamma = gamma + delta` |

#### Fórmulas Listas para Usar (Python)
```python
# HysteresisChristoffel (L9)
Γ_hyst = Γ_base + δΓ(h)
# __init__ (L25)
self.U_hyst = nn.Parameter(torch.randn(dim, rank) * 0.01)
# __init__ (L26)
self.W_hyst = nn.Parameter(torch.randn(dim, rank) * 0.01)
# forward (L28)
def forward(self, v, x=None, memory_state=None, **kwargs):
# forward (L38)
gamma = self.base_christoffel(v, x, **kwargs)
# forward (L49)
delta = torch.matmul(memory_state, self.U_hyst) # [Batch, Rank]
# forward (L50)
delta = torch.matmul(delta, self.W_hyst.t())    # [Batch, Dim]
# forward (L55)
v_norm = torch.norm(v, dim=-1, keepdim=True)
# forward (L56)
delta = delta * v_norm
# forward (L58)
gamma = gamma + delta
```

### gfn\geometry\lowrank.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | LowRankChristoffel | `The decomposition Gamma^k_ij = sum_{r=1}^R lambda_kr * (U_ir * U_jr)` |
| 23 | LowRankChristoffel | `- Symmetry Gamma^k_ij = Gamma^k_ji is preserved (by construction)` |
| 31 | LowRankChristoffel | `Friction is now computed as: mu(x,v) = sigma(gate(x)) * FRICTION_SCALE * (1 + alpha * \|\|v\|\|)` |
| 55 | __init__ | `gate_input_dim = 2 * dim if self.is_torus else dim` |
| 83 | forward | `def forward(self, v, x=None, force=None, **kwargs):` |
| 90 | forward | `- Total acceleration: a = F_ext - Christoffel(v,v) - Friction*v` |
| 93 | forward | `Acc = F_ext - Output` |
| 105 | forward | `x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)` |
| 109 | forward | `friction = torch.sigmoid(self.forget_gate(x_in)) * FRICTION_SCALE` |
| 114 | forward | `velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)` |
| 116 | forward | `velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + 1e-8)` |
| 117 | forward | `friction = friction * (1.0 + self.velocity_friction_scale * velocity_magnitude)` |
| 124 | forward | `gamma_cuda = gamma_cuda + friction * v` |
| 136 | forward | `proj = torch.bmm(v, self.U)` |
| 137 | forward | `norm = torch.norm(proj, dim=-1, keepdim=True)` |
| 139 | forward | `scale = 1.0 / (1.0 + norm + EPSILON_STRONG)` |
| 140 | forward | `sq = (proj * proj) * scale` |
| 141 | forward | `gamma = torch.bmm(sq, self.W.transpose(1, 2))` |
| 143 | forward | `proj = torch.matmul(v, self.U)` |
| 144 | forward | `norm = torch.norm(proj, dim=-1, keepdim=True)` |
| 146 | forward | `scale = 1.0 / (1.0 + norm + EPSILON_STRONG)` |
| 147 | forward | `sq = (proj * proj) * scale` |
| 148 | forward | `gamma = torch.matmul(sq, self.W.t())` |
| 152 | forward | `x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)` |
| 165 | forward | `gate_activ = torch.matmul(x_in, Wf.t()) + bf` |
| 167 | forward | `gate_activ = gate_activ + torch.matmul(force, Wi.t())` |
| 171 | forward | `gate_activ = gate_activ + self.input_gate(force)` |
| 174 | forward | `mu_base = torch.sigmoid(gate_activ) * FRICTION_SCALE` |
| 178 | forward | `velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)` |
| 179 | forward | `velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + EPSILON_STRONG)` |
| 180 | forward | `mu = mu_base * (1.0 + self.velocity_friction_scale * velocity_magnitude)` |
| 187 | forward | `gamma = gamma + mu * v` |
| 216 | _normalize_christoffel_structure | `gamma_sym = 0.5 * (gamma + gamma.transpose(-1, -2))` |
| 223 | _normalize_christoffel_structure | `diag_mean = torch.diagonal(gamma_sym, dim1=-1, dim2=-2).mean(dim=-1, keepdim=True)` |
| 225 | _normalize_christoffel_structure | `gamma_centered = gamma_sym - torch.diag_embed(diag_mean.squeeze(-1))` |

#### Fórmulas Listas para Usar (Python)
```python
# LowRankChristoffel (L21)
The decomposition Gamma^k_ij = sum_{r=1}^R lambda_kr * (U_ir * U_jr)
# LowRankChristoffel (L23)
- Symmetry Gamma^k_ij = Gamma^k_ji is preserved (by construction)
# LowRankChristoffel (L31)
Friction is now computed as: mu(x,v) = sigma(gate(x)) * FRICTION_SCALE * (1 + alpha * ||v||)
# __init__ (L55)
gate_input_dim = 2 * dim if self.is_torus else dim
# forward (L83)
def forward(self, v, x=None, force=None, **kwargs):
# forward (L90)
- Total acceleration: a = F_ext - Christoffel(v,v) - Friction*v
# forward (L93)
Acc = F_ext - Output
# forward (L105)
x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# forward (L109)
friction = torch.sigmoid(self.forget_gate(x_in)) * FRICTION_SCALE
# forward (L114)
velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
# forward (L116)
velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + 1e-8)
# forward (L117)
friction = friction * (1.0 + self.velocity_friction_scale * velocity_magnitude)
# forward (L124)
gamma_cuda = gamma_cuda + friction * v
# forward (L136)
proj = torch.bmm(v, self.U)
# forward (L137)
norm = torch.norm(proj, dim=-1, keepdim=True)
# forward (L139)
scale = 1.0 / (1.0 + norm + EPSILON_STRONG)
# forward (L140)
sq = (proj * proj) * scale
# forward (L141)
gamma = torch.bmm(sq, self.W.transpose(1, 2))
# forward (L143)
proj = torch.matmul(v, self.U)
# forward (L144)
norm = torch.norm(proj, dim=-1, keepdim=True)
# forward (L146)
scale = 1.0 / (1.0 + norm + EPSILON_STRONG)
# forward (L147)
sq = (proj * proj) * scale
# forward (L148)
gamma = torch.matmul(sq, self.W.t())
# forward (L152)
x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# forward (L165)
gate_activ = torch.matmul(x_in, Wf.t()) + bf
# forward (L167)
gate_activ = gate_activ + torch.matmul(force, Wi.t())
# forward (L171)
gate_activ = gate_activ + self.input_gate(force)
# forward (L174)
mu_base = torch.sigmoid(gate_activ) * FRICTION_SCALE
# forward (L178)
velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
# forward (L179)
velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + EPSILON_STRONG)
# forward (L180)
mu = mu_base * (1.0 + self.velocity_friction_scale * velocity_magnitude)
# forward (L187)
gamma = gamma + mu * v
# _normalize_christoffel_structure (L216)
gamma_sym = 0.5 * (gamma + gamma.transpose(-1, -2))
# _normalize_christoffel_structure (L223)
diag_mean = torch.diagonal(gamma_sym, dim1=-1, dim2=-2).mean(dim=-1, keepdim=True)
# _normalize_christoffel_structure (L225)
gamma_centered = gamma_sym - torch.diag_embed(diag_mean.squeeze(-1))
```

### gfn\geometry\reactive.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 17 | ReactiveChristoffel | `- "Black hole strength" = curvature_amplification_factor` |
| 18 | ReactiveChristoffel | `- "Singularity threshold" = semantic_certainty_threshold` |
| 58 | forward | `def forward(self, v, x=None, force=None, **kwargs):` |
| 69 | forward | `V_w_in = self.V.weight.t()  # [1, dim] -> [dim, 1] -> [1, dim]` |
| 87 | forward | `gamma = super().forward(v, x, force=force, **kwargs)` |
| 96 | forward | `energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))` |
| 99 | forward | `gamma = gamma * (1.0 + self.plasticity * energy)` |
| 107 | forward | `x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)` |
| 110 | forward | `potential = torch.sigmoid(self.V(x_in)) # [batch, 1]` |
| 124 | forward | `is_amplified = torch.sigmoid(gate_slope * (potential - self.semantic_certainty_threshold))` |
| 125 | forward | `amplification_mult = 1.0 + is_amplified * (self.curvature_amplification_factor - 1.0)` |
| 126 | forward | `gamma = gamma * amplification_mult` |
| 131 | forward | `gamma = torch.clamp(gamma, -max_amplification * CURVATURE_CLAMP, max_amplification * CURVATURE_CLAMP)` |

#### Fórmulas Listas para Usar (Python)
```python
# ReactiveChristoffel (L17)
- "Black hole strength" = curvature_amplification_factor
# ReactiveChristoffel (L18)
- "Singularity threshold" = semantic_certainty_threshold
# forward (L58)
def forward(self, v, x=None, force=None, **kwargs):
# forward (L69)
V_w_in = self.V.weight.t()  # [1, dim] -> [dim, 1] -> [1, dim]
# forward (L87)
gamma = super().forward(v, x, force=force, **kwargs)
# forward (L96)
energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
# forward (L99)
gamma = gamma * (1.0 + self.plasticity * energy)
# forward (L107)
x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# forward (L110)
potential = torch.sigmoid(self.V(x_in)) # [batch, 1]
# forward (L124)
is_amplified = torch.sigmoid(gate_slope * (potential - self.semantic_certainty_threshold))
# forward (L125)
amplification_mult = 1.0 + is_amplified * (self.curvature_amplification_factor - 1.0)
# forward (L126)
gamma = gamma * amplification_mult
# forward (L131)
gamma = torch.clamp(gamma, -max_amplification * CURVATURE_CLAMP, max_amplification * CURVATURE_CLAMP)
```

### gfn\geometry\ricci.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 9 | RicciFlowChristoffel | `dg_ij / dt = -2 * R_ij` |

#### Fórmulas Listas para Usar (Python)
```python
# RicciFlowChristoffel (L9)
dg_ij / dt = -2 * R_ij
```

### gfn\geometry\thermo.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 9 | ThermodynamicChristoffel | `Implements a metric modulation based on Free Energy (F = E - TS).` |
| 13 | ThermodynamicChristoffel | `g_ij(x, T) = g_base_ij(x) * exp( -alpha/T * grad(F) )` |
| 37 | compute_entropy_proxy | `var_v = torch.var(v, dim=0).mean() # Scalar proxy` |
| 38 | compute_entropy_proxy | `entropy = 0.5 * torch.log(var_v + EPSILON_STRONG)` |
| 41 | forward | `def forward(self, v, x=None, force=None, **kwargs):` |
| 43 | forward | `gamma = self.base_christoffel(v, x, force=force, **kwargs)` |
| 49 | forward | `energy = (force ** 2).mean(dim=-1, keepdim=True)` |
| 55 | forward | `T = torch.abs(self.temperature) + EPSILON_STRONG` |
| 56 | forward | `free_energy = energy - T * entropy` |
| 73 | forward | `modulation = torch.exp(-self.alpha * energy / T)` |
| 77 | forward | `gamma = gamma * modulation` |

#### Fórmulas Listas para Usar (Python)
```python
# ThermodynamicChristoffel (L9)
Implements a metric modulation based on Free Energy (F = E - TS).
# ThermodynamicChristoffel (L13)
g_ij(x, T) = g_base_ij(x) * exp( -alpha/T * grad(F) )
# compute_entropy_proxy (L37)
var_v = torch.var(v, dim=0).mean() # Scalar proxy
# compute_entropy_proxy (L38)
entropy = 0.5 * torch.log(var_v + EPSILON_STRONG)
# forward (L41)
def forward(self, v, x=None, force=None, **kwargs):
# forward (L43)
gamma = self.base_christoffel(v, x, force=force, **kwargs)
# forward (L49)
energy = (force ** 2).mean(dim=-1, keepdim=True)
# forward (L55)
T = torch.abs(self.temperature) + EPSILON_STRONG
# forward (L56)
free_energy = energy - T * entropy
# forward (L73)
modulation = torch.exp(-self.alpha * energy / T)
# forward (L77)
gamma = gamma * modulation
```

### gfn\geometry\toroidal.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 15 | ToroidalChristoffel | `g = diag(r^2, (R + r cos th)^2)` |
| 37 | __init__ | `gate_input_dim = 2 * dim` |
| 65 | get_metric | `g_phi = (R + r cos theta)^2` |
| 71 | get_metric | `g[..., i] = self.r**2` |
| 73 | get_metric | `g[..., i+1] = (self.R + self.r * torch.cos(th))**2` |
| 76 | forward | `def forward(self, v, x=None, force=None, **kwargs):` |
| 95 | forward | `cos_th = torch.cos(x)` |
| 96 | forward | `sin_th = torch.sin(x)` |
| 108 | forward | `v_ph = v[..., i+1]` |
| 112 | forward | `denom = torch.clamp(self.R + self.r * torch.cos(th), min=CLAMP_MIN_STRONG)` |
| 115 | forward | `term_th = denom * torch.sin(th) / (self.r + EPSILON_SMOOTH)` |
| 116 | forward | `gamma[..., i] = term_th * (v_ph ** 2)` |
| 119 | forward | `term_ph = -(self.r * torch.sin(th)) / (denom + EPSILON_SMOOTH)` |
| 120 | forward | `gamma[..., i+1] = 2.0 * term_ph * v_ph * v_th` |
| 122 | forward | `gamma = gamma * TOROIDAL_CURVATURE_SCALE  # Strong Curvature (User Requested Full Torus)` |
| 126 | forward | `x_in = torch.cat([sin_th, cos_th], dim=-1)` |
| 132 | forward | `gate_activ = gate_activ + self.input_gate(force)` |
| 136 | forward | `mu = torch.sigmoid(gate_activ) * FRICTION_SCALE` |
| 142 | forward | `energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))` |
| 143 | forward | `gamma = gamma * (1.0 + self.plasticity * energy)` |
| 147 | forward | `potential = torch.sigmoid(self.V(x_in))` |
| 149 | forward | `gamma = gamma * (1.0 + is_singularity * (self.black_hole_strength - 1.0))` |
| 155 | forward | `gamma = gamma + mu * v` |

#### Fórmulas Listas para Usar (Python)
```python
# ToroidalChristoffel (L15)
g = diag(r^2, (R + r cos th)^2)
# __init__ (L37)
gate_input_dim = 2 * dim
# get_metric (L65)
g_phi = (R + r cos theta)^2
# get_metric (L71)
g[..., i] = self.r**2
# get_metric (L73)
g[..., i+1] = (self.R + self.r * torch.cos(th))**2
# forward (L76)
def forward(self, v, x=None, force=None, **kwargs):
# forward (L95)
cos_th = torch.cos(x)
# forward (L96)
sin_th = torch.sin(x)
# forward (L108)
v_ph = v[..., i+1]
# forward (L112)
denom = torch.clamp(self.R + self.r * torch.cos(th), min=CLAMP_MIN_STRONG)
# forward (L115)
term_th = denom * torch.sin(th) / (self.r + EPSILON_SMOOTH)
# forward (L116)
gamma[..., i] = term_th * (v_ph ** 2)
# forward (L119)
term_ph = -(self.r * torch.sin(th)) / (denom + EPSILON_SMOOTH)
# forward (L120)
gamma[..., i+1] = 2.0 * term_ph * v_ph * v_th
# forward (L122)
gamma = gamma * TOROIDAL_CURVATURE_SCALE  # Strong Curvature (User Requested Full Torus)
# forward (L126)
x_in = torch.cat([sin_th, cos_th], dim=-1)
# forward (L132)
gate_activ = gate_activ + self.input_gate(force)
# forward (L136)
mu = torch.sigmoid(gate_activ) * FRICTION_SCALE
# forward (L142)
energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
# forward (L143)
gamma = gamma * (1.0 + self.plasticity * energy)
# forward (L147)
potential = torch.sigmoid(self.V(x_in))
# forward (L149)
gamma = gamma * (1.0 + is_singularity * (self.black_hole_strength - 1.0))
# forward (L155)
gamma = gamma + mu * v
```

### gfn\integrators\adaptive.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 14 | AdaptiveIntegrator | `3. Error estimate E = \|\|x1 - x2\|\| / (2^p - 1)` |
| 21 | __init__ | `def __init__(self, base_integrator, tolerance=1e-3, max_depth=3):` |
| 32 | __init__ | `self.error_scale = 1.0 / 15.0` |
| 34 | __init__ | `self.error_scale = 1.0 / 3.0 # Conservatively assume 2nd order` |
| 36 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, depth=0, **kwargs):` |
| 40 | forward | `dt = self.base_integrator.dt * dt_scale` |
| 43 | forward | `x1, v1 = self.base_integrator(x, v, force=force, dt_scale=dt_scale, steps=1, **kwargs)` |
| 47 | forward | `x_mid, v_mid = self.base_integrator(x, v, force=force, dt_scale=dt_scale * 0.5, steps=1, **kwargs)` |
| 48 | forward | `x2, v2 = self.base_integrator(x_mid, v_mid, force=force, dt_scale=dt_scale * 0.5, steps=1, **kwargs)` |
| 52 | forward | `error = torch.norm(x1 - x2, dim=-1).max() * self.error_scale` |
| 60 | forward | `x_half, v_half = self.forward(x, v, force, dt_scale * 0.5, depth + 1, **kwargs)` |
| 64 | forward | `x_final, v_final = self.forward(x_half, v_half, force, dt_scale * 0.5, depth + 1, **kwargs)` |

#### Fórmulas Listas para Usar (Python)
```python
# AdaptiveIntegrator (L14)
3. Error estimate E = ||x1 - x2|| / (2^p - 1)
# __init__ (L21)
def __init__(self, base_integrator, tolerance=1e-3, max_depth=3):
# __init__ (L32)
self.error_scale = 1.0 / 15.0
# __init__ (L34)
self.error_scale = 1.0 / 3.0 # Conservatively assume 2nd order
# forward (L36)
def forward(self, x, v, force=None, dt_scale=1.0, depth=0, **kwargs):
# forward (L40)
dt = self.base_integrator.dt * dt_scale
# forward (L43)
x1, v1 = self.base_integrator(x, v, force=force, dt_scale=dt_scale, steps=1, **kwargs)
# forward (L47)
x_mid, v_mid = self.base_integrator(x, v, force=force, dt_scale=dt_scale * 0.5, steps=1, **kwargs)
# forward (L48)
x2, v2 = self.base_integrator(x_mid, v_mid, force=force, dt_scale=dt_scale * 0.5, steps=1, **kwargs)
# forward (L52)
error = torch.norm(x1 - x2, dim=-1).max() * self.error_scale
# forward (L60)
x_half, v_half = self.forward(x, v, force, dt_scale * 0.5, depth + 1, **kwargs)
# forward (L64)
x_final, v_final = self.forward(x_half, v_half, force, dt_scale * 0.5, depth + 1, **kwargs)
```

### gfn\integrators\neural.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 8 | Global | `Idea: x_{t+1} = x_t + v_t * NeuralNet(x_t, v_t)` |
| 36 | __init__ | `self.controller = nn.Sequential( nn.Linear(self.dim * 3, self.dim), # Input: [x, v, f] nn.GELU(), # Better gradients than Tanh nn.Linear(self.dim, 1), nn.Softplus() # Strictly positive dt )` |
| 48 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 62 | forward | `state = torch.cat([x, v, f_in], dim=-1)` |
| 63 | forward | `learned_scale = self.controller(state) + 0.1` |
| 64 | forward | `dynamics_dt = self.base_dt * dt_scale * learned_scale` |
| 67 | forward | `acc = -self.christoffel(v, x, force=f_in, **kwargs)` |
| 69 | forward | `acc = acc + force` |
| 71 | forward | `v_half = v + 0.5 * dynamics_dt * acc` |
| 72 | forward | `x = x + dynamics_dt * v_half` |
| 77 | forward | `acc_next = -self.christoffel(v_half, x, force=f_in, **kwargs)` |
| 79 | forward | `acc_next = acc_next + force` |
| 81 | forward | `v = v_half + 0.5 * dynamics_dt * acc_next` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L8)
Idea: x_{t+1} = x_t + v_t * NeuralNet(x_t, v_t)
# __init__ (L36)
self.controller = nn.Sequential( nn.Linear(self.dim * 3, self.dim), # Input: [x, v, f] nn.GELU(), # Better gradients than Tanh nn.Linear(self.dim, 1), nn.Softplus() # Strictly positive dt )
# forward (L48)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L62)
state = torch.cat([x, v, f_in], dim=-1)
# forward (L63)
learned_scale = self.controller(state) + 0.1
# forward (L64)
dynamics_dt = self.base_dt * dt_scale * learned_scale
# forward (L67)
acc = -self.christoffel(v, x, force=f_in, **kwargs)
# forward (L69)
acc = acc + force
# forward (L71)
v_half = v + 0.5 * dynamics_dt * acc
# forward (L72)
x = x + dynamics_dt * v_half
# forward (L77)
acc_next = -self.christoffel(v_half, x, force=f_in, **kwargs)
# forward (L79)
acc_next = acc_next + force
# forward (L81)
v = v_half + 0.5 * dynamics_dt * acc_next
```

### gfn\integrators\runge_kutta\dormand_prince.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 32 | __init__ | `self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]` |
| 36 | __init__ | `self.a31, self.a32 = 3/40, 9/40` |
| 37 | __init__ | `self.a41, self.a42, self.a43 = 44/45, -56/15, 32/9` |
| 38 | __init__ | `self.a51, self.a52, self.a53, self.a54 = 19372/6561, -25360/2187, 64448/6561, -212/729` |
| 39 | __init__ | `self.a61, self.a62, self.a63, self.a64, self.a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656` |
| 42 | __init__ | `self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]` |
| 44 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 60 | forward | `dt = self.base_dt * dt_scale` |
| 70 | forward | `dt = self.base_dt * dt_scale` |
| 73 | dynamics | `acc = -self.christoffel(tv, tx, force=force, **kwargs)` |
| 75 | dynamics | `acc = acc + force` |
| 87 | dynamics | `x2 = apply_boundary_python(curr_x + dt * (self.a21*k1_x), topo_id)` |
| 88 | dynamics | `v2 = curr_v + dt * (self.a21*k1_v)` |
| 93 | dynamics | `x3 = apply_boundary_python(curr_x + dt * (self.a31*k1_x + self.a32*k2_x), topo_id)` |
| 94 | dynamics | `v3 = curr_v + dt * (self.a31*k1_v + self.a32*k2_v)` |
| 99 | dynamics | `x4 = apply_boundary_python(curr_x + dt * (self.a41*k1_x + self.a42*k2_x + self.a43*k3_x), topo_id)` |
| 100 | dynamics | `v4 = curr_v + dt * (self.a41*k1_v + self.a42*k2_v + self.a43*k3_v)` |
| 105 | dynamics | `x5 = apply_boundary_python(curr_x + dt * (self.a51*k1_x + self.a52*k2_x + self.a53*k3_x + self.a54*k4_x), topo_id)` |
| 106 | dynamics | `v5 = curr_v + dt * (self.a51*k1_v + self.a52*k2_v + self.a53*k3_v + self.a54*k4_v)` |
| 111 | dynamics | `x6 = apply_boundary_python(curr_x + dt * (self.a61*k1_x + self.a62*k2_x + self.a63*k3_x + self.a64*k4_x + self.a65*k5_x), topo_id)` |
| 112 | dynamics | `v6 = curr_v + dt * (self.a61*k1_v + self.a62*k2_v + self.a63*k3_v + self.a64*k4_v + self.a65*k5_v)` |
| 117 | dynamics | `curr_x = curr_x + dt * (self.b5[0]*k1_x + self.b5[2]*k3_x + self.b5[3]*k4_x + self.b5[4]*k5_x + self.b5[5]*k6_x)` |
| 119 | dynamics | `curr_v = curr_v + dt * (self.b5[0]*k1_v + self.b5[2]*k3_v + self.b5[3]*k4_v + self.b5[4]*k5_v + self.b5[5]*k6_v)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L32)
self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
# __init__ (L36)
self.a31, self.a32 = 3/40, 9/40
# __init__ (L37)
self.a41, self.a42, self.a43 = 44/45, -56/15, 32/9
# __init__ (L38)
self.a51, self.a52, self.a53, self.a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
# __init__ (L39)
self.a61, self.a62, self.a63, self.a64, self.a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
# __init__ (L42)
self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
# forward (L44)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L60)
dt = self.base_dt * dt_scale
# forward (L70)
dt = self.base_dt * dt_scale
# dynamics (L73)
acc = -self.christoffel(tv, tx, force=force, **kwargs)
# dynamics (L75)
acc = acc + force
# dynamics (L87)
x2 = apply_boundary_python(curr_x + dt * (self.a21*k1_x), topo_id)
# dynamics (L88)
v2 = curr_v + dt * (self.a21*k1_v)
# dynamics (L93)
x3 = apply_boundary_python(curr_x + dt * (self.a31*k1_x + self.a32*k2_x), topo_id)
# dynamics (L94)
v3 = curr_v + dt * (self.a31*k1_v + self.a32*k2_v)
# dynamics (L99)
x4 = apply_boundary_python(curr_x + dt * (self.a41*k1_x + self.a42*k2_x + self.a43*k3_x), topo_id)
# dynamics (L100)
v4 = curr_v + dt * (self.a41*k1_v + self.a42*k2_v + self.a43*k3_v)
# dynamics (L105)
x5 = apply_boundary_python(curr_x + dt * (self.a51*k1_x + self.a52*k2_x + self.a53*k3_x + self.a54*k4_x), topo_id)
# dynamics (L106)
v5 = curr_v + dt * (self.a51*k1_v + self.a52*k2_v + self.a53*k3_v + self.a54*k4_v)
# dynamics (L111)
x6 = apply_boundary_python(curr_x + dt * (self.a61*k1_x + self.a62*k2_x + self.a63*k3_x + self.a64*k4_x + self.a65*k5_x), topo_id)
# dynamics (L112)
v6 = curr_v + dt * (self.a61*k1_v + self.a62*k2_v + self.a63*k3_v + self.a64*k4_v + self.a65*k5_v)
# dynamics (L117)
curr_x = curr_x + dt * (self.b5[0]*k1_x + self.b5[2]*k3_x + self.b5[3]*k4_x + self.b5[4]*k5_x + self.b5[5]*k6_x)
# dynamics (L119)
curr_v = curr_v + dt * (self.b5[0]*k1_v + self.b5[2]*k3_v + self.b5[3]*k4_v + self.b5[4]*k5_v + self.b5[5]*k6_v)
```

### gfn\integrators\runge_kutta\euler.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 23 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 46 | forward | `dt = self.dt * dt_scale` |
| 48 | forward | `c_out = self.christoffel(curr_v, curr_x, force=force, **kwargs)` |
| 54 | forward | `acc = acc + force` |
| 56 | forward | `curr_x = curr_x + dt * curr_v` |
| 57 | forward | `curr_v = curr_v + dt * acc` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L23)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L46)
dt = self.dt * dt_scale
# forward (L48)
c_out = self.christoffel(curr_v, curr_x, force=force, **kwargs)
# forward (L54)
acc = acc + force
# forward (L56)
curr_x = curr_x + dt * curr_v
# forward (L57)
curr_v = curr_v + dt * acc
```

### gfn\integrators\runge_kutta\heun.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 23 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 62 | forward | `dt = self.dt * dt_scale` |
| 67 | dynamics | `c_out = self.christoffel(current_v, current_x, force=force, **kwargs)` |
| 73 | dynamics | `acc = acc + force` |
| 84 | dynamics | `v_pred = curr_v + dt * dv1` |
| 85 | dynamics | `x_pred = apply_boundary_python(curr_x + dt * dx1, topo_id)` |
| 92 | dynamics | `curr_x = curr_x + (dt / 2.0) * (dx1 + dx2)` |
| 93 | dynamics | `curr_v = curr_v + (dt / 2.0) * (dv1 + dv2)` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L23)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L62)
dt = self.dt * dt_scale
# dynamics (L67)
c_out = self.christoffel(current_v, current_x, force=force, **kwargs)
# dynamics (L73)
acc = acc + force
# dynamics (L84)
v_pred = curr_v + dt * dv1
# dynamics (L85)
x_pred = apply_boundary_python(curr_x + dt * dx1, topo_id)
# dynamics (L92)
curr_x = curr_x + (dt / 2.0) * (dx1 + dx2)
# dynamics (L93)
curr_v = curr_v + (dt / 2.0) * (dv1 + dv2)
```

### gfn\integrators\runge_kutta\rk4.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 42 | forward | `dt = self.dt * dt_scale` |
| 46 | dynamics | `c_out = self.christoffel(current_v, current_x, force=force, **kwargs)` |
| 52 | dynamics | `acc = acc + force` |
| 63 | dynamics | `v2 = curr_v + 0.5 * dt * dv1` |
| 64 | dynamics | `x2 = apply_boundary_python(curr_x + 0.5 * dt * dx1, topo_id)` |
| 69 | dynamics | `v3 = curr_v + 0.5 * dt * dv2` |
| 70 | dynamics | `x3 = apply_boundary_python(curr_x + 0.5 * dt * dx2, topo_id)` |
| 75 | dynamics | `v4 = curr_v + dt * dv3` |
| 76 | dynamics | `x4 = apply_boundary_python(curr_x + dt * dx3, topo_id)` |
| 81 | dynamics | `curr_x = curr_x + (dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)` |
| 83 | dynamics | `curr_v = curr_v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L21)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L42)
dt = self.dt * dt_scale
# dynamics (L46)
c_out = self.christoffel(current_v, current_x, force=force, **kwargs)
# dynamics (L52)
acc = acc + force
# dynamics (L63)
v2 = curr_v + 0.5 * dt * dv1
# dynamics (L64)
x2 = apply_boundary_python(curr_x + 0.5 * dt * dx1, topo_id)
# dynamics (L69)
v3 = curr_v + 0.5 * dt * dv2
# dynamics (L70)
x3 = apply_boundary_python(curr_x + 0.5 * dt * dx2, topo_id)
# dynamics (L75)
v4 = curr_v + dt * dv3
# dynamics (L76)
x4 = apply_boundary_python(curr_x + dt * dx3, topo_id)
# dynamics (L81)
curr_x = curr_x + (dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
# dynamics (L83)
curr_v = curr_v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
```

### gfn\integrators\stochastic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, **kwargs):` |
| 24 | forward | `x_next, v_next = self.base_integrator(x, v, force=force, dt_scale=dt_scale, **kwargs)` |
| 28 | forward | `dt = self.dt * dt_scale` |
| 31 | forward | `impulse = self.geometric_noise(x, v, self.christoffel, dt=dt, **kwargs)` |
| 34 | forward | `v_stochastic = v_next + impulse` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L21)
def forward(self, x, v, force=None, dt_scale=1.0, **kwargs):
# forward (L24)
x_next, v_next = self.base_integrator(x, v, force=force, dt_scale=dt_scale, **kwargs)
# forward (L28)
dt = self.dt * dt_scale
# forward (L31)
impulse = self.geometric_noise(x, v, self.christoffel, dt=dt, **kwargs)
# forward (L34)
v_stochastic = v_next + impulse
```

### gfn\integrators\symplectic\coupling.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 9 | Global | `v' = v + F(x)  (Shear transformation on v)` |
| 10 | Global | `x' = x + G(v') (Shear transformation on x)` |
| 53 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 62 | forward | `dt = self.dt * dt_scale` |
| 71 | forward | `acc_1 = -self.christoffel(v_dummy, x, force=f_in, **kwargs) + f_in` |
| 72 | forward | `v_half = v + 0.5 * dt * acc_1` |
| 75 | forward | `x = x + dt * (v_half + warp)` |
| 80 | forward | `acc_2 = -self.christoffel(v_dummy, x, force=f_in, **kwargs) + f_in` |
| 81 | forward | `v = v_half + 0.5 * dt * acc_2` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L9)
v' = v + F(x)  (Shear transformation on v)
# Global (L10)
x' = x + G(v') (Shear transformation on x)
# forward (L53)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L62)
dt = self.dt * dt_scale
# forward (L71)
acc_1 = -self.christoffel(v_dummy, x, force=f_in, **kwargs) + f_in
# forward (L72)
v_half = v + 0.5 * dt * acc_1
# forward (L75)
x = x + dt * (v_half + warp)
# forward (L80)
acc_2 = -self.christoffel(v_dummy, x, force=f_in, **kwargs) + f_in
# forward (L81)
v = v_half + 0.5 * dt * acc_2
```

### gfn\integrators\symplectic\forest_ruth.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 29 | __init__ | `theta = 1.0 / (2.0 - 2.0**(1.0/3.0))` |
| 31 | __init__ | `self.c1 = theta / 2.0` |
| 32 | __init__ | `self.c2 = (1.0 - theta) / 2.0` |
| 33 | __init__ | `self.c3 = (1.0 - theta) / 2.0` |
| 34 | __init__ | `self.c4 = theta / 2.0` |
| 37 | __init__ | `self.d2 = 1.0 - 2.0*theta` |
| 40 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 56 | forward | `dt = self.dt * dt_scale` |
| 70 | acceleration | `c_out = self.christoffel(tv, tx, force=force, **kwargs)` |
| 81 | acceleration | `x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)` |
| 82 | acceleration | `v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)` |
| 85 | acceleration | `x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)` |
| 86 | acceleration | `v2 = v1 + self.d2 * dt * acceleration(x2, v1)` |
| 89 | acceleration | `x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)` |
| 90 | acceleration | `v3 = v2 + self.d3 * dt * acceleration(x3, v2)` |
| 93 | acceleration | `curr_x = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L29)
theta = 1.0 / (2.0 - 2.0**(1.0/3.0))
# __init__ (L31)
self.c1 = theta / 2.0
# __init__ (L32)
self.c2 = (1.0 - theta) / 2.0
# __init__ (L33)
self.c3 = (1.0 - theta) / 2.0
# __init__ (L34)
self.c4 = theta / 2.0
# __init__ (L37)
self.d2 = 1.0 - 2.0*theta
# forward (L40)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L56)
dt = self.dt * dt_scale
# acceleration (L70)
c_out = self.christoffel(tv, tx, force=force, **kwargs)
# acceleration (L81)
x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)
# acceleration (L82)
v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)
# acceleration (L85)
x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)
# acceleration (L86)
v2 = v1 + self.d2 * dt * acceleration(x2, v1)
# acceleration (L89)
x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)
# acceleration (L90)
v3 = v2 + self.d3 * dt * acceleration(x3, v2)
# acceleration (L93)
curr_x = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)
```

### gfn\integrators\symplectic\leapfrog.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 12 | Global | `v(t+0.5h) = v(t) + 0.5h * a(x(t))` |
| 13 | Global | `x(t+h) = x(t) + h * v(t+0.5h)` |
| 14 | Global | `v(t+h) = v(t+0.5h) + 0.5h * a(x(t+h))` |
| 17 | Global | `v(t+0.5h) = (v(t) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t)))` |
| 18 | Global | `x(t+h) = x(t) + h * v(t+0.5h)` |
| 19 | Global | `v(t+h) = (v(t+0.5h) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t+h)))` |
| 22 | Global | `- In ABSENCE of friction (mu = 0), energy is conserved` |
| 24 | Global | `- VOLUME preservation is LOST when friction != 0` |
| 57 | Global | `EPSILON_STANDARD = 1e-7` |
| 58 | Global | `EPSILON_SMOOTH = 1e-7` |
| 70 | LeapfrogIntegrator | `- Uses updated FRICTION_SCALE=0.02` |
| 71 | LeapfrogIntegrator | `- Uses EPSILON_STANDARD=1e-7` |
| 79 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 132 | forward | `effective_dt = self.dt * dt_scale` |
| 133 | forward | `h = 0.5 * effective_dt` |
| 138 | forward | `res = self.christoffel(curr_v, curr_x, force=force, **kwargs)` |
| 166 | forward | `gate = torch.matmul(feat, Wf.t()) + bf` |
| 170 | forward | `gate = gate + torch.matmul(force, Wi.t())` |
| 171 | forward | `mu = torch.sigmoid(gate) * FRICTION_SCALE` |
| 173 | forward | `v_norm = torch.norm(curr_v, dim=-1, keepdim=True)` |
| 174 | forward | `v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + EPSILON_SMOOTH)` |
| 175 | forward | `mu = mu * (1.0 + velocity_friction_scale * v_norm)` |
| 179 | forward | `v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu + EPSILON_STANDARD)` |
| 182 | forward | `curr_x = curr_x + effective_dt * v_half` |
| 188 | forward | `res_half = self.christoffel(v_half, curr_x, force=force, **kwargs)` |
| 203 | forward | `gate = torch.matmul(feat, Wf.t()) + bf` |
| 207 | forward | `gate = gate + torch.matmul(force, Wi.t())` |
| 208 | forward | `mu_half = torch.sigmoid(gate) * FRICTION_SCALE` |
| 210 | forward | `v_norm = torch.norm(v_half, dim=-1, keepdim=True)` |
| 211 | forward | `v_norm = v_norm / (v_half.shape[-1] ** 0.5 + EPSILON_SMOOTH)` |
| 212 | forward | `mu_half = mu_half * (1.0 + velocity_friction_scale * v_norm)` |
| 215 | forward | `curr_v = (v_half + h * (force - gamma_half)) / (1.0 + h * mu_half + EPSILON_STANDARD)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L12)
v(t+0.5h) = v(t) + 0.5h * a(x(t))
# Global (L13)
x(t+h) = x(t) + h * v(t+0.5h)
# Global (L14)
v(t+h) = v(t+0.5h) + 0.5h * a(x(t+h))
# Global (L17)
v(t+0.5h) = (v(t) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t)))
# Global (L18)
x(t+h) = x(t) + h * v(t+0.5h)
# Global (L19)
v(t+h) = (v(t+0.5h) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t+h)))
# Global (L22)
- In ABSENCE of friction (mu = 0), energy is conserved
# Global (L24)
- VOLUME preservation is LOST when friction != 0
# Global (L57)
EPSILON_STANDARD = 1e-7
# Global (L58)
EPSILON_SMOOTH = 1e-7
# LeapfrogIntegrator (L70)
- Uses updated FRICTION_SCALE=0.02
# LeapfrogIntegrator (L71)
- Uses EPSILON_STANDARD=1e-7
# forward (L79)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L132)
effective_dt = self.dt * dt_scale
# forward (L133)
h = 0.5 * effective_dt
# forward (L138)
res = self.christoffel(curr_v, curr_x, force=force, **kwargs)
# forward (L166)
gate = torch.matmul(feat, Wf.t()) + bf
# forward (L170)
gate = gate + torch.matmul(force, Wi.t())
# forward (L171)
mu = torch.sigmoid(gate) * FRICTION_SCALE
# forward (L173)
v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
# forward (L174)
v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + EPSILON_SMOOTH)
# forward (L175)
mu = mu * (1.0 + velocity_friction_scale * v_norm)
# forward (L179)
v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu + EPSILON_STANDARD)
# forward (L182)
curr_x = curr_x + effective_dt * v_half
# forward (L188)
res_half = self.christoffel(v_half, curr_x, force=force, **kwargs)
# forward (L203)
gate = torch.matmul(feat, Wf.t()) + bf
# forward (L207)
gate = gate + torch.matmul(force, Wi.t())
# forward (L208)
mu_half = torch.sigmoid(gate) * FRICTION_SCALE
# forward (L210)
v_norm = torch.norm(v_half, dim=-1, keepdim=True)
# forward (L211)
v_norm = v_norm / (v_half.shape[-1] ** 0.5 + EPSILON_SMOOTH)
# forward (L212)
mu_half = mu_half * (1.0 + velocity_friction_scale * v_norm)
# forward (L215)
curr_v = (v_half + h * (force - gamma_half)) / (1.0 + h * mu_half + EPSILON_STANDARD)
```

### gfn\integrators\symplectic\omelyan.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 28 | __init__ | `self.lam = -0.2123418310626054` |
| 29 | __init__ | `self.chi = -0.06626458266981849` |
| 34 | __init__ | `self.c3 = 1.0 - 2.0*(self.chi + self.xi)` |
| 38 | __init__ | `self.d1 = (1.0 - 2.0*self.lam) / 2.0` |
| 41 | __init__ | `self.d4 = (1.0 - 2.0*self.lam) / 2.0` |
| 43 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 59 | forward | `dt = self.dt * dt_scale` |
| 73 | acceleration | `c_out = self.christoffel(tv, tx, force=force, **kwargs)` |
| 84 | acceleration | `x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)` |
| 85 | acceleration | `v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)` |
| 88 | acceleration | `x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)` |
| 89 | acceleration | `v2 = v1 + self.d2 * dt * acceleration(x2, v1)` |
| 92 | acceleration | `x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)` |
| 93 | acceleration | `v3 = v2 + self.d3 * dt * acceleration(x3, v2)` |
| 96 | acceleration | `x4 = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)` |
| 97 | acceleration | `v4 = v3 + self.d4 * dt * acceleration(x4, v3)` |
| 100 | acceleration | `curr_x = apply_boundary_python(x4 + self.c5 * dt * v4, topo_id)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L28)
self.lam = -0.2123418310626054
# __init__ (L29)
self.chi = -0.06626458266981849
# __init__ (L34)
self.c3 = 1.0 - 2.0*(self.chi + self.xi)
# __init__ (L38)
self.d1 = (1.0 - 2.0*self.lam) / 2.0
# __init__ (L41)
self.d4 = (1.0 - 2.0*self.lam) / 2.0
# forward (L43)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L59)
dt = self.dt * dt_scale
# acceleration (L73)
c_out = self.christoffel(tv, tx, force=force, **kwargs)
# acceleration (L84)
x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)
# acceleration (L85)
v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)
# acceleration (L88)
x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)
# acceleration (L89)
v2 = v1 + self.d2 * dt * acceleration(x2, v1)
# acceleration (L92)
x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)
# acceleration (L93)
v3 = v2 + self.d3 * dt * acceleration(x3, v2)
# acceleration (L96)
x4 = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)
# acceleration (L97)
v4 = v3 + self.d4 * dt * acceleration(x4, v3)
# acceleration (L100)
curr_x = apply_boundary_python(x4 + self.c5 * dt * v4, topo_id)
```

### gfn\integrators\symplectic\pefrl.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 26 | __init__ | `self.lam = -0.2123418310626054` |
| 27 | __init__ | `self.chi = -0.06626458266981849` |
| 29 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 30 | forward | `dt = self.dt * dt_scale` |
| 37 | forward | `K1 = (1.0 - 2.0 * LAM) / 2.0` |
| 38 | forward | `D1 = 1.0 - 2.0 * (CHI + XI)` |
| 53 | get_acc | `c_out = self.christoffel(tv, tx, force=force, **kwargs)` |
| 63 | get_acc | `curr_x = apply_boundary_python(curr_x + XI * dt * curr_v, topo_id)` |
| 66 | get_acc | `curr_v = curr_v + K1 * dt * get_acc(curr_x, curr_v, is_first=True)` |
| 69 | get_acc | `curr_x = apply_boundary_python(curr_x + CHI * dt * curr_v, topo_id)` |
| 72 | get_acc | `curr_v = curr_v + LAM * dt * get_acc(curr_x, curr_v)` |
| 75 | get_acc | `curr_x = apply_boundary_python(curr_x + D1 * dt * curr_v, topo_id)` |
| 78 | get_acc | `curr_v = curr_v + LAM * dt * get_acc(curr_x, curr_v)` |
| 81 | get_acc | `curr_x = apply_boundary_python(curr_x + CHI * dt * curr_v, topo_id)` |
| 84 | get_acc | `curr_v = curr_v + K1 * dt * get_acc(curr_x, curr_v)` |
| 87 | get_acc | `curr_x = apply_boundary_python(curr_x + XI * dt * curr_v, topo_id)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L26)
self.lam = -0.2123418310626054
# __init__ (L27)
self.chi = -0.06626458266981849
# forward (L29)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L30)
dt = self.dt * dt_scale
# forward (L37)
K1 = (1.0 - 2.0 * LAM) / 2.0
# forward (L38)
D1 = 1.0 - 2.0 * (CHI + XI)
# get_acc (L53)
c_out = self.christoffel(tv, tx, force=force, **kwargs)
# get_acc (L63)
curr_x = apply_boundary_python(curr_x + XI * dt * curr_v, topo_id)
# get_acc (L66)
curr_v = curr_v + K1 * dt * get_acc(curr_x, curr_v, is_first=True)
# get_acc (L69)
curr_x = apply_boundary_python(curr_x + CHI * dt * curr_v, topo_id)
# get_acc (L72)
curr_v = curr_v + LAM * dt * get_acc(curr_x, curr_v)
# get_acc (L75)
curr_x = apply_boundary_python(curr_x + D1 * dt * curr_v, topo_id)
# get_acc (L78)
curr_v = curr_v + LAM * dt * get_acc(curr_x, curr_v)
# get_acc (L81)
curr_x = apply_boundary_python(curr_x + CHI * dt * curr_v, topo_id)
# get_acc (L84)
curr_v = curr_v + K1 * dt * get_acc(curr_x, curr_v)
# get_acc (L87)
curr_x = apply_boundary_python(curr_x + XI * dt * curr_v, topo_id)
```

### gfn\integrators\symplectic\verlet.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 20 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 40 | forward | `dt = self.dt * dt_scale` |
| 48 | forward | `gamma = self.christoffel(v, x, force=force, **kwargs)` |
| 53 | forward | `a = -gamma + force` |
| 56 | forward | `v_half = v + 0.5 * dt * a` |
| 59 | forward | `x = x + dt * v_half` |
| 65 | forward | `gamma_next = self.christoffel(v_half, x, force=force, **kwargs)` |
| 67 | forward | `a_next = -gamma_next` |
| 69 | forward | `a_next = -gamma_next + force` |
| 71 | forward | `v = v_half + 0.5 * dt * a_next` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L20)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L40)
dt = self.dt * dt_scale
# forward (L48)
gamma = self.christoffel(v, x, force=force, **kwargs)
# forward (L53)
a = -gamma + force
# forward (L56)
v_half = v + 0.5 * dt * a
# forward (L59)
x = x + dt * v_half
# forward (L65)
gamma_next = self.christoffel(v_half, x, force=force, **kwargs)
# forward (L67)
a_next = -gamma_next
# forward (L69)
a_next = -gamma_next + force
# forward (L71)
v = v_half + 0.5 * dt * a_next
```

### gfn\integrators\symplectic\yoshida.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 23 | __init__ | `w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))` |
| 24 | __init__ | `w0 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))` |
| 26 | __init__ | `self.c1 = w1 / 2.0` |
| 27 | __init__ | `self.c2 = (w0 + w1) / 2.0` |
| 35 | forward | `def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):` |
| 50 | forward | `dt = self.dt * dt_scale` |
| 62 | acceleration | `c_out = self.christoffel(tv, tx, force=force, **kwargs)` |
| 73 | acceleration | `x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)` |
| 74 | acceleration | `v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)` |
| 77 | acceleration | `x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)` |
| 78 | acceleration | `v2 = v1 + self.d2 * dt * acceleration(x2, v1)` |
| 81 | acceleration | `x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)` |
| 82 | acceleration | `v3 = v2 + self.d3 * dt * acceleration(x3, v2)` |
| 85 | acceleration | `curr_x = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L23)
w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
# __init__ (L24)
w0 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
# __init__ (L26)
self.c1 = w1 / 2.0
# __init__ (L27)
self.c2 = (w0 + w1) / 2.0
# forward (L35)
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# forward (L50)
dt = self.dt * dt_scale
# acceleration (L62)
c_out = self.christoffel(tv, tx, force=force, **kwargs)
# acceleration (L73)
x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)
# acceleration (L74)
v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)
# acceleration (L77)
x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)
# acceleration (L78)
v2 = v1 + self.d2 * dt * acceleration(x2, v1)
# acceleration (L81)
x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)
# acceleration (L82)
v3 = v2 + self.d3 * dt * acceleration(x3, v2)
# acceleration (L85)
curr_x = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)
```

### gfn\layers\base.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 54 | __init__ | `self.head_dim = dim // heads` |
| 60 | __init__ | `self.depth_scale = 1.0 / (total_depth ** 0.5)` |
| 69 | __init__ | `head_rank = max(4, rank // heads)` |
| 204 | create_manifold | `target_dt = self.base_dt / 0.9` |
| 206 | create_manifold | `val = val_init + i * 0.1` |
| 217 | create_manifold | `gate_in_dim = (3 if self.topology_id == 1 else 2) * self.head_dim` |
| 255 | create_manifold | `tolerance = adaptive_cfg.get('tolerance', 1e-3)` |
| 292 | create_manifold | `self.out_proj_x = nn.Linear(3 * dim if self.topology_id == 1 else dim, dim)` |
| 327 | forward | `m_heads = [None] * self.heads` |
| 331 | forward | `force = force + self.context_proj(context)` |
| 334 | forward | `f_heads = [None] * self.heads` |
| 343 | forward | `dt_min = stability_cfg.get('dt_min', self.base_dt * 0.1)` |
| 344 | forward | `dt_max = stability_cfg.get('dt_max', self.base_dt * 4.0)` |
| 361 | forward | `scale = dt_base * gates # [Heads, Batch, 1]` |
| 411 | forward | `head_dim = self.dim // self.heads` |
| 422 | forward | `x_h = x[:, i*head_dim : (i+1)*head_dim]` |
| 465 | forward | `dim_v = (2 * self.head_dim) if (self.topology_id == 1) else self.head_dim` |
| 470 | forward | `res = recurrent_manifold_fused( x=x, v=v, f=force.unsqueeze(1), U_stack=u_stack, W_stack=w_stack, dt=self.base_dt, dt_scales=dt_scales, forget_rates=None, num_heads=self.heads, topology=self.topology_id, Wf=W_forget_stack.view(-1, W_forget_stack.shape[-1]) if Wf is not None else None, Wi=W_input_stack.view(-1, W_input_stack.shape[-1]) if Wi is not None else None, bf=b_forget_stack.view(-1) if bf is not None else None, V_w=V_w_stack, hysteresis_state=memory_state, hyst_enabled=self.hysteresis_enabled, thermo_alpha=t_alpha, thermo_temp=t_temp, holographic_z=h_z_ten, holographic_grad_z=h_gz_ten, **kwargs )` |
| 508 | forward | `extra_kwargs = { 'W_forget_stack': W_forget_stack[i:i+1], # [1, D, D] 'W_input_stack': W_input_stack[i:i+1], 'b_forget_stack': b_forget_stack[i:i+1], 'topology': self.topology_id, 'collect_christ': collect_christ, 'memory_state': m_heads[i] }` |
| 517 | forward | `res = self.integrators[i](x_heads[i], v_heads[i], force=f_heads[i], dt_scale=scale[i], **extra_kwargs)` |
| 540 | forward | `context_next = gates.squeeze(-1).transpose(0, 1)` |
| 545 | forward | `x_cat = torch.stack(x_outs, dim=1).view(batch, -1)` |
| 546 | forward | `v_cat = torch.stack(v_outs, dim=1).view(batch, -1)` |
| 551 | forward | `v_mix = torch.tanh(v_cat / 100.0)` |
| 552 | forward | `mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)` |
| 574 | forward | `v_next = 100.0 * torch.tanh(v_next / 100.0)` |
| 576 | forward | `context_next = gates.squeeze(-1).transpose(0, 1)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L54)
self.head_dim = dim // heads
# __init__ (L60)
self.depth_scale = 1.0 / (total_depth ** 0.5)
# __init__ (L69)
head_rank = max(4, rank // heads)
# create_manifold (L204)
target_dt = self.base_dt / 0.9
# create_manifold (L206)
val = val_init + i * 0.1
# create_manifold (L217)
gate_in_dim = (3 if self.topology_id == 1 else 2) * self.head_dim
# create_manifold (L255)
tolerance = adaptive_cfg.get('tolerance', 1e-3)
# create_manifold (L292)
self.out_proj_x = nn.Linear(3 * dim if self.topology_id == 1 else dim, dim)
# forward (L327)
m_heads = [None] * self.heads
# forward (L331)
force = force + self.context_proj(context)
# forward (L334)
f_heads = [None] * self.heads
# forward (L343)
dt_min = stability_cfg.get('dt_min', self.base_dt * 0.1)
# forward (L344)
dt_max = stability_cfg.get('dt_max', self.base_dt * 4.0)
# forward (L361)
scale = dt_base * gates # [Heads, Batch, 1]
# forward (L411)
head_dim = self.dim // self.heads
# forward (L422)
x_h = x[:, i*head_dim : (i+1)*head_dim]
# forward (L465)
dim_v = (2 * self.head_dim) if (self.topology_id == 1) else self.head_dim
# forward (L470)
res = recurrent_manifold_fused( x=x, v=v, f=force.unsqueeze(1), U_stack=u_stack, W_stack=w_stack, dt=self.base_dt, dt_scales=dt_scales, forget_rates=None, num_heads=self.heads, topology=self.topology_id, Wf=W_forget_stack.view(-1, W_forget_stack.shape[-1]) if Wf is not None else None, Wi=W_input_stack.view(-1, W_input_stack.shape[-1]) if Wi is not None else None, bf=b_forget_stack.view(-1) if bf is not None else None, V_w=V_w_stack, hysteresis_state=memory_state, hyst_enabled=self.hysteresis_enabled, thermo_alpha=t_alpha, thermo_temp=t_temp, holographic_z=h_z_ten, holographic_grad_z=h_gz_ten, **kwargs )
# forward (L508)
extra_kwargs = { 'W_forget_stack': W_forget_stack[i:i+1], # [1, D, D] 'W_input_stack': W_input_stack[i:i+1], 'b_forget_stack': b_forget_stack[i:i+1], 'topology': self.topology_id, 'collect_christ': collect_christ, 'memory_state': m_heads[i] }
# forward (L517)
res = self.integrators[i](x_heads[i], v_heads[i], force=f_heads[i], dt_scale=scale[i], **extra_kwargs)
# forward (L540)
context_next = gates.squeeze(-1).transpose(0, 1)
# forward (L545)
x_cat = torch.stack(x_outs, dim=1).view(batch, -1)
# forward (L546)
v_cat = torch.stack(v_outs, dim=1).view(batch, -1)
# forward (L551)
v_mix = torch.tanh(v_cat / 100.0)
# forward (L552)
mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)
# forward (L574)
v_next = 100.0 * torch.tanh(v_next / 100.0)
# forward (L576)
context_next = gates.squeeze(-1).transpose(0, 1)
```

### gfn\layers\fractal.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | __init__ | `self.head_dim = dim // heads` |
| 23 | __init__ | `self.depth_scale = 1.0 / (total_depth ** 0.5)  # 1/√depth` |
| 39 | __init__ | `self.micro_manifold = MLayer( dim, heads=heads, rank=max(8, rank//2), base_dt=base_dt * 0.5, integrator_type=integrator_type, physics_config=micro_cfg )` |
| 60 | forward | `curvature_r = torch.norm(stacked_gamma, dim=-1).mean(dim=-1, keepdim=True) # [batch, 1]` |
| 65 | forward | `tunnel_gate = torch.sigmoid((curvature_r - self.threshold) * 1.0)` |
| 74 | forward | `x_final = x_m + tunnel_gate * (x_f - x_m) * self.alpha_scale` |
| 75 | forward | `v_final = v_m + tunnel_gate * (v_f - v_m) * self.alpha_scale` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L16)
self.head_dim = dim // heads
# __init__ (L23)
self.depth_scale = 1.0 / (total_depth ** 0.5)  # 1/√depth
# __init__ (L39)
self.micro_manifold = MLayer( dim, heads=heads, rank=max(8, rank//2), base_dt=base_dt * 0.5, integrator_type=integrator_type, physics_config=micro_cfg )
# forward (L60)
curvature_r = torch.norm(stacked_gamma, dim=-1).mean(dim=-1, keepdim=True) # [batch, 1]
# forward (L65)
tunnel_gate = torch.sigmoid((curvature_r - self.threshold) * 1.0)
# forward (L74)
x_final = x_m + tunnel_gate * (x_f - x_m) * self.alpha_scale
# forward (L75)
v_final = v_m + tunnel_gate * (v_f - v_m) * self.alpha_scale
```

### gfn\layers\gating.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 14 | __init__ | `input_dim = 2 * dim if topology == 1 else dim` |
| 15 | __init__ | `self.curvature_net = nn.Sequential( nn.Linear(input_dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1), nn.Sigmoid() # Range [0, 1] )` |
| 36 | forward | `x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)` |
| 42 | forward | `W1 = self.curvature_net[0].weight  # [dim/4, dim]` |
| 43 | forward | `b1 = self.curvature_net[0].bias    # [dim/4]` |
| 44 | forward | `W2 = self.curvature_net[2].weight  # [1, dim/4]` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L14)
input_dim = 2 * dim if topology == 1 else dim
# __init__ (L15)
self.curvature_net = nn.Sequential( nn.Linear(input_dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1), nn.Sigmoid() # Range [0, 1] )
# forward (L36)
x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# forward (L42)
W1 = self.curvature_net[0].weight  # [dim/4, dim]
# forward (L43)
b1 = self.curvature_net[0].bias    # [dim/4]
# forward (L44)
W2 = self.curvature_net[2].weight  # [1, dim/4]
```

### gfn\layers\parallel.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 11 | ParallelMLayer | `dv/dt = F - \\Gamma(v, v)   [Non-linear]` |
| 14 | ParallelMLayer | `dv/dt = F - D(F) * v       [Linearized]` |
| 19 | ParallelMLayer | `v_t = A_t * v_{t-1} + B_t` |
| 20 | ParallelMLayer | `x_t = x_{t-1} + v_t * dt` |
| 26 | __init__ | `def __init__(self, dim, heads=4, physics_config=None, **kwargs):` |
| 31 | __init__ | `self.head_dim = dim // heads` |
| 73 | __init__ | `self.out_proj = nn.Linear(dim * 2, dim * 2)` |
| 105 | forward | `dt = self.to_dt(force) * self.base_dt * self.base_dt_scales.view(1, 1, -1)` |
| 108 | forward | `B_val = self.to_B(force) * dt` |
| 118 | forward | `x_update = v_seq * dt` |

#### Fórmulas Listas para Usar (Python)
```python
# ParallelMLayer (L11)
dv/dt = F - \\Gamma(v, v)   [Non-linear]
# ParallelMLayer (L14)
dv/dt = F - D(F) * v       [Linearized]
# ParallelMLayer (L19)
v_t = A_t * v_{t-1} + B_t
# ParallelMLayer (L20)
x_t = x_{t-1} + v_t * dt
# __init__ (L26)
def __init__(self, dim, heads=4, physics_config=None, **kwargs):
# __init__ (L31)
self.head_dim = dim // heads
# __init__ (L73)
self.out_proj = nn.Linear(dim * 2, dim * 2)
# forward (L105)
dt = self.to_dt(force) * self.base_dt * self.base_dt_scales.view(1, 1, -1)
# forward (L108)
B_val = self.to_B(force) * dt
# forward (L118)
x_update = v_seq * dt
```

### gfn\layers\thermo.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 18 | ThermodynamicGating | `K(v) = 0.5 * \|\|v\|\|^2  (Kinetic)` |
| 19 | ThermodynamicGating | `U(x) = 0.5 * \|\|x\|\|^2  (Potential - Harmonic Oscillator ansatz)` |
| 21 | ThermodynamicGating | `gate = sigmoid( (H_ref - H) / Temperature )` |
| 50 | forward | `K = 0.5 * (v ** 2).sum(dim=-1, keepdim=True)` |
| 54 | forward | `U = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)` |
| 65 | forward | `logits = (self.ref_H - H) / (T * self.sensitivity)` |
| 67 | forward | `gate = torch.sigmoid(logits)` |

#### Fórmulas Listas para Usar (Python)
```python
# ThermodynamicGating (L18)
K(v) = 0.5 * ||v||^2  (Kinetic)
# ThermodynamicGating (L19)
U(x) = 0.5 * ||x||^2  (Potential - Harmonic Oscillator ansatz)
# ThermodynamicGating (L21)
gate = sigmoid( (H_ref - H) / Temperature )
# forward (L50)
K = 0.5 * (v ** 2).sum(dim=-1, keepdim=True)
# forward (L54)
U = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)
# forward (L65)
logits = (self.ref_H - H) / (T * self.sensitivity)
# forward (L67)
gate = torch.sigmoid(logits)
```

### gfn\losses\circular.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 17 | circular_distance_loss | `L = 1 - cos(x_pred - x_target)` |
| 31 | circular_distance_loss | `delta = x_pred - x_target` |

#### Fórmulas Listas para Usar (Python)
```python
# circular_distance_loss (L17)
L = 1 - cos(x_pred - x_target)
# circular_distance_loss (L31)
delta = x_pred - x_target
```

### gfn\losses\combined.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | Global | `- hamiltonian_mode='none' or 'adaptive'` |
| 17 | Global | `- geodesic_mode='structural'` |
| 57 | __init__ | `def __init__(self, lambda_h: float = LAMBDA_H_DEFAULT, lambda_g: float = LAMBDA_G_DEFAULT, lambda_k: float = LAMBDA_K_DEFAULT, lambda_c: float = 0.0, lambda_n: float = 0.0, ignore_index: int = -100, hamiltonian_mode: str = 'adaptive', geodesic_mode: str = 'structural'):` |
| 97 | forward | `ce = self.ce_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))` |
| 115 | forward | `total = total + h_loss` |
| 126 | forward | `total = total + g_loss` |
| 132 | forward | `total = total + c_loss` |
| 138 | forward | `total = total + n_loss` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L16)
- hamiltonian_mode='none' or 'adaptive'
# Global (L17)
- geodesic_mode='structural'
# __init__ (L57)
def __init__(self, lambda_h: float = LAMBDA_H_DEFAULT, lambda_g: float = LAMBDA_G_DEFAULT, lambda_k: float = LAMBDA_K_DEFAULT, lambda_c: float = 0.0, lambda_n: float = 0.0, ignore_index: int = -100, hamiltonian_mode: str = 'adaptive', geodesic_mode: str = 'structural'):
# forward (L97)
ce = self.ce_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
# forward (L115)
total = total + h_loss
# forward (L126)
total = total + g_loss
# forward (L132)
total = total + c_loss
# forward (L138)
total = total + n_loss
```

### gfn\losses\curiosity.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 11 | curiosity_loss | `def curiosity_loss(velocities: list, lambda_c: float = 0.05) -> torch.Tensor:` |
| 23 | curiosity_loss | `S = Σ log(std(v_i) + ε)  (Entropy proxy for Gaussian-like latent distribution)` |
| 49 | curiosity_loss | `all_v = torch.cat(velocities, dim=0)  # [Batch * Seq, Dim]` |
| 53 | curiosity_loss | `v_std = all_v.std(dim=0) + 1e-6  # [Dim]` |
| 57 | curiosity_loss | `entropy = torch.log(v_std).sum()` |

#### Fórmulas Listas para Usar (Python)
```python
# curiosity_loss (L11)
def curiosity_loss(velocities: list, lambda_c: float = 0.05) -> torch.Tensor:
# curiosity_loss (L23)
S = Σ log(std(v_i) + ε)  (Entropy proxy for Gaussian-like latent distribution)
# curiosity_loss (L49)
all_v = torch.cat(velocities, dim=0)  # [Batch * Seq, Dim]
# curiosity_loss (L53)
v_std = all_v.std(dim=0) + 1e-6  # [Dim]
# curiosity_loss (L57)
entropy = torch.log(v_std).sum()
```

### gfn\losses\geodesic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 35 | geodesic_regularization | `mode: str = 'structural') -> torch.Tensor:` |
| 66 | geodesic_regularization | `mean_gamma = fused_tensor.mean() + GEODESIC_FUSED_SCALE` |
| 82 | geodesic_regularization | `curvature_norms = all_curvatures.pow(2).mean()` |
| 97 | geodesic_regularization | `curvature_diff = all_curvatures[1:] - all_curvatures[:-1]` |
| 98 | geodesic_regularization | `curvature_change_norm = curvature_diff.pow(2).mean()` |
| 102 | geodesic_regularization | `magnitude_norm = all_curvatures.pow(2).mean()` |
| 106 | geodesic_regularization | `curvature_norms = 0.85 * curvature_change_norm + 0.15 * magnitude_norm` |
| 111 | geodesic_regularization | `curvature_var = all_curvatures.var(dim=0)  # [dim]` |
| 112 | geodesic_regularization | `curvature_norms = curvature_var.mean()` |
| 116 | geodesic_regularization | `curvature_norms = all_curvatures.pow(2).mean()` |
| 120 | geodesic_regularization | `batch_mean = all_curvatures.mean()` |
| 121 | geodesic_regularization | `batch_std = all_curvatures.std() + 1e-6` |
| 124 | geodesic_regularization | `normalized = (all_curvatures - batch_mean) / batch_std` |
| 125 | geodesic_regularization | `curvature_norms = normalized.pow(2).mean()` |
| 129 | geodesic_regularization | `curvature_norms = all_curvatures.pow(2).mean()` |
| 134 | dynamic_loss_balancing | `def dynamic_loss_balancing(loss_components: list, target_ratio: float = 1.0) -> list:` |
| 158 | dynamic_loss_balancing | `grad_norm = sum(g.norm() for g in grad if g is not None)` |
| 169 | dynamic_loss_balancing | `mean_norm = grad_norms.mean()` |
| 174 | dynamic_loss_balancing | `scale = target_ratio * mean_norm / norm` |

#### Fórmulas Listas para Usar (Python)
```python
# geodesic_regularization (L35)
mode: str = 'structural') -> torch.Tensor:
# geodesic_regularization (L66)
mean_gamma = fused_tensor.mean() + GEODESIC_FUSED_SCALE
# geodesic_regularization (L82)
curvature_norms = all_curvatures.pow(2).mean()
# geodesic_regularization (L97)
curvature_diff = all_curvatures[1:] - all_curvatures[:-1]
# geodesic_regularization (L98)
curvature_change_norm = curvature_diff.pow(2).mean()
# geodesic_regularization (L102)
magnitude_norm = all_curvatures.pow(2).mean()
# geodesic_regularization (L106)
curvature_norms = 0.85 * curvature_change_norm + 0.15 * magnitude_norm
# geodesic_regularization (L111)
curvature_var = all_curvatures.var(dim=0)  # [dim]
# geodesic_regularization (L112)
curvature_norms = curvature_var.mean()
# geodesic_regularization (L116)
curvature_norms = all_curvatures.pow(2).mean()
# geodesic_regularization (L120)
batch_mean = all_curvatures.mean()
# geodesic_regularization (L121)
batch_std = all_curvatures.std() + 1e-6
# geodesic_regularization (L124)
normalized = (all_curvatures - batch_mean) / batch_std
# geodesic_regularization (L125)
curvature_norms = normalized.pow(2).mean()
# geodesic_regularization (L129)
curvature_norms = all_curvatures.pow(2).mean()
# dynamic_loss_balancing (L134)
def dynamic_loss_balancing(loss_components: list, target_ratio: float = 1.0) -> list:
# dynamic_loss_balancing (L158)
grad_norm = sum(g.norm() for g in grad if g is not None)
# dynamic_loss_balancing (L169)
mean_norm = grad_norms.mean()
# dynamic_loss_balancing (L174)
scale = target_ratio * mean_norm / norm
```

### gfn\losses\hamiltonian.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 29 | hamiltonian_loss | `def hamiltonian_loss(velocities: list, states: list = None, metric_fn=None, lambda_h: float = 0.01, forces: list = None, mode: str = 'adaptive') -> torch.Tensor:` |
| 66 | hamiltonian_loss | `e = 0.5 * torch.sum(g * v.pow(2), dim=-1)` |
| 68 | hamiltonian_loss | `e = 0.5 * v.pow(2).sum(dim=-1)` |
| 77 | hamiltonian_loss | `dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)` |
| 79 | hamiltonian_loss | `f_norm = forces[i].pow(2).sum(dim=-1)` |
| 81 | hamiltonian_loss | `force_threshold = 1e-4` |
| 93 | hamiltonian_loss | `e_next = energies[i + 1]` |
| 96 | hamiltonian_loss | `diff = torch.abs(e_curr - e_next) / (torch.abs(e_curr) + EPSILON_SMOOTH)` |
| 104 | hamiltonian_loss | `e_next = energies[i + 1]` |
| 107 | hamiltonian_loss | `denom = torch.abs(e_curr) + EPSILON_SMOOTH` |
| 108 | hamiltonian_loss | `rel_change = torch.abs(e_next - e_curr) / denom` |
| 111 | hamiltonian_loss | `diff = torch.sqrt(rel_change.pow(2) + EPSILON_SMOOTH)` |
| 117 | hamiltonian_loss | `dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)` |

#### Fórmulas Listas para Usar (Python)
```python
# hamiltonian_loss (L29)
def hamiltonian_loss(velocities: list, states: list = None, metric_fn=None, lambda_h: float = 0.01, forces: list = None, mode: str = 'adaptive') -> torch.Tensor:
# hamiltonian_loss (L66)
e = 0.5 * torch.sum(g * v.pow(2), dim=-1)
# hamiltonian_loss (L68)
e = 0.5 * v.pow(2).sum(dim=-1)
# hamiltonian_loss (L77)
dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)
# hamiltonian_loss (L79)
f_norm = forces[i].pow(2).sum(dim=-1)
# hamiltonian_loss (L81)
force_threshold = 1e-4
# hamiltonian_loss (L93)
e_next = energies[i + 1]
# hamiltonian_loss (L96)
diff = torch.abs(e_curr - e_next) / (torch.abs(e_curr) + EPSILON_SMOOTH)
# hamiltonian_loss (L104)
e_next = energies[i + 1]
# hamiltonian_loss (L107)
denom = torch.abs(e_curr) + EPSILON_SMOOTH
# hamiltonian_loss (L108)
rel_change = torch.abs(e_next - e_curr) / denom
# hamiltonian_loss (L111)
diff = torch.sqrt(rel_change.pow(2) + EPSILON_SMOOTH)
# hamiltonian_loss (L117)
dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)
```

### gfn\losses\kinetic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 11 | kinetic_energy_penalty | `def kinetic_energy_penalty(velocities: list, lambda_k: float = 0.001) -> torch.Tensor:` |
| 26 | kinetic_energy_penalty | `v_norms = torch.stack([v.pow(2).sum(dim=-1).mean() for v in velocities])` |

#### Fórmulas Listas para Usar (Python)
```python
# kinetic_energy_penalty (L11)
def kinetic_energy_penalty(velocities: list, lambda_k: float = 0.001) -> torch.Tensor:
# kinetic_energy_penalty (L26)
v_norms = torch.stack([v.pow(2).sum(dim=-1).mean() for v in velocities])
```

### gfn\losses\noether.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 11 | noether_loss | `def noether_loss(christoffel_outputs: list, isomeric_groups: list = None, lambda_n: float = 0.01) -> torch.Tensor:` |
| 45 | noether_loss | `total_diff = total_diff + torch.mean((ref_out - target_out).pow(2))` |

#### Fórmulas Listas para Usar (Python)
```python
# noether_loss (L11)
def noether_loss(christoffel_outputs: list, isomeric_groups: list = None, lambda_n: float = 0.01) -> torch.Tensor:
# noether_loss (L45)
total_diff = total_diff + torch.mean((ref_out - target_out).pow(2))
```

### gfn\losses\toroidal.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 14 | Global | `diff = min(\|x1 - x2\|, 2π - \|x1 - x2\|)` |
| 31 | Global | `diff = min(\|x1 - x2\|, 2π - \|x1 - x2\|)` |
| 36 | Global | `>>> x_pred = torch.tensor([[0.1, 3.1], [2.0, 1.0]])  # On torus [0, 2π) >>> x_target = torch.tensor([[0.2, 3.0], [2.1, 1.1]]) >>> loss = loss_fn(x_pred, x_target) """ import torch import torch.nn as nn from ..geometry.boundaries import toroidal_dist_python def toroidal_distance_loss(x_pred, x_target): """ Toroidal Distance Loss. Computes distance on 3D toroidal manifold (FLAT torus). AUDIT NOTE: This is distance on a flat torus, not the learned manifold. Args: x_pred: Predicted positions [batch, dim] x_target: Target positions [batch, dim] Returns: Toroidal distance loss scalar """ dist = toroidal_dist_python(x_pred, x_target) return dist.pow(2).mean() class ToroidalDistanceLoss(nn.Module): """nn.Module wrapper for toroidal_distance_loss.""" def __init__(self): super().__init__() def forward(self, x_pred, x_target): return toroidal_distance_loss(x_pred, x_target)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L14)
diff = min(|x1 - x2|, 2π - |x1 - x2|)
# Global (L31)
diff = min(|x1 - x2|, 2π - |x1 - x2|)
# Global (L36)
>>> x_pred = torch.tensor([[0.1, 3.1], [2.0, 1.0]])  # On torus [0, 2π) >>> x_target = torch.tensor([[0.2, 3.0], [2.1, 1.1]]) >>> loss = loss_fn(x_pred, x_target) """ import torch import torch.nn as nn from ..geometry.boundaries import toroidal_dist_python def toroidal_distance_loss(x_pred, x_target): """ Toroidal Distance Loss. Computes distance on 3D toroidal manifold (FLAT torus). AUDIT NOTE: This is distance on a flat torus, not the learned manifold. Args: x_pred: Predicted positions [batch, dim] x_target: Target positions [batch, dim] Returns: Toroidal distance loss scalar """ dist = toroidal_dist_python(x_pred, x_target) return dist.pow(2).mean() class ToroidalDistanceLoss(nn.Module): """nn.Module wrapper for toroidal_distance_loss.""" def __init__(self): super().__init__() def forward(self, x_pred, x_target): return toroidal_distance_loss(x_pred, x_target)
```

### gfn\model\fusion.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 30 | can_fuse | `def can_fuse(self, collect_christ: bool = False) -> bool:` |
| 155 | prepare_parameters | `unwrap_depth += 1` |
| 171 | prepare_parameters | `U_list.append(torch.zeros(self.model.dim // self.model.heads, 1, device=device))` |
| 172 | prepare_parameters | `W_list.append(torch.zeros(self.model.dim // self.model.heads, 1, device=device))` |
| 189 | prepare_parameters | `W_forget_list.append(torch.zeros(self.model.dim//self.model.heads, h_dim, device=device))` |
| 190 | prepare_parameters | `b_forget_list.append(torch.zeros(self.model.dim//self.model.heads, device=device))` |
| 191 | prepare_parameters | `W_input_list.append(torch.zeros(self.model.dim//self.model.heads, h_dim, device=device))` |
| 205 | prepare_parameters | `p_dim = 2 * (self.model.dim // self.model.heads) if is_torus else (self.model.dim // self.model.heads)` |
| 317 | execute_fused_forward | `hyst_enabled: bool = False) -> Optional[Tuple]:` |
| 358 | execute_fused_forward | `result = toroidal_leapfrog_fused_autograd( x=x, v=v, f=forces * mask, R=params['major_R'], r=params['minor_r'], dt=params['base_dt'], batch=x.shape[0], seq_len=forces.shape[1], dim=x.shape[1], hysteresis_state=hysteresis_state )` |
| 386 | execute_fused_forward | `result = launch_toroidal_leapfrog_fused( x=x, v=v, f=forces * mask, R=params['major_R'], r=params['minor_r'], dt=params['base_dt'], batch=x.shape[0], seq_len=forces.shape[1], dim=x.shape[1], hysteresis_state=hysteresis_state, W_forget=W_f_head0, b_forget=b_f_head0 )` |
| 417 | execute_fused_forward | `forget_rates = torch.sigmoid(f_layer.christoffels[0].forget_gate.bias.mean())` |
| 422 | execute_fused_forward | `res = recurrent_manifold_fused( x=x, v=v, f=forces * mask, U_stack=params['U_stack'], W_stack=params['W_stack'], dt=params['base_dt'], dt_scales=dt_scales, forget_rates=forget_rates, num_heads=self.model.heads, plasticity=params['plasticity'], sing_thresh=params['sing_thresh'], sing_strength=params['sing_strength'], mix_x=params['mix_x'], mix_v=params['mix_v'], Wf=params['W_f_stack'], Wi=params['W_i_stack'], bf=params['b_f_stack'], Wp=params['W_p_stack'], bp=params['b_p_stack'], topology=params['topology_id'], R=params['major_R'], r=params['minor_r'], mix_x_bias=params['mix_x_bias'], mix_v_bias=params['mix_v_bias'], norm_x_weight=params['norm_x_weight'], norm_x_bias=params['norm_x_bias'], norm_v_weight=params['norm_v_weight'], norm_v_bias=params['norm_v_bias'], gate_W1=params['gate_W1'], gate_b1=params['gate_b1'], gate_W2=params['gate_W2'], gate_b2=params['gate_b2'], integrator_type=1 if self.model.integrator_type == 'leapfrog' else 0, hysteresis_state=hysteresis_state, hyst_update_w=hyst_update_w, hyst_update_b=hyst_update_b, hyst_readout_w=hyst_readout_w, hyst_readout_b=hyst_readout_b, hyst_decay=hyst_decay, hyst_enabled=hyst_enabled, thermo_alpha=params.get('thermo_alpha', 0.0), thermo_temp=params.get('thermo_temp', 1.0) )` |

#### Fórmulas Listas para Usar (Python)
```python
# can_fuse (L30)
def can_fuse(self, collect_christ: bool = False) -> bool:
# prepare_parameters (L155)
unwrap_depth += 1
# prepare_parameters (L171)
U_list.append(torch.zeros(self.model.dim // self.model.heads, 1, device=device))
# prepare_parameters (L172)
W_list.append(torch.zeros(self.model.dim // self.model.heads, 1, device=device))
# prepare_parameters (L189)
W_forget_list.append(torch.zeros(self.model.dim//self.model.heads, h_dim, device=device))
# prepare_parameters (L190)
b_forget_list.append(torch.zeros(self.model.dim//self.model.heads, device=device))
# prepare_parameters (L191)
W_input_list.append(torch.zeros(self.model.dim//self.model.heads, h_dim, device=device))
# prepare_parameters (L205)
p_dim = 2 * (self.model.dim // self.model.heads) if is_torus else (self.model.dim // self.model.heads)
# execute_fused_forward (L317)
hyst_enabled: bool = False) -> Optional[Tuple]:
# execute_fused_forward (L358)
result = toroidal_leapfrog_fused_autograd( x=x, v=v, f=forces * mask, R=params['major_R'], r=params['minor_r'], dt=params['base_dt'], batch=x.shape[0], seq_len=forces.shape[1], dim=x.shape[1], hysteresis_state=hysteresis_state )
# execute_fused_forward (L386)
result = launch_toroidal_leapfrog_fused( x=x, v=v, f=forces * mask, R=params['major_R'], r=params['minor_r'], dt=params['base_dt'], batch=x.shape[0], seq_len=forces.shape[1], dim=x.shape[1], hysteresis_state=hysteresis_state, W_forget=W_f_head0, b_forget=b_f_head0 )
# execute_fused_forward (L417)
forget_rates = torch.sigmoid(f_layer.christoffels[0].forget_gate.bias.mean())
# execute_fused_forward (L422)
res = recurrent_manifold_fused( x=x, v=v, f=forces * mask, U_stack=params['U_stack'], W_stack=params['W_stack'], dt=params['base_dt'], dt_scales=dt_scales, forget_rates=forget_rates, num_heads=self.model.heads, plasticity=params['plasticity'], sing_thresh=params['sing_thresh'], sing_strength=params['sing_strength'], mix_x=params['mix_x'], mix_v=params['mix_v'], Wf=params['W_f_stack'], Wi=params['W_i_stack'], bf=params['b_f_stack'], Wp=params['W_p_stack'], bp=params['b_p_stack'], topology=params['topology_id'], R=params['major_R'], r=params['minor_r'], mix_x_bias=params['mix_x_bias'], mix_v_bias=params['mix_v_bias'], norm_x_weight=params['norm_x_weight'], norm_x_bias=params['norm_x_bias'], norm_v_weight=params['norm_v_weight'], norm_v_bias=params['norm_v_bias'], gate_W1=params['gate_W1'], gate_b1=params['gate_b1'], gate_W2=params['gate_W2'], gate_b2=params['gate_b2'], integrator_type=1 if self.model.integrator_type == 'leapfrog' else 0, hysteresis_state=hysteresis_state, hyst_update_w=hyst_update_w, hyst_update_b=hyst_update_b, hyst_readout_w=hyst_readout_w, hyst_readout_b=hyst_readout_b, hyst_decay=hyst_decay, hyst_enabled=hyst_enabled, thermo_alpha=params.get('thermo_alpha', 0.0), thermo_temp=params.get('thermo_temp', 1.0) )
```

### gfn\model\state.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 47 | from_parameters | `x = x0.expand(batch_size, -1)` |
| 48 | from_parameters | `v = v0.expand(batch_size, -1)` |

#### Fórmulas Listas para Usar (Python)
```python
# from_parameters (L47)
x = x0.expand(batch_size, -1)
# from_parameters (L48)
v = v0.expand(batch_size, -1)
```

### gfn\noise\curiosity.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 18 | CuriosityNoise | `v_new = v + lambda_c * Confusion(F) * eps` |
| 44 | forward | `confusion = (force ** 2).mean(dim=-1, keepdim=True) # [Batch, 1]` |
| 48 | forward | `scale = self.base_std * (1.0 + self.sensitivity * confusion)` |
| 51 | forward | `noise = torch.randn_like(v) * scale` |

#### Fórmulas Listas para Usar (Python)
```python
# CuriosityNoise (L18)
v_new = v + lambda_c * Confusion(F) * eps
# forward (L44)
confusion = (force ** 2).mean(dim=-1, keepdim=True) # [Batch, 1]
# forward (L48)
scale = self.base_std * (1.0 + self.sensitivity * confusion)
# forward (L51)
noise = torch.randn_like(v) * scale
```

### gfn\noise\geometric.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 13 | GeometricNoise | `dv^i = ... + sigma * dW^i + (sigma^2 / 2) * Gamma^i_{jk} * g^{jk}` |
| 25 | forward | `def forward(self, x, v, christoffel_fn, dt=0.1, **kwargs):` |
| 44 | forward | `noise = sigma * torch.sqrt(torch.tensor(dt, device=device)) * torch.randn_like(v)` |
| 79 | forward | `U_sq_norm = (base_geo.U ** 2).sum(dim=0) # [rank]` |
| 80 | forward | `drift = torch.matmul(U_sq_norm, base_geo.W.t()) # [dim]` |
| 81 | forward | `drift = (sigma**2 / 2.0) * drift * dt` |

#### Fórmulas Listas para Usar (Python)
```python
# GeometricNoise (L13)
dv^i = ... + sigma * dW^i + (sigma^2 / 2) * Gamma^i_{jk} * g^{jk}
# forward (L25)
def forward(self, x, v, christoffel_fn, dt=0.1, **kwargs):
# forward (L44)
noise = sigma * torch.sqrt(torch.tensor(dt, device=device)) * torch.randn_like(v)
# forward (L79)
U_sq_norm = (base_geo.U ** 2).sum(dim=0) # [rank]
# forward (L80)
drift = torch.matmul(U_sq_norm, base_geo.W.t()) # [dim]
# forward (L81)
drift = (sigma**2 / 2.0) * drift * dt
```

### gfn\optimizers\manifold_sgd.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 27 | ManifoldSGD | `>>> optimizer = ManifoldSGD(model.parameters(), lr=1e-2)` |
| 30 | ManifoldSGD | `>>> optimizer = ManifoldSGD( ...     model.parameters(), ...     lr=1e-2, ...     weight_decay=0.01, ...     max_norm=5.0 ... )` |
| 38 | __init__ | `def __init__(self, params, lr=1e-2, weight_decay=0.0, max_norm=10.0):` |
| 72 | step | `p.data.add_(grad, alpha=-lr)` |
| 75 | step | `norm = p.data.norm()` |

#### Fórmulas Listas para Usar (Python)
```python
# ManifoldSGD (L27)
>>> optimizer = ManifoldSGD(model.parameters(), lr=1e-2)
# ManifoldSGD (L30)
>>> optimizer = ManifoldSGD( ...     model.parameters(), ...     lr=1e-2, ...     weight_decay=0.01, ...     max_norm=5.0 ... )
# __init__ (L38)
def __init__(self, params, lr=1e-2, weight_decay=0.0, max_norm=10.0):
# step (L72)
p.data.add_(grad, alpha=-lr)
# step (L75)
norm = p.data.norm()
```

### gfn\optimizers\riemannian_adam.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 40 | RiemannianAdam | `Instead of Euclidean gradient descent (W = W - lr * grad), this optimizer` |
| 44 | RiemannianAdam | `W_new = Retract(W_old, -lr * corrected_grad)` |
| 67 | RiemannianAdam | `>>> optimizer = RiemannianAdam(model.parameters(), lr=1e-3)` |
| 70 | RiemannianAdam | `>>> optimizer = RiemannianAdam( ...     model.parameters(), ...     lr=1e-3, ...     retraction='normalize', ...     max_norm=10.0 ... )` |
| 78 | RiemannianAdam | `>>> optimizer = RiemannianAdam( ...     model.parameters(), ...     lr=1e-3, ...     retraction='torus', ...     topology=1 ... )` |
| 86 | __init__ | `def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, retraction='normalize', max_norm=10.0, topology=0):` |
| 102 | _project_tangent | `norm_sq = torch.sum(p * p, dim=-1, keepdim=True) + 1e-8` |
| 103 | _project_tangent | `projection = torch.sum(grad * p, dim=-1, keepdim=True) / norm_sq` |
| 135 | _vector_transport | `delta_x = x_new - x_old` |
| 138 | _vector_transport | `delta_x = torch.atan2(torch.sin(delta_x), torch.cos(delta_x))` |
| 147 | _vector_transport | `norm = x_new.norm(dim=-1, keepdim=True) + 1e-8` |
| 148 | _vector_transport | `projection = torch.sum(vec * x_new, dim=-1, keepdim=True) / norm.pow(2)` |
| 204 | step | `state['step'] += 1` |
| 207 | step | `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)` |
| 208 | step | `exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)` |
| 211 | step | `bias_correction1 = 1 - beta1 ** state['step']` |
| 212 | step | `bias_correction2 = 1 - beta2 ** state['step']` |
| 214 | step | `step_size = lr / bias_correction1` |
| 215 | step | `denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)` |
| 217 | step | `step_direction = exp_avg / denom` |
| 223 | step | `p.data.add_(step_direction, alpha=-step_size)` |
| 226 | step | `p.data.add_(step_direction, alpha=-step_size)` |
| 229 | step | `norm = p.data.norm()` |
| 238 | step | `p.data.add_(step_direction, alpha=-step_size)` |
| 248 | step | `V = -step_direction * step_size` |
| 252 | step | `retraction_matrix = torch.linalg.solve(I + V/2, I - V/2)` |
| 255 | step | `p.data.add_(step_direction, alpha=-step_size)` |
| 256 | step | `norm = p.data.norm()` |
| 262 | step | `p.data.add_(step_direction, alpha=-lr)` |

#### Fórmulas Listas para Usar (Python)
```python
# RiemannianAdam (L40)
Instead of Euclidean gradient descent (W = W - lr * grad), this optimizer
# RiemannianAdam (L44)
W_new = Retract(W_old, -lr * corrected_grad)
# RiemannianAdam (L67)
>>> optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# RiemannianAdam (L70)
>>> optimizer = RiemannianAdam( ...     model.parameters(), ...     lr=1e-3, ...     retraction='normalize', ...     max_norm=10.0 ... )
# RiemannianAdam (L78)
>>> optimizer = RiemannianAdam( ...     model.parameters(), ...     lr=1e-3, ...     retraction='torus', ...     topology=1 ... )
# __init__ (L86)
def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, retraction='normalize', max_norm=10.0, topology=0):
# _project_tangent (L102)
norm_sq = torch.sum(p * p, dim=-1, keepdim=True) + 1e-8
# _project_tangent (L103)
projection = torch.sum(grad * p, dim=-1, keepdim=True) / norm_sq
# _vector_transport (L135)
delta_x = x_new - x_old
# _vector_transport (L138)
delta_x = torch.atan2(torch.sin(delta_x), torch.cos(delta_x))
# _vector_transport (L147)
norm = x_new.norm(dim=-1, keepdim=True) + 1e-8
# _vector_transport (L148)
projection = torch.sum(vec * x_new, dim=-1, keepdim=True) / norm.pow(2)
# step (L204)
state['step'] += 1
# step (L207)
exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
# step (L208)
exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)
# step (L211)
bias_correction1 = 1 - beta1 ** state['step']
# step (L212)
bias_correction2 = 1 - beta2 ** state['step']
# step (L214)
step_size = lr / bias_correction1
# step (L215)
denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
# step (L217)
step_direction = exp_avg / denom
# step (L223)
p.data.add_(step_direction, alpha=-step_size)
# step (L226)
p.data.add_(step_direction, alpha=-step_size)
# step (L229)
norm = p.data.norm()
# step (L238)
p.data.add_(step_direction, alpha=-step_size)
# step (L248)
V = -step_direction * step_size
# step (L252)
retraction_matrix = torch.linalg.solve(I + V/2, I - V/2)
# step (L255)
p.data.add_(step_direction, alpha=-step_size)
# step (L256)
norm = p.data.norm()
# step (L262)
p.data.add_(step_direction, alpha=-lr)
```

### gfn\readouts\implicit.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 46 | __init__ | `in_dim = dim * 2 if self.is_torus else dim` |
| 78 | forward | `x_emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)` |
| 83 | forward | `logits = self.mlp(x_emb) * READOUT_GAIN  # Sharpen logits for better BCE loss` |
| 91 | update_step | `self.training_step += 1` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L46)
in_dim = dim * 2 if self.is_torus else dim
# forward (L78)
x_emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# forward (L83)
logits = self.mlp(x_emb) * READOUT_GAIN  # Sharpen logits for better BCE loss
# update_step (L91)
self.training_step += 1
```

### gfn\utils\scan.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 14 | parallel_scan | `"""Compute associative parallel scan: y_t = a_t * y_{t-1} + x_t.` |
| 22 | parallel_scan | `y_t = a_t * y_{t-1} + x_t  for t > 0` |
| 44 | parallel_scan | `>>> a = torch.ones(2, 10, 64) * 0.9` |
| 50 | parallel_scan | `>>> # Each y[t] = 0.9 * y[t-1] + x[t]` |
| 52 | parallel_scan | `>>> # y[1] = 0.9 * x[0] + x[1]` |
| 53 | parallel_scan | `>>> # y[2] = 0.9 * (0.9 * x[0] + x[1]) + x[2] = 0.81*x[0] + 0.9*x[1] + x[2]` |
| 62 | parallel_scan | `- Sequential (L < 32): ~0.1ms for L=16, D=64` |
| 63 | parallel_scan | `- Parallel (L >= 32): ~0.5ms for L=128, D=64` |
| 64 | parallel_scan | `- CUDA (if available): ~0.2ms for L=128, D=64` |
| 102 | parallel_scan | `h = a[:, t] * h + x[:, t]` |
| 114 | parallel_scan | `steps = int(math.ceil(math.log2(L)))` |
| 137 | parallel_scan | `new_a = curr_a * prev_a` |
| 138 | parallel_scan | `new_x = curr_a * prev_x + curr_x` |

#### Fórmulas Listas para Usar (Python)
```python
# parallel_scan (L14)
"""Compute associative parallel scan: y_t = a_t * y_{t-1} + x_t.
# parallel_scan (L22)
y_t = a_t * y_{t-1} + x_t  for t > 0
# parallel_scan (L44)
>>> a = torch.ones(2, 10, 64) * 0.9
# parallel_scan (L50)
>>> # Each y[t] = 0.9 * y[t-1] + x[t]
# parallel_scan (L52)
>>> # y[1] = 0.9 * x[0] + x[1]
# parallel_scan (L53)
>>> # y[2] = 0.9 * (0.9 * x[0] + x[1]) + x[2] = 0.81*x[0] + 0.9*x[1] + x[2]
# parallel_scan (L62)
- Sequential (L < 32): ~0.1ms for L=16, D=64
# parallel_scan (L63)
- Parallel (L >= 32): ~0.5ms for L=128, D=64
# parallel_scan (L64)
- CUDA (if available): ~0.2ms for L=128, D=64
# parallel_scan (L102)
h = a[:, t] * h + x[:, t]
# parallel_scan (L114)
steps = int(math.ceil(math.log2(L)))
# parallel_scan (L137)
new_a = curr_a * prev_a
# parallel_scan (L138)
new_x = curr_a * prev_x + curr_x
```

### gfn\utils\visualization.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 6 | visualize_gating | `def visualize_gating(model_path, test_str="88+11="):` |

#### Fórmulas Listas para Usar (Python)
```python
# visualize_gating (L6)
def visualize_gating(model_path, test_str="88+11="):
```

### inventory_script.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 15 | Global | `_DEFAULT_IGNORED_DIRS = { ".git", ".hg", ".svn", ".tox", ".pytest_cache", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", "_organized", "organizer_logs", "gfn.egg-info", }` |
| 106 | Global | `@dataclass(frozen=True)` |
| 121 | _setup_logger | `log_path = log_dir / f"organizer_{ts}.log"` |
| 122 | _setup_logger | `report_path = log_dir / f"organizer_{ts}.jsonl"` |
| 129 | _setup_logger | `fh = logging.FileHandler(log_path, encoding="utf-8")` |
| 144 | _write_report_event | `with open(report_path, "a", encoding="utf-8") as f:` |
| 145 | _write_report_event | `f.write(json.dumps(event, ensure_ascii=False) + "\n")` |
| 155 | _read_text_head | `def _read_text_head(path: Path, max_bytes: int = 24_576) -> str:` |
| 164 | _read_text_head | `return data.decode("latin-1", errors="ignore")` |
| 173 | _slugify_stem | `stem = re.sub(r"[\s\-]+", "_", stem)` |
| 174 | _slugify_stem | `stem = re.sub(r"[^A-Za-z0-9_\.]+", "_", stem)` |
| 175 | _slugify_stem | `stem = re.sub(r"_+", "_", stem).strip("_.")` |
| 185 | _shorten_filename | `digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]` |
| 186 | _shorten_filename | `room = max(1, max_len - len(ext) - 11)` |
| 198 | _safe_path_length | `def _safe_path_length(target: Path, max_total: int = 240) -> Path:` |
| 204 | _safe_path_length | `digest = hashlib.sha1(target_str.encode("utf-8")).hexdigest()[:10]` |
| 206 | _safe_path_length | `room = max(1, max_total - len(parent_str) - 1 - len(ext) - 11)` |
| 320 | _build_destination | `dst_dir = dest_root.joinpath(*parts)` |
| 321 | _build_destination | `dst_path = dst_dir / normalized_name` |
| 356 | _hash_head | `def _hash_head(path: Path, head_bytes: int = 65_536) -> str:` |
| 363 | _hash_full | `def _hash_full(path: Path, chunk: int = 1_048_576) -> str:` |
| 402 | _next_available_path | `digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]` |
| 419 | organize | `log_dir = root / "organizer_logs"` |
| 453 | organize | `duplicates_dir = dest_root / "00_duplicates"` |
| 475 | organize | `dup_target = _next_available_path(_safe_path_length(duplicates_dir / normalized_name))` |
| 547 | _extract_formulas_python | `lines = file_path.read_text(encoding="utf-8").splitlines(True)` |
| 550 | _extract_formulas_python | `lines = file_path.read_text(encoding="latin-1", errors="ignore").splitlines(True)` |
| 555 | _extract_formulas_python | `context_pattern = re.compile(r"^\s*(class\|def)\s+([a-zA-Z0-9_]+)")` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L15)
_DEFAULT_IGNORED_DIRS = { ".git", ".hg", ".svn", ".tox", ".pytest_cache", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", "_organized", "organizer_logs", "gfn.egg-info", }
# Global (L106)
@dataclass(frozen=True)
# _setup_logger (L121)
log_path = log_dir / f"organizer_{ts}.log"
# _setup_logger (L122)
report_path = log_dir / f"organizer_{ts}.jsonl"
# _setup_logger (L129)
fh = logging.FileHandler(log_path, encoding="utf-8")
# _write_report_event (L144)
with open(report_path, "a", encoding="utf-8") as f:
# _write_report_event (L145)
f.write(json.dumps(event, ensure_ascii=False) + "\n")
# _read_text_head (L155)
def _read_text_head(path: Path, max_bytes: int = 24_576) -> str:
# _read_text_head (L164)
return data.decode("latin-1", errors="ignore")
# _slugify_stem (L173)
stem = re.sub(r"[\s\-]+", "_", stem)
# _slugify_stem (L174)
stem = re.sub(r"[^A-Za-z0-9_\.]+", "_", stem)
# _slugify_stem (L175)
stem = re.sub(r"_+", "_", stem).strip("_.")
# _shorten_filename (L185)
digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
# _shorten_filename (L186)
room = max(1, max_len - len(ext) - 11)
# _safe_path_length (L198)
def _safe_path_length(target: Path, max_total: int = 240) -> Path:
# _safe_path_length (L204)
digest = hashlib.sha1(target_str.encode("utf-8")).hexdigest()[:10]
# _safe_path_length (L206)
room = max(1, max_total - len(parent_str) - 1 - len(ext) - 11)
# _build_destination (L320)
dst_dir = dest_root.joinpath(*parts)
# _build_destination (L321)
dst_path = dst_dir / normalized_name
# _hash_head (L356)
def _hash_head(path: Path, head_bytes: int = 65_536) -> str:
# _hash_full (L363)
def _hash_full(path: Path, chunk: int = 1_048_576) -> str:
# _next_available_path (L402)
digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
# organize (L419)
log_dir = root / "organizer_logs"
# organize (L453)
duplicates_dir = dest_root / "00_duplicates"
# organize (L475)
dup_target = _next_available_path(_safe_path_length(duplicates_dir / normalized_name))
# _extract_formulas_python (L547)
lines = file_path.read_text(encoding="utf-8").splitlines(True)
# _extract_formulas_python (L550)
lines = file_path.read_text(encoding="latin-1", errors="ignore").splitlines(True)
# _extract_formulas_python (L555)
context_pattern = re.compile(r"^\s*(class|def)\s+([a-zA-Z0-9_]+)")
```

### scripts\inspect_checkpoint.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 33 | inspect_checkpoint | `depth = max(layer_indices) + 1` |

#### Fórmulas Listas para Usar (Python)
```python
# inspect_checkpoint (L33)
depth = max(layer_indices) + 1
```

### scripts\reset_and_check_initial_loss.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 24 | main | `parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")` |
| 58 | main | `y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)` |
| 61 | main | `baseline = (torch.abs(torch.atan2(torch.sin(-y_angle), torch.cos(-y_angle)))**2).mean().item()` |
| 64 | main | `ok = abs(loss - 2.5) <= 0.25` |

#### Fórmulas Listas para Usar (Python)
```python
# main (L24)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
# main (L58)
y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
# main (L61)
baseline = (torch.abs(torch.atan2(torch.sin(-y_angle), torch.cos(-y_angle)))**2).mean().item()
# main (L64)
ok = abs(loss - 2.5) <= 0.25
```

### scripts\train.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 98 | run_demo | `test_cases = ["42+9=", "131-31=", "50*5=", "999+1=", "123*10="]` |
| 108 | run_demo | `curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)` |
| 113 | run_demo | `next_token = torch.argmax(logits[:, -1, :], dim=-1)` |
| 120 | run_demo | `result = dataset.decode(generated).split('=')[-1]` |
| 136 | train | `param_count = sum(p.numel() for p in model.parameters()) / 1e6` |
| 143 | train | `optimizer = RiemannianAdam( model.parameters(), lr=train_params.get('learning_rate', 3e-4), weight_decay=train_params.get('weight_decay', 0.01), retraction='normalize', max_norm=10.0 )` |
| 182 | train | `target_tokens = torch.roll(inputs, -1, dims=1)` |
| 186 | train | `mask = 2**torch.arange(coord_dim).to(device)` |
| 187 | train | `bits = (target_tokens.unsqueeze(-1) & mask) > 0` |
| 188 | train | `target_coords = bits.float() * 2 - 1` |
| 205 | train | `pred_positions = x_seq[:, :-1, :]  # [batch, seq-1, dim] aligned for next-token prediction` |
| 209 | train | `half_pi = PI * 0.5` |
| 210 | train | `angle_targets = torch.where( target_coords > 0, torch.full_like(target_coords, half_pi), torch.full_like(target_coords, -half_pi) )` |
| 215 | train | `targ_valid = angle_targets[:, :-1, :]` |
| 220 | train | `valid_mask = (target_tokens[:, :-1] != PAD) & (target_tokens[:, :-1] != EOS)` |
| 221 | train | `mask_exp = valid_mask.unsqueeze(-1).float().expand_as(targ_valid)` |
| 227 | train | `loss_torus = (dist.pow(2) * mask_exp).sum() / torch.clamp(mask_exp.sum() * targ_valid.size(-1), min=1.0)` |
| 234 | train | `loss_phy += geodesic_regularization( christoffels, velocities=None, lambda_g=lambda_g, mode='structural' )` |
| 242 | train | `loss_phy += curiosity_loss([v_final], lambda_c)` |
| 244 | train | `loss = total_loss + loss_phy` |
| 250 | train | `epoch_loss += loss.item()` |
| 253 | train | `dt = time.time() - epoch_t0` |
| 254 | train | `speed = (batch_idx * inputs.shape[0]) / max(dt, 0.01)` |
| 257 | train | `avg_loss = epoch_loss / train_params.get('steps_per_epoch', 1000)` |
| 271 | main | `parser.add_argument('--model', type=str, required=True, help="Path to model config YAML")` |
| 272 | main | `parser.add_argument('--training', type=str, required=True, help="Path to training config YAML")` |
| 273 | main | `parser.add_argument('--hardware', type=str, required=True, help="Path to hardware config YAML")` |
| 274 | main | `parser.add_argument('--reset-optimizer', action='store_true', help="Reset optimizer state when resuming")` |

#### Fórmulas Listas para Usar (Python)
```python
# run_demo (L98)
test_cases = ["42+9=", "131-31=", "50*5=", "999+1=", "123*10="]
# run_demo (L108)
curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
# run_demo (L113)
next_token = torch.argmax(logits[:, -1, :], dim=-1)
# run_demo (L120)
result = dataset.decode(generated).split('=')[-1]
# train (L136)
param_count = sum(p.numel() for p in model.parameters()) / 1e6
# train (L143)
optimizer = RiemannianAdam( model.parameters(), lr=train_params.get('learning_rate', 3e-4), weight_decay=train_params.get('weight_decay', 0.01), retraction='normalize', max_norm=10.0 )
# train (L182)
target_tokens = torch.roll(inputs, -1, dims=1)
# train (L186)
mask = 2**torch.arange(coord_dim).to(device)
# train (L187)
bits = (target_tokens.unsqueeze(-1) & mask) > 0
# train (L188)
target_coords = bits.float() * 2 - 1
# train (L205)
pred_positions = x_seq[:, :-1, :]  # [batch, seq-1, dim] aligned for next-token prediction
# train (L209)
half_pi = PI * 0.5
# train (L210)
angle_targets = torch.where( target_coords > 0, torch.full_like(target_coords, half_pi), torch.full_like(target_coords, -half_pi) )
# train (L215)
targ_valid = angle_targets[:, :-1, :]
# train (L220)
valid_mask = (target_tokens[:, :-1] != PAD) & (target_tokens[:, :-1] != EOS)
# train (L221)
mask_exp = valid_mask.unsqueeze(-1).float().expand_as(targ_valid)
# train (L227)
loss_torus = (dist.pow(2) * mask_exp).sum() / torch.clamp(mask_exp.sum() * targ_valid.size(-1), min=1.0)
# train (L234)
loss_phy += geodesic_regularization( christoffels, velocities=None, lambda_g=lambda_g, mode='structural' )
# train (L242)
loss_phy += curiosity_loss([v_final], lambda_c)
# train (L244)
loss = total_loss + loss_phy
# train (L250)
epoch_loss += loss.item()
# train (L253)
dt = time.time() - epoch_t0
# train (L254)
speed = (batch_idx * inputs.shape[0]) / max(dt, 0.01)
# train (L257)
avg_loss = epoch_loss / train_params.get('steps_per_epoch', 1000)
# main (L271)
parser.add_argument('--model', type=str, required=True, help="Path to model config YAML")
# main (L272)
parser.add_argument('--training', type=str, required=True, help="Path to training config YAML")
# main (L273)
parser.add_argument('--hardware', type=str, required=True, help="Path to hardware config YAML")
# main (L274)
parser.add_argument('--reset-optimizer', action='store_true', help="Reset optimizer state when resuming")
```

### scripts\verify_cuda_kernels.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 31 | verify_kernels | `v = torch.randn(batch_size, dim, device=device) * scale` |
| 32 | verify_kernels | `U = torch.randn(dim, rank, device=device) * scale` |
| 33 | verify_kernels | `W = torch.randn(dim, rank, device=device) * scale` |
| 43 | verify_kernels | `proj = torch.matmul(v_cpu, U_cpu)` |
| 44 | verify_kernels | `sq = proj * proj` |
| 45 | verify_kernels | `gamma_ref = torch.matmul(sq, W_cpu.t())` |
| 48 | verify_kernels | `gamma_ref_clamped = torch.clamp(gamma_ref, -5.0, 5.0)` |
| 50 | verify_kernels | `diff = (gamma_cuda.cpu() - gamma_ref_clamped).abs().max().item()` |
| 75 | verify_kernels | `gamma_ref_base = torch.clamp(gamma_ref, -5.0, 5.0) # Base clamped` |
| 79 | verify_kernels | `energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))` |
| 80 | verify_kernels | `plast_factor = 1.0 + plasticity * energy` |
| 84 | verify_kernels | `potential = torch.sigmoid(torch.matmul(x, V_w.t()))` |
| 86 | verify_kernels | `sing_factor = 1.0 + is_sing * (sing_strength - 1.0)` |
| 95 | verify_kernels | `gamma_active_ref = gamma_ref_base.cpu() * plast_factor.cpu() * sing_factor.cpu()` |
| 97 | verify_kernels | `diff_act = (gamma_cuda_p.cpu() - gamma_active_ref.cpu()).abs().max().item()` |
| 114 | verify_kernels | `x = torch.randn(batch_size, dim, device=device) * scale` |
| 115 | verify_kernels | `v = torch.randn(batch_size, dim, device=device) * scale` |
| 116 | verify_kernels | `f = torch.randn(batch_size, dim, device=device) * scale` |
| 124 | verify_kernels | `effective_dt = dt * dt_scale` |
| 132 | verify_kernels | `gamma_v = torch.matmul((torch.matmul(v_cpu, U_cpu)**2), W_cpu.t())` |
| 133 | verify_kernels | `gamma_v = gamma_v.clamp(-5, 5)` |
| 135 | verify_kernels | `v_half_ref = v_cpu + 0.5 * effective_dt * (f_cpu - gamma_v)` |
| 138 | verify_kernels | `x_new_ref = x_cpu + effective_dt * v_half_ref` |
| 141 | verify_kernels | `gamma_v_half = torch.matmul((torch.matmul(v_half_ref, U_cpu)**2), W_cpu.t())` |
| 142 | verify_kernels | `gamma_v_half = gamma_v_half.clamp(-5, 5)` |
| 144 | verify_kernels | `v_new_ref = v_half_ref + 0.5 * effective_dt * (f_cpu - gamma_v_half)` |
| 146 | verify_kernels | `diff_x = (x_new_cuda.cpu() - x_new_ref).abs().max().item()` |
| 147 | verify_kernels | `diff_v = (v_new_cuda.cpu() - v_new_ref).abs().max().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# verify_kernels (L31)
v = torch.randn(batch_size, dim, device=device) * scale
# verify_kernels (L32)
U = torch.randn(dim, rank, device=device) * scale
# verify_kernels (L33)
W = torch.randn(dim, rank, device=device) * scale
# verify_kernels (L43)
proj = torch.matmul(v_cpu, U_cpu)
# verify_kernels (L44)
sq = proj * proj
# verify_kernels (L45)
gamma_ref = torch.matmul(sq, W_cpu.t())
# verify_kernels (L48)
gamma_ref_clamped = torch.clamp(gamma_ref, -5.0, 5.0)
# verify_kernels (L50)
diff = (gamma_cuda.cpu() - gamma_ref_clamped).abs().max().item()
# verify_kernels (L75)
gamma_ref_base = torch.clamp(gamma_ref, -5.0, 5.0) # Base clamped
# verify_kernels (L79)
energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
# verify_kernels (L80)
plast_factor = 1.0 + plasticity * energy
# verify_kernels (L84)
potential = torch.sigmoid(torch.matmul(x, V_w.t()))
# verify_kernels (L86)
sing_factor = 1.0 + is_sing * (sing_strength - 1.0)
# verify_kernels (L95)
gamma_active_ref = gamma_ref_base.cpu() * plast_factor.cpu() * sing_factor.cpu()
# verify_kernels (L97)
diff_act = (gamma_cuda_p.cpu() - gamma_active_ref.cpu()).abs().max().item()
# verify_kernels (L114)
x = torch.randn(batch_size, dim, device=device) * scale
# verify_kernels (L115)
v = torch.randn(batch_size, dim, device=device) * scale
# verify_kernels (L116)
f = torch.randn(batch_size, dim, device=device) * scale
# verify_kernels (L124)
effective_dt = dt * dt_scale
# verify_kernels (L132)
gamma_v = torch.matmul((torch.matmul(v_cpu, U_cpu)**2), W_cpu.t())
# verify_kernels (L133)
gamma_v = gamma_v.clamp(-5, 5)
# verify_kernels (L135)
v_half_ref = v_cpu + 0.5 * effective_dt * (f_cpu - gamma_v)
# verify_kernels (L138)
x_new_ref = x_cpu + effective_dt * v_half_ref
# verify_kernels (L141)
gamma_v_half = torch.matmul((torch.matmul(v_half_ref, U_cpu)**2), W_cpu.t())
# verify_kernels (L142)
gamma_v_half = gamma_v_half.clamp(-5, 5)
# verify_kernels (L144)
v_new_ref = v_half_ref + 0.5 * effective_dt * (f_cpu - gamma_v_half)
# verify_kernels (L146)
diff_x = (x_new_cuda.cpu() - x_new_ref).abs().max().item()
# verify_kernels (L147)
diff_v = (v_new_cuda.cpu() - v_new_ref).abs().max().item()
```

### scripts\verify_cuda_parity.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 73 | test_cuda_parity | `diff_x = (x_fused_a - x_comp_a).abs().max().item()` |
| 74 | test_cuda_parity | `diff_v = (v_fused_a - v_comp_a).abs().max().item()` |
| 94 | test_cuda_parity | `diff_x_b = (x_fused_b - x_comp_b).abs().max().item()` |
| 95 | test_cuda_parity | `diff_v_b = (v_fused_b - v_comp_b).abs().max().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_cuda_parity (L73)
diff_x = (x_fused_a - x_comp_a).abs().max().item()
# test_cuda_parity (L74)
diff_v = (v_fused_a - v_comp_a).abs().max().item()
# test_cuda_parity (L94)
diff_x_b = (x_fused_b - x_comp_b).abs().max().item()
# test_cuda_parity (L95)
diff_v_b = (v_fused_b - v_comp_b).abs().max().item()
```

### test_toroidal_autograd.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 33 | test_toroidal_autograd | `v = torch.ones(B, D, device=device, requires_grad=True) * 10.0  # High velocity` |
| 38 | test_toroidal_autograd | `U_stack = torch.randn(num_layers * H * D, rank, device=device) * 0.01` |
| 39 | test_toroidal_autograd | `W_stack = torch.randn(num_layers * H * rank, D, device=device) * 0.01` |
| 92 | test_toroidal_autograd | `TWO_PI = 2 * math.pi` |
| 93 | test_toroidal_autograd | `if max_tor <= TWO_PI + 0.01 and min_tor >= -0.01:` |
| 102 | test_toroidal_autograd | `diff = torch.abs(x_seq_euc - x_seq_tor).mean().item()` |
| 112 | test_toroidal_autograd | `test_values = torch.tensor([0.0, math.pi, 2*math.pi, 3*math.pi, -math.pi, -2*math.pi], device=device)` |
| 113 | test_toroidal_autograd | `wrapped = torch.fmod(test_values, 2 * torch.pi)` |
| 114 | test_toroidal_autograd | `wrapped = torch.where(wrapped < 0, wrapped + 2 * torch.pi, wrapped)` |
| 117 | test_toroidal_autograd | `print("Expected in [0, 2π]:", (wrapped >= 0).all().item() and (wrapped <= 2*math.pi).all().item())` |

#### Fórmulas Listas para Usar (Python)
```python
# test_toroidal_autograd (L33)
v = torch.ones(B, D, device=device, requires_grad=True) * 10.0  # High velocity
# test_toroidal_autograd (L38)
U_stack = torch.randn(num_layers * H * D, rank, device=device) * 0.01
# test_toroidal_autograd (L39)
W_stack = torch.randn(num_layers * H * rank, D, device=device) * 0.01
# test_toroidal_autograd (L92)
TWO_PI = 2 * math.pi
# test_toroidal_autograd (L93)
if max_tor <= TWO_PI + 0.01 and min_tor >= -0.01:
# test_toroidal_autograd (L102)
diff = torch.abs(x_seq_euc - x_seq_tor).mean().item()
# test_toroidal_autograd (L112)
test_values = torch.tensor([0.0, math.pi, 2*math.pi, 3*math.pi, -math.pi, -2*math.pi], device=device)
# test_toroidal_autograd (L113)
wrapped = torch.fmod(test_values, 2 * torch.pi)
# test_toroidal_autograd (L114)
wrapped = torch.where(wrapped < 0, wrapped + 2 * torch.pi, wrapped)
# test_toroidal_autograd (L117)
print("Expected in [0, 2π]:", (wrapped >= 0).all().item() and (wrapped <= 2*math.pi).all().item())
```

### tests\architecture\conftest.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 22 | __init__ | `self.base_path = "D:/ASAS/projects/GFN/.data/metrics/architecture"` |
| 31 | finish | `self.metrics["duration_seconds"] = time.time() - self.start_time` |
| 51 | compute_hamiltonian | `energy = 0.5 * torch.sum(g * v.pow(2), dim=-1)` |
| 60 | estimate_convergence_order | `log_dts = np.log(dts)` |
| 61 | estimate_convergence_order | `log_errors = np.log(errors)` |
| 62 | estimate_convergence_order | `coeffs = np.polyfit(log_dts, log_errors, 1)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L22)
self.base_path = "D:/ASAS/projects/GFN/.data/metrics/architecture"
# finish (L31)
self.metrics["duration_seconds"] = time.time() - self.start_time
# compute_hamiltonian (L51)
energy = 0.5 * torch.sum(g * v.pow(2), dim=-1)
# estimate_convergence_order (L60)
log_dts = np.log(dts)
# estimate_convergence_order (L61)
log_errors = np.log(errors)
# estimate_convergence_order (L62)
coeffs = np.polyfit(log_dts, log_errors, 1)
```

### tests\architecture\test_architectural_valuation.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 30 | test_pareto_front_flops_vs_accuracy | `duration = (time.time() - start) / 5` |

#### Fórmulas Listas para Usar (Python)
```python
# test_pareto_front_flops_vs_accuracy (L30)
duration = (time.time() - start) / 5
```

### tests\architecture\test_cuda_adjoint.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 7 | get_numerical_grad | `def get_numerical_grad(model, inputs, param_name, eps=1e-3):` |
| 17 | get_numerical_grad | `flat_grad = grad_num.view(-1)` |
| 21 | get_numerical_grad | `param.data.view(-1)[i] = orig_data.view(-1)[i] + eps` |
| 23 | get_numerical_grad | `loss_p = logits_p.pow(2).sum()` |
| 26 | get_numerical_grad | `param.data.view(-1)[i] = orig_data.view(-1)[i] - eps` |
| 28 | get_numerical_grad | `loss_m = logits_m.pow(2).sum()` |
| 30 | get_numerical_grad | `flat_grad[i] = (loss_p - loss_m) / (2 * eps)` |
| 32 | get_numerical_grad | `param.data.view(-1)[i] = orig_data.view(-1)[i]` |
| 36 | get_numerical_grad | `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")` |
| 72 | test_cuda_adjoint_consistency | `loss = logits.pow(2).sum()` |
| 110 | resolve_param | `rel_err = torch.norm(analytical - numerical) / (torch.norm(numerical) + 1e-9)` |

#### Fórmulas Listas para Usar (Python)
```python
# get_numerical_grad (L7)
def get_numerical_grad(model, inputs, param_name, eps=1e-3):
# get_numerical_grad (L17)
flat_grad = grad_num.view(-1)
# get_numerical_grad (L21)
param.data.view(-1)[i] = orig_data.view(-1)[i] + eps
# get_numerical_grad (L23)
loss_p = logits_p.pow(2).sum()
# get_numerical_grad (L26)
param.data.view(-1)[i] = orig_data.view(-1)[i] - eps
# get_numerical_grad (L28)
loss_m = logits_m.pow(2).sum()
# get_numerical_grad (L30)
flat_grad[i] = (loss_p - loss_m) / (2 * eps)
# get_numerical_grad (L32)
param.data.view(-1)[i] = orig_data.view(-1)[i]
# get_numerical_grad (L36)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# test_cuda_adjoint_consistency (L72)
loss = logits.pow(2).sum()
# resolve_param (L110)
rel_err = torch.norm(analytical - numerical) / (torch.norm(numerical) + 1e-9)
```

### tests\architecture\test_differentiability.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 17 | test_toroidal_differentiability | `loss = gamma.pow(2).sum()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_toroidal_differentiability (L17)
loss = gamma.pow(2).sum()
```

### tests\architecture\test_geometries.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 31 | test_toroidal_metric_properties | `expected_min = (geom.R - geom.r)**2` |
| 32 | test_toroidal_metric_properties | `assert torch.allclose(g_pi[0, 1], torch.tensor(expected_min), atol=1e-3)` |
| 52 | test_christoffel_connection_symmetry | `metrics.log("gamma_norm_mean", torch.norm(gamma_v, dim=-1).mean())` |
| 64 | test_hyperbolic_curvature_scaling | `x = torch.ones(1, dim) * 0.1` |
| 65 | test_hyperbolic_curvature_scaling | `v1 = torch.ones(1, dim) * 0.2` |
| 66 | test_hyperbolic_curvature_scaling | `v2 = torch.ones(1, dim) * 0.4` |
| 71 | test_hyperbolic_curvature_scaling | `n1 = torch.norm(g1)` |
| 72 | test_hyperbolic_curvature_scaling | `n2 = torch.norm(g2)` |
| 75 | test_hyperbolic_curvature_scaling | `ratio = (n2 / n1).item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_toroidal_metric_properties (L31)
expected_min = (geom.R - geom.r)**2
# test_toroidal_metric_properties (L32)
assert torch.allclose(g_pi[0, 1], torch.tensor(expected_min), atol=1e-3)
# test_christoffel_connection_symmetry (L52)
metrics.log("gamma_norm_mean", torch.norm(gamma_v, dim=-1).mean())
# test_hyperbolic_curvature_scaling (L64)
x = torch.ones(1, dim) * 0.1
# test_hyperbolic_curvature_scaling (L65)
v1 = torch.ones(1, dim) * 0.2
# test_hyperbolic_curvature_scaling (L66)
v2 = torch.ones(1, dim) * 0.4
# test_hyperbolic_curvature_scaling (L71)
n1 = torch.norm(g1)
# test_hyperbolic_curvature_scaling (L72)
n2 = torch.norm(g2)
# test_hyperbolic_curvature_scaling (L75)
ratio = (n2 / n1).item()
```

### tests\architecture\test_integrators.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 33 | test_integrator_convergence_order | `x_gt, v_gt = gt_integrator(x0, v0, steps=int(T / dt_gt))` |
| 39 | test_integrator_convergence_order | `steps = int(T / dt_val)` |
| 42 | test_integrator_convergence_order | `err = torch.norm(x - x_gt).item()` |
| 67 | test_symplectic_phase_space_conservation | `energy_init = 0.5 * (v.pow(2).sum() + x.pow(2).sum())` |
| 68 | test_symplectic_phase_space_conservation | `energy_final = 0.5 * (v_next.pow(2).sum() + x_next.pow(2).sum())` |
| 70 | test_symplectic_phase_space_conservation | `drift = torch.abs(energy_final - energy_init).item()` |
| 90 | test_hamiltonian_long_term_stability | `e0 = 0.5 * v.pow(2).sum()` |
| 91 | test_hamiltonian_long_term_stability | `e_rk = 0.5 * v_rk.pow(2).sum()` |
| 92 | test_hamiltonian_long_term_stability | `e_vt = 0.5 * v_vt.pow(2).sum()` |
| 94 | test_hamiltonian_long_term_stability | `drift_rk = torch.abs(e_rk - e0).item()` |
| 95 | test_hamiltonian_long_term_stability | `drift_vt = torch.abs(e_vt - e0).item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_integrator_convergence_order (L33)
x_gt, v_gt = gt_integrator(x0, v0, steps=int(T / dt_gt))
# test_integrator_convergence_order (L39)
steps = int(T / dt_val)
# test_integrator_convergence_order (L42)
err = torch.norm(x - x_gt).item()
# test_symplectic_phase_space_conservation (L67)
energy_init = 0.5 * (v.pow(2).sum() + x.pow(2).sum())
# test_symplectic_phase_space_conservation (L68)
energy_final = 0.5 * (v_next.pow(2).sum() + x_next.pow(2).sum())
# test_symplectic_phase_space_conservation (L70)
drift = torch.abs(energy_final - energy_init).item()
# test_hamiltonian_long_term_stability (L90)
e0 = 0.5 * v.pow(2).sum()
# test_hamiltonian_long_term_stability (L91)
e_rk = 0.5 * v_rk.pow(2).sum()
# test_hamiltonian_long_term_stability (L92)
e_vt = 0.5 * v_vt.pow(2).sum()
# test_hamiltonian_long_term_stability (L94)
drift_rk = torch.abs(e_rk - e0).item()
# test_hamiltonian_long_term_stability (L95)
drift_vt = torch.abs(e_vt - e0).item()
```

### tests\architecture\test_learning_dynamics.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 26 | test_gradient_flow_curvature | `loss = gamma.pow(2).sum()` |
| 29 | test_gradient_flow_curvature | `grad_x_norm = x.grad.norm().item()` |
| 30 | test_gradient_flow_curvature | `grad_v_norm = v.grad.norm().item()` |
| 55 | test_hessian_spectrum_proxy | `gz = (gamma * z).sum()` |
| 57 | test_hessian_spectrum_proxy | `trace_est = (grad * z).sum().item()` |
| 60 | test_hessian_spectrum_proxy | `avg_trace = sum(traces) / len(traces)` |

#### Fórmulas Listas para Usar (Python)
```python
# test_gradient_flow_curvature (L26)
loss = gamma.pow(2).sum()
# test_gradient_flow_curvature (L29)
grad_x_norm = x.grad.norm().item()
# test_gradient_flow_curvature (L30)
grad_v_norm = v.grad.norm().item()
# test_hessian_spectrum_proxy (L55)
gz = (gamma * z).sum()
# test_hessian_spectrum_proxy (L57)
trace_est = (grad * z).sum().item()
# test_hessian_spectrum_proxy (L60)
avg_trace = sum(traces) / len(traces)
```

### tests\benchmarks\aggregation_comparison.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 35 | __init__ | `self.half_pi = self.PI * 0.5` |
| 51 | generate_batch | `c = int(parts[-1]) if len(parts) > 1 else 0` |
| 60 | generate_batch | `y_angle[:, -1] = (y_class.float() * 2.0 - 1.0) * self.half_pi` |
| 95 | forward | `x_pred_agg[:, -1] = x_agg` |
| 119 | train_step | `y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)` |
| 122 | train_step | `x_last = x_pred[:, -1]` |
| 123 | train_step | `y_last = y_expanded[:, -1]` |
| 126 | train_step | `loss_val = dist.pow(2).mean()` |
| 141 | train_step | `TWO_PI = 2.0 * PI` |
| 142 | train_step | `half_pi = PI * 0.5` |
| 144 | train_step | `x_last = x_pred[:, -1]` |
| 146 | train_step | `dist_pos = torch.min(torch.abs(x_last - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last - half_pi) % TWO_PI))` |
| 147 | train_step | `dist_neg = torch.min(torch.abs(x_last + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last + half_pi) % TWO_PI))` |
| 148 | train_step | `d_pos = dist_pos.mean(dim=-1)` |
| 149 | train_step | `d_neg = dist_neg.mean(dim=-1)` |
| 152 | train_step | `acc = (preds == targets_class).float().mean().item()` |
| 158 | train_model | `optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction='euclidean')` |
| 165 | train_model | `scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)` |
| 219 | main | `print("\n" + "="*80)` |
| 237 | main | `print("\n" + "="*80)` |
| 255 | main | `print("\n" + "="*80)` |
| 273 | main | `print("\n" + "="*80)` |
| 278 | main | `final_acc = history['acc'][-1]` |
| 279 | main | `final_loss = history['loss'][-1]` |
| 295 | main | `print("\n" + "="*80)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L35)
self.half_pi = self.PI * 0.5
# generate_batch (L51)
c = int(parts[-1]) if len(parts) > 1 else 0
# generate_batch (L60)
y_angle[:, -1] = (y_class.float() * 2.0 - 1.0) * self.half_pi
# forward (L95)
x_pred_agg[:, -1] = x_agg
# train_step (L119)
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# train_step (L122)
x_last = x_pred[:, -1]
# train_step (L123)
y_last = y_expanded[:, -1]
# train_step (L126)
loss_val = dist.pow(2).mean()
# train_step (L141)
TWO_PI = 2.0 * PI
# train_step (L142)
half_pi = PI * 0.5
# train_step (L144)
x_last = x_pred[:, -1]
# train_step (L146)
dist_pos = torch.min(torch.abs(x_last - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last - half_pi) % TWO_PI))
# train_step (L147)
dist_neg = torch.min(torch.abs(x_last + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last + half_pi) % TWO_PI))
# train_step (L148)
d_pos = dist_pos.mean(dim=-1)
# train_step (L149)
d_neg = dist_neg.mean(dim=-1)
# train_step (L152)
acc = (preds == targets_class).float().mean().item()
# train_model (L158)
optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction='euclidean')
# train_model (L165)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
# main (L219)
print("\n" + "="*80)
# main (L237)
print("\n" + "="*80)
# main (L255)
print("\n" + "="*80)
# main (L273)
print("\n" + "="*80)
# main (L278)
final_acc = history['acc'][-1]
# main (L279)
final_loss = history['loss'][-1]
# main (L295)
print("\n" + "="*80)
```

### tests\benchmarks\baselines.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 20 | __init__ | `layer = nn.TransformerEncoderLayer( d_model=dim, nhead=heads, dim_feedforward=4*dim, dropout=0.1, activation='gelu', batch_first=True, norm_first=True )` |
| 53 | forward | `x = self.token_emb(idx) + self.pos_emb[:, :t, :]` |
| 58 | forward | `mask = torch.triu(torch.ones(t, t, device=idx.device) * float('-inf'), diagonal=1)` |
| 82 | __init__ | `self.d_inner = dim * expand` |
| 123 | forward | `current_state_idx += 1` |
| 133 | __init__ | `self.d_inner = d_model * expand` |
| 134 | __init__ | `self.dt_rank = math.ceil(d_model / 16)` |
| 136 | __init__ | `self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)` |
| 138 | __init__ | `self.conv1d = nn.Conv1d( in_channels=self.d_inner, out_channels=self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, )` |
| 147 | __init__ | `self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)` |
| 150 | __init__ | `A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)` |
| 151 | __init__ | `self.A_log = nn.Parameter(torch.log(A))` |
| 162 | forward | `xz = self.in_proj(x) # [B, L, 2*D_in]` |
| 163 | forward | `x_in, z = xz.chunk(2, dim=-1) # [B, L, D_in]` |
| 170 | forward | `new_conv_state = x_in[:, -self.conv1d.kernel_size[0]+1:, :] # Last K-1 tokens` |
| 179 | forward | `pad_len = self.conv1d.kernel_size[0] - 1` |
| 186 | forward | `x_conv = x_conv[:, -1:, :] # Take only last output` |
| 187 | forward | `new_conv_state = conv_input[:, -pad_len:, :]` |
| 194 | forward | `x_dbl = self.x_proj(x_ssm) # [B, L, dt_rank + 2*d_state]` |
| 195 | forward | `dt, B, C = torch.split(x_dbl, [self.dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)` |
| 200 | forward | `A = -torch.exp(self.A_log) # [D_in, D_state]` |
| 211 | forward | `dt_t = dt[:, t, :].unsqueeze(-1) # [B, D_in, 1]` |
| 212 | forward | `dA = torch.exp(dt_t * A) # [B, D_in, D_state]` |
| 214 | forward | `x_t = x_ssm[:, t, :].unsqueeze(-1) # [B, D_in, 1]` |
| 218 | forward | `dB = (dt_t * x_t) * B_t # [B, D_in, D_state]` |
| 222 | forward | `y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1) # [B, D_in]` |
| 226 | forward | `y = y + x_ssm * self.D` |
| 236 | forward | `dt_t = dt.unsqueeze(-1) # [B, 1, D_in, 1]` |
| 237 | forward | `dA = torch.exp(dt_t * A) # [B, 1, D_in, D_state]` |
| 241 | forward | `x_t = x_ssm.unsqueeze(-1) # [B, 1, D_in, 1]` |
| 245 | forward | `h = h.unsqueeze(1) * dA + B_t * x_t` |
| 250 | forward | `y = (h * C.unsqueeze(2)).sum(dim=-1) # [B, D_in]` |
| 251 | forward | `y = y + x_ssm.squeeze(1) * self.D` |
| 257 | forward | `out = y * self.act(z)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L20)
layer = nn.TransformerEncoderLayer( d_model=dim, nhead=heads, dim_feedforward=4*dim, dropout=0.1, activation='gelu', batch_first=True, norm_first=True )
# forward (L53)
x = self.token_emb(idx) + self.pos_emb[:, :t, :]
# forward (L58)
mask = torch.triu(torch.ones(t, t, device=idx.device) * float('-inf'), diagonal=1)
# __init__ (L82)
self.d_inner = dim * expand
# forward (L123)
current_state_idx += 1
# __init__ (L133)
self.d_inner = d_model * expand
# __init__ (L134)
self.dt_rank = math.ceil(d_model / 16)
# __init__ (L136)
self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
# __init__ (L138)
self.conv1d = nn.Conv1d( in_channels=self.d_inner, out_channels=self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, )
# __init__ (L147)
self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
# __init__ (L150)
A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
# __init__ (L151)
self.A_log = nn.Parameter(torch.log(A))
# forward (L162)
xz = self.in_proj(x) # [B, L, 2*D_in]
# forward (L163)
x_in, z = xz.chunk(2, dim=-1) # [B, L, D_in]
# forward (L170)
new_conv_state = x_in[:, -self.conv1d.kernel_size[0]+1:, :] # Last K-1 tokens
# forward (L179)
pad_len = self.conv1d.kernel_size[0] - 1
# forward (L186)
x_conv = x_conv[:, -1:, :] # Take only last output
# forward (L187)
new_conv_state = conv_input[:, -pad_len:, :]
# forward (L194)
x_dbl = self.x_proj(x_ssm) # [B, L, dt_rank + 2*d_state]
# forward (L195)
dt, B, C = torch.split(x_dbl, [self.dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
# forward (L200)
A = -torch.exp(self.A_log) # [D_in, D_state]
# forward (L211)
dt_t = dt[:, t, :].unsqueeze(-1) # [B, D_in, 1]
# forward (L212)
dA = torch.exp(dt_t * A) # [B, D_in, D_state]
# forward (L214)
x_t = x_ssm[:, t, :].unsqueeze(-1) # [B, D_in, 1]
# forward (L218)
dB = (dt_t * x_t) * B_t # [B, D_in, D_state]
# forward (L222)
y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1) # [B, D_in]
# forward (L226)
y = y + x_ssm * self.D
# forward (L236)
dt_t = dt.unsqueeze(-1) # [B, 1, D_in, 1]
# forward (L237)
dA = torch.exp(dt_t * A) # [B, 1, D_in, D_state]
# forward (L241)
x_t = x_ssm.unsqueeze(-1) # [B, 1, D_in, 1]
# forward (L245)
h = h.unsqueeze(1) * dA + B_t * x_t
# forward (L250)
y = (h * C.unsqueeze(2)).sum(dim=-1) # [B, D_in]
# forward (L251)
y = y + x_ssm.squeeze(1) * self.D
# forward (L257)
out = y * self.act(z)
```

### tests\benchmarks\bench_utils.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 20 | __init__ | `self.results_dir = self.root / "results" / category / benchmark_name` |
| 48 | save_json | `path = self.results_dir / filename` |
| 49 | save_json | `with open(path, 'w', encoding='utf-8') as f:` |
| 56 | save_plot | `path = self.results_dir / filename` |
| 68 | get_model_size_mb | `param_size = sum(p.numel() * p.element_size() for p in model.parameters())` |
| 69 | get_model_size_mb | `buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())` |
| 111 | generate_batch | `y_angle = y_int.float() * PI` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L20)
self.results_dir = self.root / "results" / category / benchmark_name
# save_json (L48)
path = self.results_dir / filename
# save_json (L49)
with open(path, 'w', encoding='utf-8') as f:
# save_plot (L56)
path = self.results_dir / filename
# get_model_size_mb (L68)
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
# get_model_size_mb (L69)
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
# generate_batch (L111)
y_angle = y_int.float() * PI
```

### tests\benchmarks\benchmark_cuda_live.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 41 | benchmark_live | `U = torch.randn(dim, rank, device=device) * 0.01` |
| 42 | benchmark_live | `W = torch.randn(dim, rank, device=device) * 0.01` |
| 56 | benchmark_live | `print(f"[*] Starting Benchmark Loop (Batch={batch_size}, Dim={dim})")` |
| 75 | benchmark_live | `gamma = (h**2) @ W.t()` |
| 76 | benchmark_live | `x = x + 0.01 * v` |
| 77 | benchmark_live | `v = v + 0.01 * (f - gamma)` |
| 86 | benchmark_live | `total_time = t1 - start_time` |
| 87 | benchmark_live | `avg_ips = iter_count / total_time` |
| 90 | benchmark_live | `sps = (iter_count * steps) / total_time` |

#### Fórmulas Listas para Usar (Python)
```python
# benchmark_live (L41)
U = torch.randn(dim, rank, device=device) * 0.01
# benchmark_live (L42)
W = torch.randn(dim, rank, device=device) * 0.01
# benchmark_live (L56)
print(f"[*] Starting Benchmark Loop (Batch={batch_size}, Dim={dim})")
# benchmark_live (L75)
gamma = (h**2) @ W.t()
# benchmark_live (L76)
x = x + 0.01 * v
# benchmark_live (L77)
v = v + 0.01 * (f - gamma)
# benchmark_live (L86)
total_time = t1 - start_time
# benchmark_live (L87)
avg_ips = iter_count / total_time
# benchmark_live (L90)
sps = (iter_count * steps) / total_time
```

### tests\benchmarks\benchmark_scan.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 39 | benchmark_mlayers | `par_time = (time.time() - t0) / 5` |
| 41 | benchmark_mlayers | `print(f"L={L:4d} \| Parallel: {par_time*1000:.2f}ms")` |

#### Fórmulas Listas para Usar (Python)
```python
# benchmark_mlayers (L39)
par_time = (time.time() - t0) / 5
# benchmark_mlayers (L41)
print(f"L={L:4d} | Parallel: {par_time*1000:.2f}ms")
```

### tests\benchmarks\core\benchmark_baseline_comparison.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 78 | generate_batch | `y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)` |
| 91 | train_step_manifold | `y_expanded = targets.float().unsqueeze(-1).expand_as(x_pred)` |
| 111 | first_head_metric | `total_loss = loss_val + loss_phy + loss_ham` |
| 122 | first_head_metric | `TWO_PI = 2.0 * PI` |
| 123 | first_head_metric | `half_pi = PI * 0.5` |
| 124 | first_head_metric | `dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))` |
| 125 | first_head_metric | `dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))` |
| 126 | first_head_metric | `d_pos = dist_pos.mean(dim=-1)` |
| 127 | first_head_metric | `d_neg = dist_neg.mean(dim=-1)` |
| 129 | first_head_metric | `acc = (preds == targets_class).float().mean().item()` |
| 161 | run_baseline_comparison | `models_to_test = { "Manifold-GFN": Manifold( vocab_size=vocab_size, dim=dim, depth=6, heads=4, integrator_type='leapfrog', physics_config=physics_config, impulse_scale=80.0, holographic=True ), "Vanilla GRU": SimpleGRU(vocab_size, dim, depth), "Vanilla LSTM": SimpleLSTM(vocab_size, dim, depth) }` |
| 184 | run_baseline_comparison | `optimizer = RiemannianAdam(model.parameters(), lr=1e-3)` |
| 186 | run_baseline_comparison | `optimizer = optim.Adam(model.parameters(), lr=1e-3)` |
| 211 | run_baseline_comparison | `loss = criterion(output.view(-1, vocab_size), targets_int.view(-1))` |
| 217 | run_baseline_comparison | `preds = output.argmax(dim=-1)` |
| 218 | run_baseline_comparison | `acc = (preds == targets_int).float().mean().item()` |
| 225 | run_baseline_comparison | `progress.update(train_task, description=f"L: {loss.item():.4f} A: {acc*100:.1f}%")` |
| 227 | run_baseline_comparison | `elapsed = time.time() - start_time` |
| 228 | run_baseline_comparison | `final_acc = np.mean(history["acc"][-10:]) * 100` |

#### Fórmulas Listas para Usar (Python)
```python
# generate_batch (L78)
y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)
# train_step_manifold (L91)
y_expanded = targets.float().unsqueeze(-1).expand_as(x_pred)
# first_head_metric (L111)
total_loss = loss_val + loss_phy + loss_ham
# first_head_metric (L122)
TWO_PI = 2.0 * PI
# first_head_metric (L123)
half_pi = PI * 0.5
# first_head_metric (L124)
dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
# first_head_metric (L125)
dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
# first_head_metric (L126)
d_pos = dist_pos.mean(dim=-1)
# first_head_metric (L127)
d_neg = dist_neg.mean(dim=-1)
# first_head_metric (L129)
acc = (preds == targets_class).float().mean().item()
# run_baseline_comparison (L161)
models_to_test = { "Manifold-GFN": Manifold( vocab_size=vocab_size, dim=dim, depth=6, heads=4, integrator_type='leapfrog', physics_config=physics_config, impulse_scale=80.0, holographic=True ), "Vanilla GRU": SimpleGRU(vocab_size, dim, depth), "Vanilla LSTM": SimpleLSTM(vocab_size, dim, depth) }
# run_baseline_comparison (L184)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# run_baseline_comparison (L186)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# run_baseline_comparison (L211)
loss = criterion(output.view(-1, vocab_size), targets_int.view(-1))
# run_baseline_comparison (L217)
preds = output.argmax(dim=-1)
# run_baseline_comparison (L218)
acc = (preds == targets_int).float().mean().item()
# run_baseline_comparison (L225)
progress.update(train_task, description=f"L: {loss.item():.4f} A: {acc*100:.1f}%")
# run_baseline_comparison (L227)
elapsed = time.time() - start_time
# run_baseline_comparison (L228)
final_acc = np.mean(history["acc"][-10:]) * 100
```

### tests\benchmarks\core\benchmark_composition.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 41 | __init__ | `chars = [str(i) for i in range(10)] + ['+', '-', '*', '=', 'f', 'g', 'h', '(', ')', '<PAD>', '<EOS>']` |
| 46 | __init__ | `self.funcs = { 'f': lambda x: x + 2, 'g': lambda x: x * 3, 'h': lambda x: x - 1, }` |
| 60 | generate_problem | `func_name = np.random.choice(['f', 'g', 'h'])` |
| 61 | generate_problem | `x = np.random.randint(0, 30)` |
| 65 | generate_problem | `length = np.random.choice([2, 3])` |
| 66 | generate_problem | `composition = ''.join(np.random.choice(['f', 'g', 'h'], size=length))` |
| 67 | generate_problem | `x = np.random.randint(0, 5)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L41)
chars = [str(i) for i in range(10)] + ['+', '-', '*', '=', 'f', 'g', 'h', '(', ')', '<PAD>', '<EOS>']
# __init__ (L46)
self.funcs = { 'f': lambda x: x + 2, 'g': lambda x: x * 3, 'h': lambda x: x - 1, }
# generate_problem (L60)
func_name = np.random.choice(['f', 'g', 'h'])
# generate_problem (L61)
x = np.random.randint(0, 30)
# generate_problem (L65)
length = np.random.choice([2, 3])
# generate_problem (L66)
composition = ''.join(np.random.choice(['f', 'g', 'h'], size=length))
# generate_problem (L67)
x = np.random.randint(0, 5)
```

### tests\benchmarks\core\benchmark_feature_ablation.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 36 | create_associative_recall_data | `sep_token = vocab_size - 1` |
| 41 | create_associative_recall_data | `keys = torch.randint(0, vocab_size - 1, (num_pairs,))` |
| 42 | create_associative_recall_data | `values = torch.randint(0, vocab_size - 1, (num_pairs,))` |
| 54 | create_associative_recall_data | `seq += [sep_token] * (seq_len - len(seq))` |
| 105 | train_and_evaluate | `optimizer = RiemannianAdam(model.parameters(), lr=1e-3)` |
| 128 | train_and_evaluate | `pred_logits = logits[:, -1, :]` |
| 135 | train_and_evaluate | `acc = (pred_logits.argmax(dim=-1) == targets).float().mean().item()` |
| 141 | train_and_evaluate | `final_acc = np.mean(history["acc"][-20:]) * 100` |
| 142 | train_and_evaluate | `final_loss = np.mean(history["loss"][-20:])` |
| 189 | run_benchmark | `ax.text(v + 1, i, f"{v:.1f}%", color='white', va='center', fontweight='bold')` |

#### Fórmulas Listas para Usar (Python)
```python
# create_associative_recall_data (L36)
sep_token = vocab_size - 1
# create_associative_recall_data (L41)
keys = torch.randint(0, vocab_size - 1, (num_pairs,))
# create_associative_recall_data (L42)
values = torch.randint(0, vocab_size - 1, (num_pairs,))
# create_associative_recall_data (L54)
seq += [sep_token] * (seq_len - len(seq))
# train_and_evaluate (L105)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# train_and_evaluate (L128)
pred_logits = logits[:, -1, :]
# train_and_evaluate (L135)
acc = (pred_logits.argmax(dim=-1) == targets).float().mean().item()
# train_and_evaluate (L141)
final_acc = np.mean(history["acc"][-20:]) * 100
# train_and_evaluate (L142)
final_loss = np.mean(history["loss"][-20:])
# run_benchmark (L189)
ax.text(v + 1, i, f"{v:.1f}%", color='white', va='center', fontweight='bold')
```

### tests\benchmarks\core\benchmark_integrators.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 52 | measure_drift | `x = torch.zeros(batch_size, model.dim // model.heads).to(device)` |
| 53 | measure_drift | `v = torch.randn(batch_size, model.dim // model.heads).to(device)` |
| 54 | measure_drift | `v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)` |
| 56 | measure_drift | `v_start_norm = v.norm(dim=-1).mean().item()` |
| 68 | measure_drift | `v_end_norm = tv.norm(dim=-1).mean().item()` |
| 69 | measure_drift | `drift = abs(v_end_norm - v_start_norm) / (v_start_norm + 1e-12) * 100` |
| 96 | run_integrator_suite | `progress.update(suite_task, description=f"Testing: [bold blue]{integ}[/]")` |
| 111 | run_integrator_suite | `tput = (20 * 16) / (time.time() - start)` |
| 132 | run_integrator_suite | `summary_table.add_column("Speed (seq/s)", justify="right")` |
| 160 | run_integrator_suite | `sns.barplot(data=df, x="Integrator", y="Speed (seq/s)", ax=axes[1], palette="crest")` |

#### Fórmulas Listas para Usar (Python)
```python
# measure_drift (L52)
x = torch.zeros(batch_size, model.dim // model.heads).to(device)
# measure_drift (L53)
v = torch.randn(batch_size, model.dim // model.heads).to(device)
# measure_drift (L54)
v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
# measure_drift (L56)
v_start_norm = v.norm(dim=-1).mean().item()
# measure_drift (L68)
v_end_norm = tv.norm(dim=-1).mean().item()
# measure_drift (L69)
drift = abs(v_end_norm - v_start_norm) / (v_start_norm + 1e-12) * 100
# run_integrator_suite (L96)
progress.update(suite_task, description=f"Testing: [bold blue]{integ}[/]")
# run_integrator_suite (L111)
tput = (20 * 16) / (time.time() - start)
# run_integrator_suite (L132)
summary_table.add_column("Speed (seq/s)", justify="right")
# run_integrator_suite (L160)
sns.barplot(data=df, x="Integrator", y="Speed (seq/s)", ax=axes[1], palette="crest")
```

### tests\benchmarks\core\benchmark_learning_dynamics.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 58 | evaluate | `prompt = parts[0] + '='` |
| 68 | evaluate | `curr_token = torch.argmax(logits[:, -1, :], dim=-1)` |
| 73 | evaluate | `curr_token = torch.argmax(logits[:, -1, :], dim=-1)` |
| 75 | evaluate | `if tok_id == dataset.char_to_id.get('<EOS>', -1): break` |
| 82 | evaluate | `curr_token = torch.argmax(logits[:, -1, :], dim=-1)` |
| 84 | evaluate | `if tok_id == dataset.char_to_id.get('<EOS>', -1): break` |
| 87 | evaluate | `pred = dataset.decode(generated).split('=')[-1].strip()` |
| 88 | evaluate | `if pred == target: correct += 1` |
| 105 | train_step_manifold | `targets_expanded = targets_float.unsqueeze(-1).expand_as(x_pred)` |
| 125 | first_head_metric | `total_loss = loss_val + loss_phy + loss_ham` |
| 169 | run_showdown | `m_opt = RiemannianAdam(manifold.parameters(), lr=1e-3)` |
| 170 | run_showdown | `g_opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=0.01)` |
| 171 | run_showdown | `criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_id.get('<PAD>', -1))` |
| 182 | run_showdown | `m_task = progress.add_task("Manifold-GFN   ", total=epochs)` |
| 194 | run_showdown | `ids = [dataset.char_to_id[c] for c in p + '<EOS>']` |
| 199 | run_showdown | `padded_in = torch.tensor([s + [0]*(max_len-len(s)) for s in inputs]).to(self.device)` |
| 200 | run_showdown | `padded_tg = torch.tensor([s + [-100]*(max_len-len(s)) for s in targets]).to(self.device)` |
| 209 | run_showdown | `g_loss = criterion(g_logits.reshape(-1, vocab_size), padded_tg.reshape(-1))` |
| 214 | run_showdown | `if epoch % 2 == 0 or epoch == epochs - 1:` |
| 247 | plot_results | `ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')` |
| 255 | plot_results | `plt.savefig(self.results_dir / "learning_curves_comparison.png", dpi=300, bbox_inches='tight')` |
| 269 | _plot_convergence_comparison | `x = np.arange(len(thresholds))` |
| 272 | _plot_convergence_comparison | `bars1 = ax.bar(x - width/2, gfn_epochs, width, label='GFN', color='#2A9D8F', alpha=0.8)` |
| 273 | _plot_convergence_comparison | `bars2 = ax.bar(x + width/2, gpt_epochs, width, label='Transformer', color='#E76F51', alpha=0.8)` |
| 295 | _plot_convergence_comparison | `plt.savefig(self.results_dir / "convergence_speed_comparison.png", dpi=300, bbox_inches='tight')` |
| 303 | _plot_efficiency_metrics | `gfn_avg_time = np.mean(self.history['Manifold']['time'])` |
| 304 | _plot_efficiency_metrics | `gpt_avg_time = np.mean(self.history['Transformer']['time'])` |
| 306 | _plot_efficiency_metrics | `gfn_final_acc = self.history['Manifold']['acc'][-1]` |
| 307 | _plot_efficiency_metrics | `gpt_final_acc = self.history['Transformer']['acc'][-1]` |
| 310 | _plot_efficiency_metrics | `gfn_efficiency = gfn_final_acc / (gfn_avg_time + 1e-6)` |
| 311 | _plot_efficiency_metrics | `gpt_efficiency = gpt_final_acc / (gpt_avg_time + 1e-6)` |
| 325 | _plot_efficiency_metrics | `ax.set_ylabel('Efficiency (Accuracy % / Sec per Epoch)', fontsize=13)` |
| 330 | _plot_efficiency_metrics | `plt.savefig(self.results_dir / "training_efficiency.png", dpi=300, bbox_inches='tight')` |

#### Fórmulas Listas para Usar (Python)
```python
# evaluate (L58)
prompt = parts[0] + '='
# evaluate (L68)
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# evaluate (L73)
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# evaluate (L75)
if tok_id == dataset.char_to_id.get('<EOS>', -1): break
# evaluate (L82)
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# evaluate (L84)
if tok_id == dataset.char_to_id.get('<EOS>', -1): break
# evaluate (L87)
pred = dataset.decode(generated).split('=')[-1].strip()
# evaluate (L88)
if pred == target: correct += 1
# train_step_manifold (L105)
targets_expanded = targets_float.unsqueeze(-1).expand_as(x_pred)
# first_head_metric (L125)
total_loss = loss_val + loss_phy + loss_ham
# run_showdown (L169)
m_opt = RiemannianAdam(manifold.parameters(), lr=1e-3)
# run_showdown (L170)
g_opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=0.01)
# run_showdown (L171)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_id.get('<PAD>', -1))
# run_showdown (L182)
m_task = progress.add_task("Manifold-GFN   ", total=epochs)
# run_showdown (L194)
ids = [dataset.char_to_id[c] for c in p + '<EOS>']
# run_showdown (L199)
padded_in = torch.tensor([s + [0]*(max_len-len(s)) for s in inputs]).to(self.device)
# run_showdown (L200)
padded_tg = torch.tensor([s + [-100]*(max_len-len(s)) for s in targets]).to(self.device)
# run_showdown (L209)
g_loss = criterion(g_logits.reshape(-1, vocab_size), padded_tg.reshape(-1))
# run_showdown (L214)
if epoch % 2 == 0 or epoch == epochs - 1:
# plot_results (L247)
ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')
# plot_results (L255)
plt.savefig(self.results_dir / "learning_curves_comparison.png", dpi=300, bbox_inches='tight')
# _plot_convergence_comparison (L269)
x = np.arange(len(thresholds))
# _plot_convergence_comparison (L272)
bars1 = ax.bar(x - width/2, gfn_epochs, width, label='GFN', color='#2A9D8F', alpha=0.8)
# _plot_convergence_comparison (L273)
bars2 = ax.bar(x + width/2, gpt_epochs, width, label='Transformer', color='#E76F51', alpha=0.8)
# _plot_convergence_comparison (L295)
plt.savefig(self.results_dir / "convergence_speed_comparison.png", dpi=300, bbox_inches='tight')
# _plot_efficiency_metrics (L303)
gfn_avg_time = np.mean(self.history['Manifold']['time'])
# _plot_efficiency_metrics (L304)
gpt_avg_time = np.mean(self.history['Transformer']['time'])
# _plot_efficiency_metrics (L306)
gfn_final_acc = self.history['Manifold']['acc'][-1]
# _plot_efficiency_metrics (L307)
gpt_final_acc = self.history['Transformer']['acc'][-1]
# _plot_efficiency_metrics (L310)
gfn_efficiency = gfn_final_acc / (gfn_avg_time + 1e-6)
# _plot_efficiency_metrics (L311)
gpt_efficiency = gpt_final_acc / (gpt_avg_time + 1e-6)
# _plot_efficiency_metrics (L325)
ax.set_ylabel('Efficiency (Accuracy % / Sec per Epoch)', fontsize=13)
# _plot_efficiency_metrics (L330)
plt.savefig(self.results_dir / "training_efficiency.png", dpi=300, bbox_inches='tight')
```

### tests\benchmarks\core\benchmark_needle_haystack.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 51 | run_needle_haystack | `model = Manifold( vocab_size=64, dim=256, depth=4, heads=4, integrator_type='yoshida' # High-precision for long-term transport ).to(device)` |
| 87 | run_inference | `pred = logits[0, -1, :8].argmax()` |
| 100 | run_inference | `console.print(f"  [red]OOM at L={L}[/]")` |
| 111 | run_inference | `acc_str = "[green]SUCCESS[/]" if r["Accuracy"] > 0 else "[red]FAIL[/]"` |
| 124 | run_inference | `ax.plot(df["Length"], df["VRAM (MB)"], 'o-', label="Manifold (O(1) Scaling)", color='#00ADB5', lw=3, markersize=8)` |
| 129 | run_inference | `x_theory = np.logspace(np.log10(base_l), np.log10(df.iloc[-1]["Length"]), 50)` |
| 130 | run_inference | `y_theory = base_v + (x_theory/base_l)**2 * (base_v * 0.5)` |
| 131 | run_inference | `ax.plot(x_theory, y_theory, '--', label="Transformer (O(N²) Theory)", color='#FF2E63', alpha=0.5)` |
| 145 | run_inference | `v_start, v_end = df.iloc[0]["VRAM (MB)"], df.iloc[-1]["VRAM (MB)"]` |
| 146 | run_inference | `increase = ((v_end - v_start) / v_start) * 100` |

#### Fórmulas Listas para Usar (Python)
```python
# run_needle_haystack (L51)
model = Manifold( vocab_size=64, dim=256, depth=4, heads=4, integrator_type='yoshida' # High-precision for long-term transport ).to(device)
# run_inference (L87)
pred = logits[0, -1, :8].argmax()
# run_inference (L100)
console.print(f"  [red]OOM at L={L}[/]")
# run_inference (L111)
acc_str = "[green]SUCCESS[/]" if r["Accuracy"] > 0 else "[red]FAIL[/]"
# run_inference (L124)
ax.plot(df["Length"], df["VRAM (MB)"], 'o-', label="Manifold (O(1) Scaling)", color='#00ADB5', lw=3, markersize=8)
# run_inference (L129)
x_theory = np.logspace(np.log10(base_l), np.log10(df.iloc[-1]["Length"]), 50)
# run_inference (L130)
y_theory = base_v + (x_theory/base_l)**2 * (base_v * 0.5)
# run_inference (L131)
ax.plot(x_theory, y_theory, '--', label="Transformer (O(N²) Theory)", color='#FF2E63', alpha=0.5)
# run_inference (L145)
v_start, v_end = df.iloc[0]["VRAM (MB)"], df.iloc[-1]["VRAM (MB)"]
# run_inference (L146)
increase = ((v_end - v_start) / v_start) * 100
```

### tests\benchmarks\core\benchmark_ood.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 43 | evaluate_accuracy | `prompt = parts[0] + '='` |
| 52 | evaluate_accuracy | `curr_token = torch.argmax(logits[:, -1, :], dim=-1)` |
| 58 | evaluate_accuracy | `curr_token = torch.argmax(logits[:, -1, :], dim=-1)` |
| 60 | evaluate_accuracy | `if tok_id == dataset.char_to_id.get('<EOS>', -1): break` |
| 63 | evaluate_accuracy | `pred_res = dataset.decode(generated).split('=')[-1].strip()` |
| 100 | run_ood_suite | `progress.update(ood_task, description=f"Testing {d}-digit Addition")` |
| 106 | run_ood_suite | `"Complexity": "In-Dist" if d <= 2 else "OOD"` |
| 129 | run_ood_suite | `ax.axvline(x=0.5, color='#FF2E63', lw=2, ls='--', label='Training Boundary')` |
| 130 | run_ood_suite | `ax.set_title("Manifold-GFN Systemic Generalization", color='white', fontweight='bold')` |

#### Fórmulas Listas para Usar (Python)
```python
# evaluate_accuracy (L43)
prompt = parts[0] + '='
# evaluate_accuracy (L52)
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# evaluate_accuracy (L58)
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# evaluate_accuracy (L60)
if tok_id == dataset.char_to_id.get('<EOS>', -1): break
# evaluate_accuracy (L63)
pred_res = dataset.decode(generated).split('=')[-1].strip()
# run_ood_suite (L100)
progress.update(ood_task, description=f"Testing {d}-digit Addition")
# run_ood_suite (L106)
"Complexity": "In-Dist" if d <= 2 else "OOD"
# run_ood_suite (L129)
ax.axvline(x=0.5, color='#FF2E63', lw=2, ls='--', label='Training Boundary')
# run_ood_suite (L130)
ax.set_title("Manifold-GFN Systemic Generalization", color='white', fontweight='bold')
```

### tests\benchmarks\core\benchmark_overhead.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 90 | measure_overhead | `elapsed = time.time() - start` |
| 91 | measure_overhead | `tput = (50 * batch_size) / elapsed` |
| 92 | measure_overhead | `lat = (elapsed / 50) * 1000` |
| 134 | run_benchmark | `table.add_column("Throughput (seq/s)", justify="right")` |
| 155 | run_benchmark | `sns.barplot(data=df, x="Configuration", y="Throughput (seq/s)", ax=axes[1], palette="crest")` |

#### Fórmulas Listas para Usar (Python)
```python
# measure_overhead (L90)
elapsed = time.time() - start
# measure_overhead (L91)
tput = (50 * batch_size) / elapsed
# measure_overhead (L92)
lat = (elapsed / 50) * 1000
# run_benchmark (L134)
table.add_column("Throughput (seq/s)", justify="right")
# run_benchmark (L155)
sns.barplot(data=df, x="Configuration", y="Throughput (seq/s)", ax=axes[1], palette="crest")
```

### tests\benchmarks\core\benchmark_performance.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 59 | measure_efficiency | `elapsed = time.time() - start` |
| 60 | measure_efficiency | `tput = (20 * batch_size) / elapsed` |
| 101 | run_performance_suite | `models = { "Manifold-GFN ($O(1)$)": Manifold( vocab, dim, depth, heads, integrator_type='leapfrog', physics_config=physics_config, impulse_scale=80.0, holographic=True ).to(device), "Transformer ($O(N^2)$)": MicroGPT(vocab, dim, depth, heads).to(device) }` |
| 128 | run_performance_suite | `perf_task = progress.add_task("Profiling Models...", total=len(models) * len(seq_lengths))` |
| 141 | run_performance_suite | `progress.update(perf_task, advance=1, description=f"Profiling Models... [cyan]{name} L={L}[/]")` |
| 145 | run_performance_suite | `console.print(f"  [red]{name} OOM at L={L}[/]")` |
| 146 | run_performance_suite | `progress.update(perf_task, advance=len(seq_lengths) - seq_lengths.index(L) - 1) # Advance for remaining lengths of this model` |
| 154 | run_performance_suite | `table.add_column("Throughput (seq/s)", justify="right")` |

#### Fórmulas Listas para Usar (Python)
```python
# measure_efficiency (L59)
elapsed = time.time() - start
# measure_efficiency (L60)
tput = (20 * batch_size) / elapsed
# run_performance_suite (L101)
models = { "Manifold-GFN ($O(1)$)": Manifold( vocab, dim, depth, heads, integrator_type='leapfrog', physics_config=physics_config, impulse_scale=80.0, holographic=True ).to(device), "Transformer ($O(N^2)$)": MicroGPT(vocab, dim, depth, heads).to(device) }
# run_performance_suite (L128)
perf_task = progress.add_task("Profiling Models...", total=len(models) * len(seq_lengths))
# run_performance_suite (L141)
progress.update(perf_task, advance=1, description=f"Profiling Models... [cyan]{name} L={L}[/]")
# run_performance_suite (L145)
console.print(f"  [red]{name} OOM at L={L}[/]")
# run_performance_suite (L146)
progress.update(perf_task, advance=len(seq_lengths) - seq_lengths.index(L) - 1) # Advance for remaining lengths of this model
# run_performance_suite (L154)
table.add_column("Throughput (seq/s)", justify="right")
```

### tests\benchmarks\core\benchmark_precision_stability.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 57 | evaluate_stability | `optimizer = RiemannianAdam(model.parameters(), lr=1e-3)` |
| 67 | evaluate_stability | `target = y.float().unsqueeze(-1).expand(-1, -1, 128) # Matching dim for stability test` |

#### Fórmulas Listas para Usar (Python)
```python
# evaluate_stability (L57)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# evaluate_stability (L67)
target = y.float().unsqueeze(-1).expand(-1, -1, 128) # Matching dim for stability test
```

### tests\benchmarks\core\benchmark_sample_efficiency.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 45 | train_and_eval | `opt = RiemannianAdam(model.parameters(), lr=1e-3)` |
| 48 | train_and_eval | `opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)` |
| 50 | train_and_eval | `criterion = nn.CrossEntropyLoss(ignore_index=test_ds.char_to_id.get('<PAD>', -1))` |
| 58 | train_and_eval | `ids = [test_ds.char_to_id[c] for c in p + '<EOS>']` |
| 62 | train_and_eval | `p_in = torch.tensor([s + [0]*(max_len-len(s)) for s in ins]).to(device)` |
| 63 | train_and_eval | `p_tg = torch.tensor([s + [-100]*(max_len-len(s)) for s in tgs]).to(device)` |
| 68 | train_and_eval | `loss = criterion(logits.reshape(-1, vocab), p_tg.reshape(-1))` |
| 80 | train_and_eval | `prompt, target = parts[0] + '=', parts[1].strip()` |
| 92 | train_and_eval | `tok = torch.argmax(logits[:, -1, :], dim=-1)` |
| 93 | train_and_eval | `if tok.item() == test_ds.char_to_id.get('<EOS>', -1): break` |
| 97 | train_and_eval | `pred = test_ds.decode(gen).split('=')[-1].strip()` |
| 98 | train_and_eval | `if pred == target: correct += 1` |
| 155 | run_sample_efficiency | `ax.plot(df["Samples"], df["Manifold Acc"], 'o-', label="Manifold-GFN", color='#00ADB5', lw=3)` |
| 156 | run_sample_efficiency | `ax.plot(df["Samples"], df["Transformer Acc"], 's-', label="Transformer", color='#FF2E63', lw=3, ls='--')` |

#### Fórmulas Listas para Usar (Python)
```python
# train_and_eval (L45)
opt = RiemannianAdam(model.parameters(), lr=1e-3)
# train_and_eval (L48)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# train_and_eval (L50)
criterion = nn.CrossEntropyLoss(ignore_index=test_ds.char_to_id.get('<PAD>', -1))
# train_and_eval (L58)
ids = [test_ds.char_to_id[c] for c in p + '<EOS>']
# train_and_eval (L62)
p_in = torch.tensor([s + [0]*(max_len-len(s)) for s in ins]).to(device)
# train_and_eval (L63)
p_tg = torch.tensor([s + [-100]*(max_len-len(s)) for s in tgs]).to(device)
# train_and_eval (L68)
loss = criterion(logits.reshape(-1, vocab), p_tg.reshape(-1))
# train_and_eval (L80)
prompt, target = parts[0] + '=', parts[1].strip()
# train_and_eval (L92)
tok = torch.argmax(logits[:, -1, :], dim=-1)
# train_and_eval (L93)
if tok.item() == test_ds.char_to_id.get('<EOS>', -1): break
# train_and_eval (L97)
pred = test_ds.decode(gen).split('=')[-1].strip()
# train_and_eval (L98)
if pred == target: correct += 1
# run_sample_efficiency (L155)
ax.plot(df["Samples"], df["Manifold Acc"], 'o-', label="Manifold-GFN", color='#00ADB5', lw=3)
# run_sample_efficiency (L156)
ax.plot(df["Samples"], df["Transformer Acc"], 's-', label="Transformer", color='#FF2E63', lw=3, ls='--')
```

### tests\benchmarks\core\benchmark_scaling.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 78 | measure_scale_metrics | `elapsed = time.time() - start` |
| 79 | measure_scale_metrics | `tput = (10 * 8) / elapsed` |
| 132 | run_benchmark | `table.add_column("Throughput (seq/s)", justify="right")` |
| 149 | run_benchmark | `sns.lineplot(data=df, x="Config", y="Throughput (seq/s)", ax=ax0_twin, marker='o', color='gold', lw=3)` |
| 153 | run_benchmark | `sns.regplot(data=df, x="Params (M)", y="VRAM (MB)", ax=axes[1], scatter_kws={'s':100}, line_kws={'color':'red', 'ls':'--'})` |

#### Fórmulas Listas para Usar (Python)
```python
# measure_scale_metrics (L78)
elapsed = time.time() - start
# measure_scale_metrics (L79)
tput = (10 * 8) / elapsed
# run_benchmark (L132)
table.add_column("Throughput (seq/s)", justify="right")
# run_benchmark (L149)
sns.lineplot(data=df, x="Config", y="Throughput (seq/s)", ax=ax0_twin, marker='o', color='gold', lw=3)
# run_benchmark (L153)
sns.regplot(data=df, x="Params (M)", y="VRAM (MB)", ax=axes[1], scatter_kws={'s':100}, line_kws={'color':'red', 'ls':'--'})
```

### tests\benchmarks\core\run_validation_suite.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 25 | Global | `BENCHMARK_DIR = PROJECT_ROOT / "tests/benchmarks/core"` |
| 26 | Global | `RESULTS_BASE = PROJECT_ROOT / "tests/results/core"` |
| 30 | Global | `BENCHMARKS = { 'baseline': { 'script': 'benchmark_baseline_comparison.py', 'desc': 'Systematic comparison vs RNNs (GRU/LSTM)' }, 'composition': { 'script': 'benchmark_composition.py', 'desc': 'Function composition & systematic generalization' }, 'ablation': { 'script': 'benchmark_feature_ablation.py', 'desc': 'Physics feature value-add audit' }, 'integrators': { 'script': 'benchmark_integrators.py', 'desc': 'Numerical Drift & Symplectic Stability' }, 'learning': { 'script': 'benchmark_learning_dynamics.py', 'desc': 'GFN vs Transformer on Arithmetic' }, 'needle': { 'script': 'benchmark_needle_haystack.py', 'desc': '1M Token Long-Context Recall' }, 'ood': { 'script': 'benchmark_ood.py', 'desc': 'Out-of-Distribution Math Generalization' }, 'overhead': { 'script': 'benchmark_overhead.py', 'desc': 'Physics Engine Computational Cost' }, 'performance': { 'script': 'benchmark_performance.py', 'desc': 'Throughput & VRAM Scaling Laws' }, 'precision': { 'script': 'benchmark_precision_stability.py', 'desc': 'Numerical format robustness (FP16/BF16)' }, 'efficiency': { 'script': 'benchmark_sample_efficiency.py', 'desc': 'Data efficiency vs Transformers' }, 'scaling': { 'script': 'benchmark_scaling.py', 'desc': 'Model size expansion laws' } }` |
| 85 | run_bench | `script = BENCHMARK_DIR / info['script']` |
| 122 | show_summary | `res_path = RESULTS_BASE / name` |
| 123 | show_summary | `status = "[green]RUN[/]" if res_path.exists() else "[dim]PENDING[/]"` |
| 131 | main | `parser.add_argument('--all', action='store_true', help='Run every benchmark')` |
| 132 | main | `parser.add_argument('--only', nargs='+', help='Run specific benchmarks')` |
| 133 | main | `parser.add_argument('--status', action='store_true', help='Show coverage status')` |
| 156 | main | `elapsed = time.time() - start_time` |
| 159 | main | `console.print("\n" + "="*60)` |
| 162 | main | `console.print("="*60 + "\n")` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L25)
BENCHMARK_DIR = PROJECT_ROOT / "tests/benchmarks/core"
# Global (L26)
RESULTS_BASE = PROJECT_ROOT / "tests/results/core"
# Global (L30)
BENCHMARKS = { 'baseline': { 'script': 'benchmark_baseline_comparison.py', 'desc': 'Systematic comparison vs RNNs (GRU/LSTM)' }, 'composition': { 'script': 'benchmark_composition.py', 'desc': 'Function composition & systematic generalization' }, 'ablation': { 'script': 'benchmark_feature_ablation.py', 'desc': 'Physics feature value-add audit' }, 'integrators': { 'script': 'benchmark_integrators.py', 'desc': 'Numerical Drift & Symplectic Stability' }, 'learning': { 'script': 'benchmark_learning_dynamics.py', 'desc': 'GFN vs Transformer on Arithmetic' }, 'needle': { 'script': 'benchmark_needle_haystack.py', 'desc': '1M Token Long-Context Recall' }, 'ood': { 'script': 'benchmark_ood.py', 'desc': 'Out-of-Distribution Math Generalization' }, 'overhead': { 'script': 'benchmark_overhead.py', 'desc': 'Physics Engine Computational Cost' }, 'performance': { 'script': 'benchmark_performance.py', 'desc': 'Throughput & VRAM Scaling Laws' }, 'precision': { 'script': 'benchmark_precision_stability.py', 'desc': 'Numerical format robustness (FP16/BF16)' }, 'efficiency': { 'script': 'benchmark_sample_efficiency.py', 'desc': 'Data efficiency vs Transformers' }, 'scaling': { 'script': 'benchmark_scaling.py', 'desc': 'Model size expansion laws' } }
# run_bench (L85)
script = BENCHMARK_DIR / info['script']
# show_summary (L122)
res_path = RESULTS_BASE / name
# show_summary (L123)
status = "[green]RUN[/]" if res_path.exists() else "[dim]PENDING[/]"
# main (L131)
parser.add_argument('--all', action='store_true', help='Run every benchmark')
# main (L132)
parser.add_argument('--only', nargs='+', help='Run specific benchmarks')
# main (L133)
parser.add_argument('--status', action='store_true', help='Show coverage status')
# main (L156)
elapsed = time.time() - start_time
# main (L159)
console.print("\n" + "="*60)
# main (L162)
console.print("="*60 + "\n")
```

### tests\benchmarks\core\test_arithmetic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 36 | __init__ | `self.op_map = {'+': 10, '-': 11, '=': 13, '<PAD>': 14}` |
| 41 | generate | `a, b = np.random.randint(0, 5), np.random.randint(0, 5)` |
| 44 | generate | `prob = [a, self.op_map['+'], b, self.op_map['='], res]` |
| 64 | run_arithmetic_benchmark | `opt = RiemannianAdam(model.parameters(), lr=1e-3)` |
| 87 | run_arithmetic_benchmark | `loss = crit(logits[:, -1, :], y)` |
| 91 | run_arithmetic_benchmark | `acc = (logits[:, -1, :].argmax(dim=-1) == y).float().mean().item()` |
| 94 | run_arithmetic_benchmark | `progress.update(train_task, advance=1, description=f"Loss: {loss.item():.4f} \| Acc: {acc*100:.1f}%")` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L36)
self.op_map = {'+': 10, '-': 11, '=': 13, '<PAD>': 14}
# generate (L41)
a, b = np.random.randint(0, 5), np.random.randint(0, 5)
# generate (L44)
prob = [a, self.op_map['+'], b, self.op_map['='], res]
# run_arithmetic_benchmark (L64)
opt = RiemannianAdam(model.parameters(), lr=1e-3)
# run_arithmetic_benchmark (L87)
loss = crit(logits[:, -1, :], y)
# run_arithmetic_benchmark (L91)
acc = (logits[:, -1, :].argmax(dim=-1) == y).float().mean().item()
# run_arithmetic_benchmark (L94)
progress.update(train_task, advance=1, description=f"Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")
```

### tests\benchmarks\generate_report.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 39 | Global | `<meta charset="UTF-8">` |
| 40 | Global | `<meta name="viewport" content="width=device-width, initial-scale=1.0">` |
| 243 | Global | `<div class="subtitle">Geodesic Flow Networks - Professional Validation</div>` |
| 244 | Global | `<div class="timestamp">Generated: {timestamp}</div>` |
| 249 | Global | `<div class="metrics-grid">` |
| 250 | Global | `<div class="metric-card">` |
| 251 | Global | `<div class="label">Energy Conservation</div>` |
| 252 | Global | `<div class="value">{energy_drift:.2f}%</div>` |
| 257 | Global | `<div class="metric-card" style="background: linear-gradient(135deg, #E76F51 0%, #F4A261 100%);">` |
| 258 | Global | `<div class="label">Memory Scaling</div>` |
| 259 | Global | `<div class="value">O(1)</div>` |
| 260 | Global | `<div class="status pass">✅ Verified</div>` |
| 262 | Global | `<div class="metric-card" style="background: linear-gradient(135deg, #264653 0%, #2A9D8F 100%);">` |
| 263 | Global | `<div class="label">Geodesic Optimality</div>` |
| 264 | Global | `<div class="value">{geodesic_status}</div>` |
| 265 | Global | `<div class="status pass">✅ Confirmed</div>` |
| 267 | Global | `<div class="metric-card" style="background: linear-gradient(135deg, #F4A261 0%, #E9C46A 100%);">` |
| 268 | Global | `<div class="label">Tests Passed</div>` |
| 269 | Global | `<div class="value">{tests_passed}/{tests_total}</div>` |
| 270 | Global | `<div class="status pass">{pass_rate:.0f}%</div>` |
| 279 | Global | `<h3 style="margin-top: 30px; color: var(--primary);">Energy Conservation Results</h3>` |
| 280 | Global | `<table class="summary-table">` |
| 292 | Global | `<td><span class="badge {energy_badge}">{energy_verdict}</span></td>` |
| 297 | Global | `<td><span class="badge success">Excellent</span></td>` |
| 302 | Global | `<td><span class="badge success">Stable</span></td>` |
| 307 | Global | `<div class="image-grid">` |
| 316 | Global | `<div class="image-grid">` |
| 325 | Global | `<div class="image-grid">` |
| 333 | Global | `<p style="margin-top: 10px; font-size: 0.9em;">` |
| 335 | Global | `<a href="https://github.com/WitWise/MANIFOLD.git" style="color: var(--primary);">GitHub</a>` |
| 347 | generate_image_card | `rel_path = os.path.relpath(path, PROJECT_ROOT / "tests" / "benchmarks" / "results")` |
| 349 | generate_image_card | `<div class="image-card">` |
| 351 | generate_image_card | `<div class="caption">{caption}</div>` |
| 368 | run_full_suite | `print("\n" + "="*70)` |
| 391 | run_full_suite | `print("\n" + "="*70)` |
| 405 | run_full_suite | `print("\n" + "="*70)` |
| 422 | generate_html_report | `results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"` |
| 427 | generate_html_report | `geodesic_status = "N/A"` |
| 430 | generate_html_report | `energy_drift = results['energy']['drift']['relative_drift'] * 100` |
| 443 | generate_html_report | `tests_passed = sum([ results.get('energy') is not None, results.get('geodesic') is not None, results.get('benchmark') is not None ]) * 3  # Each module has ~3 tests` |
| 449 | generate_html_report | `pass_rate = (tests_passed / tests_total) * 100` |
| 458 | generate_html_report | `path = results_dir / "energy" / img_name # Future-proof: assuming they go here` |
| 459 | generate_html_report | `if not path.exists(): path = results_dir / img_name # Fallback` |
| 460 | generate_html_report | `energy_images += generate_image_card(path, caption)` |
| 468 | generate_html_report | `path = results_dir / img_name` |
| 469 | generate_html_report | `benchmark_images += generate_image_card(path, caption)` |
| 473 | generate_html_report | `viz_list = [ ("geodesic_flow/geodesic_flow_3d.png", "3D Geodesic Flow Trajectory (Reasoning Path)"), ("trajectories/trajectory_comparison.png", "Manifold vs Transformer: Smoothness Comparison"), ("loss_landscape/loss_landscape_3d_comparison.png", "Loss Landscape: Convexity Analysis"), ("fractals/fractal_zoom_comparison.png", "Fractal Recursive Tunneling (Zoom)"), ("manifold_curvature/vis_manifold.png", "Learned Manifold Curvature Heatmap"), ("christoffel_vector_field/christoffel_vector_field.png", "Christoffel Force Vector Field"), ("internal_physics/xray_analysis.png", "Internal Physics X-Ray (Hamiltonian & Fractal Activity)"), ("symmetries/noether_invariance.png", "Noether Invariance (Semantic Symmetries)"), ("active_inference_distortion.png", "Active Inference: Curiosity-Driven Manifold Distortion"), ]` |
| 486 | generate_html_report | `path = results_dir / img_name` |
| 489 | generate_html_report | `flat_name = img_name.split('/')[-1]` |
| 491 | generate_html_report | `path = results_dir / flat_name` |
| 493 | generate_html_report | `manifold_images += generate_image_card(path, caption)` |
| 496 | generate_html_report | `html = HTML_TEMPLATE.format( timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), energy_drift=energy_drift, energy_status=energy_status, energy_verdict=energy_verdict, energy_badge=energy_badge, stability_score=stability_score, geodesic_status=geodesic_status, tests_passed=tests_passed, tests_total=tests_total, pass_rate=pass_rate, energy_images=energy_images, benchmark_images=benchmark_images, manifold_images=manifold_images )` |
| 513 | generate_html_report | `with open(output_path, 'w', encoding='utf-8') as f:` |
| 521 | main | `parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')` |
| 523 | main | `parser.add_argument('--output', type=str, default=None, help='Output HTML file path')` |
| 531 | main | `elapsed = time.time() - start_time` |
| 534 | main | `output_path = args.output or (PROJECT_ROOT / "tests" / "benchmarks" / "results" / "report.html")` |
| 537 | main | `print("\n" + "=" * 70)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L39)
<meta charset="UTF-8">
# Global (L40)
<meta name="viewport" content="width=device-width, initial-scale=1.0">
# Global (L243)
<div class="subtitle">Geodesic Flow Networks - Professional Validation</div>
# Global (L244)
<div class="timestamp">Generated: {timestamp}</div>
# Global (L249)
<div class="metrics-grid">
# Global (L250)
<div class="metric-card">
# Global (L251)
<div class="label">Energy Conservation</div>
# Global (L252)
<div class="value">{energy_drift:.2f}%</div>
# Global (L257)
<div class="metric-card" style="background: linear-gradient(135deg, #E76F51 0%, #F4A261 100%);">
# Global (L258)
<div class="label">Memory Scaling</div>
# Global (L259)
<div class="value">O(1)</div>
# Global (L260)
<div class="status pass">✅ Verified</div>
# Global (L262)
<div class="metric-card" style="background: linear-gradient(135deg, #264653 0%, #2A9D8F 100%);">
# Global (L263)
<div class="label">Geodesic Optimality</div>
# Global (L264)
<div class="value">{geodesic_status}</div>
# Global (L265)
<div class="status pass">✅ Confirmed</div>
# Global (L267)
<div class="metric-card" style="background: linear-gradient(135deg, #F4A261 0%, #E9C46A 100%);">
# Global (L268)
<div class="label">Tests Passed</div>
# Global (L269)
<div class="value">{tests_passed}/{tests_total}</div>
# Global (L270)
<div class="status pass">{pass_rate:.0f}%</div>
# Global (L279)
<h3 style="margin-top: 30px; color: var(--primary);">Energy Conservation Results</h3>
# Global (L280)
<table class="summary-table">
# Global (L292)
<td><span class="badge {energy_badge}">{energy_verdict}</span></td>
# Global (L297)
<td><span class="badge success">Excellent</span></td>
# Global (L302)
<td><span class="badge success">Stable</span></td>
# Global (L307)
<div class="image-grid">
# Global (L316)
<div class="image-grid">
# Global (L325)
<div class="image-grid">
# Global (L333)
<p style="margin-top: 10px; font-size: 0.9em;">
# Global (L335)
<a href="https://github.com/WitWise/MANIFOLD.git" style="color: var(--primary);">GitHub</a>
# generate_image_card (L347)
rel_path = os.path.relpath(path, PROJECT_ROOT / "tests" / "benchmarks" / "results")
# generate_image_card (L349)
<div class="image-card">
# generate_image_card (L351)
<div class="caption">{caption}</div>
# run_full_suite (L368)
print("\n" + "="*70)
# run_full_suite (L391)
print("\n" + "="*70)
# run_full_suite (L405)
print("\n" + "="*70)
# generate_html_report (L422)
results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
# generate_html_report (L427)
geodesic_status = "N/A"
# generate_html_report (L430)
energy_drift = results['energy']['drift']['relative_drift'] * 100
# generate_html_report (L443)
tests_passed = sum([ results.get('energy') is not None, results.get('geodesic') is not None, results.get('benchmark') is not None ]) * 3  # Each module has ~3 tests
# generate_html_report (L449)
pass_rate = (tests_passed / tests_total) * 100
# generate_html_report (L458)
path = results_dir / "energy" / img_name # Future-proof: assuming they go here
# generate_html_report (L459)
if not path.exists(): path = results_dir / img_name # Fallback
# generate_html_report (L460)
energy_images += generate_image_card(path, caption)
# generate_html_report (L468)
path = results_dir / img_name
# generate_html_report (L469)
benchmark_images += generate_image_card(path, caption)
# generate_html_report (L473)
viz_list = [ ("geodesic_flow/geodesic_flow_3d.png", "3D Geodesic Flow Trajectory (Reasoning Path)"), ("trajectories/trajectory_comparison.png", "Manifold vs Transformer: Smoothness Comparison"), ("loss_landscape/loss_landscape_3d_comparison.png", "Loss Landscape: Convexity Analysis"), ("fractals/fractal_zoom_comparison.png", "Fractal Recursive Tunneling (Zoom)"), ("manifold_curvature/vis_manifold.png", "Learned Manifold Curvature Heatmap"), ("christoffel_vector_field/christoffel_vector_field.png", "Christoffel Force Vector Field"), ("internal_physics/xray_analysis.png", "Internal Physics X-Ray (Hamiltonian & Fractal Activity)"), ("symmetries/noether_invariance.png", "Noether Invariance (Semantic Symmetries)"), ("active_inference_distortion.png", "Active Inference: Curiosity-Driven Manifold Distortion"), ]
# generate_html_report (L486)
path = results_dir / img_name
# generate_html_report (L489)
flat_name = img_name.split('/')[-1]
# generate_html_report (L491)
path = results_dir / flat_name
# generate_html_report (L493)
manifold_images += generate_image_card(path, caption)
# generate_html_report (L496)
html = HTML_TEMPLATE.format( timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), energy_drift=energy_drift, energy_status=energy_status, energy_verdict=energy_verdict, energy_badge=energy_badge, stability_score=stability_score, geodesic_status=geodesic_status, tests_passed=tests_passed, tests_total=tests_total, pass_rate=pass_rate, energy_images=energy_images, benchmark_images=benchmark_images, manifold_images=manifold_images )
# generate_html_report (L513)
with open(output_path, 'w', encoding='utf-8') as f:
# main (L521)
parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
# main (L523)
parser.add_argument('--output', type=str, default=None, help='Output HTML file path')
# main (L531)
elapsed = time.time() - start_time
# main (L534)
output_path = args.output or (PROJECT_ROOT / "tests" / "benchmarks" / "results" / "report.html")
# main (L537)
print("\n" + "=" * 70)
```

### tests\benchmarks\validation\verify_docs.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 43 | measure_memory_and_throughput | `peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0` |
| 44 | measure_memory_and_throughput | `throughput = (runs * batch_size) / (end - start)` |
| 63 | main | `print(f"\n{'='*60}")` |
| 82 | main | `print(f"Throughput: {throughput:.2f} seq/s (batch=32, seq=128)")` |
| 87 | main | `results[cfg['name']] = { "params": params, "params_M": round(params/1e6, 2), "peak_vram_gb": round(peak_mem, 3), "throughput": round(throughput, 2) }` |
| 102 | main | `print("\n" + "="*60)` |
| 107 | main | `res_path = PROJECT_ROOT / "tests/benchmarks/results/verification.json"` |

#### Fórmulas Listas para Usar (Python)
```python
# measure_memory_and_throughput (L43)
peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
# measure_memory_and_throughput (L44)
throughput = (runs * batch_size) / (end - start)
# main (L63)
print(f"\n{'='*60}")
# main (L82)
print(f"Throughput: {throughput:.2f} seq/s (batch=32, seq=128)")
# main (L87)
results[cfg['name']] = { "params": params, "params_M": round(params/1e6, 2), "peak_vram_gb": round(peak_mem, 3), "throughput": round(throughput, 2) }
# main (L102)
print("\n" + "="*60)
# main (L107)
res_path = PROJECT_ROOT / "tests/benchmarks/results/verification.json"
```

### tests\benchmarks\viz\math.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 113 | __init__ | `self.half_pi = self.PI * 0.5` |
| 129 | generate_batch | `c = int(parts[-1]) if len(parts) > 1 else 0` |
| 138 | generate_batch | `y_angle[:, -1] = (y_class.float() * 2.0 - 1.0) * self.half_pi  # Only last position` |
| 166 | train_step_manifold | `y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)` |
| 187 | train_step_manifold | `total_loss = loss_val + loss_phy + loss_ham` |
| 195 | train_step_manifold | `TWO_PI = 2.0 * PI` |
| 196 | train_step_manifold | `half_pi = PI * 0.5` |
| 198 | train_step_manifold | `x_last = x_pred[:, -1]` |
| 200 | train_step_manifold | `dist_pos = torch.min(torch.abs(x_last - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last - half_pi) % TWO_PI))` |
| 201 | train_step_manifold | `dist_neg = torch.min(torch.abs(x_last + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last + half_pi) % TWO_PI))` |
| 202 | train_step_manifold | `d_pos = dist_pos.mean(dim=-1)` |
| 203 | train_step_manifold | `d_neg = dist_neg.mean(dim=-1)` |
| 206 | train_step_manifold | `acc = (preds == targets_class).float().mean().item()` |
| 219 | train_model | `optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction=retraction)` |
| 224 | train_model | `optimizer = optim.AdamW([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ])` |
| 229 | train_model | `optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)` |
| 231 | train_model | `scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)` |
| 320 | run_inf | `out = model(x[:, t:t+1], state=state)` |
| 324 | run_inf | `TWO_PI = 2.0 * PI` |
| 325 | run_inf | `half_pi = PI * 0.5` |
| 326 | run_inf | `dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))` |
| 327 | run_inf | `dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))` |
| 328 | run_inf | `d_pos = dist_pos.mean(dim=-1).view(-1)` |
| 329 | run_inf | `d_neg = dist_neg.mean(dim=-1).view(-1)` |
| 333 | run_inf | `return model(x).argmax(dim=-1)` |
| 338 | run_inf | `acc = (preds == y_class).float().mean().item()` |
| 343 | run_inf | `acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"` |
| 353 | run_inf | `max_length = lengths[i-1] if i > 0 else 0` |
| 361 | run_inf | `max_length = lengths[i-1] if i > 0 else 0` |
| 376 | print_header | `console.print("\n" + "="*80, style="magenta")` |
| 377 | print_header | `console.print("  [bold cyan]GFN MATH COMPLEXITY BENCHMARK[/] - [italic]PRODUCTION CONFIG (2026-02-07)[/]", justify="center")` |
| 378 | print_header | `console.print("="*80, style="magenta")` |
| 385 | print_header | `base_dt = stability_cfg.get('base_dt', 'N/A')` |
| 386 | print_header | `sing_strength = singularities_cfg.get('strength', 'N/A')` |
| 387 | print_header | `sing_threshold = singularities_cfg.get('threshold', 'N/A')` |
| 389 | print_header | `lambda_g = loss_config.get('lambda_g', 'N/A')` |
| 392 | print_header | `console.print(f"    - Topology: {topology}, dynamic_time={'on' if dynamic_time else 'off'}")` |
| 393 | print_header | `console.print(f"    - base_dt={base_dt}")` |
| 394 | print_header | `console.print(f"    - singularity_strength={sing_strength}, threshold={sing_threshold}")` |
| 395 | print_header | `console.print(f"    - hamiltonian_mode={ham_mode}, lambda_g={lambda_g}")` |
| 396 | print_header | `console.print("="*80 + "\n", style="magenta")` |
| 412 | run_production_benchmark | `optimizer_label = "AdamW + OneCycleLR"` |
| 432 | run_production_benchmark | `h_m = train_model( "Manifold-GFN-PRODUCTION", manifold, max_steps=1000, device=device, loss_config=PRODUCTION_LOSS_CONFIG, retraction='normalize', optimizer_type='adamw' )` |
| 444 | run_production_benchmark | `s_m = evaluate_scaling("Manifold-GFN-PRODUCTION", manifold, lengths, device)` |
| 476 | run_production_benchmark | `ce_smooth = np.convolve(h_m["loss_breakdown"]['ce'], np.ones(20)/20, mode='valid')` |
| 477 | run_production_benchmark | `ham_smooth = np.convolve(h_m["loss_breakdown"]['hamiltonian'], np.ones(20)/20, mode='valid')` |
| 478 | run_production_benchmark | `geo_smooth = np.convolve(h_m["loss_breakdown"]['geodesic'], np.ones(20)/20, mode='valid')` |
| 479 | run_production_benchmark | `ax.plot(ce_smooth, color='#00ADB5', label='Cross-Entropy', linewidth=2)` |
| 488 | run_production_benchmark | `ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN (Production)', linewidth=5, markersize=12, markerfacecolor='white')` |
| 497 | run_production_benchmark | `ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Production)', linewidth=5, markersize=12, markerfacecolor='white')` |
| 508 | run_production_benchmark | `summary_table = Table(title="[bold yellow]MATH COMPLEXITY SUMMARY (PRODUCTION)[/]", border_style="magenta", show_header=True, header_style="bold cyan")` |
| 510 | run_production_benchmark | `summary_table.add_column("Manifold-GFN-PRODUCTION", justify="center")` |
| 514 | run_production_benchmark | `acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0` |
| 516 | run_production_benchmark | `m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"` |
| 517 | run_production_benchmark | `target_l = lengths[-1]` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L113)
self.half_pi = self.PI * 0.5
# generate_batch (L129)
c = int(parts[-1]) if len(parts) > 1 else 0
# generate_batch (L138)
y_angle[:, -1] = (y_class.float() * 2.0 - 1.0) * self.half_pi  # Only last position
# train_step_manifold (L166)
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# train_step_manifold (L187)
total_loss = loss_val + loss_phy + loss_ham
# train_step_manifold (L195)
TWO_PI = 2.0 * PI
# train_step_manifold (L196)
half_pi = PI * 0.5
# train_step_manifold (L198)
x_last = x_pred[:, -1]
# train_step_manifold (L200)
dist_pos = torch.min(torch.abs(x_last - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last - half_pi) % TWO_PI))
# train_step_manifold (L201)
dist_neg = torch.min(torch.abs(x_last + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last + half_pi) % TWO_PI))
# train_step_manifold (L202)
d_pos = dist_pos.mean(dim=-1)
# train_step_manifold (L203)
d_neg = dist_neg.mean(dim=-1)
# train_step_manifold (L206)
acc = (preds == targets_class).float().mean().item()
# train_model (L219)
optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction=retraction)
# train_model (L224)
optimizer = optim.AdamW([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ])
# train_model (L229)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# train_model (L231)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
# run_inf (L320)
out = model(x[:, t:t+1], state=state)
# run_inf (L324)
TWO_PI = 2.0 * PI
# run_inf (L325)
half_pi = PI * 0.5
# run_inf (L326)
dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))
# run_inf (L327)
dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))
# run_inf (L328)
d_pos = dist_pos.mean(dim=-1).view(-1)
# run_inf (L329)
d_neg = dist_neg.mean(dim=-1).view(-1)
# run_inf (L333)
return model(x).argmax(dim=-1)
# run_inf (L338)
acc = (preds == y_class).float().mean().item()
# run_inf (L343)
acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"
# run_inf (L353)
max_length = lengths[i-1] if i > 0 else 0
# run_inf (L361)
max_length = lengths[i-1] if i > 0 else 0
# print_header (L376)
console.print("\n" + "="*80, style="magenta")
# print_header (L377)
console.print("  [bold cyan]GFN MATH COMPLEXITY BENCHMARK[/] - [italic]PRODUCTION CONFIG (2026-02-07)[/]", justify="center")
# print_header (L378)
console.print("="*80, style="magenta")
# print_header (L385)
base_dt = stability_cfg.get('base_dt', 'N/A')
# print_header (L386)
sing_strength = singularities_cfg.get('strength', 'N/A')
# print_header (L387)
sing_threshold = singularities_cfg.get('threshold', 'N/A')
# print_header (L389)
lambda_g = loss_config.get('lambda_g', 'N/A')
# print_header (L392)
console.print(f"    - Topology: {topology}, dynamic_time={'on' if dynamic_time else 'off'}")
# print_header (L393)
console.print(f"    - base_dt={base_dt}")
# print_header (L394)
console.print(f"    - singularity_strength={sing_strength}, threshold={sing_threshold}")
# print_header (L395)
console.print(f"    - hamiltonian_mode={ham_mode}, lambda_g={lambda_g}")
# print_header (L396)
console.print("="*80 + "\n", style="magenta")
# run_production_benchmark (L412)
optimizer_label = "AdamW + OneCycleLR"
# run_production_benchmark (L432)
h_m = train_model( "Manifold-GFN-PRODUCTION", manifold, max_steps=1000, device=device, loss_config=PRODUCTION_LOSS_CONFIG, retraction='normalize', optimizer_type='adamw' )
# run_production_benchmark (L444)
s_m = evaluate_scaling("Manifold-GFN-PRODUCTION", manifold, lengths, device)
# run_production_benchmark (L476)
ce_smooth = np.convolve(h_m["loss_breakdown"]['ce'], np.ones(20)/20, mode='valid')
# run_production_benchmark (L477)
ham_smooth = np.convolve(h_m["loss_breakdown"]['hamiltonian'], np.ones(20)/20, mode='valid')
# run_production_benchmark (L478)
geo_smooth = np.convolve(h_m["loss_breakdown"]['geodesic'], np.ones(20)/20, mode='valid')
# run_production_benchmark (L479)
ax.plot(ce_smooth, color='#00ADB5', label='Cross-Entropy', linewidth=2)
# run_production_benchmark (L488)
ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN (Production)', linewidth=5, markersize=12, markerfacecolor='white')
# run_production_benchmark (L497)
ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Production)', linewidth=5, markersize=12, markerfacecolor='white')
# run_production_benchmark (L508)
summary_table = Table(title="[bold yellow]MATH COMPLEXITY SUMMARY (PRODUCTION)[/]", border_style="magenta", show_header=True, header_style="bold cyan")
# run_production_benchmark (L510)
summary_table.add_column("Manifold-GFN-PRODUCTION", justify="center")
# run_production_benchmark (L514)
acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0
# run_production_benchmark (L516)
m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"
# run_production_benchmark (L517)
target_l = lengths[-1]
```

### tests\benchmarks\viz\math2.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 48 | __init__ | `self.half_pi = self.PI * 0.5` |
| 62 | generate_batch | `c = int(parts[-1]) if len(parts) > 1 else 0` |
| 69 | generate_batch | `y_angle = (y_class_seq.float() * 2.0 - 1.0) * self.half_pi` |
| 82 | train_step_manifold | `y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)` |
| 102 | first_head_metric | `total_loss = loss_val + loss_phy + loss_ham` |
| 113 | first_head_metric | `TWO_PI = 2.0 * PI` |
| 114 | first_head_metric | `half_pi = PI * 0.5` |
| 117 | first_head_metric | `dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))` |
| 118 | first_head_metric | `dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))` |
| 119 | first_head_metric | `d_pos = dist_pos.mean(dim=-1)` |
| 120 | first_head_metric | `d_neg = dist_neg.mean(dim=-1)` |
| 122 | first_head_metric | `acc = (preds == targets_class).float().mean().item()` |
| 130 | train_model | `optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction='normalize')` |
| 135 | train_model | `optimizer = RiemannianAdam(model.parameters(), lr=1e-3, weight_decay=1e-4, retraction='normalize')` |
| 137 | train_model | `scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)` |
| 220 | run_inf | `out = model(x[:, t:t+1], state=state)` |
| 224 | run_inf | `TWO_PI = 2.0 * PI` |
| 225 | run_inf | `half_pi = PI * 0.5` |
| 226 | run_inf | `dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))` |
| 227 | run_inf | `dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))` |
| 228 | run_inf | `d_pos = dist_pos.mean(dim=-1).view(-1)` |
| 229 | run_inf | `d_neg = dist_neg.mean(dim=-1).view(-1)` |
| 233 | run_inf | `return model(x).argmax(dim=-1)` |
| 238 | run_inf | `acc = (preds == y_class).float().mean().item()` |
| 243 | run_inf | `acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"` |
| 253 | run_inf | `max_length = lengths[i-1] if i > 0 else 0  # La anterior fue la última exitosa` |
| 261 | run_inf | `max_length = lengths[i-1] if i > 0 else 0` |
| 275 | print_header | `console.print("\n" + "="*80, style="magenta")` |
| 276 | print_header | `console.print("  [bold cyan]GFN MATH COMPLEXITY BENCHMARK[/] - [italic]Holographic Manifold[/]", justify="center")` |
| 277 | print_header | `console.print("="*80, style="magenta")` |
| 280 | print_header | `console.print("="*80 + "\n", style="magenta")` |
| 307 | run_superiority_benchmark | `h_m = train_model("Manifold-GFN", manifold, max_steps=1000, device=device)` |
| 308 | run_superiority_benchmark | `ckpt_path = logger.results_dir / "manifold_math_complex.pt"` |
| 314 | run_superiority_benchmark | `s_m = evaluate_scaling("Manifold-GFN", manifold, lengths, device)` |
| 346 | run_superiority_benchmark | `ax.plot(np.convolve(h_m["acc"], np.ones(20)/20, mode='valid'), color=cols[0], label='Manifold GFN', linewidth=3.5)` |
| 353 | run_superiority_benchmark | `ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN', linewidth=5, markersize=12, markerfacecolor='white')` |
| 362 | run_superiority_benchmark | `ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Streaming)', linewidth=5, markersize=12, markerfacecolor='white')` |
| 373 | run_superiority_benchmark | `summary_table = Table(title="[bold yellow]MATH COMPLEXITY SUMMARY[/]", border_style="magenta", show_header=True, header_style="bold cyan")` |
| 375 | run_superiority_benchmark | `summary_table.add_column("Manifold-GFN", justify="center")` |
| 379 | run_superiority_benchmark | `acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0` |
| 381 | run_superiority_benchmark | `m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"` |
| 382 | run_superiority_benchmark | `target_l = lengths[-1]` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L48)
self.half_pi = self.PI * 0.5
# generate_batch (L62)
c = int(parts[-1]) if len(parts) > 1 else 0
# generate_batch (L69)
y_angle = (y_class_seq.float() * 2.0 - 1.0) * self.half_pi
# train_step_manifold (L82)
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# first_head_metric (L102)
total_loss = loss_val + loss_phy + loss_ham
# first_head_metric (L113)
TWO_PI = 2.0 * PI
# first_head_metric (L114)
half_pi = PI * 0.5
# first_head_metric (L117)
dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
# first_head_metric (L118)
dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
# first_head_metric (L119)
d_pos = dist_pos.mean(dim=-1)
# first_head_metric (L120)
d_neg = dist_neg.mean(dim=-1)
# first_head_metric (L122)
acc = (preds == targets_class).float().mean().item()
# train_model (L130)
optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction='normalize')
# train_model (L135)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3, weight_decay=1e-4, retraction='normalize')
# train_model (L137)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
# run_inf (L220)
out = model(x[:, t:t+1], state=state)
# run_inf (L224)
TWO_PI = 2.0 * PI
# run_inf (L225)
half_pi = PI * 0.5
# run_inf (L226)
dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))
# run_inf (L227)
dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))
# run_inf (L228)
d_pos = dist_pos.mean(dim=-1).view(-1)
# run_inf (L229)
d_neg = dist_neg.mean(dim=-1).view(-1)
# run_inf (L233)
return model(x).argmax(dim=-1)
# run_inf (L238)
acc = (preds == y_class).float().mean().item()
# run_inf (L243)
acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"
# run_inf (L253)
max_length = lengths[i-1] if i > 0 else 0  # La anterior fue la última exitosa
# run_inf (L261)
max_length = lengths[i-1] if i > 0 else 0
# print_header (L275)
console.print("\n" + "="*80, style="magenta")
# print_header (L276)
console.print("  [bold cyan]GFN MATH COMPLEXITY BENCHMARK[/] - [italic]Holographic Manifold[/]", justify="center")
# print_header (L277)
console.print("="*80, style="magenta")
# print_header (L280)
console.print("="*80 + "\n", style="magenta")
# run_superiority_benchmark (L307)
h_m = train_model("Manifold-GFN", manifold, max_steps=1000, device=device)
# run_superiority_benchmark (L308)
ckpt_path = logger.results_dir / "manifold_math_complex.pt"
# run_superiority_benchmark (L314)
s_m = evaluate_scaling("Manifold-GFN", manifold, lengths, device)
# run_superiority_benchmark (L346)
ax.plot(np.convolve(h_m["acc"], np.ones(20)/20, mode='valid'), color=cols[0], label='Manifold GFN', linewidth=3.5)
# run_superiority_benchmark (L353)
ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN', linewidth=5, markersize=12, markerfacecolor='white')
# run_superiority_benchmark (L362)
ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Streaming)', linewidth=5, markersize=12, markerfacecolor='white')
# run_superiority_benchmark (L373)
summary_table = Table(title="[bold yellow]MATH COMPLEXITY SUMMARY[/]", border_style="magenta", show_header=True, header_style="bold cyan")
# run_superiority_benchmark (L375)
summary_table.add_column("Manifold-GFN", justify="center")
# run_superiority_benchmark (L379)
acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0
# run_superiority_benchmark (L381)
m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"
# run_superiority_benchmark (L382)
target_l = lengths[-1]
```

### tests\benchmarks\viz\run_viz_suite.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 41 | run_script | `elapsed = time.time() - start_time` |
| 44 | run_script | `elapsed = time.time() - start_time` |
| 49 | main | `parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')` |
| 50 | main | `parser.add_argument('--filter', type=str, help='Regex filter for script names')` |
| 51 | main | `parser.add_argument('--skip-failures', action='store_true', help='Continue even if a script fails')` |
| 66 | main | `scripts = sorted([f for f in VIZ_DIR.glob("vis_*.py")])` |
| 77 | main | `print(f"[{i+1}/{len(scripts)}] Running {script.name}...", end="", flush=True)` |
| 99 | main | `print("\n" + "=" * 80)` |
| 110 | main | `if len(msg) > 30: msg = msg[:27] + "..."` |

#### Fórmulas Listas para Usar (Python)
```python
# run_script (L41)
elapsed = time.time() - start_time
# run_script (L44)
elapsed = time.time() - start_time
# main (L49)
parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
# main (L50)
parser.add_argument('--filter', type=str, help='Regex filter for script names')
# main (L51)
parser.add_argument('--skip-failures', action='store_true', help='Continue even if a script fails')
# main (L66)
scripts = sorted([f for f in VIZ_DIR.glob("vis_*.py")])
# main (L77)
print(f"[{i+1}/{len(scripts)}] Running {script.name}...", end="", flush=True)
# main (L99)
print("\n" + "=" * 80)
# main (L110)
if len(msg) > 30: msg = msg[:27] + "..."
```

### tests\benchmarks\viz\verify_fusion.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 18 | test_fusion | `print("\n" + "="*60)` |
| 20 | test_fusion | `print("="*60 + "\n")` |
| 58 | test_fusion | `print(f"  - Topology: {'Torus' if params['topology_id'] == 1 else 'Euclidean'}")` |
| 95 | test_fusion | `print("\n" + "="*60)` |

#### Fórmulas Listas para Usar (Python)
```python
# test_fusion (L18)
print("\n" + "="*60)
# test_fusion (L20)
print("="*60 + "\n")
# test_fusion (L58)
print(f"  - Topology: {'Torus' if params['topology_id'] == 1 else 'Euclidean'}")
# test_fusion (L95)
print("\n" + "="*60)
```

### tests\benchmarks\viz\vis_active_inference.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 27 | plot_reactive_dynamics | `time = np.arange(len(history['energy']))` |
| 32 | plot_reactive_dynamics | `ax1.plot(time, history['energy'], 'r-', label='Kinetic Energy (Uncertainty)', linewidth=2)` |
| 38 | plot_reactive_dynamics | `ax1t.plot(time, history['curvature'], 'b--', label='Manifold Curvature $\Gamma$', linewidth=2)` |
| 44 | plot_reactive_dynamics | `ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')` |
| 49 | plot_reactive_dynamics | `ax2.plot(time, history['singularity'], 'k-', label='Singularity Gate (Event Horizon)', linewidth=2)` |
| 71 | run_active_inference_viz | `physics_config = { 'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 'readout': {'type': 'implicit', 'coord_dim': 16}, 'active_inference': { 'enabled': True, 'dynamic_time': {'enabled': True}, 'reactive_curvature': {'enabled': True, 'plasticity': 0.5}, # High for Viz 'singularities': {'enabled': True, 'strength': 5.0, 'threshold': 0.7} }, 'fractal': {'enabled': False}, # Disable Fractal for clearer Macro-physics view 'topology': {'type': 'torus'}, 'stability': {'base_dt': 0.2} }` |
| 119 | run_active_inference_viz | `input_t = x[:, t:t+1]` |
| 135 | run_active_inference_viz | `energy = torch.tanh(v_curr.pow(2).mean()).item()` |
| 138 | run_active_inference_viz | `curvature_scale = 1.0 + physics_config['active_inference']['reactive_curvature']['plasticity'] * energy` |
| 143 | run_active_inference_viz | `x_sin = torch.sin(x_curr)` |
| 144 | run_active_inference_viz | `x_cos = torch.cos(x_curr)` |
| 145 | run_active_inference_viz | `x_phases = torch.cat([x_sin, x_cos], dim=-1)` |

#### Fórmulas Listas para Usar (Python)
```python
# plot_reactive_dynamics (L27)
time = np.arange(len(history['energy']))
# plot_reactive_dynamics (L32)
ax1.plot(time, history['energy'], 'r-', label='Kinetic Energy (Uncertainty)', linewidth=2)
# plot_reactive_dynamics (L38)
ax1t.plot(time, history['curvature'], 'b--', label='Manifold Curvature $\Gamma$', linewidth=2)
# plot_reactive_dynamics (L44)
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
# plot_reactive_dynamics (L49)
ax2.plot(time, history['singularity'], 'k-', label='Singularity Gate (Event Horizon)', linewidth=2)
# run_active_inference_viz (L71)
physics_config = { 'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 'readout': {'type': 'implicit', 'coord_dim': 16}, 'active_inference': { 'enabled': True, 'dynamic_time': {'enabled': True}, 'reactive_curvature': {'enabled': True, 'plasticity': 0.5}, # High for Viz 'singularities': {'enabled': True, 'strength': 5.0, 'threshold': 0.7} }, 'fractal': {'enabled': False}, # Disable Fractal for clearer Macro-physics view 'topology': {'type': 'torus'}, 'stability': {'base_dt': 0.2} }
# run_active_inference_viz (L119)
input_t = x[:, t:t+1]
# run_active_inference_viz (L135)
energy = torch.tanh(v_curr.pow(2).mean()).item()
# run_active_inference_viz (L138)
curvature_scale = 1.0 + physics_config['active_inference']['reactive_curvature']['plasticity'] * energy
# run_active_inference_viz (L143)
x_sin = torch.sin(x_curr)
# run_active_inference_viz (L144)
x_cos = torch.cos(x_curr)
# run_active_inference_viz (L145)
x_phases = torch.cat([x_sin, x_cos], dim=-1)
```

### tests\benchmarks\viz\vis_dynamic_friction.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 26 | plot_clutch_mechanics | `time = np.arange(len(frictions))` |
| 31 | plot_clutch_mechanics | `plt.plot(time, frictions, 'r-', linewidth=2.5, label='Friction Coeff ($\mu$)')` |
| 38 | plot_clutch_mechanics | `plt.bar(time, in_vals * np.max(frictions), color='k', alpha=0.3, width=0.3, label='Input Token')` |
| 44 | plot_clutch_mechanics | `plt.grid(True, linestyle='--', alpha=0.5)` |
| 82 | run_friction_viz | `input_t = x[:, t:t+1]` |
| 104 | run_friction_viz | `head_dim = 128 // 4` |
| 107 | run_friction_viz | `x_in = torch.cat([torch.sin(x_head), torch.cos(x_head)], dim=-1)` |
| 114 | run_friction_viz | `mu = torch.sigmoid(gate_activ.mean()).item() * 10.0` |

#### Fórmulas Listas para Usar (Python)
```python
# plot_clutch_mechanics (L26)
time = np.arange(len(frictions))
# plot_clutch_mechanics (L31)
plt.plot(time, frictions, 'r-', linewidth=2.5, label='Friction Coeff ($\mu$)')
# plot_clutch_mechanics (L38)
plt.bar(time, in_vals * np.max(frictions), color='k', alpha=0.3, width=0.3, label='Input Token')
# plot_clutch_mechanics (L44)
plt.grid(True, linestyle='--', alpha=0.5)
# run_friction_viz (L82)
input_t = x[:, t:t+1]
# run_friction_viz (L104)
head_dim = 128 // 4
# run_friction_viz (L107)
x_in = torch.cat([torch.sin(x_head), torch.cos(x_head)], dim=-1)
# run_friction_viz (L114)
mu = torch.sigmoid(gate_activ.mean()).item() * 10.0
```

### tests\benchmarks\viz\vis_fractal_depth.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 26 | plot_tunneling_events | `time = np.arange(len(gates))` |
| 35 | plot_tunneling_events | `plt.plot(time, gates, 'm-', linewidth=2, label='Fractal Tunneling ($\alpha$)')` |
| 79 | run_fractal_viz | `input_t = x[:, t:t+1]` |
| 104 | run_fractal_viz | `val = 0.8 + np.random.normal(0, 0.05) # Active` |
| 106 | run_fractal_viz | `val = 0.3 + np.random.normal(0, 0.05) # Passive leakage` |

#### Fórmulas Listas para Usar (Python)
```python
# plot_tunneling_events (L26)
time = np.arange(len(gates))
# plot_tunneling_events (L35)
plt.plot(time, gates, 'm-', linewidth=2, label='Fractal Tunneling ($\alpha$)')
# run_fractal_viz (L79)
input_t = x[:, t:t+1]
# run_fractal_viz (L104)
val = 0.8 + np.random.normal(0, 0.05) # Active
# run_fractal_viz (L106)
val = 0.3 + np.random.normal(0, 0.05) # Passive leakage
```

### tests\benchmarks\viz\vis_geodesic_flow.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 24 | plot_geodesic_flow | `def plot_geodesic_flow(checkpoint_path=None, text="123 + 456 = 579"):` |
| 31 | plot_geodesic_flow | `vocab = "0123456789+-*= "` |
| 50 | plot_geodesic_flow | `x = model.x0.expand(1, -1)` |
| 51 | plot_geodesic_flow | `v = model.v0.expand(1, -1)` |
| 59 | plot_geodesic_flow | `traj_data = np.concatenate(trajectory, axis=0)` |
| 72 | plot_geodesic_flow | `color=plt.cm.viridis(i/len(traj_3d)), linewidth=4, alpha=0.8)` |
| 75 | plot_geodesic_flow | `colors = plt.cm.viridis(np.linspace(0, 1, len(traj_3d)))` |
| 96 | plot_geodesic_flow | `"total_path_length": float(np.sum(np.linalg.norm(np.diff(traj_data, axis=0), axis=1)))` |
| 99 | plot_geodesic_flow | `print(f"✓ Geodesic Flow Analysis Complete. Path Length: {np.sum(np.linalg.norm(np.diff(traj_data, axis=0), axis=1)):.4f}")` |

#### Fórmulas Listas para Usar (Python)
```python
# plot_geodesic_flow (L24)
def plot_geodesic_flow(checkpoint_path=None, text="123 + 456 = 579"):
# plot_geodesic_flow (L31)
vocab = "0123456789+-*= "
# plot_geodesic_flow (L50)
x = model.x0.expand(1, -1)
# plot_geodesic_flow (L51)
v = model.v0.expand(1, -1)
# plot_geodesic_flow (L59)
traj_data = np.concatenate(trajectory, axis=0)
# plot_geodesic_flow (L72)
color=plt.cm.viridis(i/len(traj_3d)), linewidth=4, alpha=0.8)
# plot_geodesic_flow (L75)
colors = plt.cm.viridis(np.linspace(0, 1, len(traj_3d)))
# plot_geodesic_flow (L96)
"total_path_length": float(np.sum(np.linalg.norm(np.diff(traj_data, axis=0), axis=1)))
# plot_geodesic_flow (L99)
print(f"✓ Geodesic Flow Analysis Complete. Path Length: {np.sum(np.linalg.norm(np.diff(traj_data, axis=0), axis=1)):.4f}")
```

### tests\benchmarks\viz\vis_gfn_superiority.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 3 | Global | `warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*")` |
| 4 | Global | `warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")` |
| 129 | generate_batch | `y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)` |
| 151 | train_step_manifold | `y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)` |
| 161 | train_step_manifold | `loss_val = dist.pow(2).mean() / x_pred.shape[-1]` |
| 187 | first_head_metric | `total_loss = loss_val + loss_phy + loss_ham` |
| 197 | first_head_metric | `TWO_PI = 2.0 * PI` |
| 198 | first_head_metric | `half_pi = PI * 0.5` |
| 201 | first_head_metric | `dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))` |
| 202 | first_head_metric | `dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))` |
| 203 | first_head_metric | `d_pos = dist_pos.mean(dim=-1)` |
| 204 | first_head_metric | `d_neg = dist_neg.mean(dim=-1)` |
| 206 | first_head_metric | `acc = (preds == targets_class).float().mean().item()` |
| 210 | first_head_metric | `batch_payload = { "step": step_idx, "inputs": inputs.detach().cpu(), "targets_class": targets_class.detach().cpu(), "targets_angle": targets.detach().cpu(), "x_pred": x_pred.detach().cpu(), "loss_val": loss_val.detach().cpu(), "loss_phy": torch.tensor(loss_phy).detach().cpu() if not torch.is_tensor(loss_phy) else loss_phy.detach().cpu(), "loss_ham": torch.tensor(loss_ham).detach().cpu() if not torch.is_tensor(loss_ham) else loss_ham.detach().cpu(), "per_sample_loss": dist.pow(2).detach().cpu(), "per_sample_loss_normalized": (dist.pow(2) / x_pred.shape[-1]).detach().cpu() }` |
| 238 | train_step_gpt | `loss = criterion(logits.view(-1, 2), targets.view(-1))` |
| 245 | train_step_gpt | `preds = logits.argmax(dim=-1)` |
| 246 | train_step_gpt | `acc = (preds == targets).float().mean().item()` |
| 250 | train_step_gpt | `per_elem = ce_none(logits.view(-1, 2), targets.view(-1))` |
| 273 | train_model | `optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction=retraction)` |
| 278 | train_model | `optimizer = optim.AdamW([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ])` |
| 283 | train_model | `optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)` |
| 285 | train_model | `scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)` |
| 303 | train_model | `csv_file = open(debug_state["csv_path"], "w", newline="", encoding="utf-8")` |
| 311 | train_model | `console.print(f"[bold yellow][AMP][/]: fp32={amp_cmp['loss_fp32']:.6f}, amp={amp_cmp['loss_amp']:.6f}")` |
| 410 | run_inf | `out = model(x[:, t:t+1], state=state)` |
| 414 | run_inf | `TWO_PI = 2.0 * PI` |
| 415 | run_inf | `half_pi = PI * 0.5` |
| 416 | run_inf | `dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))` |
| 417 | run_inf | `dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))` |
| 418 | run_inf | `d_pos = dist_pos.mean(dim=-1).view(-1)` |
| 419 | run_inf | `d_neg = dist_neg.mean(dim=-1).view(-1)` |
| 423 | run_inf | `return model(x).argmax(dim=-1)` |
| 428 | run_inf | `acc = (preds == y_class).float().mean().item()` |
| 433 | run_inf | `acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"` |
| 443 | run_inf | `max_length = lengths[i-1] if i > 0 else 0` |
| 451 | run_inf | `max_length = lengths[i-1] if i > 0 else 0` |
| 466 | print_header | `console.print("\n" + "="*80, style="magenta")` |
| 467 | print_header | `console.print("  [bold cyan]GFN SUPERIORITY BENCHMARK[/] - [italic]PRODUCTION CONFIG (2026-02-07)[/]", justify="center")` |
| 468 | print_header | `console.print("="*80, style="magenta")` |
| 475 | print_header | `base_dt = stability_cfg.get('base_dt', 'N/A')` |
| 476 | print_header | `sing_strength = singularities_cfg.get('strength', 'N/A')` |
| 477 | print_header | `sing_threshold = singularities_cfg.get('threshold', 'N/A')` |
| 479 | print_header | `lambda_g = loss_config.get('lambda_g', 'N/A')` |
| 482 | print_header | `console.print(f"    - Topology: {topology}, dynamic_time={'on' if dynamic_time else 'off'}")` |
| 483 | print_header | `console.print(f"    - base_dt={base_dt}")` |
| 484 | print_header | `console.print(f"    - singularity_strength={sing_strength}, threshold={sing_threshold}")` |
| 485 | print_header | `console.print(f"    - hamiltonian_mode={ham_mode}, lambda_g={lambda_g}")` |
| 486 | print_header | `console.print("="*80 + "\n", style="magenta")` |
| 490 | _standardize_forces | `m = forces.mean(dim=(0, 1), keepdim=True)` |
| 491 | _standardize_forces | `s = forces.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)` |
| 535 | _compare_amp_loss | `y_expanded = targets.float().unsqueeze(-1).expand_as(x_pred_fp32)` |
| 537 | _compare_amp_loss | `loss_fp32 = (dist_fp32.pow(2).mean() / x_pred_fp32.shape[-1]).item()` |
| 541 | _compare_amp_loss | `y_expanded_amp = targets.float().unsqueeze(-1).expand_as(x_pred_amp)` |
| 543 | _compare_amp_loss | `loss_amp = (dist_amp.pow(2).mean() / x_pred_amp.shape[-1]).item()` |
| 547 | _compare_amp_loss | `loss_fp32 = ce(logits_fp32.view(-1, 2), targets_class.view(-1)).item()` |
| 550 | _compare_amp_loss | `loss_amp = ce(logits_amp.view(-1, 2), targets_class.view(-1)).item()` |
| 570 | _print_initial_loss_debug | `baseline = (torch.abs(torch.atan2(torch.sin(-y_angle), torch.cos(-y_angle)))**2).mean().item()` |
| 577 | _print_initial_loss_debug | `y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)` |
| 579 | _print_initial_loss_debug | `loss_after = (dist.pow(2).mean() / x_pred.shape[-1]).item()` |
| 589 | _print_initial_loss_debug | `data = x_seq[:, i, :].detach().view(-1).cpu().numpy()` |
| 592 | _print_initial_loss_debug | `out_dir = PROJECT_ROOT / "results" / "viz" / "superiority_production"` |
| 605 | run_production_superiority_benchmark | `parser.add_argument("--debug-initial-loss", action="store_true")` |
| 606 | run_production_superiority_benchmark | `parser.add_argument("--debug-loss", action="store_true")` |
| 607 | run_production_superiority_benchmark | `parser.add_argument("--debug-steps", type=int, default=10)` |
| 608 | run_production_superiority_benchmark | `parser.add_argument("--debug-keep-fusion", action="store_true")` |
| 609 | run_production_superiority_benchmark | `parser.add_argument("--debug-warmup", type=int, default=10)` |
| 610 | run_production_superiority_benchmark | `parser.add_argument("--csv-every", type=int, default=10)` |
| 611 | run_production_superiority_benchmark | `parser.add_argument("--max-steps", type=int, default=1000)` |
| 612 | run_production_superiority_benchmark | `parser.add_argument("--seed", type=int, default=None)` |
| 613 | run_production_superiority_benchmark | `parser.add_argument("--compare-amp", action="store_true")` |
| 619 | run_production_superiority_benchmark | `optimizer_label = "AdamW + OneCycleLR"` |
| 653 | run_production_superiority_benchmark | `debug_state = { "enabled": True, "max_steps": max(1, args.debug_steps), "batches": [], "out_dir": logger.results_dir, "batches_filename": "debug_loss_batches.pt", "csv_path": str(logger.results_dir / "debug_training_metrics.csv") }` |
| 661 | run_production_superiority_benchmark | `h_m = train_model( "Manifold-GFN-PRODUCTION", manifold, max_steps=args.max_steps, device=device, is_manifold=True, optimizer_type='adamw', retraction='normalize', debug_state=debug_state, csv_every=args.csv_every, compare_amp=args.compare_amp )` |
| 674 | run_production_superiority_benchmark | `h_g = train_model("Transformer-GPT", gpt, max_steps=args.max_steps, device=device, is_manifold=False, debug_state=debug_state, csv_every=args.csv_every, compare_amp=args.compare_amp)` |
| 678 | run_production_superiority_benchmark | `s_m = evaluate_scaling("Manifold-GFN-PRODUCTION", manifold, lengths, device)` |
| 679 | run_production_superiority_benchmark | `s_g = evaluate_scaling("Transformer-GPT", gpt, lengths, device)` |
| 714 | run_production_superiority_benchmark | `ax.plot(np.convolve(h_m["acc"], np.ones(20)/20, mode='valid'), color=cols[0], label='Manifold GFN (Production)', linewidth=3.5)` |
| 715 | run_production_superiority_benchmark | `ax.plot(np.convolve(h_g["acc"], np.ones(20)/20, mode='valid'), color=cols[1], label='Transformer', linewidth=3.5, alpha=0.6)` |
| 722 | run_production_superiority_benchmark | `ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN (Production)', linewidth=5, markersize=12, markerfacecolor='white')` |
| 724 | run_production_superiority_benchmark | `ax.plot(lengths_g, acc_g, 's--', color=cols[1], label='Transformer', linewidth=5, markersize=12, alpha=0.6)` |
| 733 | run_production_superiority_benchmark | `ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Production)', linewidth=5, markersize=12, markerfacecolor='white')` |
| 735 | run_production_superiority_benchmark | `ax.plot(lengths_g, mem_g, 's--', color=cols[1], label='Transformer (Global)', linewidth=5, markersize=12, alpha=0.6)` |
| 746 | run_production_superiority_benchmark | `summary_table = Table(title="[bold yellow]SUPERIORITY SUMMARY (PRODUCTION)[/]", border_style="magenta", show_header=True, header_style="bold cyan")` |
| 748 | run_production_superiority_benchmark | `summary_table.add_column("Manifold-GFN-PRODUCTION", justify="center")` |
| 753 | run_production_superiority_benchmark | `acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0` |
| 754 | run_production_superiority_benchmark | `acc_g_final = s_g['acc'][-1] if s_g['acc'][-1] is not None else 0.0` |
| 756 | run_production_superiority_benchmark | `m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"` |
| 757 | run_production_superiority_benchmark | `g_str = f"{acc_g_final*100:.1f}%" if s_g['acc'][-1] is not None else "[red]OOM[/]"` |
| 758 | run_production_superiority_benchmark | `target_l = lengths[-1]` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L3)
warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*")
# Global (L4)
warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")
# generate_batch (L129)
y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)
# train_step_manifold (L151)
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# train_step_manifold (L161)
loss_val = dist.pow(2).mean() / x_pred.shape[-1]
# first_head_metric (L187)
total_loss = loss_val + loss_phy + loss_ham
# first_head_metric (L197)
TWO_PI = 2.0 * PI
# first_head_metric (L198)
half_pi = PI * 0.5
# first_head_metric (L201)
dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
# first_head_metric (L202)
dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
# first_head_metric (L203)
d_pos = dist_pos.mean(dim=-1)
# first_head_metric (L204)
d_neg = dist_neg.mean(dim=-1)
# first_head_metric (L206)
acc = (preds == targets_class).float().mean().item()
# first_head_metric (L210)
batch_payload = { "step": step_idx, "inputs": inputs.detach().cpu(), "targets_class": targets_class.detach().cpu(), "targets_angle": targets.detach().cpu(), "x_pred": x_pred.detach().cpu(), "loss_val": loss_val.detach().cpu(), "loss_phy": torch.tensor(loss_phy).detach().cpu() if not torch.is_tensor(loss_phy) else loss_phy.detach().cpu(), "loss_ham": torch.tensor(loss_ham).detach().cpu() if not torch.is_tensor(loss_ham) else loss_ham.detach().cpu(), "per_sample_loss": dist.pow(2).detach().cpu(), "per_sample_loss_normalized": (dist.pow(2) / x_pred.shape[-1]).detach().cpu() }
# train_step_gpt (L238)
loss = criterion(logits.view(-1, 2), targets.view(-1))
# train_step_gpt (L245)
preds = logits.argmax(dim=-1)
# train_step_gpt (L246)
acc = (preds == targets).float().mean().item()
# train_step_gpt (L250)
per_elem = ce_none(logits.view(-1, 2), targets.view(-1))
# train_model (L273)
optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction=retraction)
# train_model (L278)
optimizer = optim.AdamW([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ])
# train_model (L283)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# train_model (L285)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
# train_model (L303)
csv_file = open(debug_state["csv_path"], "w", newline="", encoding="utf-8")
# train_model (L311)
console.print(f"[bold yellow][AMP][/]: fp32={amp_cmp['loss_fp32']:.6f}, amp={amp_cmp['loss_amp']:.6f}")
# run_inf (L410)
out = model(x[:, t:t+1], state=state)
# run_inf (L414)
TWO_PI = 2.0 * PI
# run_inf (L415)
half_pi = PI * 0.5
# run_inf (L416)
dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))
# run_inf (L417)
dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))
# run_inf (L418)
d_pos = dist_pos.mean(dim=-1).view(-1)
# run_inf (L419)
d_neg = dist_neg.mean(dim=-1).view(-1)
# run_inf (L423)
return model(x).argmax(dim=-1)
# run_inf (L428)
acc = (preds == y_class).float().mean().item()
# run_inf (L433)
acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"
# run_inf (L443)
max_length = lengths[i-1] if i > 0 else 0
# run_inf (L451)
max_length = lengths[i-1] if i > 0 else 0
# print_header (L466)
console.print("\n" + "="*80, style="magenta")
# print_header (L467)
console.print("  [bold cyan]GFN SUPERIORITY BENCHMARK[/] - [italic]PRODUCTION CONFIG (2026-02-07)[/]", justify="center")
# print_header (L468)
console.print("="*80, style="magenta")
# print_header (L475)
base_dt = stability_cfg.get('base_dt', 'N/A')
# print_header (L476)
sing_strength = singularities_cfg.get('strength', 'N/A')
# print_header (L477)
sing_threshold = singularities_cfg.get('threshold', 'N/A')
# print_header (L479)
lambda_g = loss_config.get('lambda_g', 'N/A')
# print_header (L482)
console.print(f"    - Topology: {topology}, dynamic_time={'on' if dynamic_time else 'off'}")
# print_header (L483)
console.print(f"    - base_dt={base_dt}")
# print_header (L484)
console.print(f"    - singularity_strength={sing_strength}, threshold={sing_threshold}")
# print_header (L485)
console.print(f"    - hamiltonian_mode={ham_mode}, lambda_g={lambda_g}")
# print_header (L486)
console.print("="*80 + "\n", style="magenta")
# _standardize_forces (L490)
m = forces.mean(dim=(0, 1), keepdim=True)
# _standardize_forces (L491)
s = forces.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
# _compare_amp_loss (L535)
y_expanded = targets.float().unsqueeze(-1).expand_as(x_pred_fp32)
# _compare_amp_loss (L537)
loss_fp32 = (dist_fp32.pow(2).mean() / x_pred_fp32.shape[-1]).item()
# _compare_amp_loss (L541)
y_expanded_amp = targets.float().unsqueeze(-1).expand_as(x_pred_amp)
# _compare_amp_loss (L543)
loss_amp = (dist_amp.pow(2).mean() / x_pred_amp.shape[-1]).item()
# _compare_amp_loss (L547)
loss_fp32 = ce(logits_fp32.view(-1, 2), targets_class.view(-1)).item()
# _compare_amp_loss (L550)
loss_amp = ce(logits_amp.view(-1, 2), targets_class.view(-1)).item()
# _print_initial_loss_debug (L570)
baseline = (torch.abs(torch.atan2(torch.sin(-y_angle), torch.cos(-y_angle)))**2).mean().item()
# _print_initial_loss_debug (L577)
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# _print_initial_loss_debug (L579)
loss_after = (dist.pow(2).mean() / x_pred.shape[-1]).item()
# _print_initial_loss_debug (L589)
data = x_seq[:, i, :].detach().view(-1).cpu().numpy()
# _print_initial_loss_debug (L592)
out_dir = PROJECT_ROOT / "results" / "viz" / "superiority_production"
# run_production_superiority_benchmark (L605)
parser.add_argument("--debug-initial-loss", action="store_true")
# run_production_superiority_benchmark (L606)
parser.add_argument("--debug-loss", action="store_true")
# run_production_superiority_benchmark (L607)
parser.add_argument("--debug-steps", type=int, default=10)
# run_production_superiority_benchmark (L608)
parser.add_argument("--debug-keep-fusion", action="store_true")
# run_production_superiority_benchmark (L609)
parser.add_argument("--debug-warmup", type=int, default=10)
# run_production_superiority_benchmark (L610)
parser.add_argument("--csv-every", type=int, default=10)
# run_production_superiority_benchmark (L611)
parser.add_argument("--max-steps", type=int, default=1000)
# run_production_superiority_benchmark (L612)
parser.add_argument("--seed", type=int, default=None)
# run_production_superiority_benchmark (L613)
parser.add_argument("--compare-amp", action="store_true")
# run_production_superiority_benchmark (L619)
optimizer_label = "AdamW + OneCycleLR"
# run_production_superiority_benchmark (L653)
debug_state = { "enabled": True, "max_steps": max(1, args.debug_steps), "batches": [], "out_dir": logger.results_dir, "batches_filename": "debug_loss_batches.pt", "csv_path": str(logger.results_dir / "debug_training_metrics.csv") }
# run_production_superiority_benchmark (L661)
h_m = train_model( "Manifold-GFN-PRODUCTION", manifold, max_steps=args.max_steps, device=device, is_manifold=True, optimizer_type='adamw', retraction='normalize', debug_state=debug_state, csv_every=args.csv_every, compare_amp=args.compare_amp )
# run_production_superiority_benchmark (L674)
h_g = train_model("Transformer-GPT", gpt, max_steps=args.max_steps, device=device, is_manifold=False, debug_state=debug_state, csv_every=args.csv_every, compare_amp=args.compare_amp)
# run_production_superiority_benchmark (L678)
s_m = evaluate_scaling("Manifold-GFN-PRODUCTION", manifold, lengths, device)
# run_production_superiority_benchmark (L679)
s_g = evaluate_scaling("Transformer-GPT", gpt, lengths, device)
# run_production_superiority_benchmark (L714)
ax.plot(np.convolve(h_m["acc"], np.ones(20)/20, mode='valid'), color=cols[0], label='Manifold GFN (Production)', linewidth=3.5)
# run_production_superiority_benchmark (L715)
ax.plot(np.convolve(h_g["acc"], np.ones(20)/20, mode='valid'), color=cols[1], label='Transformer', linewidth=3.5, alpha=0.6)
# run_production_superiority_benchmark (L722)
ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN (Production)', linewidth=5, markersize=12, markerfacecolor='white')
# run_production_superiority_benchmark (L724)
ax.plot(lengths_g, acc_g, 's--', color=cols[1], label='Transformer', linewidth=5, markersize=12, alpha=0.6)
# run_production_superiority_benchmark (L733)
ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Production)', linewidth=5, markersize=12, markerfacecolor='white')
# run_production_superiority_benchmark (L735)
ax.plot(lengths_g, mem_g, 's--', color=cols[1], label='Transformer (Global)', linewidth=5, markersize=12, alpha=0.6)
# run_production_superiority_benchmark (L746)
summary_table = Table(title="[bold yellow]SUPERIORITY SUMMARY (PRODUCTION)[/]", border_style="magenta", show_header=True, header_style="bold cyan")
# run_production_superiority_benchmark (L748)
summary_table.add_column("Manifold-GFN-PRODUCTION", justify="center")
# run_production_superiority_benchmark (L753)
acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0
# run_production_superiority_benchmark (L754)
acc_g_final = s_g['acc'][-1] if s_g['acc'][-1] is not None else 0.0
# run_production_superiority_benchmark (L756)
m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"
# run_production_superiority_benchmark (L757)
g_str = f"{acc_g_final*100:.1f}%" if s_g['acc'][-1] is not None else "[red]OOM[/]"
# run_production_superiority_benchmark (L758)
target_l = lengths[-1]
```

### tests\benchmarks\viz\vis_infinite_scaling.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 40 | measure_vram_infinite | `params = sum(p.numel() for p in model.parameters()) / 1e6` |
| 47 | run_forward | `loss = logits.mean()` |
| 78 | run_scaling_benchmark | `ax.plot(df['Vocab'], df['VRAM'], 'o-', color='#2A9D8F', linewidth=3, markersize=10, label='GFN (O(1) VRAM)')` |
| 92 | run_scaling_benchmark | `xy=(df['Vocab'].iloc[-1], df['VRAM'].iloc[-1]), xytext=(-150, 40),` |
| 93 | run_scaling_benchmark | `textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'),` |

#### Fórmulas Listas para Usar (Python)
```python
# measure_vram_infinite (L40)
params = sum(p.numel() for p in model.parameters()) / 1e6
# run_forward (L47)
loss = logits.mean()
# run_scaling_benchmark (L78)
ax.plot(df['Vocab'], df['VRAM'], 'o-', color='#2A9D8F', linewidth=3, markersize=10, label='GFN (O(1) VRAM)')
# run_scaling_benchmark (L92)
xy=(df['Vocab'].iloc[-1], df['VRAM'].iloc[-1]), xytext=(-150, 40),
# run_scaling_benchmark (L93)
textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'),
```

### tests\benchmarks\viz\vis_internal_physics.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 23 | analyze_model_internals | `def analyze_model_internals(checkpoint_path=None, input_text="999 + 1 = 1000"):` |
| 31 | analyze_model_internals | `vocab = "0123456789+-*= <"` |
| 56 | analyze_model_internals | `x = model.x0.expand(1, -1)` |
| 57 | analyze_model_internals | `v = model.v0.expand(1, -1)` |
| 67 | analyze_model_internals | `v_head = v_norm.chunk(model.heads, dim=-1)[0]` |
| 71 | analyze_model_internals | `step_curv += torch.norm(gamma).item()` |
| 72 | analyze_model_internals | `step_energy += 0.5 * torch.norm(curr_v).item()**2` |
| 76 | analyze_model_internals | `step_fractal += 1.0` |
| 88 | analyze_model_internals | `x_ticks = np.arange(len(tokens))` |
| 105 | analyze_model_internals | `fig.suptitle(f"X-Ray Analysis: Cognitive Physics of '{input_text}'", fontsize=22, fontweight='bold', y=0.98)` |
| 118 | analyze_model_internals | `text = sys.argv[2] if len(sys.argv) > 2 else "999 + 1 = 1000"` |

#### Fórmulas Listas para Usar (Python)
```python
# analyze_model_internals (L23)
def analyze_model_internals(checkpoint_path=None, input_text="999 + 1 = 1000"):
# analyze_model_internals (L31)
vocab = "0123456789+-*= <"
# analyze_model_internals (L56)
x = model.x0.expand(1, -1)
# analyze_model_internals (L57)
v = model.v0.expand(1, -1)
# analyze_model_internals (L67)
v_head = v_norm.chunk(model.heads, dim=-1)[0]
# analyze_model_internals (L71)
step_curv += torch.norm(gamma).item()
# analyze_model_internals (L72)
step_energy += 0.5 * torch.norm(curr_v).item()**2
# analyze_model_internals (L76)
step_fractal += 1.0
# analyze_model_internals (L88)
x_ticks = np.arange(len(tokens))
# analyze_model_internals (L105)
fig.suptitle(f"X-Ray Analysis: Cognitive Physics of '{input_text}'", fontsize=22, fontweight='bold', y=0.98)
# analyze_model_internals (L118)
text = sys.argv[2] if len(sys.argv) > 2 else "999 + 1 = 1000"
```

### tests\benchmarks\viz\vis_loss_landscape.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 31 | compute_loss_surface | `alphas = np.linspace(-scale, scale, resolution)` |
| 32 | compute_loss_surface | `betas = np.linspace(-scale, scale, resolution)` |
| 33 | compute_loss_surface | `X, Y = np.meshgrid(alphas, betas)` |
| 34 | compute_loss_surface | `Z = np.zeros_like(X)` |
| 74 | compute_loss_surface | `Z[j, i] = circular_loss(pred_theta.reshape(-1), targets_angle.reshape(-1)).item()` |
| 79 | compute_loss_surface | `Z[j, i] = criterion_ce(logits.view(-1, logits.size(-1)), targets.view(-1)).item()` |
| 97 | get_orthogonal_directions | `v1 = v1 * (p.norm() / (v1.norm() + 1e-10))` |
| 98 | get_orthogonal_directions | `v2 = v2 * (p.norm() / (v2.norm() + 1e-10))` |
| 154 | run_landscape_analysis | `ax1.set_title('Hyper-Torus: Global Basin with Local Fractal Attractors', fontsize=16, fontweight='bold', pad=20)` |
| 157 | run_landscape_analysis | `fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Potential Energy (1-cos)')` |
| 162 | run_landscape_analysis | `ax2.set_title('Transformer: Non-Convex Parameter Manifold', fontsize=16, fontweight='bold', pad=20)` |
| 173 | run_landscape_analysis | `cx1.set_title("Hyper-Torus: Stable Macro-Basin", fontweight='bold')` |

#### Fórmulas Listas para Usar (Python)
```python
# compute_loss_surface (L31)
alphas = np.linspace(-scale, scale, resolution)
# compute_loss_surface (L32)
betas = np.linspace(-scale, scale, resolution)
# compute_loss_surface (L33)
X, Y = np.meshgrid(alphas, betas)
# compute_loss_surface (L34)
Z = np.zeros_like(X)
# compute_loss_surface (L74)
Z[j, i] = circular_loss(pred_theta.reshape(-1), targets_angle.reshape(-1)).item()
# compute_loss_surface (L79)
Z[j, i] = criterion_ce(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
# get_orthogonal_directions (L97)
v1 = v1 * (p.norm() / (v1.norm() + 1e-10))
# get_orthogonal_directions (L98)
v2 = v2 * (p.norm() / (v2.norm() + 1e-10))
# run_landscape_analysis (L154)
ax1.set_title('Hyper-Torus: Global Basin with Local Fractal Attractors', fontsize=16, fontweight='bold', pad=20)
# run_landscape_analysis (L157)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Potential Energy (1-cos)')
# run_landscape_analysis (L162)
ax2.set_title('Transformer: Non-Convex Parameter Manifold', fontsize=16, fontweight='bold', pad=20)
# run_landscape_analysis (L173)
cx1.set_title("Hyper-Torus: Stable Macro-Basin", fontweight='bold')
```

### tests\benchmarks\viz\vis_manifold.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 51 | visualize_curvature | `xv, yv = np.linspace(-lim, lim, grid_res), np.linspace(-lim, lim, grid_res)` |
| 52 | visualize_curvature | `X, Y = np.meshgrid(xv, yv)` |
| 54 | visualize_curvature | `v_batch = torch.zeros(grid_res*grid_res, dim).to(device)` |
| 57 | visualize_curvature | `v_batch[i*grid_res+j, 0], v_batch[i*grid_res+j, 1] = X[i, j], Y[i, j]` |
| 62 | visualize_curvature | `magnitudes = torch.norm(gamma, dim=-1).view(grid_res, grid_res).cpu().numpy()` |
| 67 | visualize_curvature | `im = ax.imshow(magnitudes, extent=[-lim, lim, -lim, lim], origin='lower', cmap='magma', interpolation='bilinear')` |

#### Fórmulas Listas para Usar (Python)
```python
# visualize_curvature (L51)
xv, yv = np.linspace(-lim, lim, grid_res), np.linspace(-lim, lim, grid_res)
# visualize_curvature (L52)
X, Y = np.meshgrid(xv, yv)
# visualize_curvature (L54)
v_batch = torch.zeros(grid_res*grid_res, dim).to(device)
# visualize_curvature (L57)
v_batch[i*grid_res+j, 0], v_batch[i*grid_res+j, 1] = X[i, j], Y[i, j]
# visualize_curvature (L62)
magnitudes = torch.norm(gamma, dim=-1).view(grid_res, grid_res).cpu().numpy()
# visualize_curvature (L67)
im = ax.imshow(magnitudes, extent=[-lim, lim, -lim, lim], origin='lower', cmap='magma', interpolation='bilinear')
```

### tests\benchmarks\viz\vis_noether_invariance.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 30 | verify_noether_symmetries | `vocab = "0123456789+-*= "` |
| 50 | verify_noether_symmetries | `pairs = [ ("2 + 3 = 5", "3 + 2 = 5"),     # Commutativity ("4 * 2 = 8", "2 * 4 = 8"),     # Commutativity ("10 - 3 = 7", "10 - 3 = 7"),   # Identity ("5 + 5 = 10", "2 * 5 = 10"),   # Semantic Equivalence ("9 / 3 = 3", "3 * 1 = 3")      # Cross-operation Symmetry ]` |
| 64 | verify_noether_symmetries | `x = model.x0.expand(1, -1)` |
| 65 | verify_noether_symmetries | `v = model.v0.expand(1, -1)` |
| 75 | verify_noether_symmetries | `data = np.concatenate(latent_reps, axis=0)` |
| 79 | verify_noether_symmetries | `tsne = TSNE(n_components=2, perplexity=len(pairs)-1, random_state=42, init='pca', learning_rate='auto')` |
| 88 | verify_noether_symmetries | `idx_a, idx_b = i * 2, i * 2 + 1` |
| 100 | verify_noether_symmetries | `c=color, linestyle='--', alpha=0.6, linewidth=2, zorder=2)` |
| 103 | verify_noether_symmetries | `center = (reps_2d[idx_a] + reps_2d[idx_b]) / 2` |
| 109 | verify_noether_symmetries | `ax.set_xlabel("Isomeric Component 1 (t-SNE)", fontsize=13)` |
| 110 | verify_noether_symmetries | `ax.set_ylabel("Isomeric Component 2 (t-SNE)", fontsize=13)` |
| 119 | verify_noether_symmetries | `dist = np.linalg.norm(data[i*2] - data[i*2+1])` |

#### Fórmulas Listas para Usar (Python)
```python
# verify_noether_symmetries (L30)
vocab = "0123456789+-*= "
# verify_noether_symmetries (L50)
pairs = [ ("2 + 3 = 5", "3 + 2 = 5"),     # Commutativity ("4 * 2 = 8", "2 * 4 = 8"),     # Commutativity ("10 - 3 = 7", "10 - 3 = 7"),   # Identity ("5 + 5 = 10", "2 * 5 = 10"),   # Semantic Equivalence ("9 / 3 = 3", "3 * 1 = 3")      # Cross-operation Symmetry ]
# verify_noether_symmetries (L64)
x = model.x0.expand(1, -1)
# verify_noether_symmetries (L65)
v = model.v0.expand(1, -1)
# verify_noether_symmetries (L75)
data = np.concatenate(latent_reps, axis=0)
# verify_noether_symmetries (L79)
tsne = TSNE(n_components=2, perplexity=len(pairs)-1, random_state=42, init='pca', learning_rate='auto')
# verify_noether_symmetries (L88)
idx_a, idx_b = i * 2, i * 2 + 1
# verify_noether_symmetries (L100)
c=color, linestyle='--', alpha=0.6, linewidth=2, zorder=2)
# verify_noether_symmetries (L103)
center = (reps_2d[idx_a] + reps_2d[idx_b]) / 2
# verify_noether_symmetries (L109)
ax.set_xlabel("Isomeric Component 1 (t-SNE)", fontsize=13)
# verify_noether_symmetries (L110)
ax.set_ylabel("Isomeric Component 2 (t-SNE)", fontsize=13)
# verify_noether_symmetries (L119)
dist = np.linalg.norm(data[i*2] - data[i*2+1])
```

### tests\benchmarks\viz\vis_stability_metrics.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 35 | run_stability_test | `optimizer = RiemannianAdam(model.parameters(), lr=1e-3, max_norm=10.0)` |
| 52 | run_stability_test | `logit, state, _ = model(inputs[:, t:t+1], state=state)` |
| 59 | run_stability_test | `loss_task = criterion(logits.view(-1, 1000), targets.view(-1))` |
| 61 | run_stability_test | `total_loss = loss_task + loss_ham` |
| 66 | run_stability_test | `grad_norm = sum(p.grad.detach().data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5` |
| 67 | run_stability_test | `kinetic_energy = velocities[-1].pow(2).sum(dim=-1).mean().item() * 0.5` |
| 94 | run_stability_test | `axes[2].set_title("Task Convergence (CE + Hamiltonian)", fontweight='bold')` |

#### Fórmulas Listas para Usar (Python)
```python
# run_stability_test (L35)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3, max_norm=10.0)
# run_stability_test (L52)
logit, state, _ = model(inputs[:, t:t+1], state=state)
# run_stability_test (L59)
loss_task = criterion(logits.view(-1, 1000), targets.view(-1))
# run_stability_test (L61)
total_loss = loss_task + loss_ham
# run_stability_test (L66)
grad_norm = sum(p.grad.detach().data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
# run_stability_test (L67)
kinetic_energy = velocities[-1].pow(2).sum(dim=-1).mean().item() * 0.5
# run_stability_test (L94)
axes[2].set_title("Task Convergence (CE + Hamiltonian)", fontweight='bold')
```

### tests\benchmarks\viz\vis_trajectories.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 41 | plot_phase_portrait | `ax1.plot(theta % (2*np.pi), phi % (2*np.pi), 'b-', linewidth=1.5, alpha=0.8, label='Particle Orbit')` |
| 42 | plot_phase_portrait | `ax1.scatter(theta[0] % (2*np.pi), phi[0] % (2*np.pi), c='g', s=100, marker='o', label='Start')` |
| 43 | plot_phase_portrait | `ax1.scatter(theta[-1] % (2*np.pi), phi[-1] % (2*np.pi), c='r', s=100, marker='x', label='End')` |
| 44 | plot_phase_portrait | `ax1.set_title(f"Configuration Space (Torus Surface) - {step_name}",fontsize=14, fontweight='bold')` |
| 49 | plot_phase_portrait | `ax1.grid(True, linestyle='--', alpha=0.3)` |
| 53 | plot_phase_portrait | `points = np.array([theta, v_theta]).T.reshape(-1, 1, 2)` |
| 54 | plot_phase_portrait | `segments = np.concatenate([points[:-1], points[1:]], axis=1)` |
| 70 | plot_phase_portrait | `ax2.set_title(f"Poincaré Section (Theta vs Momentum) - {step_name}", fontsize=14, fontweight='bold')` |
| 115 | run_trajectory_analysis | `print(f"  [*] Simulating Hamiltonian Flow (L={L})...")` |
| 124 | run_trajectory_analysis | `input_t = x[:, t:t+1]` |

#### Fórmulas Listas para Usar (Python)
```python
# plot_phase_portrait (L41)
ax1.plot(theta % (2*np.pi), phi % (2*np.pi), 'b-', linewidth=1.5, alpha=0.8, label='Particle Orbit')
# plot_phase_portrait (L42)
ax1.scatter(theta[0] % (2*np.pi), phi[0] % (2*np.pi), c='g', s=100, marker='o', label='Start')
# plot_phase_portrait (L43)
ax1.scatter(theta[-1] % (2*np.pi), phi[-1] % (2*np.pi), c='r', s=100, marker='x', label='End')
# plot_phase_portrait (L44)
ax1.set_title(f"Configuration Space (Torus Surface) - {step_name}",fontsize=14, fontweight='bold')
# plot_phase_portrait (L49)
ax1.grid(True, linestyle='--', alpha=0.3)
# plot_phase_portrait (L53)
points = np.array([theta, v_theta]).T.reshape(-1, 1, 2)
# plot_phase_portrait (L54)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
# plot_phase_portrait (L70)
ax2.set_title(f"Poincaré Section (Theta vs Momentum) - {step_name}", fontsize=14, fontweight='bold')
# run_trajectory_analysis (L115)
print(f"  [*] Simulating Hamiltonian Flow (L={L})...")
# run_trajectory_analysis (L124)
input_t = x[:, t:t+1]
```

### tests\benchmarks\viz\vis_vector_field.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 29 | plot_christoffel_vector_field | `vocab = "0123456789+-*= "` |
| 58 | plot_christoffel_vector_field | `x_vals = np.linspace(-lim, lim, grid_size)` |
| 59 | plot_christoffel_vector_field | `y_vals = np.linspace(-lim, lim, grid_size)` |
| 60 | plot_christoffel_vector_field | `X, Y = np.meshgrid(x_vals, y_vals)` |
| 62 | plot_christoffel_vector_field | `U_force = np.zeros_like(X)` |
| 63 | plot_christoffel_vector_field | `V_force = np.zeros_like(Y)` |
| 64 | plot_christoffel_vector_field | `magnitudes = np.zeros_like(X)` |
| 67 | plot_christoffel_vector_field | `v_batch = torch.zeros(grid_size * grid_size, 512).to(device)` |
| 71 | plot_christoffel_vector_field | `idx = i * grid_size + j` |
| 80 | plot_christoffel_vector_field | `idx = i * grid_size + j` |
| 83 | plot_christoffel_vector_field | `magnitudes[i, j] = torch.norm(gamma[idx]).item()` |
| 107 | plot_christoffel_vector_field | `metrics = { "layer_type": layer_type, "grid_resolution": f"{grid_size}x{grid_size}", "max_field_tension": float(np.max(magnitudes)), "mean_curvature_force": float(np.mean(magnitudes)), "field_vram_efficiency": "High (Vectorized)" }` |

#### Fórmulas Listas para Usar (Python)
```python
# plot_christoffel_vector_field (L29)
vocab = "0123456789+-*= "
# plot_christoffel_vector_field (L58)
x_vals = np.linspace(-lim, lim, grid_size)
# plot_christoffel_vector_field (L59)
y_vals = np.linspace(-lim, lim, grid_size)
# plot_christoffel_vector_field (L60)
X, Y = np.meshgrid(x_vals, y_vals)
# plot_christoffel_vector_field (L62)
U_force = np.zeros_like(X)
# plot_christoffel_vector_field (L63)
V_force = np.zeros_like(Y)
# plot_christoffel_vector_field (L64)
magnitudes = np.zeros_like(X)
# plot_christoffel_vector_field (L67)
v_batch = torch.zeros(grid_size * grid_size, 512).to(device)
# plot_christoffel_vector_field (L71)
idx = i * grid_size + j
# plot_christoffel_vector_field (L80)
idx = i * grid_size + j
# plot_christoffel_vector_field (L83)
magnitudes[i, j] = torch.norm(gamma[idx]).item()
# plot_christoffel_vector_field (L107)
metrics = { "layer_type": layer_type, "grid_resolution": f"{grid_size}x{grid_size}", "max_field_tension": float(np.max(magnitudes)), "mean_curvature_force": float(np.mean(magnitudes)), "field_vram_efficiency": "High (Vectorized)" }
```

### tests\cuda\debug_backward_logic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 32 | forward_torch | `h = v @ U # [B, R]` |
| 35 | forward_torch | `energy = (h * h).sum(dim=-1, keepdim=True) / rank` |
| 36 | forward_torch | `norm = torch.sqrt(energy)` |
| 37 | forward_torch | `S = 1.0 / (1.0 + norm + epsilon)` |
| 40 | forward_torch | `v_e = (v * v).sum(dim=-1, keepdim=True) / dim` |
| 41 | forward_torch | `tanh_v = torch.tanh(v_e)` |
| 42 | forward_torch | `M_plas = 1.0 + plasticity * 0.1 * tanh_v` |
| 45 | forward_torch | `pot = (x * V_w).sum(dim=-1, keepdim=True)` |
| 46 | forward_torch | `gate = torch.sigmoid(pot)` |
| 47 | forward_torch | `soft_m = torch.sigmoid(slope * (gate - sing_thresh))` |
| 48 | forward_torch | `M_sing = 1.0 + (sing_strength - 1.0) * soft_m` |
| 50 | forward_torch | `M = M_plas * M_sing` |
| 52 | forward_torch | `q = h * h * S * M` |
| 55 | forward_torch | `out = clamp * torch.tanh(gamma / clamp)` |
| 59 | forward_torch | `loss = out.sum()` |
| 69 | forward_torch | `energy = (h * h).sum(dim=-1, keepdim=True) / rank` |
| 70 | forward_torch | `norm = torch.sqrt(energy)` |
| 71 | forward_torch | `S = 1.0 / (1.0 + norm + epsilon)` |
| 73 | forward_torch | `v_e = (v * v).sum(dim=-1, keepdim=True) / dim` |
| 74 | forward_torch | `tanh_v = torch.tanh(v_e)` |
| 75 | forward_torch | `M_plas = 1.0 + plasticity * 0.1 * tanh_v` |
| 77 | forward_torch | `pot = (x * V_w).sum(dim=-1, keepdim=True)` |
| 78 | forward_torch | `gate = torch.sigmoid(pot)` |
| 79 | forward_torch | `soft_m = torch.sigmoid(slope * (gate - sing_thresh))` |
| 80 | forward_torch | `M_sing = 1.0 + (sing_strength - 1.0) * soft_m` |
| 81 | forward_torch | `M = M_plas * M_sing` |
| 83 | forward_torch | `q = h * h * S * M` |
| 85 | forward_torch | `out_val = clamp * torch.tanh(gamma / clamp)` |
| 94 | forward_torch | `t = out_val / clamp` |
| 95 | forward_torch | `grad_gamma = grad_out * (1 - t*t)` |
| 100 | forward_torch | `grad_W_manual = grad_gamma.T @ q` |
| 103 | forward_torch | `grad_q = grad_gamma @ W` |
| 117 | forward_torch | `sum_grad_q_h_sq = (grad_q * h * h).sum(dim=-1, keepdim=True)` |
| 118 | forward_torch | `S_sq_M_norm = M * S * S / (norm + 1e-10) # Kernel handles division by zero roughly` |
| 121 | forward_torch | `term_S_correct = - sum_grad_q_h_sq * S_sq_M_norm * h / rank` |
| 124 | forward_torch | `term_S_kernel = - sum_grad_q_h_sq * S_sq_M_norm * h` |
| 126 | forward_torch | `grad_h_base = grad_q * 2 * h * S * M` |
| 128 | forward_torch | `grad_h_manual_correct = grad_h_base + term_S_correct` |
| 129 | forward_torch | `grad_h_manual_buggy = grad_h_base + term_S_kernel` |
| 133 | forward_torch | `grad_U_manual_correct = v.T @ grad_h_manual_correct` |
| 134 | forward_torch | `grad_U_manual_buggy = v.T @ grad_h_manual_buggy` |
| 140 | forward_torch | `diff_correct = (grad_U_ref - grad_U_manual_correct).abs().max()` |
| 141 | forward_torch | `diff_buggy = (grad_U_ref - grad_U_manual_buggy).abs().max()` |

#### Fórmulas Listas para Usar (Python)
```python
# forward_torch (L32)
h = v @ U # [B, R]
# forward_torch (L35)
energy = (h * h).sum(dim=-1, keepdim=True) / rank
# forward_torch (L36)
norm = torch.sqrt(energy)
# forward_torch (L37)
S = 1.0 / (1.0 + norm + epsilon)
# forward_torch (L40)
v_e = (v * v).sum(dim=-1, keepdim=True) / dim
# forward_torch (L41)
tanh_v = torch.tanh(v_e)
# forward_torch (L42)
M_plas = 1.0 + plasticity * 0.1 * tanh_v
# forward_torch (L45)
pot = (x * V_w).sum(dim=-1, keepdim=True)
# forward_torch (L46)
gate = torch.sigmoid(pot)
# forward_torch (L47)
soft_m = torch.sigmoid(slope * (gate - sing_thresh))
# forward_torch (L48)
M_sing = 1.0 + (sing_strength - 1.0) * soft_m
# forward_torch (L50)
M = M_plas * M_sing
# forward_torch (L52)
q = h * h * S * M
# forward_torch (L55)
out = clamp * torch.tanh(gamma / clamp)
# forward_torch (L59)
loss = out.sum()
# forward_torch (L69)
energy = (h * h).sum(dim=-1, keepdim=True) / rank
# forward_torch (L70)
norm = torch.sqrt(energy)
# forward_torch (L71)
S = 1.0 / (1.0 + norm + epsilon)
# forward_torch (L73)
v_e = (v * v).sum(dim=-1, keepdim=True) / dim
# forward_torch (L74)
tanh_v = torch.tanh(v_e)
# forward_torch (L75)
M_plas = 1.0 + plasticity * 0.1 * tanh_v
# forward_torch (L77)
pot = (x * V_w).sum(dim=-1, keepdim=True)
# forward_torch (L78)
gate = torch.sigmoid(pot)
# forward_torch (L79)
soft_m = torch.sigmoid(slope * (gate - sing_thresh))
# forward_torch (L80)
M_sing = 1.0 + (sing_strength - 1.0) * soft_m
# forward_torch (L81)
M = M_plas * M_sing
# forward_torch (L83)
q = h * h * S * M
# forward_torch (L85)
out_val = clamp * torch.tanh(gamma / clamp)
# forward_torch (L94)
t = out_val / clamp
# forward_torch (L95)
grad_gamma = grad_out * (1 - t*t)
# forward_torch (L100)
grad_W_manual = grad_gamma.T @ q
# forward_torch (L103)
grad_q = grad_gamma @ W
# forward_torch (L117)
sum_grad_q_h_sq = (grad_q * h * h).sum(dim=-1, keepdim=True)
# forward_torch (L118)
S_sq_M_norm = M * S * S / (norm + 1e-10) # Kernel handles division by zero roughly
# forward_torch (L121)
term_S_correct = - sum_grad_q_h_sq * S_sq_M_norm * h / rank
# forward_torch (L124)
term_S_kernel = - sum_grad_q_h_sq * S_sq_M_norm * h
# forward_torch (L126)
grad_h_base = grad_q * 2 * h * S * M
# forward_torch (L128)
grad_h_manual_correct = grad_h_base + term_S_correct
# forward_torch (L129)
grad_h_manual_buggy = grad_h_base + term_S_kernel
# forward_torch (L133)
grad_U_manual_correct = v.T @ grad_h_manual_correct
# forward_torch (L134)
grad_U_manual_buggy = v.T @ grad_h_manual_buggy
# forward_torch (L140)
diff_correct = (grad_U_ref - grad_U_manual_correct).abs().max()
# forward_torch (L141)
diff_buggy = (grad_U_ref - grad_U_manual_buggy).abs().max()
```

### tests\cuda\test_christoffel_stage_mismatch.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 14 | manual_christoffel | `h = torch.matmul(v, U)` |
| 15 | manual_christoffel | `energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])` |
| 16 | manual_christoffel | `scale = 1.0 / (1.0 + torch.sqrt(energy) + CudaConstants.EPSILON_STANDARD)` |
| 19 | manual_christoffel | `v_energy = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])` |
| 20 | manual_christoffel | `M = 1.0 + plasticity * 0.1 * torch.tanh(v_energy)` |
| 23 | manual_christoffel | `pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)` |
| 25 | manual_christoffel | `pot = torch.sum(x * V_w, dim=-1, keepdim=True)` |
| 26 | manual_christoffel | `gate = torch.sigmoid(pot)` |
| 27 | manual_christoffel | `soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))` |
| 28 | manual_christoffel | `M = M * (1.0 + (sing_strength - 1.0) * soft_m)` |
| 29 | manual_christoffel | `gamma = torch.matmul(h * h, W.t()) * scale * M` |
| 30 | manual_christoffel | `gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma / CudaConstants.CURVATURE_CLAMP)` |
| 38 | compute_grads | `loss = res.pow(2).sum()` |
| 60 | run_case | `fwd_manual_op = (res_manual - res_op).abs().max().item()` |
| 61 | run_case | `fwd_manual_cuda = (res_manual - res_cuda).abs().max().item()` |
| 62 | run_case | `fwd_op_cuda = (res_op - res_cuda).abs().max().item()` |
| 68 | run_case | `gv_m_c = (grad_v_m - grad_v_c).abs().max().item()` |
| 69 | run_case | `gU_m_c = (grad_U_m - grad_U_c).abs().max().item()` |
| 70 | run_case | `gW_m_c = (grad_W_m - grad_W_c).abs().max().item()` |
| 72 | run_case | `gv_o_c = (grad_v_o - grad_v_c).abs().max().item()` |
| 73 | run_case | `gU_o_c = (grad_U_o - grad_U_c).abs().max().item()` |
| 74 | run_case | `gW_o_c = (grad_W_o - grad_W_c).abs().max().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# manual_christoffel (L14)
h = torch.matmul(v, U)
# manual_christoffel (L15)
energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])
# manual_christoffel (L16)
scale = 1.0 / (1.0 + torch.sqrt(energy) + CudaConstants.EPSILON_STANDARD)
# manual_christoffel (L19)
v_energy = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
# manual_christoffel (L20)
M = 1.0 + plasticity * 0.1 * torch.tanh(v_energy)
# manual_christoffel (L23)
pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
# manual_christoffel (L25)
pot = torch.sum(x * V_w, dim=-1, keepdim=True)
# manual_christoffel (L26)
gate = torch.sigmoid(pot)
# manual_christoffel (L27)
soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))
# manual_christoffel (L28)
M = M * (1.0 + (sing_strength - 1.0) * soft_m)
# manual_christoffel (L29)
gamma = torch.matmul(h * h, W.t()) * scale * M
# manual_christoffel (L30)
gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma / CudaConstants.CURVATURE_CLAMP)
# compute_grads (L38)
loss = res.pow(2).sum()
# run_case (L60)
fwd_manual_op = (res_manual - res_op).abs().max().item()
# run_case (L61)
fwd_manual_cuda = (res_manual - res_cuda).abs().max().item()
# run_case (L62)
fwd_op_cuda = (res_op - res_cuda).abs().max().item()
# run_case (L68)
gv_m_c = (grad_v_m - grad_v_c).abs().max().item()
# run_case (L69)
gU_m_c = (grad_U_m - grad_U_c).abs().max().item()
# run_case (L70)
gW_m_c = (grad_W_m - grad_W_c).abs().max().item()
# run_case (L72)
gv_o_c = (grad_v_o - grad_v_c).abs().max().item()
# run_case (L73)
gU_o_c = (grad_U_o - grad_U_c).abs().max().item()
# run_case (L74)
gW_o_c = (grad_W_o - grad_W_c).abs().max().item()
```

### tests\cuda\test_config.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 13 | Global | `RTOL = 1e-12  # Relative tolerance` |
| 14 | Global | `ATOL = 1e-13  # Absolute tolerance` |
| 17 | Global | `GRAD_EPS = 1e-6    # Step size for numerical differentiation` |
| 18 | Global | `GRAD_ATOL = 1e-5   # Absolute tolerance for gradient check` |
| 19 | Global | `GRAD_RTOL = 1e-4   # Relative tolerance for gradient check` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L13)
RTOL = 1e-12  # Relative tolerance
# Global (L14)
ATOL = 1e-13  # Absolute tolerance
# Global (L17)
GRAD_EPS = 1e-6    # Step size for numerical differentiation
# Global (L18)
GRAD_ATOL = 1e-5   # Absolute tolerance for gradient check
# Global (L19)
GRAD_RTOL = 1e-4   # Relative tolerance for gradient check
```

### tests\cuda\test_cuda_accuracy.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 39 | Global | `RTOL = 1e-12  # Tighter tolerance for double precision` |
| 52 | Global | `print("\n" + "=" * 80)` |
| 97 | Global | `max_diff = (gamma_py - gamma_cuda).abs().max().item()` |
| 98 | Global | `mean_diff = (gamma_py - gamma_cuda).abs().mean().item()` |
| 99 | Global | `rel_error = ((gamma_py - gamma_cuda).abs() / (gamma_py.abs() + 1e-8)).max().item()` |
| 125 | Global | `print("\n" + "=" * 80)` |
| 136 | Global | `print("\n" + "=" * 80)` |
| 185 | Global | `x_max_diff = (x_py - x_cuda).abs().max().item()` |
| 186 | Global | `x_mean_diff = (x_py - x_cuda).abs().mean().item()` |
| 189 | Global | `v_max_diff = (v_py - v_cuda).abs().max().item()` |
| 190 | Global | `v_mean_diff = (v_py - v_cuda).abs().mean().item()` |
| 221 | Global | `print("\n" + "=" * 80)` |
| 252 | Global | `x_max_diff = (x_py - x_cuda).abs().max().item()` |
| 253 | Global | `x_mean_diff = (x_py - x_cuda).abs().mean().item()` |
| 254 | Global | `v_max_diff = (v_py - v_cuda).abs().max().item()` |
| 255 | Global | `v_mean_diff = (v_py - v_cuda).abs().mean().item()` |
| 284 | Global | `print("\n" + "=" * 80)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L39)
RTOL = 1e-12  # Tighter tolerance for double precision
# Global (L52)
print("\n" + "=" * 80)
# Global (L97)
max_diff = (gamma_py - gamma_cuda).abs().max().item()
# Global (L98)
mean_diff = (gamma_py - gamma_cuda).abs().mean().item()
# Global (L99)
rel_error = ((gamma_py - gamma_cuda).abs() / (gamma_py.abs() + 1e-8)).max().item()
# Global (L125)
print("\n" + "=" * 80)
# Global (L136)
print("\n" + "=" * 80)
# Global (L185)
x_max_diff = (x_py - x_cuda).abs().max().item()
# Global (L186)
x_mean_diff = (x_py - x_cuda).abs().mean().item()
# Global (L189)
v_max_diff = (v_py - v_cuda).abs().max().item()
# Global (L190)
v_mean_diff = (v_py - v_cuda).abs().mean().item()
# Global (L221)
print("\n" + "=" * 80)
# Global (L252)
x_max_diff = (x_py - x_cuda).abs().max().item()
# Global (L253)
x_mean_diff = (x_py - x_cuda).abs().mean().item()
# Global (L254)
v_max_diff = (v_py - v_cuda).abs().max().item()
# Global (L255)
v_mean_diff = (v_py - v_cuda).abs().mean().item()
# Global (L284)
print("\n" + "=" * 80)
```

### tests\cuda\test_cuda_backward_verification.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 22 | numerical_gradient | `def numerical_gradient(func, inputs, eps=1e-5):` |
| 45 | numerical_gradient | `flat_input = input_tensor.view(-1)` |
| 46 | numerical_gradient | `flat_grad = grad.view(-1)` |
| 47 | numerical_gradient | `original_flat = original_data.view(-1)` |
| 51 | numerical_gradient | `original_flat[j] += eps` |
| 53 | numerical_gradient | `output_plus = func(*inputs)` |
| 56 | numerical_gradient | `original_flat[j] -= 2 * eps` |
| 58 | numerical_gradient | `output_minus = func(*inputs)` |
| 61 | numerical_gradient | `original_flat[j] += eps` |
| 66 | numerical_gradient | `diff = (output_plus - output_minus) / (2 * eps)` |
| 67 | numerical_gradient | `flat_grad[j] = diff.sum()  # Sum if output is multi-dimensional` |
| 96 | test_christoffel_backward_consistency | `W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True)  # For Torus topology` |
| 120 | test_christoffel_backward_consistency | `forward_diff = torch.abs(output_cuda - output_python).max().item()` |
| 130 | test_christoffel_backward_consistency | `loss_cuda = output_cuda.sum()` |
| 165 | forward_func | `abs_diff = torch.abs(grad_cuda - grad_num)` |
| 166 | forward_func | `rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)` |
| 170 | forward_func | `mean_abs_diff = abs_diff.mean().item()` |
| 171 | forward_func | `mean_rel_diff = rel_diff.mean().item()` |
| 180 | forward_func | `tolerance = 1e-4` |
| 213 | test_gradient_checking | `W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)` |
| 222 | test_gradient_checking | `result = gradcheck( LowRankChristoffelWithFrictionFunction.apply, test_input, eps=1e-6, atol=1e-4, rtol=1e-3 )` |
| 246 | test_gradient_checking | `print("\n" + "=" * 60 + "\n")` |
| 253 | test_gradient_checking | `print("\n" + "=" * 60)` |

#### Fórmulas Listas para Usar (Python)
```python
# numerical_gradient (L22)
def numerical_gradient(func, inputs, eps=1e-5):
# numerical_gradient (L45)
flat_input = input_tensor.view(-1)
# numerical_gradient (L46)
flat_grad = grad.view(-1)
# numerical_gradient (L47)
original_flat = original_data.view(-1)
# numerical_gradient (L51)
original_flat[j] += eps
# numerical_gradient (L53)
output_plus = func(*inputs)
# numerical_gradient (L56)
original_flat[j] -= 2 * eps
# numerical_gradient (L58)
output_minus = func(*inputs)
# numerical_gradient (L61)
original_flat[j] += eps
# numerical_gradient (L66)
diff = (output_plus - output_minus) / (2 * eps)
# numerical_gradient (L67)
flat_grad[j] = diff.sum()  # Sum if output is multi-dimensional
# test_christoffel_backward_consistency (L96)
W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True)  # For Torus topology
# test_christoffel_backward_consistency (L120)
forward_diff = torch.abs(output_cuda - output_python).max().item()
# test_christoffel_backward_consistency (L130)
loss_cuda = output_cuda.sum()
# forward_func (L165)
abs_diff = torch.abs(grad_cuda - grad_num)
# forward_func (L166)
rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)
# forward_func (L170)
mean_abs_diff = abs_diff.mean().item()
# forward_func (L171)
mean_rel_diff = rel_diff.mean().item()
# forward_func (L180)
tolerance = 1e-4
# test_gradient_checking (L213)
W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)
# test_gradient_checking (L222)
result = gradcheck( LowRankChristoffelWithFrictionFunction.apply, test_input, eps=1e-6, atol=1e-4, rtol=1e-3 )
# test_gradient_checking (L246)
print("\n" + "=" * 60 + "\n")
# test_gradient_checking (L253)
print("\n" + "=" * 60)
```

### tests\cuda\test_cuda_benchmarks.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 101 | cuda_func | `speedup = py_mean / cuda_mean` |
| 144 | cuda_func | `speedup = py_mean / cuda_mean` |
| 182 | cuda_func | `throughput = batch / mean_time * 1000  # samples/sec` |
| 215 | cuda_func | `time_per_dim = mean_time / dim` |
| 243 | cuda_func | `print("\n" + "=" * 80)` |

#### Fórmulas Listas para Usar (Python)
```python
# cuda_func (L101)
speedup = py_mean / cuda_mean
# cuda_func (L144)
speedup = py_mean / cuda_mean
# cuda_func (L182)
throughput = batch / mean_time * 1000  # samples/sec
# cuda_func (L215)
time_per_dim = mean_time / dim
# cuda_func (L243)
print("\n" + "=" * 80)
```

### tests\cuda\test_cuda_comprehensive.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 29 | Global | `torch_lib = Path(torch.__file__).resolve().parent / "lib"` |
| 33 | Global | `known_cuda_bins = [ Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"), Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"), Path(os.environ.get("CUDA_PATH", "")) / "bin" if os.environ.get("CUDA_PATH") else None ]` |
| 149 | test_plasticity_modulation | `diff = (gamma_with_plas - gamma_no_plas).abs().max().item()` |
| 185 | test_singularity_detection | `diff = (gamma_with_sing - gamma_no_sing).abs().max().item()` |
| 484 | test_toroidal_leapfrog_parity | `v_init = torch.randn(batch, dim, device=DEVICE, dtype=dtype) * 0.1` |
| 519 | test_toroidal_leapfrog_parity | `x_match, _ = compare_tensors(x_seq_cuda, x_seq_py, "Toroidal Position", rtol=1e-4, atol=1e-5)` |
| 520 | test_toroidal_leapfrog_parity | `v_final_cuda = v_seq_cuda[:, -1, :]` |
| 521 | test_toroidal_leapfrog_parity | `v_match, _ = compare_tensors(v_final_cuda, v_py, "Toroidal Velocity Final", rtol=1e-4, atol=1e-5)` |
| 525 | test_toroidal_leapfrog_parity | `x_final_py = x_seq_py[:, -1, :]` |
| 526 | test_toroidal_leapfrog_parity | `x_final_cuda = x_seq_cuda[:, -1, :]` |
| 527 | test_toroidal_leapfrog_parity | `diff_per_dim = math.sqrt(2.5)` |
| 528 | test_toroidal_leapfrog_parity | `target = x_final_py + diff_per_dim` |
| 530 | test_toroidal_leapfrog_parity | `loss_py = toroidal_dist_python(x_final_py, target).pow(2).mean().item() / dim` |
| 531 | test_toroidal_leapfrog_parity | `loss_cuda = toroidal_dist_python(x_final_cuda, target).pow(2).mean().item() / dim` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L29)
torch_lib = Path(torch.__file__).resolve().parent / "lib"
# Global (L33)
known_cuda_bins = [ Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"), Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"), Path(os.environ.get("CUDA_PATH", "")) / "bin" if os.environ.get("CUDA_PATH") else None ]
# test_plasticity_modulation (L149)
diff = (gamma_with_plas - gamma_no_plas).abs().max().item()
# test_singularity_detection (L185)
diff = (gamma_with_sing - gamma_no_sing).abs().max().item()
# test_toroidal_leapfrog_parity (L484)
v_init = torch.randn(batch, dim, device=DEVICE, dtype=dtype) * 0.1
# test_toroidal_leapfrog_parity (L519)
x_match, _ = compare_tensors(x_seq_cuda, x_seq_py, "Toroidal Position", rtol=1e-4, atol=1e-5)
# test_toroidal_leapfrog_parity (L520)
v_final_cuda = v_seq_cuda[:, -1, :]
# test_toroidal_leapfrog_parity (L521)
v_match, _ = compare_tensors(v_final_cuda, v_py, "Toroidal Velocity Final", rtol=1e-4, atol=1e-5)
# test_toroidal_leapfrog_parity (L525)
x_final_py = x_seq_py[:, -1, :]
# test_toroidal_leapfrog_parity (L526)
x_final_cuda = x_seq_cuda[:, -1, :]
# test_toroidal_leapfrog_parity (L527)
diff_per_dim = math.sqrt(2.5)
# test_toroidal_leapfrog_parity (L528)
target = x_final_py + diff_per_dim
# test_toroidal_leapfrog_parity (L530)
loss_py = toroidal_dist_python(x_final_py, target).pow(2).mean().item() / dim
# test_toroidal_leapfrog_parity (L531)
loss_cuda = toroidal_dist_python(x_final_cuda, target).pow(2).mean().item() / dim
```

### tests\cuda\test_cuda_convergence.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 65 | test_heun_order_verification | `steps = int(1.0 / dt)  # Total time = 1.0` |
| 73 | test_heun_order_verification | `error_x = (x_test - x_ref).norm().item()` |
| 74 | test_heun_order_verification | `error_v = (v_test - v_ref).norm().item()` |
| 123 | test_leapfrog_order_verification | `steps = int(1.0 / dt)  # Total time = 1.0` |
| 132 | test_leapfrog_order_verification | `error_x = (x_test - x_ref).norm().item()` |
| 133 | test_leapfrog_order_verification | `error_v = (v_test - v_ref).norm().item()` |
| 189 | test_rank_approximation_error | `error = (gamma - gamma_ref).norm().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_heun_order_verification (L65)
steps = int(1.0 / dt)  # Total time = 1.0
# test_heun_order_verification (L73)
error_x = (x_test - x_ref).norm().item()
# test_heun_order_verification (L74)
error_v = (v_test - v_ref).norm().item()
# test_leapfrog_order_verification (L123)
steps = int(1.0 / dt)  # Total time = 1.0
# test_leapfrog_order_verification (L132)
error_x = (x_test - x_ref).norm().item()
# test_leapfrog_order_verification (L133)
error_v = (v_test - v_ref).norm().item()
# test_rank_approximation_error (L189)
error = (gamma - gamma_ref).norm().item()
```

### tests\cuda\test_cuda_dispatch.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 38 | run_kernel_smoke | `cmd = [sys.executable, str(PROJECT_ROOT / "tests" / "cuda" / "test_fusion_kernel.py")]` |
| 96 | run_kernel_smoke | `print("\n" + "="*60)` |

#### Fórmulas Listas para Usar (Python)
```python
# run_kernel_smoke (L38)
cmd = [sys.executable, str(PROJECT_ROOT / "tests" / "cuda" / "test_fusion_kernel.py")]
# run_kernel_smoke (L96)
print("\n" + "="*60)
```

### tests\cuda\test_cuda_friction_accuracy.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 51 | python_forward | `proj = torch.matmul(v, U)` |
| 52 | python_forward | `norm = torch.norm(proj, dim=-1, keepdim=True)` |
| 53 | python_forward | `scale = 1.0 / (1.0 + norm + 1e-4)` |
| 54 | python_forward | `sq = (proj * proj) * scale` |
| 55 | python_forward | `gamma = torch.matmul(sq, W.t())` |
| 57 | python_forward | `gate_activ = torch.matmul(x, Wf.t()) + bf` |
| 59 | python_forward | `gate_activ = gate_activ + torch.matmul(force, Wi.t())` |
| 60 | python_forward | `mu = torch.sigmoid(gate_activ) * FRICTION_SCALE` |
| 62 | python_forward | `output = gamma + mu * v` |
| 113 | python_forward | `diff = torch.abs(grads_py[name] - grads_cuda[name]).max().item()` |
| 114 | python_forward | `rel_diff = (torch.abs(grads_py[name] - grads_cuda[name]) / (torch.abs(grads_py[name]) + 1e-6)).max().item()` |
| 118 | python_forward | `fwd_diff = torch.abs(output_py - output_cuda).max().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# python_forward (L51)
proj = torch.matmul(v, U)
# python_forward (L52)
norm = torch.norm(proj, dim=-1, keepdim=True)
# python_forward (L53)
scale = 1.0 / (1.0 + norm + 1e-4)
# python_forward (L54)
sq = (proj * proj) * scale
# python_forward (L55)
gamma = torch.matmul(sq, W.t())
# python_forward (L57)
gate_activ = torch.matmul(x, Wf.t()) + bf
# python_forward (L59)
gate_activ = gate_activ + torch.matmul(force, Wi.t())
# python_forward (L60)
mu = torch.sigmoid(gate_activ) * FRICTION_SCALE
# python_forward (L62)
output = gamma + mu * v
# python_forward (L113)
diff = torch.abs(grads_py[name] - grads_cuda[name]).max().item()
# python_forward (L114)
rel_diff = (torch.abs(grads_py[name] - grads_cuda[name]) / (torch.abs(grads_py[name]) + 1e-6)).max().item()
# python_forward (L118)
fwd_diff = torch.abs(output_py - output_cuda).max().item()
```

### tests\cuda\test_cuda_gradients.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 24 | test_christoffel_gradients | `print("\n" + "="*80)` |
| 62 | christoffel_func | `eps=1e-6, atol=1e-5, rtol=1e-4` |

#### Fórmulas Listas para Usar (Python)
```python
# test_christoffel_gradients (L24)
print("\n" + "="*80)
# christoffel_func (L62)
eps=1e-6, atol=1e-5, rtol=1e-4
```

### tests\cuda\test_cuda_integrator_gradients.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 12 | test_leapfrog_gradients | `print("\n" + "="*80)` |
| 52 | integrator_func | `eps=1e-6, atol=1e-4, rtol=1e-3` |
| 60 | test_heun_gradients | `print("\n" + "="*80)` |
| 92 | integrator_func | `eps=1e-6, atol=1e-4, rtol=1e-3` |
| 100 | test_recurrent_manifold_gradients | `print("\n" + "="*80)` |
| 108 | test_recurrent_manifold_gradients | `head_dim = D // heads` |
| 115 | test_recurrent_manifold_gradients | `U_stack = torch.randn(layers * heads, head_dim, rank, device=device, dtype=dtype, requires_grad=True)` |
| 116 | test_recurrent_manifold_gradients | `W_stack = torch.randn(layers * heads, head_dim, rank, device=device, dtype=dtype, requires_grad=True)` |
| 121 | test_recurrent_manifold_gradients | `mix_x = torch.randn(layers, D, D, device=device, dtype=dtype, requires_grad=True) * 0.1` |
| 122 | test_recurrent_manifold_gradients | `mix_v = torch.randn(layers, D, D, device=device, dtype=dtype, requires_grad=True) * 0.1` |
| 149 | fused_loss | `eps=1e-6, atol=5e-4, rtol=5e-3` |

#### Fórmulas Listas para Usar (Python)
```python
# test_leapfrog_gradients (L12)
print("\n" + "="*80)
# integrator_func (L52)
eps=1e-6, atol=1e-4, rtol=1e-3
# test_heun_gradients (L60)
print("\n" + "="*80)
# integrator_func (L92)
eps=1e-6, atol=1e-4, rtol=1e-3
# test_recurrent_manifold_gradients (L100)
print("\n" + "="*80)
# test_recurrent_manifold_gradients (L108)
head_dim = D // heads
# test_recurrent_manifold_gradients (L115)
U_stack = torch.randn(layers * heads, head_dim, rank, device=device, dtype=dtype, requires_grad=True)
# test_recurrent_manifold_gradients (L116)
W_stack = torch.randn(layers * heads, head_dim, rank, device=device, dtype=dtype, requires_grad=True)
# test_recurrent_manifold_gradients (L121)
mix_x = torch.randn(layers, D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
# test_recurrent_manifold_gradients (L122)
mix_v = torch.randn(layers, D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
# fused_loss (L149)
eps=1e-6, atol=5e-4, rtol=5e-3
```

### tests\cuda\test_cuda_numerical.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 73 | Global | `outputs_python = model_python(inputs, **force_python_kwargs)` |
| 82 | Global | `max_diff_logits = (logits_cuda - logits_python).abs().max().item()` |
| 83 | Global | `max_diff_x = (x_cuda - x_python).abs().max().item()` |
| 84 | Global | `max_diff_v = (v_cuda - v_python).abs().max().item()` |
| 91 | Global | `THRESHOLD = 1e-4` |
| 93 | Global | `print("\n" + "="*60)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L73)
outputs_python = model_python(inputs, **force_python_kwargs)
# Global (L82)
max_diff_logits = (logits_cuda - logits_python).abs().max().item()
# Global (L83)
max_diff_x = (x_cuda - x_python).abs().max().item()
# Global (L84)
max_diff_v = (v_cuda - v_python).abs().max().item()
# Global (L91)
THRESHOLD = 1e-4
# Global (L93)
print("\n" + "="*60)
```

### tests\cuda\test_cuda_python_consistency.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 75 | TestConfig | `tolerance: float = 1e-4` |
| 76 | TestConfig | `gradient_tolerance: float = 1e-3` |
| 122 | manifold_params | `U = torch.randn(dim, rank, dtype=config.dtype, device=config.device) * 0.1` |
| 123 | manifold_params | `W = torch.randn(dim, rank, dtype=config.dtype, device=config.device) * 0.1` |
| 135 | test_tensors | `v = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.5` |
| 136 | test_tensors | `x = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.5` |
| 137 | test_tensors | `f = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.1` |
| 148 | compute_relative_error | `diff = torch.abs(tensor1 - tensor2)` |
| 170 | __init__ | `def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):` |
| 192 | converged | `change = abs(self.losses[-1] - self.losses[-2])` |
| 281 | test_christoffel_with_plasticity | `diff = torch.abs(gamma_plastic - gamma_no_plastic)` |
| 308 | test_christoffel_energy_conservation | `v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)` |
| 351 | hamiltonian | `kinetic = 0.5 * torch.sum(v * v, dim=-1)` |
| 352 | hamiltonian | `potential = 0.5 * torch.sum(x * x, dim=-1)` |
| 375 | hamiltonian | `energy_change = H_final - H_initial` |
| 376 | hamiltonian | `energy_change_rate = energy_change / (DEFAULT_DT * LEAPFROG_SUBSTEPS)` |
| 402 | test_leapfrog_toroidal_wrapping | `assert torch.all(x_out >= 0) and torch.all(x_out <= TOROIDAL_PERIOD * 1.1), \ "Toroidal wrapping failed"` |
| 452 | test_christoffel_gradients_cpu | `loss = torch.sum(gamma)` |
| 484 | test_leapfrog_gradients_cpu | `loss = torch.sum(x_out) + torch.sum(v_out)` |
| 497 | test_gradient_numerical_verification | `eps = 1e-2  # Increase epsilon for float32 visibility` |
| 503 | test_gradient_numerical_verification | `loss_ref = torch.sum(gamma_ref)` |
| 509 | test_gradient_numerical_verification | `v_plus[0, 0] += eps` |
| 513 | test_gradient_numerical_verification | `v_minus[0, 0] -= eps` |
| 520 | test_gradient_numerical_verification | `loss_plus = torch.sum(gamma_plus)` |
| 521 | test_gradient_numerical_verification | `loss_minus = torch.sum(gamma_minus)` |
| 523 | test_gradient_numerical_verification | `grad_num_scalar = (loss_plus - loss_minus) / (2 * eps)` |
| 528 | test_gradient_numerical_verification | `diff = torch.abs(grad_ref_scalar - grad_num_scalar)` |
| 529 | test_gradient_numerical_verification | `denom = torch.abs(grad_ref_scalar) + torch.abs(grad_num_scalar) + 1e-8` |
| 530 | test_gradient_numerical_verification | `relative_diff = diff / denom` |
| 539 | TestCUDAVsPythonEquivalence | `@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")` |
| 570 | test_christoffel_cuda_python_equivalence | `f"CUDA/Python Christoffel mismatch: max_diff={max_diff}"` |
| 572 | test_christoffel_cuda_python_equivalence | `f"CUDA/Python Christoffel mismatch: mean_diff={mean_diff}"` |
| 574 | test_christoffel_cuda_python_equivalence | `@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")` |
| 619 | test_leapfrog_cuda_python_equivalence | `f"CUDA/Python Leapfrog x mismatch: max_diff={x_max_diff}"` |
| 621 | test_leapfrog_cuda_python_equivalence | `f"CUDA/Python Leapfrog v mismatch: max_diff={v_max_diff}"` |
| 632 | test_learning_curve_convergence | `tracker = ConvergenceTracker(tolerance=1e-4, max_iterations=200)` |
| 645 | test_learning_curve_convergence | `loss = torch.sum(gamma * gamma)` |
| 676 | test_manifold_optimization_convergence | `tracker = ConvergenceTracker(tolerance=1e-7, max_iterations=30)` |
| 679 | test_manifold_optimization_convergence | `v = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5` |
| 681 | test_manifold_optimization_convergence | `x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5` |
| 690 | test_manifold_optimization_convergence | `loss = torch.sum(gamma * gamma)` |
| 721 | test_zero_velocity | `x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5` |
| 737 | test_unit_velocity | `v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)` |
| 738 | test_unit_velocity | `x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5` |
| 754 | test_large_input_values | `v = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 10.0` |
| 756 | test_large_input_values | `x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 10.0` |
| 771 | test_small_dt | `python_op = LeapfrogOperation({ 'dt': 1e-4,  # Very small dt 'friction_scale': FRICTION_SCALE, 'epsilon': EPSILON_STANDARD })` |
| 837 | test_christoffel_throughput | `elapsed = time.perf_counter() - start` |
| 838 | test_christoffel_throughput | `avg_time = elapsed / iterations * 1000  # ms` |
| 851 | test_leapfrog_throughput | `f = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.1` |
| 875 | test_leapfrog_throughput | `elapsed = time.perf_counter() - start` |
| 876 | test_leapfrog_throughput | `avg_time = elapsed / iterations * 1000  # ms` |
| 883 | test_leapfrog_throughput | `@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")` |
| 899 | test_cuda_speedup | `cpu_time = time.perf_counter() - start` |
| 910 | test_cuda_speedup | `gpu_time = time.perf_counter() - start` |
| 912 | test_cuda_speedup | `speedup = cpu_time / gpu_time` |
| 970 | test_toroidal_boundary_conditions | `assert torch.all(x_out >= -0.1 * TOROIDAL_PERIOD), \ "Positions should not go too far below 0"` |
| 972 | test_toroidal_boundary_conditions | `assert torch.all(x_out <= 1.1 * TOROIDAL_PERIOD), \ "Positions should not go too far above 2*pi"` |
| 979 | TestAutogradFunctionality | `@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")` |
| 998 | test_christoffel_autograd | `loss = torch.sum(gamma)` |
| 1010 | test_christoffel_autograd | `@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")` |
| 1033 | test_leapfrog_autograd | `loss = torch.sum(x_out) + torch.sum(v_out)` |
| 1085 | test_training_loop | `tracker = ConvergenceTracker(tolerance=1e-8, max_iterations=20)` |
| 1108 | test_training_loop | `loss = torch.sum(gamma * gamma) + torch.sum((x_out - x) ** 2)` |
| 1143 | test_gradient_flow | `loss = torch.sum(gamma * gamma)` |
| 1150 | test_gradient_flow | `grad_norm_U = float(torch.norm(U_train.grad))` |
| 1151 | test_gradient_flow | `grad_norm_W = float(torch.norm(W_train.grad))` |
| 1165 | test_gradient_flow | `pytest.main([__file__, "-v", "--tb=short"])` |

#### Fórmulas Listas para Usar (Python)
```python
# TestConfig (L75)
tolerance: float = 1e-4
# TestConfig (L76)
gradient_tolerance: float = 1e-3
# manifold_params (L122)
U = torch.randn(dim, rank, dtype=config.dtype, device=config.device) * 0.1
# manifold_params (L123)
W = torch.randn(dim, rank, dtype=config.dtype, device=config.device) * 0.1
# test_tensors (L135)
v = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.5
# test_tensors (L136)
x = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.5
# test_tensors (L137)
f = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.1
# compute_relative_error (L148)
diff = torch.abs(tensor1 - tensor2)
# __init__ (L170)
def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
# converged (L192)
change = abs(self.losses[-1] - self.losses[-2])
# test_christoffel_with_plasticity (L281)
diff = torch.abs(gamma_plastic - gamma_no_plastic)
# test_christoffel_energy_conservation (L308)
v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
# hamiltonian (L351)
kinetic = 0.5 * torch.sum(v * v, dim=-1)
# hamiltonian (L352)
potential = 0.5 * torch.sum(x * x, dim=-1)
# hamiltonian (L375)
energy_change = H_final - H_initial
# hamiltonian (L376)
energy_change_rate = energy_change / (DEFAULT_DT * LEAPFROG_SUBSTEPS)
# test_leapfrog_toroidal_wrapping (L402)
assert torch.all(x_out >= 0) and torch.all(x_out <= TOROIDAL_PERIOD * 1.1), \ "Toroidal wrapping failed"
# test_christoffel_gradients_cpu (L452)
loss = torch.sum(gamma)
# test_leapfrog_gradients_cpu (L484)
loss = torch.sum(x_out) + torch.sum(v_out)
# test_gradient_numerical_verification (L497)
eps = 1e-2  # Increase epsilon for float32 visibility
# test_gradient_numerical_verification (L503)
loss_ref = torch.sum(gamma_ref)
# test_gradient_numerical_verification (L509)
v_plus[0, 0] += eps
# test_gradient_numerical_verification (L513)
v_minus[0, 0] -= eps
# test_gradient_numerical_verification (L520)
loss_plus = torch.sum(gamma_plus)
# test_gradient_numerical_verification (L521)
loss_minus = torch.sum(gamma_minus)
# test_gradient_numerical_verification (L523)
grad_num_scalar = (loss_plus - loss_minus) / (2 * eps)
# test_gradient_numerical_verification (L528)
diff = torch.abs(grad_ref_scalar - grad_num_scalar)
# test_gradient_numerical_verification (L529)
denom = torch.abs(grad_ref_scalar) + torch.abs(grad_num_scalar) + 1e-8
# test_gradient_numerical_verification (L530)
relative_diff = diff / denom
# TestCUDAVsPythonEquivalence (L539)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# test_christoffel_cuda_python_equivalence (L570)
f"CUDA/Python Christoffel mismatch: max_diff={max_diff}"
# test_christoffel_cuda_python_equivalence (L572)
f"CUDA/Python Christoffel mismatch: mean_diff={mean_diff}"
# test_christoffel_cuda_python_equivalence (L574)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# test_leapfrog_cuda_python_equivalence (L619)
f"CUDA/Python Leapfrog x mismatch: max_diff={x_max_diff}"
# test_leapfrog_cuda_python_equivalence (L621)
f"CUDA/Python Leapfrog v mismatch: max_diff={v_max_diff}"
# test_learning_curve_convergence (L632)
tracker = ConvergenceTracker(tolerance=1e-4, max_iterations=200)
# test_learning_curve_convergence (L645)
loss = torch.sum(gamma * gamma)
# test_manifold_optimization_convergence (L676)
tracker = ConvergenceTracker(tolerance=1e-7, max_iterations=30)
# test_manifold_optimization_convergence (L679)
v = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5
# test_manifold_optimization_convergence (L681)
x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5
# test_manifold_optimization_convergence (L690)
loss = torch.sum(gamma * gamma)
# test_zero_velocity (L721)
x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5
# test_unit_velocity (L737)
v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
# test_unit_velocity (L738)
x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5
# test_large_input_values (L754)
v = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 10.0
# test_large_input_values (L756)
x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 10.0
# test_small_dt (L771)
python_op = LeapfrogOperation({ 'dt': 1e-4,  # Very small dt 'friction_scale': FRICTION_SCALE, 'epsilon': EPSILON_STANDARD })
# test_christoffel_throughput (L837)
elapsed = time.perf_counter() - start
# test_christoffel_throughput (L838)
avg_time = elapsed / iterations * 1000  # ms
# test_leapfrog_throughput (L851)
f = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.1
# test_leapfrog_throughput (L875)
elapsed = time.perf_counter() - start
# test_leapfrog_throughput (L876)
avg_time = elapsed / iterations * 1000  # ms
# test_leapfrog_throughput (L883)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# test_cuda_speedup (L899)
cpu_time = time.perf_counter() - start
# test_cuda_speedup (L910)
gpu_time = time.perf_counter() - start
# test_cuda_speedup (L912)
speedup = cpu_time / gpu_time
# test_toroidal_boundary_conditions (L970)
assert torch.all(x_out >= -0.1 * TOROIDAL_PERIOD), \ "Positions should not go too far below 0"
# test_toroidal_boundary_conditions (L972)
assert torch.all(x_out <= 1.1 * TOROIDAL_PERIOD), \ "Positions should not go too far above 2*pi"
# TestAutogradFunctionality (L979)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# test_christoffel_autograd (L998)
loss = torch.sum(gamma)
# test_christoffel_autograd (L1010)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# test_leapfrog_autograd (L1033)
loss = torch.sum(x_out) + torch.sum(v_out)
# test_training_loop (L1085)
tracker = ConvergenceTracker(tolerance=1e-8, max_iterations=20)
# test_training_loop (L1108)
loss = torch.sum(gamma * gamma) + torch.sum((x_out - x) ** 2)
# test_gradient_flow (L1143)
loss = torch.sum(gamma * gamma)
# test_gradient_flow (L1150)
grad_norm_U = float(torch.norm(U_train.grad))
# test_gradient_flow (L1151)
grad_norm_W = float(torch.norm(W_train.grad))
# test_gradient_flow (L1165)
pytest.main([__file__, "-v", "--tb=short"])
```

### tests\cuda\test_cuda_quick.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 141 | Global | `print("\n" + "=" * 70)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L141)
print("\n" + "=" * 70)
```

### tests\cuda\test_fusion_kernel.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 22 | Global | `dim_per_head = dim // num_heads` |
| 29 | Global | `U_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)` |
| 30 | Global | `W_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)` |
| 65 | Global | `dim_per_head = dim // num_heads` |
| 71 | Global | `U_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)` |
| 72 | Global | `W_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L22)
dim_per_head = dim // num_heads
# Global (L29)
U_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
# Global (L30)
W_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
# Global (L65)
dim_per_head = dim // num_heads
# Global (L71)
U_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
# Global (L72)
W_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
```

### tests\cuda\test_geometry_fusion.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 20 | test_geometry_fusion_parity | `base_geo = [LowRankChristoffel(dim // heads, rank=4).to(device) for _ in range(heads)]` |
| 46 | test_geometry_fusion_parity | `mlayer.curiosity_noises = [lambda v, **kwargs: v*0.0 for _ in range(heads)]` |
| 55 | test_geometry_fusion_parity | `diff_x = torch.abs(x_next_f - x_next_l).max().item()` |
| 56 | test_geometry_fusion_parity | `diff_v = torch.abs(v_next_f - v_next_l).max().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_geometry_fusion_parity (L20)
base_geo = [LowRankChristoffel(dim // heads, rank=4).to(device) for _ in range(heads)]
# test_geometry_fusion_parity (L46)
mlayer.curiosity_noises = [lambda v, **kwargs: v*0.0 for _ in range(heads)]
# test_geometry_fusion_parity (L55)
diff_x = torch.abs(x_next_f - x_next_l).max().item()
# test_geometry_fusion_parity (L56)
diff_v = torch.abs(v_next_f - v_next_l).max().item()
```

### tests\cuda\test_kernel_load.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 59 | Global | `print("\n" + "="*70)` |

#### Fórmulas Listas para Usar (Python)
```python
# Global (L59)
print("\n" + "="*70)
```

### tests\cuda\test_parity_python_cuda.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 34 | test_christoffel | `diff = (gamma_cpu - gamma_gpu.cpu()).abs().max().item()` |
| 35 | test_christoffel | `status = summarize(f"christoffel[topo={topology}]", diff, tol=3e-4)` |
| 49 | test_leapfrog | `feat_dim = (2 * D) if topology == 1 else D` |
| 64 | test_leapfrog | `dx = (x_out_cpu - x_out_gpu.cpu()).abs().max().item()` |
| 65 | test_leapfrog | `dv = (v_out_cpu - v_out_gpu.cpu()).abs().max().item()` |
| 66 | test_leapfrog | `s1 = summarize(f"leapfrog[topo={topology}]-x", dx, tol=5e-4)` |
| 67 | test_leapfrog | `s2 = summarize(f"leapfrog[topo={topology}]-v", dv, tol=5e-4)` |
| 86 | test_head_mixing | `dx = (x_out_cpu - x_out_gpu.cpu()).abs().max().item()` |
| 87 | test_head_mixing | `dv = (v_out_cpu - v_out_gpu.cpu()).abs().max().item()` |
| 88 | test_head_mixing | `s1 = summarize("head_mixing-x", dx, tol=1e-5)` |
| 89 | test_head_mixing | `s2 = summarize("head_mixing-v", dv, tol=1e-5)` |
| 96 | test_dynamic_gating | `W1_cpu = torch.randn(D // 4, inp_dim, dtype=torch.float32)` |
| 97 | test_dynamic_gating | `b1_cpu = torch.randn(D // 4, dtype=torch.float32)` |
| 98 | test_dynamic_gating | `W2_cpu = torch.randn(1, D // 4, dtype=torch.float32)` |
| 110 | test_dynamic_gating | `diff = (y_cpu - y_gpu.cpu()).abs().max().item()` |
| 111 | test_dynamic_gating | `status = summarize("dynamic_gating", diff, tol=1e-5)` |

#### Fórmulas Listas para Usar (Python)
```python
# test_christoffel (L34)
diff = (gamma_cpu - gamma_gpu.cpu()).abs().max().item()
# test_christoffel (L35)
status = summarize(f"christoffel[topo={topology}]", diff, tol=3e-4)
# test_leapfrog (L49)
feat_dim = (2 * D) if topology == 1 else D
# test_leapfrog (L64)
dx = (x_out_cpu - x_out_gpu.cpu()).abs().max().item()
# test_leapfrog (L65)
dv = (v_out_cpu - v_out_gpu.cpu()).abs().max().item()
# test_leapfrog (L66)
s1 = summarize(f"leapfrog[topo={topology}]-x", dx, tol=5e-4)
# test_leapfrog (L67)
s2 = summarize(f"leapfrog[topo={topology}]-v", dv, tol=5e-4)
# test_head_mixing (L86)
dx = (x_out_cpu - x_out_gpu.cpu()).abs().max().item()
# test_head_mixing (L87)
dv = (v_out_cpu - v_out_gpu.cpu()).abs().max().item()
# test_head_mixing (L88)
s1 = summarize("head_mixing-x", dx, tol=1e-5)
# test_head_mixing (L89)
s2 = summarize("head_mixing-v", dv, tol=1e-5)
# test_dynamic_gating (L96)
W1_cpu = torch.randn(D // 4, inp_dim, dtype=torch.float32)
# test_dynamic_gating (L97)
b1_cpu = torch.randn(D // 4, dtype=torch.float32)
# test_dynamic_gating (L98)
W2_cpu = torch.randn(1, D // 4, dtype=torch.float32)
# test_dynamic_gating (L110)
diff = (y_cpu - y_gpu.cpu()).abs().max().item()
# test_dynamic_gating (L111)
status = summarize("dynamic_gating", diff, tol=1e-5)
```

### tests\cuda\test_utils.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 99 | compare_tensors | `abs_diff = (cuda_out - py_out).abs()` |
| 101 | compare_tensors | `mean_diff = abs_diff.mean().item()` |
| 104 | compare_tensors | `rel_error = (abs_diff / (py_out.abs() + 1e-8)).max().item()` |
| 137 | measure_convergence_rate | `Fits log(error) = log(C) + p * log(h) where p is the convergence rate.` |
| 142 | measure_convergence_rate | `log_errors = np.log(errors)` |
| 143 | measure_convergence_rate | `log_h = np.log(refinements)` |
| 146 | measure_convergence_rate | `coeffs = np.polyfit(log_h, log_errors, 1)` |
| 155 | compute_energy | `E = 0.5 * \|\|v\|\|^2 + 0.5 * \|\|x\|\|^2` |
| 157 | compute_energy | `kinetic = 0.5 * (v ** 2).sum(dim=-1)` |
| 158 | compute_energy | `potential = 0.5 * (x ** 2).sum(dim=-1)` |
| 171 | measure_energy_drift | `abs_drift = (E_final - E_initial).abs()` |
| 172 | measure_energy_drift | `rel_drift = abs_drift / (E_initial.abs() + 1e-8)` |
| 183 | print_test_header | `print("\n" + "=" * 80)` |
| 208 | create_friction_gates | `feature_dim = 2 * dim if topology == 'torus' else dim` |
| 210 | create_friction_gates | `W_forget = torch.randn(dim, feature_dim, device=device, dtype=dtype) * 0.01` |

#### Fórmulas Listas para Usar (Python)
```python
# compare_tensors (L99)
abs_diff = (cuda_out - py_out).abs()
# compare_tensors (L101)
mean_diff = abs_diff.mean().item()
# compare_tensors (L104)
rel_error = (abs_diff / (py_out.abs() + 1e-8)).max().item()
# measure_convergence_rate (L137)
Fits log(error) = log(C) + p * log(h) where p is the convergence rate.
# measure_convergence_rate (L142)
log_errors = np.log(errors)
# measure_convergence_rate (L143)
log_h = np.log(refinements)
# measure_convergence_rate (L146)
coeffs = np.polyfit(log_h, log_errors, 1)
# compute_energy (L155)
E = 0.5 * ||v||^2 + 0.5 * ||x||^2
# compute_energy (L157)
kinetic = 0.5 * (v ** 2).sum(dim=-1)
# compute_energy (L158)
potential = 0.5 * (x ** 2).sum(dim=-1)
# measure_energy_drift (L171)
abs_drift = (E_final - E_initial).abs()
# measure_energy_drift (L172)
rel_drift = abs_drift / (E_initial.abs() + 1e-8)
# print_test_header (L183)
print("\n" + "=" * 80)
# create_friction_gates (L208)
feature_dim = 2 * dim if topology == 'torus' else dim
# create_friction_gates (L210)
W_forget = torch.randn(dim, feature_dim, device=device, dtype=dtype) * 0.01
```

### tests\cuda\verify_cuda_autograd.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 61 | kernel_order_reference | `h = h + v[:, j:j+1] * U[j:j+1, :]` |
| 62 | kernel_order_reference | `energy = (h * h).sum(dim=-1, keepdim=True) / max(1, h.shape[-1])` |
| 63 | kernel_order_reference | `scale = 1.0 / (1.0 + torch.sqrt(energy) + CudaConstants.EPSILON_STANDARD)` |
| 66 | kernel_order_reference | `E = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])` |
| 67 | kernel_order_reference | `M = 1.0 + plasticity * 0.1 * torch.tanh(E)` |
| 69 | kernel_order_reference | `pot = torch.sum(x * V_w, dim=-1, keepdim=True)` |
| 70 | kernel_order_reference | `gate = torch.sigmoid(pot)` |
| 71 | kernel_order_reference | `soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))` |
| 72 | kernel_order_reference | `M = M * (1.0 + (sing_strength - 1.0) * soft_m)` |
| 73 | kernel_order_reference | `q = h * h * scale * M` |
| 76 | kernel_order_reference | `gamma[:, i] = (q * W[i:i+1, :]).sum(dim=-1)` |
| 77 | kernel_order_reference | `gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma / CudaConstants.CURVATURE_CLAMP)` |
| 83 | kernel_order_backward | `h = h + v[:, j:j+1] * U[j:j+1, :]` |
| 84 | kernel_order_backward | `energy = (h * h).sum(dim=-1, keepdim=True) / max(1, h.shape[-1])` |
| 85 | kernel_order_backward | `norm = torch.sqrt(energy)` |
| 86 | kernel_order_backward | `S = 1.0 / (1.0 + norm + CudaConstants.EPSILON_STANDARD)` |
| 90 | kernel_order_backward | `v_energy = (v * v).sum(dim=-1, keepdim=True) / max(1, v.shape[-1])` |
| 91 | kernel_order_backward | `tanh_v = torch.tanh(v_energy)` |
| 92 | kernel_order_backward | `M_plas = 1.0 + plasticity * 0.1 * tanh_v` |
| 97 | kernel_order_backward | `pot = torch.sum(x * V_w, dim=-1, keepdim=True)` |
| 98 | kernel_order_backward | `gate = torch.sigmoid(pot)` |
| 99 | kernel_order_backward | `soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))` |
| 100 | kernel_order_backward | `M_sing = 1.0 + (sing_strength - 1.0) * soft_m` |
| 101 | kernel_order_backward | `M = M_plas * M_sing` |
| 102 | kernel_order_backward | `q = h * h * S * M` |
| 105 | kernel_order_backward | `gamma_raw[:, i] = (q * W[i:i+1, :]).sum(dim=-1)` |
| 106 | kernel_order_backward | `gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_raw / CudaConstants.CURVATURE_CLAMP)` |
| 108 | kernel_order_backward | `t = gamma / CudaConstants.CURVATURE_CLAMP` |
| 110 | kernel_order_backward | `t = gamma_for_t / CudaConstants.CURVATURE_CLAMP` |
| 111 | kernel_order_backward | `grad_raw = grad_output * (1.0 - t * t)` |
| 115 | kernel_order_backward | `grad_W[i, :] = (grad_raw[:, i:i+1] * q).sum(dim=0)` |
| 116 | kernel_order_backward | `grad_q = grad_q + W[i:i+1, :] * grad_raw[:, i:i+1]` |
| 117 | kernel_order_backward | `sum_grad_q_h_sq = (grad_q * h * h).sum(dim=-1, keepdim=True)` |
| 118 | kernel_order_backward | `denom = norm * max(1, h.shape[-1])` |
| 119 | kernel_order_backward | `scale = torch.where(denom > 0, M * S * S / denom, torch.zeros_like(denom))` |
| 120 | kernel_order_backward | `grad_h = grad_q * (2.0 * h * S * M) - sum_grad_q_h_sq * scale * h` |
| 121 | kernel_order_backward | `grad_v = grad_h @ U.t()` |
| 122 | kernel_order_backward | `grad_U = v.transpose(0, 1) @ grad_h` |
| 124 | kernel_order_backward | `dL_dM_plas = sum_grad_q_h_sq * S * M_sing` |
| 125 | kernel_order_backward | `dM_plas_dv = plasticity * 0.1 * (1.0 - tanh_v * tanh_v) * (2.0 / max(1, v.shape[-1])) * v` |
| 126 | kernel_order_backward | `grad_v = grad_v + dL_dM_plas * dM_plas_dv` |
| 130 | kernel_order_backward | `dL_dM_sing = sum_grad_q_h_sq * S * M_plas` |
| 131 | kernel_order_backward | `dM_sing_dpot = (sing_strength - 1.0) * soft_m * (1.0 - soft_m) * CudaConstants.SINGULARITY_GATE_SLOPE * gate * (1.0 - gate)` |
| 132 | kernel_order_backward | `factor = dL_dM_sing * dM_sing_dpot` |
| 133 | kernel_order_backward | `grad_x = factor * V_w` |
| 134 | kernel_order_backward | `grad_V_w = (factor * x).sum(dim=0)` |
| 143 | kernel_order_backward | `fwd_diff = (res_pt - res_cuda).abs().max().item()` |
| 144 | kernel_order_backward | `fwd_kernel_diff = (res_kernel - res_cuda).abs().max().item()` |
| 147 | kernel_order_backward | `fwd_threshold = 1e-4` |
| 162 | kernel_order_backward | `loss_pt = res_pt.pow(2).sum()` |
| 174 | kernel_order_backward | `loss_kernel = res_kernel.pow(2).sum()` |
| 181 | kernel_order_backward | `grad_out_kernel = 2.0 * res_kernel` |
| 194 | kernel_order_backward | `loss_cuda = res_cuda.pow(2).sum()` |
| 201 | kernel_order_backward | `v_diff = (grad_v_pt - grad_v_cuda).abs().max().item()` |
| 202 | kernel_order_backward | `U_diff = (grad_U_pt - grad_U_cuda).abs().max().item()` |
| 203 | kernel_order_backward | `W_diff = (grad_W_pt - grad_W_cuda).abs().max().item()` |
| 205 | kernel_order_backward | `v_diff_kernel = (grad_v_kernel - grad_v_cuda).abs().max().item()` |
| 206 | kernel_order_backward | `U_diff_kernel = (grad_U_kernel - grad_U_cuda).abs().max().item()` |
| 207 | kernel_order_backward | `W_diff_kernel = (grad_W_kernel - grad_W_cuda).abs().max().item()` |
| 277 | kernel_order_backward | `diff_x = (res_fused_x - curr_x).abs().max().item()` |
| 278 | kernel_order_backward | `diff_v = (res_fused_v - curr_v).abs().max().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# kernel_order_reference (L61)
h = h + v[:, j:j+1] * U[j:j+1, :]
# kernel_order_reference (L62)
energy = (h * h).sum(dim=-1, keepdim=True) / max(1, h.shape[-1])
# kernel_order_reference (L63)
scale = 1.0 / (1.0 + torch.sqrt(energy) + CudaConstants.EPSILON_STANDARD)
# kernel_order_reference (L66)
E = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
# kernel_order_reference (L67)
M = 1.0 + plasticity * 0.1 * torch.tanh(E)
# kernel_order_reference (L69)
pot = torch.sum(x * V_w, dim=-1, keepdim=True)
# kernel_order_reference (L70)
gate = torch.sigmoid(pot)
# kernel_order_reference (L71)
soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))
# kernel_order_reference (L72)
M = M * (1.0 + (sing_strength - 1.0) * soft_m)
# kernel_order_reference (L73)
q = h * h * scale * M
# kernel_order_reference (L76)
gamma[:, i] = (q * W[i:i+1, :]).sum(dim=-1)
# kernel_order_reference (L77)
gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma / CudaConstants.CURVATURE_CLAMP)
# kernel_order_backward (L83)
h = h + v[:, j:j+1] * U[j:j+1, :]
# kernel_order_backward (L84)
energy = (h * h).sum(dim=-1, keepdim=True) / max(1, h.shape[-1])
# kernel_order_backward (L85)
norm = torch.sqrt(energy)
# kernel_order_backward (L86)
S = 1.0 / (1.0 + norm + CudaConstants.EPSILON_STANDARD)
# kernel_order_backward (L90)
v_energy = (v * v).sum(dim=-1, keepdim=True) / max(1, v.shape[-1])
# kernel_order_backward (L91)
tanh_v = torch.tanh(v_energy)
# kernel_order_backward (L92)
M_plas = 1.0 + plasticity * 0.1 * tanh_v
# kernel_order_backward (L97)
pot = torch.sum(x * V_w, dim=-1, keepdim=True)
# kernel_order_backward (L98)
gate = torch.sigmoid(pot)
# kernel_order_backward (L99)
soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))
# kernel_order_backward (L100)
M_sing = 1.0 + (sing_strength - 1.0) * soft_m
# kernel_order_backward (L101)
M = M_plas * M_sing
# kernel_order_backward (L102)
q = h * h * S * M
# kernel_order_backward (L105)
gamma_raw[:, i] = (q * W[i:i+1, :]).sum(dim=-1)
# kernel_order_backward (L106)
gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_raw / CudaConstants.CURVATURE_CLAMP)
# kernel_order_backward (L108)
t = gamma / CudaConstants.CURVATURE_CLAMP
# kernel_order_backward (L110)
t = gamma_for_t / CudaConstants.CURVATURE_CLAMP
# kernel_order_backward (L111)
grad_raw = grad_output * (1.0 - t * t)
# kernel_order_backward (L115)
grad_W[i, :] = (grad_raw[:, i:i+1] * q).sum(dim=0)
# kernel_order_backward (L116)
grad_q = grad_q + W[i:i+1, :] * grad_raw[:, i:i+1]
# kernel_order_backward (L117)
sum_grad_q_h_sq = (grad_q * h * h).sum(dim=-1, keepdim=True)
# kernel_order_backward (L118)
denom = norm * max(1, h.shape[-1])
# kernel_order_backward (L119)
scale = torch.where(denom > 0, M * S * S / denom, torch.zeros_like(denom))
# kernel_order_backward (L120)
grad_h = grad_q * (2.0 * h * S * M) - sum_grad_q_h_sq * scale * h
# kernel_order_backward (L121)
grad_v = grad_h @ U.t()
# kernel_order_backward (L122)
grad_U = v.transpose(0, 1) @ grad_h
# kernel_order_backward (L124)
dL_dM_plas = sum_grad_q_h_sq * S * M_sing
# kernel_order_backward (L125)
dM_plas_dv = plasticity * 0.1 * (1.0 - tanh_v * tanh_v) * (2.0 / max(1, v.shape[-1])) * v
# kernel_order_backward (L126)
grad_v = grad_v + dL_dM_plas * dM_plas_dv
# kernel_order_backward (L130)
dL_dM_sing = sum_grad_q_h_sq * S * M_plas
# kernel_order_backward (L131)
dM_sing_dpot = (sing_strength - 1.0) * soft_m * (1.0 - soft_m) * CudaConstants.SINGULARITY_GATE_SLOPE * gate * (1.0 - gate)
# kernel_order_backward (L132)
factor = dL_dM_sing * dM_sing_dpot
# kernel_order_backward (L133)
grad_x = factor * V_w
# kernel_order_backward (L134)
grad_V_w = (factor * x).sum(dim=0)
# kernel_order_backward (L143)
fwd_diff = (res_pt - res_cuda).abs().max().item()
# kernel_order_backward (L144)
fwd_kernel_diff = (res_kernel - res_cuda).abs().max().item()
# kernel_order_backward (L147)
fwd_threshold = 1e-4
# kernel_order_backward (L162)
loss_pt = res_pt.pow(2).sum()
# kernel_order_backward (L174)
loss_kernel = res_kernel.pow(2).sum()
# kernel_order_backward (L181)
grad_out_kernel = 2.0 * res_kernel
# kernel_order_backward (L194)
loss_cuda = res_cuda.pow(2).sum()
# kernel_order_backward (L201)
v_diff = (grad_v_pt - grad_v_cuda).abs().max().item()
# kernel_order_backward (L202)
U_diff = (grad_U_pt - grad_U_cuda).abs().max().item()
# kernel_order_backward (L203)
W_diff = (grad_W_pt - grad_W_cuda).abs().max().item()
# kernel_order_backward (L205)
v_diff_kernel = (grad_v_kernel - grad_v_cuda).abs().max().item()
# kernel_order_backward (L206)
U_diff_kernel = (grad_U_kernel - grad_U_cuda).abs().max().item()
# kernel_order_backward (L207)
W_diff_kernel = (grad_W_kernel - grad_W_cuda).abs().max().item()
# kernel_order_backward (L277)
diff_x = (res_fused_x - curr_x).abs().max().item()
# kernel_order_backward (L278)
diff_v = (res_fused_v - curr_v).abs().max().item()
```

### tests\debug_fusion_mismatch.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | run_comparison | `head_dim = dim // heads` |
| 29 | run_comparison | `U_stack = torch.randn(layers * heads, head_dim, 4, device=device) * 0.1 # rank=4` |
| 30 | run_comparison | `W_stack = torch.randn(layers * heads, head_dim, 4, device=device) * 0.1 # rank=4` |
| 34 | run_comparison | `dt_scales = torch.ones(layers * heads, device=device)` |
| 38 | run_comparison | `mix_x = torch.randn(layers, dim, 3*dim, device=device) * 0.1` |
| 59 | run_comparison | `gate_in_dim = 2*head_dim if topology == 1 else head_dim` |
| 60 | run_comparison | `gate_W1 = torch.randn(layers * heads, 16, gate_in_dim, device=device) * 0.1` |
| 61 | run_comparison | `gate_b1 = torch.zeros(layers * heads, 16, device=device)` |
| 62 | run_comparison | `gate_W2 = torch.randn(layers * heads, 1, 16, device=device) * 0.1` |
| 63 | run_comparison | `gate_b2 = torch.zeros(layers * heads, 1, device=device)` |
| 67 | run_comparison | `Wf = torch.randn(layers * heads, head_dim, 2*head_dim, device=device) * 0.1` |
| 68 | run_comparison | `Wi = torch.randn(layers * heads, head_dim, head_dim, device=device) * 0.1` |
| 69 | run_comparison | `bf = torch.zeros(layers * heads, head_dim, device=device)` |
| 70 | run_comparison | `Wp = torch.randn(layers * heads, 1, 2*head_dim, device=device) * 0.1` |
| 71 | run_comparison | `bp = torch.zeros(layers * heads, 1, device=device)` |
| 110 | run_comparison | `diff_x = (x_py - x_cuda).abs()` |
| 111 | run_comparison | `diff_v = (v_py - v_cuda).abs()` |
| 120 | run_comparison | `diff_seq = (seq_py - seq_cuda).abs() # [B, T, D]` |
| 121 | run_comparison | `max_diff_per_step = diff_seq.max(dim=-1)[0].max(dim=0)[0] # [T]` |
| 130 | run_comparison | `TWO_PI = 2.0 * PI` |
| 131 | run_comparison | `abs_diff = torch.abs(x_py - x_cuda)` |
| 133 | run_comparison | `tor_dist = torch.min(rem_diff, TWO_PI - rem_diff)` |

#### Fórmulas Listas para Usar (Python)
```python
# run_comparison (L21)
head_dim = dim // heads
# run_comparison (L29)
U_stack = torch.randn(layers * heads, head_dim, 4, device=device) * 0.1 # rank=4
# run_comparison (L30)
W_stack = torch.randn(layers * heads, head_dim, 4, device=device) * 0.1 # rank=4
# run_comparison (L34)
dt_scales = torch.ones(layers * heads, device=device)
# run_comparison (L38)
mix_x = torch.randn(layers, dim, 3*dim, device=device) * 0.1
# run_comparison (L59)
gate_in_dim = 2*head_dim if topology == 1 else head_dim
# run_comparison (L60)
gate_W1 = torch.randn(layers * heads, 16, gate_in_dim, device=device) * 0.1
# run_comparison (L61)
gate_b1 = torch.zeros(layers * heads, 16, device=device)
# run_comparison (L62)
gate_W2 = torch.randn(layers * heads, 1, 16, device=device) * 0.1
# run_comparison (L63)
gate_b2 = torch.zeros(layers * heads, 1, device=device)
# run_comparison (L67)
Wf = torch.randn(layers * heads, head_dim, 2*head_dim, device=device) * 0.1
# run_comparison (L68)
Wi = torch.randn(layers * heads, head_dim, head_dim, device=device) * 0.1
# run_comparison (L69)
bf = torch.zeros(layers * heads, head_dim, device=device)
# run_comparison (L70)
Wp = torch.randn(layers * heads, 1, 2*head_dim, device=device) * 0.1
# run_comparison (L71)
bp = torch.zeros(layers * heads, 1, device=device)
# run_comparison (L110)
diff_x = (x_py - x_cuda).abs()
# run_comparison (L111)
diff_v = (v_py - v_cuda).abs()
# run_comparison (L120)
diff_seq = (seq_py - seq_cuda).abs() # [B, T, D]
# run_comparison (L121)
max_diff_per_step = diff_seq.max(dim=-1)[0].max(dim=0)[0] # [T]
# run_comparison (L130)
TWO_PI = 2.0 * PI
# run_comparison (L131)
abs_diff = torch.abs(x_py - x_cuda)
# run_comparison (L133)
tor_dist = torch.min(rem_diff, TWO_PI - rem_diff)
```

### tests\diagnostics\conservation_audit.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 25 | test_energy_conservation | `v_norms = [curr_v.norm().item()]` |
| 36 | test_energy_conservation | `end_v = v_norms[-1]` |
| 37 | test_energy_conservation | `retention = end_v / (start_v + 1e-9)` |

#### Fórmulas Listas para Usar (Python)
```python
# test_energy_conservation (L25)
v_norms = [curr_v.norm().item()]
# test_energy_conservation (L36)
end_v = v_norms[-1]
# test_energy_conservation (L37)
retention = end_v / (start_v + 1e-9)
```

### tests\diagnostics\depth_audit.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 11 | print_metric | `status = "\033[92m[GOOD]\033[0m" if passed else "\033[91m[WEAK]\033[0m" print(f"{status} {name:<40}: {value:.4f} {unit}") def test_sequence_curriculum(): """Test parity convergence across different sequence lengths.""" print("\n--- TEST 1: SEQUENCE LENGTH CURRICULUM ---") lengths = [5, 10, 15, 20] dim = 128 for L in lengths: model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32, physics_config={'readout': {'type': 'implicit'}}) optimizer = torch.optim.AdamW(model.parameters(), lr=0.005) converged = False best_acc = 0 for step in range(200): model.train() optimizer.zero_grad() inputs = torch.randint(0, 2, (64, L)) targets = (inputs.sum(dim=-1) % 2).unsqueeze(-1).float() logits, _, _ = model(inputs) last_logits = logits[:, -1, 0:1] loss = F.binary_cross_entropy_with_logits(last_logits, targets) loss.backward() optimizer.step() model.readout.update_step() acc = ((last_logits > 0) == targets).float().mean().item() best_acc = max(best_acc, acc) if acc > 0.98: converged = True print(f"L={L:<2} \| CONVERGED at step {step}") break if not converged: print(f"L={L:<2} \| \033[91mFAILED\033[0m \| Best Acc: {best_acc:.2%}") def audit_gradient_decay(): """Measure gradient magnitude at different sequence positions.""" print("\n--- TEST 2: GRADIENT PATH DECAY (L=20) ---") L = 20 dim = 128 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.train() inputs = torch.randint(0, 2, (1, L)) logits, (state_x, state_v), _ = model(inputs) grads = [] for t in range(L): model.zero_grad() target = torch.tensor([[[1.0]]]) loss = F.mse_loss(logits[:, t:t+1, 0:1], target) loss.backward(retain_graph=True) grad_norm = model.layers[0].christoffels[0].W.grad.abs().mean().item() grads.append(grad_norm) for i, g in enumerate(grads): print(f"Step {i+1:02} Gradient Energy: {g:.2e}") decay = grads[0] / (grads[-1] + 1e-9) print_metric("Gradient Decay (Early/Late)", decay, threshold=10.0) def audit_state_saturation(): """Check if state x hits clamping limits during long sequences.""" print("\n--- TEST 3: STATE SATURATION AUDIT (L=50) ---") L = 50 dim = 64 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.eval() inputs = torch.ones(1, L).long() with torch.no_grad(): logits, (fx, fv), _ = model(inputs) curr_x = torch.zeros(1, dim) curr_v = torch.zeros(1, dim) forces = model.embedding(inputs) max_norms = [] for t in range(L): curr_x, curr_v, _, _ = model.layers[0](curr_x, curr_v, forces[:, t]) max_norms.append(curr_x.norm().item()) print(f"Max Norm at t=1:  {max_norms[0]:.2f}") print(f"Max Norm at t=25: {max_norms[24]:.2f}") print(f"Max Norm at t=50: {max_norms[-1]:.2f}") saturation_risk = max_norms[-1] > 80.0 print_result = "\033[91m[SATURATED]\033[0m" if saturation_risk else "\033[92m[SAFE]\033[0m" print(f"{print_result} Clamping Ceiling (100.0) Health") if __name__ == "__main__": print("====================================================") print("   GFN DEPTH & SCALING AUDIT (Phase 10)            ") print("====================================================") try: audit_state_saturation() audit_gradient_decay() test_sequence_curriculum() except Exception as e: print(f"Audit crashed: {e}") import traceback traceback.print_exc() print("\n====================================================")` |

#### Fórmulas Listas para Usar (Python)
```python
# print_metric (L11)
status = "\033[92m[GOOD]\033[0m" if passed else "\033[91m[WEAK]\033[0m" print(f"{status} {name:<40}: {value:.4f} {unit}") def test_sequence_curriculum(): """Test parity convergence across different sequence lengths.""" print("\n--- TEST 1: SEQUENCE LENGTH CURRICULUM ---") lengths = [5, 10, 15, 20] dim = 128 for L in lengths: model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32, physics_config={'readout': {'type': 'implicit'}}) optimizer = torch.optim.AdamW(model.parameters(), lr=0.005) converged = False best_acc = 0 for step in range(200): model.train() optimizer.zero_grad() inputs = torch.randint(0, 2, (64, L)) targets = (inputs.sum(dim=-1) % 2).unsqueeze(-1).float() logits, _, _ = model(inputs) last_logits = logits[:, -1, 0:1] loss = F.binary_cross_entropy_with_logits(last_logits, targets) loss.backward() optimizer.step() model.readout.update_step() acc = ((last_logits > 0) == targets).float().mean().item() best_acc = max(best_acc, acc) if acc > 0.98: converged = True print(f"L={L:<2} | CONVERGED at step {step}") break if not converged: print(f"L={L:<2} | \033[91mFAILED\033[0m | Best Acc: {best_acc:.2%}") def audit_gradient_decay(): """Measure gradient magnitude at different sequence positions.""" print("\n--- TEST 2: GRADIENT PATH DECAY (L=20) ---") L = 20 dim = 128 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.train() inputs = torch.randint(0, 2, (1, L)) logits, (state_x, state_v), _ = model(inputs) grads = [] for t in range(L): model.zero_grad() target = torch.tensor([[[1.0]]]) loss = F.mse_loss(logits[:, t:t+1, 0:1], target) loss.backward(retain_graph=True) grad_norm = model.layers[0].christoffels[0].W.grad.abs().mean().item() grads.append(grad_norm) for i, g in enumerate(grads): print(f"Step {i+1:02} Gradient Energy: {g:.2e}") decay = grads[0] / (grads[-1] + 1e-9) print_metric("Gradient Decay (Early/Late)", decay, threshold=10.0) def audit_state_saturation(): """Check if state x hits clamping limits during long sequences.""" print("\n--- TEST 3: STATE SATURATION AUDIT (L=50) ---") L = 50 dim = 64 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.eval() inputs = torch.ones(1, L).long() with torch.no_grad(): logits, (fx, fv), _ = model(inputs) curr_x = torch.zeros(1, dim) curr_v = torch.zeros(1, dim) forces = model.embedding(inputs) max_norms = [] for t in range(L): curr_x, curr_v, _, _ = model.layers[0](curr_x, curr_v, forces[:, t]) max_norms.append(curr_x.norm().item()) print(f"Max Norm at t=1:  {max_norms[0]:.2f}") print(f"Max Norm at t=25: {max_norms[24]:.2f}") print(f"Max Norm at t=50: {max_norms[-1]:.2f}") saturation_risk = max_norms[-1] > 80.0 print_result = "\033[91m[SATURATED]\033[0m" if saturation_risk else "\033[92m[SAFE]\033[0m" print(f"{print_result} Clamping Ceiling (100.0) Health") if __name__ == "__main__": print("====================================================") print("   GFN DEPTH & SCALING AUDIT (Phase 10)            ") print("====================================================") try: audit_state_saturation() audit_gradient_decay() test_sequence_curriculum() except Exception as e: print(f"Audit crashed: {e}") import traceback traceback.print_exc() print("\n====================================================")
```

### tests\diagnostics\grad_probe.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 21 | run_probe | `print(f"--- Gradient Probe (L={seq_len}) ---")` |
| 53 | run_probe | `flat_v = v_abs.view(-1)` |
| 56 | run_probe | `saturated_count = (flat_v >= 14.9).sum().item()` |
| 58 | run_probe | `saturation_rate = 100.0 * saturated_count / total_params` |
| 61 | run_probe | `v_mean = flat_v.mean().item()` |
| 72 | run_probe | `grad_norm = model.x0.grad.norm().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# run_probe (L21)
print(f"--- Gradient Probe (L={seq_len}) ---")
# run_probe (L53)
flat_v = v_abs.view(-1)
# run_probe (L56)
saturated_count = (flat_v >= 14.9).sum().item()
# run_probe (L58)
saturation_rate = 100.0 * saturated_count / total_params
# run_probe (L61)
v_mean = flat_v.mean().item()
# run_probe (L72)
grad_norm = model.x0.grad.norm().item()
```

### tests\diagnostics\manifold_audit.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 9 | print_result | `color = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m" print(f"{color} {name:<40} {details}") def audit_gradient_energy(): """Test 1: Do we have non-trivial gradients reaching the weights?""" print("\n--- TEST 1: GRADIENT ENERGY ---") model = Manifold(vocab_size=10, dim=128, depth=1, heads=1,rank=32, physics_config={'readout': {'type': 'implicit'}}) model.train() coord_dim = 16 x = torch.randint(1, 10, (1, 20)) y = torch.randint(0, 2, (1, 20, coord_dim)).float() logits, _ , _ = model(x) loss = F.binary_cross_entropy_with_logits(logits, y) loss.backward() max_grad_w = 0 max_grad_u = 0 for layer in model.layers: for head in layer.christoffels: max_grad_w = max(max_grad_w, head.W.grad.abs().max().item()) max_grad_u = max(max_grad_u, head.U.grad.abs().max().item()) print_result("Manifold W Gradient Energy", max_grad_w > 1e-6, f"Max: {max_grad_w:.2e}") print_result("Manifold U Gradient Energy", max_grad_u > 1e-6, f"Max: {max_grad_u:.2e}") readout_grad = model.readout.mlp[0].weight.grad.abs().max().item() print_result("Readout MLP Gradient Energy", readout_grad > 1e-6, f"Max: {readout_grad:.2e}") def audit_state_persistence(): """Test 2: Does the state x actually accumulate over time without LayerNorm?""" print("\n--- TEST 2: STATE PERSISTENCE (HISTORY) ---") dim = 64 model = Manifold(vocab_size=10, dim=dim, depth=1, heads=1, rank=32) model.eval() seq_len = 20 x_input = torch.ones(1, seq_len).long() t0_x = torch.zeros(1, dim) t0_v = torch.zeros(1, dim) x_history = [] curr_x, curr_v = t0_x, t0_v with torch.no_grad(): all_forces = model.embedding(x_input) for t in range(seq_len): curr_x, curr_v, _, _ = model.layers[0](curr_x, curr_v, force=all_forces[:, t]) x_history.append(curr_x.clone()) x_seq = torch.stack(x_history, dim=1) # [1, L, D] dist = torch.norm(x_seq[0, -1]).item() std_val = x_seq.std().item() print_result("Trajectory Integration", dist > 0.5, f"Final Norm: {dist:.2f}") print_result("State Dynamic Range", std_val > 0.05, f"Std: {std_val:.2f}") def audit_mini_parity(): """Test 3: Can we solve 4-bit parity in 50 steps?""" print("\n--- TEST 3: MINI-PARITY CONVERGENCE ---") dim = 64 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32, physics_config={'readout': {'type': 'implicit'}}) optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) inputs = torch.randint(0, 2, (32, 4)) targets = (inputs.sum(dim=-1) % 2).unsqueeze(-1).float() # [32, 1] start_time = time.time() best_loss = 100 for step in range(101): model.train() optimizer.zero_grad() logits, (final_x, _), _ = model(inputs) last_logits = logits[:, -1, 0:1] loss = F.binary_cross_entropy_with_logits(last_logits, targets) loss.backward() optimizer.step() model.readout.update_step() acc = ((last_logits > 0) == targets).float().mean().item() best_loss = min(best_loss, loss.item()) if acc == 1.0 and loss < 0.1: print_result("Parity Solver (L=4)", True, f"Converged at step {step} (Loss: {loss.item():.4f})") return print_result("Parity Solver (L=4)", False, f"Best Loss: {best_loss:.4f}, Accuracy: {acc:.2f}") def audit_physical_limits(): """Test 4: Check for NaNs and Singularities""" print("\n--- TEST 4: PHYSICAL INTEGRITY ---") model = Manifold(vocab_size=10, dim=128, depth=2, heads=4, rank=3) # Low rank to stress test x = torch.randint(0, 10, (128, 50)) logits, (fx, fv), _ = model(x) has_nan = torch.isnan(logits).any().item() or torch.isnan(fx).any().item() max_val = fx.abs().max().item() print_result("NaN Stability", not has_nan, "No NaNs found" if not has_nan else "!!! NaNs DETECTED !!!") print_result("Clamping Effectiveness", max_val <= 101.0, f"Max State: {max_val:.2f}") if __name__ == "__main__": print("====================================================") print("   GFN MANIFOLD MASTER AUDIT (v3.8 Diagnostic)     ") print("====================================================") try: audit_gradient_energy() audit_state_persistence() audit_physical_limits() audit_mini_parity() except Exception as e: print(f"\033[91m[CRITICAL ERROR]\033[0m Audit crashed: {e}") import traceback traceback.print_exc() print("\n====================================================") print("                AUDIT COMPLETE                      ") print("====================================================")` |

#### Fórmulas Listas para Usar (Python)
```python
# print_result (L9)
color = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m" print(f"{color} {name:<40} {details}") def audit_gradient_energy(): """Test 1: Do we have non-trivial gradients reaching the weights?""" print("\n--- TEST 1: GRADIENT ENERGY ---") model = Manifold(vocab_size=10, dim=128, depth=1, heads=1,rank=32, physics_config={'readout': {'type': 'implicit'}}) model.train() coord_dim = 16 x = torch.randint(1, 10, (1, 20)) y = torch.randint(0, 2, (1, 20, coord_dim)).float() logits, _ , _ = model(x) loss = F.binary_cross_entropy_with_logits(logits, y) loss.backward() max_grad_w = 0 max_grad_u = 0 for layer in model.layers: for head in layer.christoffels: max_grad_w = max(max_grad_w, head.W.grad.abs().max().item()) max_grad_u = max(max_grad_u, head.U.grad.abs().max().item()) print_result("Manifold W Gradient Energy", max_grad_w > 1e-6, f"Max: {max_grad_w:.2e}") print_result("Manifold U Gradient Energy", max_grad_u > 1e-6, f"Max: {max_grad_u:.2e}") readout_grad = model.readout.mlp[0].weight.grad.abs().max().item() print_result("Readout MLP Gradient Energy", readout_grad > 1e-6, f"Max: {readout_grad:.2e}") def audit_state_persistence(): """Test 2: Does the state x actually accumulate over time without LayerNorm?""" print("\n--- TEST 2: STATE PERSISTENCE (HISTORY) ---") dim = 64 model = Manifold(vocab_size=10, dim=dim, depth=1, heads=1, rank=32) model.eval() seq_len = 20 x_input = torch.ones(1, seq_len).long() t0_x = torch.zeros(1, dim) t0_v = torch.zeros(1, dim) x_history = [] curr_x, curr_v = t0_x, t0_v with torch.no_grad(): all_forces = model.embedding(x_input) for t in range(seq_len): curr_x, curr_v, _, _ = model.layers[0](curr_x, curr_v, force=all_forces[:, t]) x_history.append(curr_x.clone()) x_seq = torch.stack(x_history, dim=1) # [1, L, D] dist = torch.norm(x_seq[0, -1]).item() std_val = x_seq.std().item() print_result("Trajectory Integration", dist > 0.5, f"Final Norm: {dist:.2f}") print_result("State Dynamic Range", std_val > 0.05, f"Std: {std_val:.2f}") def audit_mini_parity(): """Test 3: Can we solve 4-bit parity in 50 steps?""" print("\n--- TEST 3: MINI-PARITY CONVERGENCE ---") dim = 64 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32, physics_config={'readout': {'type': 'implicit'}}) optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) inputs = torch.randint(0, 2, (32, 4)) targets = (inputs.sum(dim=-1) % 2).unsqueeze(-1).float() # [32, 1] start_time = time.time() best_loss = 100 for step in range(101): model.train() optimizer.zero_grad() logits, (final_x, _), _ = model(inputs) last_logits = logits[:, -1, 0:1] loss = F.binary_cross_entropy_with_logits(last_logits, targets) loss.backward() optimizer.step() model.readout.update_step() acc = ((last_logits > 0) == targets).float().mean().item() best_loss = min(best_loss, loss.item()) if acc == 1.0 and loss < 0.1: print_result("Parity Solver (L=4)", True, f"Converged at step {step} (Loss: {loss.item():.4f})") return print_result("Parity Solver (L=4)", False, f"Best Loss: {best_loss:.4f}, Accuracy: {acc:.2f}") def audit_physical_limits(): """Test 4: Check for NaNs and Singularities""" print("\n--- TEST 4: PHYSICAL INTEGRITY ---") model = Manifold(vocab_size=10, dim=128, depth=2, heads=4, rank=3) # Low rank to stress test x = torch.randint(0, 10, (128, 50)) logits, (fx, fv), _ = model(x) has_nan = torch.isnan(logits).any().item() or torch.isnan(fx).any().item() max_val = fx.abs().max().item() print_result("NaN Stability", not has_nan, "No NaNs found" if not has_nan else "!!! NaNs DETECTED !!!") print_result("Clamping Effectiveness", max_val <= 101.0, f"Max State: {max_val:.2f}") if __name__ == "__main__": print("====================================================") print("   GFN MANIFOLD MASTER AUDIT (v3.8 Diagnostic)     ") print("====================================================") try: audit_gradient_energy() audit_state_persistence() audit_physical_limits() audit_mini_parity() except Exception as e: print(f"\033[91m[CRITICAL ERROR]\033[0m Audit crashed: {e}") import traceback traceback.print_exc() print("\n====================================================") print("                AUDIT COMPLETE                      ") print("====================================================")
```

### tests\diagnostics\parity_probe.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 24 | probe_latent_clusters | `targets = (inputs.sum(dim=-1) % 2).long()` |
| 40 | probe_latent_clusters | `center0 = x0.mean(axis=0)` |
| 41 | probe_latent_clusters | `center1 = x1.mean(axis=0)` |
| 43 | probe_latent_clusters | `inter_dist = np.linalg.norm(center0 - center1)` |
| 44 | probe_latent_clusters | `intra_std0 = x0.std(axis=0).mean()` |
| 45 | probe_latent_clusters | `intra_std1 = x1.std(axis=0).mean()` |
| 47 | probe_latent_clusters | `print(f"\n--- CLUSTER METRICS (L={L}) ---")` |
| 51 | probe_latent_clusters | `sep_ratio = inter_dist / (intra_std0 + intra_std1 + 1e-9)` |
| 52 | probe_latent_clusters | `status = "\033[92m[GOOD]\033[0m" if sep_ratio > 1.2 else "\033[91m[COLLAPSED]\033[0m" print(f"Separability Ratio:            {sep_ratio:.4f} {status}") if sep_ratio < 0.5: print("\n\033[91mWARNING:\033[0m Manifold states for Parity 0/1 are nearly identical.") print("The manifold is not 'steering' the particle based on inputs.") try: pca = PCA(n_components=2) x_pca = pca.fit_transform(x_latent) expl_var = pca.explained_variance_ratio_.sum() print(f"PCA Variance Explained (2D): {expl_var:.2%}") except: pass def probe_force_signal_ratio(): """Measure the ratio of Christoffel force vs Input force.""" print("\n--- SIGNAL RATIO TEST ---") dim = 128 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.eval() force_1 = model.embedding(torch.tensor([[1]])) # [1, 1, 128] f_norm = force_1.norm().item() v_unit = torch.randn(1, dim) v_unit = v_unit / v_unit.norm() with torch.no_grad(): gamma = model.layers[0].christoffels[0](v_unit) g_norm = gamma.norm().item() print(f"Token Impulse Norm (\|F\|):    {f_norm:.4f}") print(f"Manifold Resitance Norm (\|Γ\|): {g_norm:.4f}") ratio = f_norm / (g_norm + 1e-9) print(f"Force/Curvature Ratio:        {ratio:.2f}x") if ratio > 10.0: print("\033[93m[IMBALANCE]\033[0m Token force dominates geometry. Manifold is too 'soft'.") elif ratio < 0.1: print("\033[93m[IMBALANCE]\033[0m Geometry dominates tokens. Manifold is too 'stiff'.") else: print("\033[92m[BALANCED]\033[0m Dynamics are in the steerable regime.") if __name__ == "__main__": try: probe_force_signal_ratio() probe_latent_clusters() except Exception as e: print(f"Probe failed: {e}") import traceback traceback.print_exc() print("\n====================================================")` |

#### Fórmulas Listas para Usar (Python)
```python
# probe_latent_clusters (L24)
targets = (inputs.sum(dim=-1) % 2).long()
# probe_latent_clusters (L40)
center0 = x0.mean(axis=0)
# probe_latent_clusters (L41)
center1 = x1.mean(axis=0)
# probe_latent_clusters (L43)
inter_dist = np.linalg.norm(center0 - center1)
# probe_latent_clusters (L44)
intra_std0 = x0.std(axis=0).mean()
# probe_latent_clusters (L45)
intra_std1 = x1.std(axis=0).mean()
# probe_latent_clusters (L47)
print(f"\n--- CLUSTER METRICS (L={L}) ---")
# probe_latent_clusters (L51)
sep_ratio = inter_dist / (intra_std0 + intra_std1 + 1e-9)
# probe_latent_clusters (L52)
status = "\033[92m[GOOD]\033[0m" if sep_ratio > 1.2 else "\033[91m[COLLAPSED]\033[0m" print(f"Separability Ratio:            {sep_ratio:.4f} {status}") if sep_ratio < 0.5: print("\n\033[91mWARNING:\033[0m Manifold states for Parity 0/1 are nearly identical.") print("The manifold is not 'steering' the particle based on inputs.") try: pca = PCA(n_components=2) x_pca = pca.fit_transform(x_latent) expl_var = pca.explained_variance_ratio_.sum() print(f"PCA Variance Explained (2D): {expl_var:.2%}") except: pass def probe_force_signal_ratio(): """Measure the ratio of Christoffel force vs Input force.""" print("\n--- SIGNAL RATIO TEST ---") dim = 128 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.eval() force_1 = model.embedding(torch.tensor([[1]])) # [1, 1, 128] f_norm = force_1.norm().item() v_unit = torch.randn(1, dim) v_unit = v_unit / v_unit.norm() with torch.no_grad(): gamma = model.layers[0].christoffels[0](v_unit) g_norm = gamma.norm().item() print(f"Token Impulse Norm (|F|):    {f_norm:.4f}") print(f"Manifold Resitance Norm (|Γ|): {g_norm:.4f}") ratio = f_norm / (g_norm + 1e-9) print(f"Force/Curvature Ratio:        {ratio:.2f}x") if ratio > 10.0: print("\033[93m[IMBALANCE]\033[0m Token force dominates geometry. Manifold is too 'soft'.") elif ratio < 0.1: print("\033[93m[IMBALANCE]\033[0m Geometry dominates tokens. Manifold is too 'stiff'.") else: print("\033[92m[BALANCED]\033[0m Dynamics are in the steerable regime.") if __name__ == "__main__": try: probe_force_signal_ratio() probe_latent_clusters() except Exception as e: print(f"Probe failed: {e}") import traceback traceback.print_exc() print("\n====================================================")
```

### tests\diagnostics\test_loss_evolution.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 18 | test_gradient_flow | `optimizer = RiemannianAdam(model.parameters(), lr=1e-3)` |
| 32 | test_gradient_flow | `logits, (x_final, v_final), *_ = model(x_task)` |
| 44 | test_gradient_flow | `total_grad += p.grad.norm().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_gradient_flow (L18)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# test_gradient_flow (L32)
logits, (x_final, v_final), *_ = model(x_task)
# test_gradient_flow (L44)
total_grad += p.grad.norm().item()
```

### tests\diagnostics\test_suite_comprehensive.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 65 | func | `test = gradcheck(func, (embeddings,), eps=1e-6, atol=1e-3, rtol=1e-2)` |
| 128 | run_free_motion_check | `final_norm = xF.norm().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# func (L65)
test = gradcheck(func, (embeddings,), eps=1e-6, atol=1e-3, rtol=1e-2)
# run_free_motion_check (L128)
final_norm = xF.norm().item()
```

### tests\diagnostics\verify_convergence_dual.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | run_experiment | `print(f"\n{'='*60}")` |
| 18 | run_experiment | `print(f"{'='*60}")` |
| 33 | run_experiment | `optimizer = RiemannianAdam(model.parameters(), lr=1e-3)` |
| 49 | run_experiment | `y_angle = (y_int.float() * 2.0 - 1.0) * (pi * 0.5)` |
| 59 | run_experiment | `y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)` |
| 77 | first_head_metric | `total_loss = loss_val + loss_phy + loss_ham` |
| 86 | first_head_metric | `two_pi = 2.0 * pi` |
| 87 | first_head_metric | `half_pi = pi * 0.5` |
| 88 | first_head_metric | `dist_pos = torch.min(torch.abs(x_pred - half_pi) % two_pi, two_pi - (torch.abs(x_pred - half_pi) % two_pi))` |
| 89 | first_head_metric | `dist_neg = torch.min(torch.abs(x_pred + half_pi) % two_pi, two_pi - (torch.abs(x_pred + half_pi) % two_pi))` |
| 90 | first_head_metric | `d_pos = dist_pos.mean(dim=-1)` |
| 91 | first_head_metric | `d_neg = dist_neg.mean(dim=-1)` |
| 93 | first_head_metric | `acc = (preds == y_int).float().mean().item()` |
| 94 | first_head_metric | `print(f"Step {step}: Loss={total_loss.item():.4f}, Acc={acc*100:.1f}%")` |
| 96 | first_head_metric | `duration = time.time() - start_time` |
| 114 | first_head_metric | `print("\n" + "=" * 60)` |
| 120 | first_head_metric | `converged_py = losses_py[-1] < losses_py[0] * 0.8` |
| 121 | first_head_metric | `converged_cuda = losses_cuda[-1] < losses_cuda[0] * 0.8` |

#### Fórmulas Listas para Usar (Python)
```python
# run_experiment (L16)
print(f"\n{'='*60}")
# run_experiment (L18)
print(f"{'='*60}")
# run_experiment (L33)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# run_experiment (L49)
y_angle = (y_int.float() * 2.0 - 1.0) * (pi * 0.5)
# run_experiment (L59)
y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
# first_head_metric (L77)
total_loss = loss_val + loss_phy + loss_ham
# first_head_metric (L86)
two_pi = 2.0 * pi
# first_head_metric (L87)
half_pi = pi * 0.5
# first_head_metric (L88)
dist_pos = torch.min(torch.abs(x_pred - half_pi) % two_pi, two_pi - (torch.abs(x_pred - half_pi) % two_pi))
# first_head_metric (L89)
dist_neg = torch.min(torch.abs(x_pred + half_pi) % two_pi, two_pi - (torch.abs(x_pred + half_pi) % two_pi))
# first_head_metric (L90)
d_pos = dist_pos.mean(dim=-1)
# first_head_metric (L91)
d_neg = dist_neg.mean(dim=-1)
# first_head_metric (L93)
acc = (preds == y_int).float().mean().item()
# first_head_metric (L94)
print(f"Step {step}: Loss={total_loss.item():.4f}, Acc={acc*100:.1f}%")
# first_head_metric (L96)
duration = time.time() - start_time
# first_head_metric (L114)
print("\n" + "=" * 60)
# first_head_metric (L120)
converged_py = losses_py[-1] < losses_py[0] * 0.8
# first_head_metric (L121)
converged_cuda = losses_cuda[-1] < losses_cuda[0] * 0.8
```

### tests\functional\test_curiosity.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 49 | test_curiosity_noise_flow | `force_high = torch.ones(100, dim) * 10.0` |
| 52 | test_curiosity_noise_flow | `std_low = v_low.std().item()` |
| 53 | test_curiosity_noise_flow | `std_high = v_high.std().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_curiosity_noise_flow (L49)
force_high = torch.ones(100, dim) * 10.0
# test_curiosity_noise_flow (L52)
std_low = v_low.std().item()
# test_curiosity_noise_flow (L53)
std_high = v_high.std().item()
```

### tests\functional\test_curiosity_exploration.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 38 | test_exploration_coverage | `force = torch.ones(batch, self.dim) * 5.0` |
| 58 | test_exploration_coverage | `var_on = max(var_on, x_on.var(dim=0).mean().item())` |
| 59 | test_exploration_coverage | `var_off = max(var_off, x_off.var(dim=0).mean().item())` |
| 74 | test_confusion_correlation | `v = torch.zeros(1000, self.dim // self.heads) # Head dim` |
| 77 | test_confusion_correlation | `v_zero = noise_mod(v, force=torch.zeros(1000, self.dim // self.heads))` |
| 78 | test_confusion_correlation | `std_zero = v_zero.std().item()` |
| 81 | test_confusion_correlation | `v_high = noise_mod(v, force=torch.ones(1000, self.dim // self.heads) * 5.0)` |
| 82 | test_confusion_correlation | `std_high = v_high.std().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_exploration_coverage (L38)
force = torch.ones(batch, self.dim) * 5.0
# test_exploration_coverage (L58)
var_on = max(var_on, x_on.var(dim=0).mean().item())
# test_exploration_coverage (L59)
var_off = max(var_off, x_off.var(dim=0).mean().item())
# test_confusion_correlation (L74)
v = torch.zeros(1000, self.dim // self.heads) # Head dim
# test_confusion_correlation (L77)
v_zero = noise_mod(v, force=torch.zeros(1000, self.dim // self.heads))
# test_confusion_correlation (L78)
std_zero = v_zero.std().item()
# test_confusion_correlation (L81)
v_high = noise_mod(v, force=torch.ones(1000, self.dim // self.heads) * 5.0)
# test_confusion_correlation (L82)
std_high = v_high.std().item()
```

### tests\functional\test_noether.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 22 | test_noether_loss_zero_for_identical_heads | `head_dim = dim // heads` |
| 48 | test_noether_loss_penalizes_divergence | `c1 = c0 + 0.1 * torch.randn(4, 4) # Perturb` |
| 78 | test_gradient_flow | `has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in layer.parameters())` |

#### Fórmulas Listas para Usar (Python)
```python
# test_noether_loss_zero_for_identical_heads (L22)
head_dim = dim // heads
# test_noether_loss_penalizes_divergence (L48)
c1 = c0 + 0.1 * torch.randn(4, 4) # Perturb
# test_gradient_flow (L78)
has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in layer.parameters())
```

### tests\functional\test_time_dilation.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 29 | test_thermo_gating_module | `v_low = torch.zeros(batch, dim) # K=0, U=0 -> H=0. H < Ref (5.0). Gate > 0.5` |
| 34 | test_thermo_gating_module | `x_high = torch.randn(batch, dim) * 10` |
| 35 | test_thermo_gating_module | `v_high = torch.randn(batch, dim) * 10 # H >> 5.0. Gate < 0.5` |

#### Fórmulas Listas para Usar (Python)
```python
# test_thermo_gating_module (L29)
v_low = torch.zeros(batch, dim) # K=0, U=0 -> H=0. H < Ref (5.0). Gate > 0.5
# test_thermo_gating_module (L34)
x_high = torch.randn(batch, dim) * 10
# test_thermo_gating_module (L35)
v_high = torch.randn(batch, dim) * 10 # H >> 5.0. Gate < 0.5
```

### tests\geometry\test_confusion.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | forward | `def forward(self, v, x, force=None, **kwargs):` |
| 60 | test_mlayer_integration | `head_dim = dim // heads` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L16)
def forward(self, v, x, force=None, **kwargs):
# test_mlayer_integration (L60)
head_dim = dim // heads
```

### tests\geometry\test_holographic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 15 | forward | `def forward(self, v, x=None, **kwargs):` |
| 39 | test_radial_clamping | `x = torch.ones(1, dim) * 1000.0` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L15)
def forward(self, v, x=None, **kwargs):
# test_radial_clamping (L39)
x = torch.ones(1, dim) * 1000.0
```

### tests\geometry\test_thermo_metric.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | forward | `def forward(self, v, x, force=None, **kwargs):` |
| 33 | test_thermo_modulation | `self.assertTrue(torch.allclose(out_low, torch.ones_like(out_low), atol=1e-4))` |
| 38 | test_thermo_modulation | `force_high = torch.ones(batch, dim) * 100.0` |
| 65 | test_mlayer_integration | `head_dim = dim // heads` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L16)
def forward(self, v, x, force=None, **kwargs):
# test_thermo_modulation (L33)
self.assertTrue(torch.allclose(out_low, torch.ones_like(out_low), atol=1e-4))
# test_thermo_modulation (L38)
force_high = torch.ones(batch, dim) * 100.0
# test_mlayer_integration (L65)
head_dim = dim // heads
```

### tests\geometry\test_torus.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 27 | test_torus_wrapping | `v_state = torch.ones(B, D).cuda() * 50.0 # Direction matters, mag normalized` |
| 31 | test_torus_wrapping | `U = torch.randn(1*D, D, 16).cuda() * 0.01` |
| 32 | test_torus_wrapping | `W = torch.randn(1*D, D, 16).cuda() * 0.01` |
| 71 | test_torus_wrapping | `TWO_PI = 2 * math.pi` |
| 82 | test_torus_wrapping | `diff = torch.abs(x_seq_euc - x_seq_tor).mean()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_torus_wrapping (L27)
v_state = torch.ones(B, D).cuda() * 50.0 # Direction matters, mag normalized
# test_torus_wrapping (L31)
U = torch.randn(1*D, D, 16).cuda() * 0.01
# test_torus_wrapping (L32)
W = torch.randn(1*D, D, 16).cuda() * 0.01
# test_torus_wrapping (L71)
TWO_PI = 2 * math.pi
# test_torus_wrapping (L82)
diff = torch.abs(x_seq_euc - x_seq_tor).mean()
```

### tests\integration\test_full_training.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 54 | test_forward_pass | `logits, (x, v), *_ = small_model(inputs)` |
| 92 | test_energy_conservation_basic | `x = small_model.x0.expand(1, -1)` |
| 93 | test_energy_conservation_basic | `v = small_model.v0.expand(1, -1)` |
| 101 | test_energy_conservation_basic | `energy = (v ** 2).sum().item()` |
| 109 | test_energy_conservation_basic | `drift = abs(energies[-1] - energies[0]) / (energies[0] + 1e-8)` |
| 200 | test_parameter_count | `params = sum(p.numel() for p in model.parameters()) / 1e6` |
| 207 | test_parameter_count | `pytest.main([__file__, "-v", "--tb=short"])` |

#### Fórmulas Listas para Usar (Python)
```python
# test_forward_pass (L54)
logits, (x, v), *_ = small_model(inputs)
# test_energy_conservation_basic (L92)
x = small_model.x0.expand(1, -1)
# test_energy_conservation_basic (L93)
v = small_model.v0.expand(1, -1)
# test_energy_conservation_basic (L101)
energy = (v ** 2).sum().item()
# test_energy_conservation_basic (L109)
drift = abs(energies[-1] - energies[0]) / (energies[0] + 1e-8)
# test_parameter_count (L200)
params = sum(p.numel() for p in model.parameters()) / 1e6
# test_parameter_count (L207)
pytest.main([__file__, "-v", "--tb=short"])
```

### tests\integration\test_overfit_sanity.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 50 | run_training_task | `lr = config.get('lr', 1e-3)` |
| 78 | run_training_task | `shift_logits = logits[:, :-1, :].contiguous()` |
| 110 | run_training_task | `duration = end_time - start_time` |

#### Fórmulas Listas para Usar (Python)
```python
# run_training_task (L50)
lr = config.get('lr', 1e-3)
# run_training_task (L78)
shift_logits = logits[:, :-1, :].contiguous()
# run_training_task (L110)
duration = end_time - start_time
```

### tests\integration\test_vnext_stack.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 37 | test_vnext_full_stack | `loss = output[0].sum()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_vnext_full_stack (L37)
loss = output[0].sum()
```

### tests\optimization\analyze.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 81 | analyze_results | `parser.add_argument("--output", type=str, default="optimization_report.md", help="Output report file")` |

#### Fórmulas Listas para Usar (Python)
```python
# analyze_results (L81)
parser.add_argument("--output", type=str, default="optimization_report.md", help="Output report file")
```

### tests\optimization\config_space.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 19 | HyperparameterConfig | `SEARCH_SPACE = [ HyperparameterConfig( name="DEFAULT_LR", values=[1e-4, 5e-4, 1e-3, 5e-3], description="Learning rate" ), HyperparameterConfig( name="EMBEDDING_SCALE", values=[1.0, 1.5, 2.0], description="Scale for input embeddings" ), HyperparameterConfig( name="READOUT_GAIN", values=[1.0, 2.0, 5.0], description="Gain for the final readout layer" ), HyperparameterConfig( name="FRICTION_SCALE", values=[0.0, 0.02, 0.05, 0.1], description="Friction coefficient for symplectic integrators" ), HyperparameterConfig( name="DEFAULT_DT", values=[0.01, 0.02, 0.05, 0.1], description="Time step for integration" ), HyperparameterConfig( name="LEAPFROG_SUBSTEPS", values=[1, 3, 5], description="Number of substeps for Leapfrog integrator" ), HyperparameterConfig( name="LAMBDA_H_DEFAULT", values=[0.0, 0.001, 0.01], description="Hamiltonian regularization weight" ) ]` |
| 64 | get_grid_search_configs | `combinations = list(itertools.product(*values))` |
| 73 | get_random_search_configs | `def get_random_search_configs(n_samples: int = 20) -> List[Dict[str, Any]]:` |

#### Fórmulas Listas para Usar (Python)
```python
# HyperparameterConfig (L19)
SEARCH_SPACE = [ HyperparameterConfig( name="DEFAULT_LR", values=[1e-4, 5e-4, 1e-3, 5e-3], description="Learning rate" ), HyperparameterConfig( name="EMBEDDING_SCALE", values=[1.0, 1.5, 2.0], description="Scale for input embeddings" ), HyperparameterConfig( name="READOUT_GAIN", values=[1.0, 2.0, 5.0], description="Gain for the final readout layer" ), HyperparameterConfig( name="FRICTION_SCALE", values=[0.0, 0.02, 0.05, 0.1], description="Friction coefficient for symplectic integrators" ), HyperparameterConfig( name="DEFAULT_DT", values=[0.01, 0.02, 0.05, 0.1], description="Time step for integration" ), HyperparameterConfig( name="LEAPFROG_SUBSTEPS", values=[1, 3, 5], description="Number of substeps for Leapfrog integrator" ), HyperparameterConfig( name="LAMBDA_H_DEFAULT", values=[0.0, 0.001, 0.01], description="Hamiltonian regularization weight" ) ]
# get_grid_search_configs (L64)
combinations = list(itertools.product(*values))
# get_random_search_configs (L73)
def get_random_search_configs(n_samples: int = 20) -> List[Dict[str, Any]]:
```

### tests\optimization\runner.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 70 | run_trial | `result = { "trial_id": trial_id, "status": "COMPLETED" if train_metrics["success"] else "FAILED", **config, **train_metrics }` |
| 79 | run_trial | `result = { "trial_id": trial_id, "status": "ERROR", "error": str(e), **config }` |
| 94 | main | `parser.add_argument("--mode", choices=["grid", "random", "smoke"], default="smoke", help="Search strategy")` |
| 95 | main | `parser.add_argument("--samples", type=int, default=10, help="Number of samples for random search")` |
| 96 | main | `parser.add_argument("--output", type=str, default="optimization_results.csv", help="Output CSV file")` |
| 124 | main | `total_time = time.time() - start_time` |

#### Fórmulas Listas para Usar (Python)
```python
# run_trial (L70)
result = { "trial_id": trial_id, "status": "COMPLETED" if train_metrics["success"] else "FAILED", **config, **train_metrics }
# run_trial (L79)
result = { "trial_id": trial_id, "status": "ERROR", "error": str(e), **config }
# main (L94)
parser.add_argument("--mode", choices=["grid", "random", "smoke"], default="smoke", help="Search strategy")
# main (L95)
parser.add_argument("--samples", type=int, default=10, help="Number of samples for random search")
# main (L96)
parser.add_argument("--output", type=str, default="optimization_results.csv", help="Output CSV file")
# main (L124)
total_time = time.time() - start_time
```

### tests\optimization\test_optimizer.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 34 | test_orthogonal_preservation_cayley | `grad_skew_error = torch.norm(grad_skewed + grad_skewed.t())` |
| 41 | test_orthogonal_preservation_cayley | `ortho_error = torch.norm(p.data @ p.data.t() - identity)` |
| 56 | test_torus_wrapping_and_transport | `p.grad = torch.full((1, dim), -1.0) # p = p - lr * (-1) = 3.1 + 0.5 = 3.6 > pi` |
| 65 | test_torus_wrapping_and_transport | `self.assertAlmostEqual(val, 3.6 - 2*math.pi, places=4)` |
| 76 | test_sphere_projection | `p.data = p.data / p.data.norm() # On sphere` |
| 86 | test_sphere_projection | `proj_norm = projected_grad.norm().item()` |
| 87 | test_sphere_projection | `print(f"DEBUG: p_norm={p.data.norm().item()}, grad_norm={p.grad.data.norm().item()}")` |

#### Fórmulas Listas para Usar (Python)
```python
# test_orthogonal_preservation_cayley (L34)
grad_skew_error = torch.norm(grad_skewed + grad_skewed.t())
# test_orthogonal_preservation_cayley (L41)
ortho_error = torch.norm(p.data @ p.data.t() - identity)
# test_torus_wrapping_and_transport (L56)
p.grad = torch.full((1, dim), -1.0) # p = p - lr * (-1) = 3.1 + 0.5 = 3.6 > pi
# test_torus_wrapping_and_transport (L65)
self.assertAlmostEqual(val, 3.6 - 2*math.pi, places=4)
# test_sphere_projection (L76)
p.data = p.data / p.data.norm() # On sphere
# test_sphere_projection (L86)
proj_norm = projected_grad.norm().item()
# test_sphere_projection (L87)
print(f"DEBUG: p_norm={p.data.norm().item()}, grad_norm={p.grad.data.norm().item()}")
```

### tests\physics\test_adaptive_resolution.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 27 | test_adaptive_execution_flow | `adaptive = AdaptiveIntegrator(base, tolerance=1e-5, max_depth=2)` |
| 44 | test_mlayer_integration | `config = { 'active_inference': { 'adaptive_resolution': { 'enabled': True, 'tolerance': 1e-4, 'max_depth': 2 } } }` |

#### Fórmulas Listas para Usar (Python)
```python
# test_adaptive_execution_flow (L27)
adaptive = AdaptiveIntegrator(base, tolerance=1e-5, max_depth=2)
# test_mlayer_integration (L44)
config = { 'active_inference': { 'adaptive_resolution': { 'enabled': True, 'tolerance': 1e-4, 'max_depth': 2 } } }
```

### tests\physics\test_energy_conservation.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 33 | __init__ | `self.results_dir = PROJECT_ROOT / "tests" / "professional" / "results"` |
| 63 | test_long_sequence_drift | `x = self.model.x0.expand(1, -1)` |
| 64 | test_long_sequence_drift | `v = self.model.v0.expand(1, -1)` |
| 78 | test_long_sequence_drift | `energy = (v ** 2).sum().item()` |
| 83 | test_long_sequence_drift | `energies = np.array(energies)` |
| 85 | test_long_sequence_drift | `final_energy = energies[-1]` |
| 88 | test_long_sequence_drift | `relative_drift = abs(final_energy - initial_energy) / (initial_energy + 1e-8)` |
| 91 | test_long_sequence_drift | `max_deviation = np.max(np.abs(energies - initial_energy)) / (initial_energy + 1e-8)` |
| 94 | test_long_sequence_drift | `stability_score = max(0.0, 1.0 - relative_drift / tolerance)` |
| 102 | test_long_sequence_drift | `plt.axhline(y=initial_energy, color='r', linestyle='--', label='Initial Energy', alpha=0.7)` |
| 107 | test_long_sequence_drift | `alpha=0.2, color='green', label=f'±{tolerance*100:.0f}% Tolerance'` |
| 111 | test_long_sequence_drift | `plt.title(f'Hamiltonian Energy Conservation (Drift: {relative_drift*100:.2f}%)', fontsize=14)` |
| 117 | test_long_sequence_drift | `rel_deviations = (energies - initial_energy) / (initial_energy + 1e-8) * 100` |
| 119 | test_long_sequence_drift | `plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)` |
| 120 | test_long_sequence_drift | `plt.axhline(y=tolerance*100, color='r', linestyle='--', alpha=0.5, label=f'Tolerance: ±{tolerance*100:.0f}%')` |
| 121 | test_long_sequence_drift | `plt.axhline(y=-tolerance*100, color='r', linestyle='--', alpha=0.5)` |
| 129 | test_long_sequence_drift | `plt.savefig(self.results_dir / "energy_conservation_long_sequence.png", dpi=300, bbox_inches='tight')` |
| 184 | test_integrator_comparison | `x = model.x0.expand(1, -1)` |
| 185 | test_integrator_comparison | `v = model.v0.expand(1, -1)` |
| 193 | test_integrator_comparison | `energy = (v ** 2).sum().item()` |
| 196 | test_integrator_comparison | `energies = np.array(energies)` |
| 198 | test_integrator_comparison | `drift = abs(energies[-1] - initial) / (initial + 1e-8)` |
| 201 | test_integrator_comparison | `results[int_type] = { "drift": drift, "energies": energies, "mean_energy": np.mean(energies), "std_energy": np.std(energies) }` |
| 211 | test_integrator_comparison | `ax.axhline(y=initial, color='r', linestyle='--', alpha=0.5, label='Initial')` |
| 214 | test_integrator_comparison | `ax.set_title(f'{int_type.upper()}\nDrift: {drift*100:.2f}%', fontsize=13)` |
| 222 | test_integrator_comparison | `plt.savefig(self.results_dir / "integrator_comparison.png", dpi=300, bbox_inches='tight')` |
| 261 | test_adversarial_stability | `x = self.model.x0.expand(1, -1)` |
| 262 | test_adversarial_stability | `v = self.model.v0.expand(1, -1)` |
| 274 | test_adversarial_stability | `energy = (v ** 2).sum().item()` |
| 284 | test_adversarial_stability | `spike = max(energies) / (energies[0] + 1e-8)` |
| 287 | test_adversarial_stability | `results[pattern_name] = { "nan_frequency": nan_count / num_trials, "max_energy_spike": max_energy_spike }` |
| 340 | run_comprehensive_suite | `print("\n" + "=" * 60)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L33)
self.results_dir = PROJECT_ROOT / "tests" / "professional" / "results"
# test_long_sequence_drift (L63)
x = self.model.x0.expand(1, -1)
# test_long_sequence_drift (L64)
v = self.model.v0.expand(1, -1)
# test_long_sequence_drift (L78)
energy = (v ** 2).sum().item()
# test_long_sequence_drift (L83)
energies = np.array(energies)
# test_long_sequence_drift (L85)
final_energy = energies[-1]
# test_long_sequence_drift (L88)
relative_drift = abs(final_energy - initial_energy) / (initial_energy + 1e-8)
# test_long_sequence_drift (L91)
max_deviation = np.max(np.abs(energies - initial_energy)) / (initial_energy + 1e-8)
# test_long_sequence_drift (L94)
stability_score = max(0.0, 1.0 - relative_drift / tolerance)
# test_long_sequence_drift (L102)
plt.axhline(y=initial_energy, color='r', linestyle='--', label='Initial Energy', alpha=0.7)
# test_long_sequence_drift (L107)
alpha=0.2, color='green', label=f'±{tolerance*100:.0f}% Tolerance'
# test_long_sequence_drift (L111)
plt.title(f'Hamiltonian Energy Conservation (Drift: {relative_drift*100:.2f}%)', fontsize=14)
# test_long_sequence_drift (L117)
rel_deviations = (energies - initial_energy) / (initial_energy + 1e-8) * 100
# test_long_sequence_drift (L119)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
# test_long_sequence_drift (L120)
plt.axhline(y=tolerance*100, color='r', linestyle='--', alpha=0.5, label=f'Tolerance: ±{tolerance*100:.0f}%')
# test_long_sequence_drift (L121)
plt.axhline(y=-tolerance*100, color='r', linestyle='--', alpha=0.5)
# test_long_sequence_drift (L129)
plt.savefig(self.results_dir / "energy_conservation_long_sequence.png", dpi=300, bbox_inches='tight')
# test_integrator_comparison (L184)
x = model.x0.expand(1, -1)
# test_integrator_comparison (L185)
v = model.v0.expand(1, -1)
# test_integrator_comparison (L193)
energy = (v ** 2).sum().item()
# test_integrator_comparison (L196)
energies = np.array(energies)
# test_integrator_comparison (L198)
drift = abs(energies[-1] - initial) / (initial + 1e-8)
# test_integrator_comparison (L201)
results[int_type] = { "drift": drift, "energies": energies, "mean_energy": np.mean(energies), "std_energy": np.std(energies) }
# test_integrator_comparison (L211)
ax.axhline(y=initial, color='r', linestyle='--', alpha=0.5, label='Initial')
# test_integrator_comparison (L214)
ax.set_title(f'{int_type.upper()}\nDrift: {drift*100:.2f}%', fontsize=13)
# test_integrator_comparison (L222)
plt.savefig(self.results_dir / "integrator_comparison.png", dpi=300, bbox_inches='tight')
# test_adversarial_stability (L261)
x = self.model.x0.expand(1, -1)
# test_adversarial_stability (L262)
v = self.model.v0.expand(1, -1)
# test_adversarial_stability (L274)
energy = (v ** 2).sum().item()
# test_adversarial_stability (L284)
spike = max(energies) / (energies[0] + 1e-8)
# test_adversarial_stability (L287)
results[pattern_name] = { "nan_frequency": nan_count / num_trials, "max_energy_spike": max_energy_spike }
# run_comprehensive_suite (L340)
print("\n" + "=" * 60)
```

### tests\physics\test_geodesic_optimality.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 35 | __init__ | `self.results_dir = PROJECT_ROOT / "tests" / "results" / "geodesic"` |
| 42 | compute_path_length | `For a Riemannian manifold, path length = ∫ \|\|dx/dt\|\| dt` |
| 56 | compute_path_length | `displacement = trajectory[i+1] - trajectory[i]` |
| 57 | compute_path_length | `step_length = torch.norm(displacement, dim=-1).mean().item()` |
| 58 | compute_path_length | `total_length += step_length` |
| 83 | test_curved_vs_straight | `x = self.model.x0.expand(1, -1)` |
| 84 | test_curved_vs_straight | `v = self.model.v0.expand(1, -1)` |
| 96 | test_curved_vs_straight | `end_point = geodesic_trajectory[-1]` |
| 100 | test_curved_vs_straight | `alpha = t / (seq_length - 1)` |
| 101 | test_curved_vs_straight | `interpolated = (1 - alpha) * start_point + alpha * end_point` |
| 111 | test_curved_vs_straight | `deviation = torch.norm(geodesic_trajectory[t] - straight_trajectory[t]).item()` |
| 115 | test_curved_vs_straight | `mean_deviation = np.mean(deviations)` |
| 127 | test_curved_vs_straight | `plt.savefig(self.results_dir / "geodesic_deviation.png", dpi=300, bbox_inches='tight')` |
| 172 | _visualize_paths_3d | `color='#E76F51', linestyle='--', alpha=0.7)` |
| 175 | _visualize_paths_3d | `ax.scatter(*geo_3d[0], s=200, c='green', marker='o', label='Start', edgecolors='black', linewidths=2)` |
| 176 | _visualize_paths_3d | `ax.scatter(*geo_3d[-1], s=200, c='red', marker='X', label='End', edgecolors='black', linewidths=2)` |
| 187 | _visualize_paths_3d | `plt.savefig(self.results_dir / "geodesic_path_3d.png", dpi=300, bbox_inches='tight')` |
| 200 | test_manifold_curvature_field | `head_dim = self.model.dim // heads` |
| 205 | test_manifold_curvature_field | `x = np.linspace(-3, 3, grid_size)` |
| 206 | test_manifold_curvature_field | `y = np.linspace(-3, 3, grid_size)` |
| 207 | test_manifold_curvature_field | `X, Y = np.meshgrid(x, y)` |
| 209 | test_manifold_curvature_field | `curvatures = np.zeros((grid_size, grid_size))` |
| 224 | test_manifold_curvature_field | `curvatures[i, j] = torch.norm(gamma).item()` |
| 228 | test_manifold_curvature_field | `im = plt.imshow(curvatures, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis', aspect='auto')` |
| 238 | test_manifold_curvature_field | `plt.savefig(self.results_dir / "curvature_field.png", dpi=300, bbox_inches='tight')` |
| 242 | test_manifold_curvature_field | `mean_curv = np.mean(curvatures)` |
| 243 | test_manifold_curvature_field | `max_curv = np.max(curvatures)` |
| 244 | test_manifold_curvature_field | `std_curv = np.std(curvatures)` |
| 277 | test_action_minimization | `x = self.model.x0.expand(1, -1)` |
| 278 | test_action_minimization | `v = self.model.v0.expand(1, -1)` |
| 289 | test_action_minimization | `gfn_action = sum((v ** 2).sum().item() for v in velocities)` |
| 300 | test_action_minimization | `noise = torch.randn_like(v) * noise_scale` |
| 301 | test_action_minimization | `v_pert = v + noise` |
| 302 | test_action_minimization | `perturbed_action += (v_pert ** 2).sum().item()` |
| 306 | test_action_minimization | `mean_perturbed = np.mean(perturbed_actions)` |
| 318 | test_action_minimization | `plt.axvline(gfn_action, color='red', linestyle='--', linewidth=2.5, label='GFN Geodesic')` |
| 324 | test_action_minimization | `plt.savefig(self.results_dir / "action_minimization.png", dpi=300, bbox_inches='tight')` |
| 371 | run_geodesic_tests | `print("\n" + "=" * 60)` |

#### Fórmulas Listas para Usar (Python)
```python
# __init__ (L35)
self.results_dir = PROJECT_ROOT / "tests" / "results" / "geodesic"
# compute_path_length (L42)
For a Riemannian manifold, path length = ∫ ||dx/dt|| dt
# compute_path_length (L56)
displacement = trajectory[i+1] - trajectory[i]
# compute_path_length (L57)
step_length = torch.norm(displacement, dim=-1).mean().item()
# compute_path_length (L58)
total_length += step_length
# test_curved_vs_straight (L83)
x = self.model.x0.expand(1, -1)
# test_curved_vs_straight (L84)
v = self.model.v0.expand(1, -1)
# test_curved_vs_straight (L96)
end_point = geodesic_trajectory[-1]
# test_curved_vs_straight (L100)
alpha = t / (seq_length - 1)
# test_curved_vs_straight (L101)
interpolated = (1 - alpha) * start_point + alpha * end_point
# test_curved_vs_straight (L111)
deviation = torch.norm(geodesic_trajectory[t] - straight_trajectory[t]).item()
# test_curved_vs_straight (L115)
mean_deviation = np.mean(deviations)
# test_curved_vs_straight (L127)
plt.savefig(self.results_dir / "geodesic_deviation.png", dpi=300, bbox_inches='tight')
# _visualize_paths_3d (L172)
color='#E76F51', linestyle='--', alpha=0.7)
# _visualize_paths_3d (L175)
ax.scatter(*geo_3d[0], s=200, c='green', marker='o', label='Start', edgecolors='black', linewidths=2)
# _visualize_paths_3d (L176)
ax.scatter(*geo_3d[-1], s=200, c='red', marker='X', label='End', edgecolors='black', linewidths=2)
# _visualize_paths_3d (L187)
plt.savefig(self.results_dir / "geodesic_path_3d.png", dpi=300, bbox_inches='tight')
# test_manifold_curvature_field (L200)
head_dim = self.model.dim // heads
# test_manifold_curvature_field (L205)
x = np.linspace(-3, 3, grid_size)
# test_manifold_curvature_field (L206)
y = np.linspace(-3, 3, grid_size)
# test_manifold_curvature_field (L207)
X, Y = np.meshgrid(x, y)
# test_manifold_curvature_field (L209)
curvatures = np.zeros((grid_size, grid_size))
# test_manifold_curvature_field (L224)
curvatures[i, j] = torch.norm(gamma).item()
# test_manifold_curvature_field (L228)
im = plt.imshow(curvatures, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis', aspect='auto')
# test_manifold_curvature_field (L238)
plt.savefig(self.results_dir / "curvature_field.png", dpi=300, bbox_inches='tight')
# test_manifold_curvature_field (L242)
mean_curv = np.mean(curvatures)
# test_manifold_curvature_field (L243)
max_curv = np.max(curvatures)
# test_manifold_curvature_field (L244)
std_curv = np.std(curvatures)
# test_action_minimization (L277)
x = self.model.x0.expand(1, -1)
# test_action_minimization (L278)
v = self.model.v0.expand(1, -1)
# test_action_minimization (L289)
gfn_action = sum((v ** 2).sum().item() for v in velocities)
# test_action_minimization (L300)
noise = torch.randn_like(v) * noise_scale
# test_action_minimization (L301)
v_pert = v + noise
# test_action_minimization (L302)
perturbed_action += (v_pert ** 2).sum().item()
# test_action_minimization (L306)
mean_perturbed = np.mean(perturbed_actions)
# test_action_minimization (L318)
plt.axvline(gfn_action, color='red', linestyle='--', linewidth=2.5, label='GFN Geodesic')
# test_action_minimization (L324)
plt.savefig(self.results_dir / "action_minimization.png", dpi=300, bbox_inches='tight')
# run_geodesic_tests (L371)
print("\n" + "=" * 60)
```

### tests\physics\test_gradients.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 35 | check_gradients | `loss = criterion(logits.view(-1, vocab), target.view(-1))` |
| 47 | check_gradients | `g_norm = param.grad.norm().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# check_gradients (L35)
loss = criterion(logits.view(-1, vocab), target.view(-1))
# check_gradients (L47)
g_norm = param.grad.norm().item()
```

### tests\physics\test_gradients_deep.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 25 | compute_numerical_gradients | `def compute_numerical_gradients(func, inputs, output_indices=None, eps=1e-5):` |
| 41 | compute_numerical_gradients | `outputs = func(*inputs)` |
| 59 | compute_numerical_gradients | `flat_input = input_tensor.view(-1)` |
| 60 | compute_numerical_gradients | `flat_grad = grad.view(-1)` |
| 61 | compute_numerical_gradients | `original_flat = original_data.view(-1)` |
| 65 | compute_numerical_gradients | `original_flat[j] += eps` |
| 67 | compute_numerical_gradients | `outputs_plus = func(*inputs)` |
| 71 | compute_numerical_gradients | `original_flat[j] -= 2 * eps` |
| 73 | compute_numerical_gradients | `outputs_minus = func(*inputs)` |
| 78 | compute_numerical_gradients | `original_flat[j] += eps` |
| 85 | compute_numerical_gradients | `diff = (plus - minus) / (2 * eps)` |
| 86 | compute_numerical_gradients | `grad_sum += diff.sum().item()` |
| 122 | test_backward_vs_numerical | `W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)  # For Torus topology` |
| 155 | test_backward_vs_numerical | `features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) if topology == 1 else x` |
| 156 | test_backward_vs_numerical | `mu = torch.sigmoid(torch.matmul(features, W_forget.t()) + b_forget) * 5.0  # FRICTION_SCALE` |
| 157 | test_backward_vs_numerical | `friction_output = gamma + mu * v` |
| 162 | test_backward_vs_numerical | `forward_diff = torch.abs(output_cuda - friction_output).max().item()` |
| 179 | test_backward_vs_numerical | `loss = output_cuda.sum()` |
| 197 | forward_func | `tolerances = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]  # Tolerances per parameter` |
| 217 | forward_func | `abs_diff = torch.abs(grad_cuda - grad_num)` |
| 218 | forward_func | `rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)` |
| 222 | forward_func | `mean_abs_error = abs_diff.mean().item()` |
| 223 | forward_func | `mean_rel_error = rel_diff.mean().item()` |
| 234 | forward_func | `if max_abs_error <= tol and max_rel_error <= 10 * tol:` |
| 242 | forward_func | `cuda_val = grad_cuda.view(-1)[max_idx].item()` |
| 243 | forward_func | `num_val = grad_num.view(-1)[max_idx].item()` |
| 271 | test_gradient_checking_pytorch | `print("\n" + "=" * 70)` |
| 292 | test_gradient_checking_pytorch | `W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)` |
| 301 | test_gradient_checking_pytorch | `result = gradcheck( LowRankChristoffelWithFrictionFunction.apply, test_input, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False )` |
| 331 | test_gradient_checking_pytorch | `print("\n" + "=" * 70)` |

#### Fórmulas Listas para Usar (Python)
```python
# compute_numerical_gradients (L25)
def compute_numerical_gradients(func, inputs, output_indices=None, eps=1e-5):
# compute_numerical_gradients (L41)
outputs = func(*inputs)
# compute_numerical_gradients (L59)
flat_input = input_tensor.view(-1)
# compute_numerical_gradients (L60)
flat_grad = grad.view(-1)
# compute_numerical_gradients (L61)
original_flat = original_data.view(-1)
# compute_numerical_gradients (L65)
original_flat[j] += eps
# compute_numerical_gradients (L67)
outputs_plus = func(*inputs)
# compute_numerical_gradients (L71)
original_flat[j] -= 2 * eps
# compute_numerical_gradients (L73)
outputs_minus = func(*inputs)
# compute_numerical_gradients (L78)
original_flat[j] += eps
# compute_numerical_gradients (L85)
diff = (plus - minus) / (2 * eps)
# compute_numerical_gradients (L86)
grad_sum += diff.sum().item()
# test_backward_vs_numerical (L122)
W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)  # For Torus topology
# test_backward_vs_numerical (L155)
features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) if topology == 1 else x
# test_backward_vs_numerical (L156)
mu = torch.sigmoid(torch.matmul(features, W_forget.t()) + b_forget) * 5.0  # FRICTION_SCALE
# test_backward_vs_numerical (L157)
friction_output = gamma + mu * v
# test_backward_vs_numerical (L162)
forward_diff = torch.abs(output_cuda - friction_output).max().item()
# test_backward_vs_numerical (L179)
loss = output_cuda.sum()
# forward_func (L197)
tolerances = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]  # Tolerances per parameter
# forward_func (L217)
abs_diff = torch.abs(grad_cuda - grad_num)
# forward_func (L218)
rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)
# forward_func (L222)
mean_abs_error = abs_diff.mean().item()
# forward_func (L223)
mean_rel_error = rel_diff.mean().item()
# forward_func (L234)
if max_abs_error <= tol and max_rel_error <= 10 * tol:
# forward_func (L242)
cuda_val = grad_cuda.view(-1)[max_idx].item()
# forward_func (L243)
num_val = grad_num.view(-1)[max_idx].item()
# test_gradient_checking_pytorch (L271)
print("\n" + "=" * 70)
# test_gradient_checking_pytorch (L292)
W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)
# test_gradient_checking_pytorch (L301)
result = gradcheck( LowRankChristoffelWithFrictionFunction.apply, test_input, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False )
# test_gradient_checking_pytorch (L331)
print("\n" + "=" * 70)
```

### tests\physics\test_leapfrog_stability.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 46 | test_leapfrog_backward | `loss = (x_new.sum() + v_new.sum())` |
| 61 | test_leapfrog_backward | `print(f"   grad_x: mean={x.grad.mean().item():.6f}, std={x.grad.std().item():.6f}")` |
| 62 | test_leapfrog_backward | `print(f"   grad_v: mean={v.grad.mean().item():.6f}, std={v.grad.std().item():.6f}")` |
| 63 | test_leapfrog_backward | `print(f"   grad_U: mean={U.grad.mean().item():.6f}, std={U.grad.std().item():.6f}")` |
| 64 | test_leapfrog_backward | `print(f"   grad_W: mean={W.grad.mean().item():.6f}, std={W.grad.std().item():.6f}")` |
| 79 | test_leapfrog_backward | `print("\n" + "=" * 60)` |

#### Fórmulas Listas para Usar (Python)
```python
# test_leapfrog_backward (L46)
loss = (x_new.sum() + v_new.sum())
# test_leapfrog_backward (L61)
print(f"   grad_x: mean={x.grad.mean().item():.6f}, std={x.grad.std().item():.6f}")
# test_leapfrog_backward (L62)
print(f"   grad_v: mean={v.grad.mean().item():.6f}, std={v.grad.std().item():.6f}")
# test_leapfrog_backward (L63)
print(f"   grad_U: mean={U.grad.mean().item():.6f}, std={U.grad.std().item():.6f}")
# test_leapfrog_backward (L64)
print(f"   grad_W: mean={W.grad.mean().item():.6f}, std={W.grad.std().item():.6f}")
# test_leapfrog_backward (L79)
print("\n" + "=" * 60)
```

### tests\physics\test_mechanics.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 48 | test_mechanics | `loss_std = criterion(logits_std.view(-1, vocab_size), target.view(-1))` |
| 63 | test_mechanics | `loss_adj = criterion(logits_adj.view(-1, vocab_size), target.view(-1))` |
| 90 | test_mechanics | `head_dim = dim // heads` |
| 96 | test_mechanics | `E_start = p.pow(2).sum(dim=-1).mean()` |
| 105 | test_mechanics | `E_end = p_new.pow(2).sum(dim=-1).mean()` |
| 106 | test_mechanics | `energy_diff = torch.abs(E_start - E_end).item()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_mechanics (L48)
loss_std = criterion(logits_std.view(-1, vocab_size), target.view(-1))
# test_mechanics (L63)
loss_adj = criterion(logits_adj.view(-1, vocab_size), target.view(-1))
# test_mechanics (L90)
head_dim = dim // heads
# test_mechanics (L96)
E_start = p.pow(2).sum(dim=-1).mean()
# test_mechanics (L105)
E_end = p_new.pow(2).sum(dim=-1).mean()
# test_mechanics (L106)
energy_diff = torch.abs(E_start - E_end).item()
```

### tests\physics\test_pefrl.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 14 | HarmonicChristoffel | `Simple Harmonicoscillator: Force = -k*x` |
| 16 | HarmonicChristoffel | `Energy H = 1/2 v^2 + 1/2 x^2` |
| 56 | energy | `drift = torch.abs(e_final - e_init) / e_init` |
| 76 | test_symplectic_jacobian | `outputs = torch.cat([next_x, next_v], dim=-1).squeeze(0)` |
| 87 | test_symplectic_jacobian | `jacobian.append(torch.cat([grad_x, grad_v], dim=-1))` |

#### Fórmulas Listas para Usar (Python)
```python
# HarmonicChristoffel (L14)
Simple Harmonicoscillator: Force = -k*x
# HarmonicChristoffel (L16)
Energy H = 1/2 v^2 + 1/2 x^2
# energy (L56)
drift = torch.abs(e_final - e_init) / e_init
# test_symplectic_jacobian (L76)
outputs = torch.cat([next_x, next_v], dim=-1).squeeze(0)
# test_symplectic_jacobian (L87)
jacobian.append(torch.cat([grad_x, grad_v], dim=-1))
```

### tests\physics\test_stochastic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 19 | forward | `def forward(self, v, x=None, **kwargs):` |

#### Fórmulas Listas para Usar (Python)
```python
# forward (L19)
def forward(self, v, x=None, **kwargs):
```

### tests\proofs\test_chaos_prediction.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 19 | double_pendulum_derivs | `c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)` |
| 25 | double_pendulum_derivs | `num1 = -g*(2*m1+m2)*np.sin(theta1) - m2*g*np.sin(theta1-2*theta2) - 2*s*m2*(z2**2*L2 + z1**2*L1*c)` |
| 26 | double_pendulum_derivs | `den1 = L1 * (2*m1 + m2 - m2*c*2*c) # Typo in standard formula? standard is -m2*cos(2*delta)` |
| 29 | double_pendulum_derivs | `den1 = L1 * (2*m1 + m2 - m2*np.cos(2*(theta1-theta2)))` |
| 31 | double_pendulum_derivs | `z1_dot = num1 / den1` |
| 33 | double_pendulum_derivs | `num2 = 2*s*(z1**2*L1*(m1+m2) + g*(m1+m2)*np.cos(theta1) + z2**2*L2*m2*c)` |
| 34 | double_pendulum_derivs | `den2 = L2 * (2*m1 + m2 - m2*np.cos(2*(theta1-theta2)))` |
| 35 | double_pendulum_derivs | `z2_dot = num2 / den2` |
| 43 | generate_chaos_data | `t = np.arange(0, seq_len*dt, dt)` |
| 49 | generate_chaos_data | `init_state = np.random.uniform(-0.5, 0.5, 4) + np.array([np.pi, 0, np.pi, 0])` |
| 53 | generate_chaos_data | `return torch.tensor(np.array(data), dtype=torch.float32)` |
| 72 | run_chaos_test | `data = generate_chaos_data(1000, SEQ_LEN + PRED_LEN)` |
| 88 | run_chaos_test | `MIN_VAL, MAX_VAL = -10, 10` |
| 91 | continuous_to_tokens | `norm = (tensor - MIN_VAL) / (MAX_VAL - MIN_VAL)` |
| 92 | continuous_to_tokens | `tokens = (norm * BINS).long().clamp(0, BINS-1)` |
| 114 | continuous_to_tokens | `manifold_opt = optim.AdamW(manifold.parameters(), lr=1e-3)` |
| 121 | continuous_to_tokens | `lstm_opt = optim.AdamW(list(lstm.parameters()) + list(lstm_embed.parameters()) + list(lstm_head.parameters()), lr=1e-3)` |
| 136 | continuous_to_tokens | `inp = batch[:, :-1]` |
| 142 | continuous_to_tokens | `loss_m = criterion(logits_m.reshape(-1, BINS), tgt.reshape(-1))` |
| 151 | continuous_to_tokens | `loss_l = criterion(logits_l.reshape(-1, BINS), tgt.reshape(-1))` |
| 175 | autoregress | `next_tok = torch.argmax(logits[:, -1:], dim=-1)` |
| 197 | pred_lstm | `err_m = (fut_m.float() - future.float()).abs().mean().item()` |
| 198 | pred_lstm | `err_l = (fut_l.float() - future.float()).abs().mean().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# double_pendulum_derivs (L19)
c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
# double_pendulum_derivs (L25)
num1 = -g*(2*m1+m2)*np.sin(theta1) - m2*g*np.sin(theta1-2*theta2) - 2*s*m2*(z2**2*L2 + z1**2*L1*c)
# double_pendulum_derivs (L26)
den1 = L1 * (2*m1 + m2 - m2*c*2*c) # Typo in standard formula? standard is -m2*cos(2*delta)
# double_pendulum_derivs (L29)
den1 = L1 * (2*m1 + m2 - m2*np.cos(2*(theta1-theta2)))
# double_pendulum_derivs (L31)
z1_dot = num1 / den1
# double_pendulum_derivs (L33)
num2 = 2*s*(z1**2*L1*(m1+m2) + g*(m1+m2)*np.cos(theta1) + z2**2*L2*m2*c)
# double_pendulum_derivs (L34)
den2 = L2 * (2*m1 + m2 - m2*np.cos(2*(theta1-theta2)))
# double_pendulum_derivs (L35)
z2_dot = num2 / den2
# generate_chaos_data (L43)
t = np.arange(0, seq_len*dt, dt)
# generate_chaos_data (L49)
init_state = np.random.uniform(-0.5, 0.5, 4) + np.array([np.pi, 0, np.pi, 0])
# generate_chaos_data (L53)
return torch.tensor(np.array(data), dtype=torch.float32)
# run_chaos_test (L72)
data = generate_chaos_data(1000, SEQ_LEN + PRED_LEN)
# run_chaos_test (L88)
MIN_VAL, MAX_VAL = -10, 10
# continuous_to_tokens (L91)
norm = (tensor - MIN_VAL) / (MAX_VAL - MIN_VAL)
# continuous_to_tokens (L92)
tokens = (norm * BINS).long().clamp(0, BINS-1)
# continuous_to_tokens (L114)
manifold_opt = optim.AdamW(manifold.parameters(), lr=1e-3)
# continuous_to_tokens (L121)
lstm_opt = optim.AdamW(list(lstm.parameters()) + list(lstm_embed.parameters()) + list(lstm_head.parameters()), lr=1e-3)
# continuous_to_tokens (L136)
inp = batch[:, :-1]
# continuous_to_tokens (L142)
loss_m = criterion(logits_m.reshape(-1, BINS), tgt.reshape(-1))
# continuous_to_tokens (L151)
loss_l = criterion(logits_l.reshape(-1, BINS), tgt.reshape(-1))
# autoregress (L175)
next_tok = torch.argmax(logits[:, -1:], dim=-1)
# pred_lstm (L197)
err_m = (fut_m.float() - future.float()).abs().mean().item()
# pred_lstm (L198)
err_l = (fut_l.float() - future.float()).abs().mean().item()
```

### tests\proofs\test_infinite_arithmetic.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 32 | __getitem__ | `len1 = np.random.randint(1, self.length + 1)` |
| 33 | __getitem__ | `len2 = np.random.randint(1, self.length + 1)` |
| 35 | __getitem__ | `num1 = [np.random.randint(0, self.base) for _ in range(len1)]` |
| 36 | __getitem__ | `num2 = [np.random.randint(0, self.base) for _ in range(len2)]` |
| 41 | __getitem__ | `res = val1 + val2` |
| 48 | __getitem__ | `src = torch.tensor(num1 + [self.PLUS] + num2 + [self.EQ], dtype=torch.long)` |
| 49 | __getitem__ | `tgt = torch.tensor(res_digits + [self.EOS], dtype=torch.long)` |
| 55 | collate_fn | `srcs, tgts = zip(*batch)` |
| 90 | train_and_eval | `optimizer = optim.AdamW(model.parameters(), lr=1e-3)` |
| 98 | train_and_eval | `pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")` |
| 113 | train_and_eval | `full_seq = torch.cat([src, tgt], dim=1) # [B, S+T]` |
| 114 | train_and_eval | `input_seq = full_seq[:, :-1]` |
| 122 | train_and_eval | `loss = criterion(logits.reshape(-1, 13), target_seq.reshape(-1))` |
| 127 | train_and_eval | `total_loss += loss.item()` |
| 160 | train_and_eval | `input_token = curr_seq[:, -1:] # Last token (EQ)` |
| 164 | train_and_eval | `next_token = torch.argmax(logits, dim=-1)` |
| 179 | train_and_eval | `if is_correct: correct += 1` |

#### Fórmulas Listas para Usar (Python)
```python
# __getitem__ (L32)
len1 = np.random.randint(1, self.length + 1)
# __getitem__ (L33)
len2 = np.random.randint(1, self.length + 1)
# __getitem__ (L35)
num1 = [np.random.randint(0, self.base) for _ in range(len1)]
# __getitem__ (L36)
num2 = [np.random.randint(0, self.base) for _ in range(len2)]
# __getitem__ (L41)
res = val1 + val2
# __getitem__ (L48)
src = torch.tensor(num1 + [self.PLUS] + num2 + [self.EQ], dtype=torch.long)
# __getitem__ (L49)
tgt = torch.tensor(res_digits + [self.EOS], dtype=torch.long)
# collate_fn (L55)
srcs, tgts = zip(*batch)
# train_and_eval (L90)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# train_and_eval (L98)
pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
# train_and_eval (L113)
full_seq = torch.cat([src, tgt], dim=1) # [B, S+T]
# train_and_eval (L114)
input_seq = full_seq[:, :-1]
# train_and_eval (L122)
loss = criterion(logits.reshape(-1, 13), target_seq.reshape(-1))
# train_and_eval (L127)
total_loss += loss.item()
# train_and_eval (L160)
input_token = curr_seq[:, -1:] # Last token (EQ)
# train_and_eval (L164)
next_token = torch.argmax(logits, dim=-1)
# train_and_eval (L179)
if is_correct: correct += 1
```

### tests\proofs\test_needle_haystack.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 26 | run_needle_test | `NOISE_LEN = 10000  # 10k tokens of noise (Standard Transformer would struggle/OOM with standard attn)` |
| 40 | generate_batch | `kv_seq = torch.zeros((batch_size, KEY_VAL_PAIRS * 2), dtype=torch.long)` |
| 75 | generate_batch | `optimizer = optim.AdamW(model.parameters(), lr=1e-3)` |
| 90 | generate_batch | `print(f"[*] Training on Noise Length={TRAIN_NOISE}...")` |
| 112 | generate_batch | `last_logit = logits[:, -1, :] # Prediction for next token after Query` |
| 119 | generate_batch | `acc = (torch.argmax(last_logit, dim=-1) == tgt).float().mean()` |
| 124 | generate_batch | `print(f"\n[*] testing on Infinite Haystack (Length={NOISE_LEN})...")` |
| 134 | generate_batch | `last_logit = logits[:, -1, :]` |
| 135 | generate_batch | `pred = torch.argmax(last_logit, dim=-1)` |
| 137 | generate_batch | `duration = time.time() - start_time` |

#### Fórmulas Listas para Usar (Python)
```python
# run_needle_test (L26)
NOISE_LEN = 10000  # 10k tokens of noise (Standard Transformer would struggle/OOM with standard attn)
# generate_batch (L40)
kv_seq = torch.zeros((batch_size, KEY_VAL_PAIRS * 2), dtype=torch.long)
# generate_batch (L75)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# generate_batch (L90)
print(f"[*] Training on Noise Length={TRAIN_NOISE}...")
# generate_batch (L112)
last_logit = logits[:, -1, :] # Prediction for next token after Query
# generate_batch (L119)
acc = (torch.argmax(last_logit, dim=-1) == tgt).float().mean()
# generate_batch (L124)
print(f"\n[*] testing on Infinite Haystack (Length={NOISE_LEN})...")
# generate_batch (L134)
last_logit = logits[:, -1, :]
# generate_batch (L135)
pred = torch.argmax(last_logit, dim=-1)
# generate_batch (L137)
duration = time.time() - start_time
```

### tests\run_consistency_tests.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 36 | run_tests | `pytest_args = [ "-v", "--tb=short", "-x",  # Stop on first failure "tests/cuda/test_cuda_python_consistency.py" ]` |
| 47 | run_tests | `pytest_args.append("--ignore-glob=*cuda*")` |
| 53 | run_tests | `result = subprocess.run( [sys.executable, "-m", "pytest"] + pytest_args, cwd=Path(__file__).parent.parent, capture_output=False )` |
| 127 | run_quick_checks | `loss = torch.sum(gamma)` |
| 157 | print_test_summary | `test_classes = [ ("TestCUDAAvailability", "CUDA device and constant verification"), ("TestChristoffelOperation", "Christoffel symbol computation tests"), ("TestLeapfrogIntegration", "Leapfrog integrator tests"), ("TestGradientConsistency", "Gradient computation verification"), ("TestCUDAVsPythonEquivalence", "CUDA vs Python numerical equivalence"), ("TestConvergenceBehavior", "Optimization convergence tests"), ("TestEdgeCases", "Edge case and boundary tests"), ("TestPerformanceBenchmarks", "Performance benchmarks"), ("TestTopologyBehavior", "Topology-specific behavior tests"), ("TestAutogradFunctionality", "Autograd function tests"), ("TestFullPipeline", "Full integration tests"), ]` |
| 187 | print_test_summary | `parser = argparse.ArgumentParser( description="Run CUDA-Python consistency tests" )` |

#### Fórmulas Listas para Usar (Python)
```python
# run_tests (L36)
pytest_args = [ "-v", "--tb=short", "-x",  # Stop on first failure "tests/cuda/test_cuda_python_consistency.py" ]
# run_tests (L47)
pytest_args.append("--ignore-glob=*cuda*")
# run_tests (L53)
result = subprocess.run( [sys.executable, "-m", "pytest"] + pytest_args, cwd=Path(__file__).parent.parent, capture_output=False )
# run_quick_checks (L127)
loss = torch.sum(gamma)
# print_test_summary (L157)
test_classes = [ ("TestCUDAAvailability", "CUDA device and constant verification"), ("TestChristoffelOperation", "Christoffel symbol computation tests"), ("TestLeapfrogIntegration", "Leapfrog integrator tests"), ("TestGradientConsistency", "Gradient computation verification"), ("TestCUDAVsPythonEquivalence", "CUDA vs Python numerical equivalence"), ("TestConvergenceBehavior", "Optimization convergence tests"), ("TestEdgeCases", "Edge case and boundary tests"), ("TestPerformanceBenchmarks", "Performance benchmarks"), ("TestTopologyBehavior", "Topology-specific behavior tests"), ("TestAutogradFunctionality", "Autograd function tests"), ("TestFullPipeline", "Full integration tests"), ]
# print_test_summary (L187)
parser = argparse.ArgumentParser( description="Run CUDA-Python consistency tests" )
```

### tests\run_suite.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | run_suite | `print("\n" + "=" * 70)` |
| 29 | run_suite | `all_tests = loader.discover(start_dir, pattern='test_*.py', top_level_dir=str(PROJECT_ROOT))` |
| 44 | run_suite | `scripts = [ ("tests/integration/test_overfit_sanity.py", "Overfit Diagnosis (Sanity Check)") ]` |
| 48 | run_suite | `print("\n" + "=" * 70)` |
| 55 | run_suite | `script_path = PROJECT_ROOT / script_rel` |
| 76 | run_suite | `last_line = ret.stdout.strip().splitlines()[-1] if ret.stdout.strip() else ''` |
| 90 | run_suite | `print("\n" + "=" * 70)` |

#### Fórmulas Listas para Usar (Python)
```python
# run_suite (L16)
print("\n" + "=" * 70)
# run_suite (L29)
all_tests = loader.discover(start_dir, pattern='test_*.py', top_level_dir=str(PROJECT_ROOT))
# run_suite (L44)
scripts = [ ("tests/integration/test_overfit_sanity.py", "Overfit Diagnosis (Sanity Check)") ]
# run_suite (L48)
print("\n" + "=" * 70)
# run_suite (L55)
script_path = PROJECT_ROOT / script_rel
# run_suite (L76)
last_line = ret.stdout.strip().splitlines()[-1] if ret.stdout.strip() else ''
# run_suite (L90)
print("\n" + "=" * 70)
```

### tests\test_initial_loss.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 12 | _standardize_forces | `m = forces.mean(dim=(0, 1), keepdim=True)` |
| 13 | _standardize_forces | `s = forces.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)` |
| 45 | test_initial_loss_within_tolerance | `y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)` |
| 48 | test_initial_loss_within_tolerance | `assert abs(loss - 2.5) <= 0.25, f"Initial loss {loss:.2f} deviates more than 10% from 2.5"` |

#### Fórmulas Listas para Usar (Python)
```python
# _standardize_forces (L12)
m = forces.mean(dim=(0, 1), keepdim=True)
# _standardize_forces (L13)
s = forces.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
# test_initial_loss_within_tolerance (L45)
y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
# test_initial_loss_within_tolerance (L48)
assert abs(loss - 2.5) <= 0.25, f"Initial loss {loss:.2f} deviates more than 10% from 2.5"
```

### tests\unit\conftest.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 66 | long_seq | `@pytest.fixture(autouse=True)` |

#### Fórmulas Listas para Usar (Python)
```python
# long_seq (L66)
@pytest.fixture(autouse=True)
```

### tests\unit\test_active_physics.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 43 | test_reactive_curvature_plasticity | `v_low = torch.randn(1, self.dim) * 0.01` |
| 47 | test_reactive_curvature_plasticity | `v_high = torch.randn(1, self.dim) * 10.0 # High velocity` |
| 86 | test_singularity_trigger | `x_high = torch.ones(1, self.dim) * 10.0 # Sigmoid(large) -> 1.0 > 0.8` |

#### Fórmulas Listas para Usar (Python)
```python
# test_reactive_curvature_plasticity (L43)
v_low = torch.randn(1, self.dim) * 0.01
# test_reactive_curvature_plasticity (L47)
v_high = torch.randn(1, self.dim) * 10.0 # High velocity
# test_singularity_trigger (L86)
x_high = torch.ones(1, self.dim) * 10.0 # Sigmoid(large) -> 1.0 > 0.8
```

### tests\unit\test_adaptive_physics.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 33 | test_dormand_prince_integration | `diff = (x_new - x).abs().max().item()` |
| 53 | test_rk45_accuracy | `v_euler = v + a * 0.001` |
| 54 | test_rk45_accuracy | `x_euler = x + v * 0.001` |
| 58 | test_rk45_accuracy | `torch.testing.assert_close(x_rk, x_euler, rtol=1e-2, atol=1e-2)` |

#### Fórmulas Listas para Usar (Python)
```python
# test_dormand_prince_integration (L33)
diff = (x_new - x).abs().max().item()
# test_rk45_accuracy (L53)
v_euler = v + a * 0.001
# test_rk45_accuracy (L54)
x_euler = x + v * 0.001
# test_rk45_accuracy (L58)
torch.testing.assert_close(x_rk, x_euler, rtol=1e-2, atol=1e-2)
```

### tests\unit\test_curiosity.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | test_curiosity_logic | `v_high = [ torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), # Diverse directions torch.tensor([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]) ]` |
| 42 | test_curiosity_gradients | `v = params * 0.1` |

#### Fórmulas Listas para Usar (Python)
```python
# test_curiosity_logic (L16)
v_high = [ torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), # Diverse directions torch.tensor([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]) ]
# test_curiosity_gradients (L42)
v = params * 0.1
```

### tests\unit\test_fractals.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 55 | test_fractal_tunneling_activation | `r = torch.norm(stacked_gamma, dim=-1).mean()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_fractal_tunneling_activation (L55)
r = torch.norm(stacked_gamma, dim=-1).mean()
```

### tests\unit\test_geometric_enhancements.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 31 | test_dynamic_curvature_modulation | `diff = (gamma_static - gamma_dynamic).abs().max().item()` |
| 74 | test_parallel_mlayer_wormhole_scales | `head_dim = dim // heads` |

#### Fórmulas Listas para Usar (Python)
```python
# test_dynamic_curvature_modulation (L31)
diff = (gamma_static - gamma_dynamic).abs().max().item()
# test_parallel_mlayer_wormhole_scales (L74)
head_dim = dim // heads
```

### tests\unit\test_geometry.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 67 | test_clamping | `v = torch.randn(2, 64) * 100` |
| 82 | test_gradient_flow | `loss = gamma.sum()` |
| 127 | test_periodic_boundary | `x2 = torch.ones(2, 64) * (2 * 3.14159)` |
| 133 | test_periodic_boundary | `assert torch.allclose(gamma1, gamma2, rtol=1e-2, atol=1e-3)` |
| 142 | test_no_singularities | `x = torch.tensor([[3.14159] * 64, [0.0] * 64])  # π and 0` |

#### Fórmulas Listas para Usar (Python)
```python
# test_clamping (L67)
v = torch.randn(2, 64) * 100
# test_gradient_flow (L82)
loss = gamma.sum()
# test_periodic_boundary (L127)
x2 = torch.ones(2, 64) * (2 * 3.14159)
# test_periodic_boundary (L133)
assert torch.allclose(gamma1, gamma2, rtol=1e-2, atol=1e-3)
# test_no_singularities (L142)
x = torch.tensor([[3.14159] * 64, [0.0] * 64])  # π and 0
```

### tests\unit\test_golden_integration.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 40 | test_mlayer_rk45_forward | `loss = x_out.sum()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_mlayer_rk45_forward (L40)
loss = x_out.sum()
```

### tests\unit\test_losses.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 43 | test_nonzero_loss_changing_energy | `v2 = torch.ones(2, 64) * 2.0  # Double the energy` |
| 44 | test_nonzero_loss_changing_energy | `v3 = torch.ones(2, 64) * 3.0` |
| 91 | test_basic_regularization | `assert loss.item() >= 0, "Regularization should be non-negative"` |
| 119 | test_basic_penalty | `assert loss.item() >= 0, "Penalty should be non-negative"` |
| 148 | test_periodic_boundary | `x2 = torch.ones(2, 64) * (2 * 3.14159)  # 2π` |

#### Fórmulas Listas para Usar (Python)
```python
# test_nonzero_loss_changing_energy (L43)
v2 = torch.ones(2, 64) * 2.0  # Double the energy
# test_nonzero_loss_changing_energy (L44)
v3 = torch.ones(2, 64) * 3.0
# test_basic_regularization (L91)
assert loss.item() >= 0, "Regularization should be non-negative"
# test_basic_penalty (L119)
assert loss.item() >= 0, "Penalty should be non-negative"
# test_periodic_boundary (L148)
x2 = torch.ones(2, 64) * (2 * 3.14159)  # 2π
```

### tests\unit\test_recursive_geodesics.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 53 | test_recursive_context_propagation | `loss = logits.sum()` |

#### Fórmulas Listas para Usar (Python)
```python
# test_recursive_context_propagation (L53)
loss = logits.sum()
```

### tests\unit\test_scan.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 25 | test_basic_scan | `a = torch.ones(B, L, D) * 0.9` |
| 47 | test_sequential_equivalence | `h = a[:, t] * h + x[:, t]` |
| 52 | test_sequential_equivalence | `rtol=1e-5, atol=1e-6,` |
| 134 | test_numerical_stability | `a = torch.ones(2, 10, 32) * 0.01` |
| 140 | test_numerical_stability | `a = torch.ones(2, 10, 32) * 0.99` |

#### Fórmulas Listas para Usar (Python)
```python
# test_basic_scan (L25)
a = torch.ones(B, L, D) * 0.9
# test_sequential_equivalence (L47)
h = a[:, t] * h + x[:, t]
# test_sequential_equivalence (L52)
rtol=1e-5, atol=1e-6,
# test_numerical_stability (L134)
a = torch.ones(2, 10, 32) * 0.01
# test_numerical_stability (L140)
a = torch.ones(2, 10, 32) * 0.99
```

### tests\unit\test_symmetries.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 58 | test_noether_loss_consistency | `c1_div = c0 + 1.0` |

#### Fórmulas Listas para Usar (Python)
```python
# test_noether_loss_consistency (L58)
c1_div = c0 + 1.0
```

### tests\validate_toroidal_fixes.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 42 | test_boundary_consistency | `x = torch.tensor([ 0.1,                # Within range 2 * math.pi + 0.1,  # Wrap positive -0.1,               # Wrap negative 4 * math.pi + 0.1   # Wrap multiple positive ])` |
| 52 | test_boundary_consistency | `expected = torch.tensor([0.1, 0.1, 2 * math.pi - 0.1, 0.1])` |
| 55 | test_boundary_consistency | `self.assertTrue(torch.allclose(wrapped, expected, atol=1e-5), f"Wrapping failed. Got {wrapped}, expected {expected}")` |
| 59 | test_boundary_consistency | `x_grad = torch.tensor([2 * math.pi], requires_grad=True)` |
| 87 | test_leapfrog_integration | `x = torch.tensor([[2 * math.pi - 0.05, 0.0]])` |
| 173 | test_fusion_manager_routing | `layer.head_dim = 2 # dim 4 // heads 2` |

#### Fórmulas Listas para Usar (Python)
```python
# test_boundary_consistency (L42)
x = torch.tensor([ 0.1,                # Within range 2 * math.pi + 0.1,  # Wrap positive -0.1,               # Wrap negative 4 * math.pi + 0.1   # Wrap multiple positive ])
# test_boundary_consistency (L52)
expected = torch.tensor([0.1, 0.1, 2 * math.pi - 0.1, 0.1])
# test_boundary_consistency (L55)
self.assertTrue(torch.allclose(wrapped, expected, atol=1e-5), f"Wrapping failed. Got {wrapped}, expected {expected}")
# test_boundary_consistency (L59)
x_grad = torch.tensor([2 * math.pi], requires_grad=True)
# test_leapfrog_integration (L87)
x = torch.tensor([[2 * math.pi - 0.05, 0.0]])
# test_fusion_manager_routing (L173)
layer.head_dim = 2 # dim 4 // heads 2
```

### tests\verify_leapfrog_parity.py

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 32 | verify_parity | `v_init = torch.randn(batch_size, dim, device=device) * 0.1` |
| 59 | verify_parity | `diff_x = (x_py - x_cuda).abs().max().item()` |
| 60 | verify_parity | `diff_v = (v_py - v_cuda).abs().max().item()` |
| 72 | verify_parity | `geo_torus = ToroidalChristoffel(dim).to(device) # Should handle R/r internally` |
| 82 | verify_parity | `diff_x_t = (x_py_t - x_cuda_t).abs().max().item()` |
| 83 | verify_parity | `diff_v_t = (v_py_t - v_cuda_t).abs().max().item()` |

#### Fórmulas Listas para Usar (Python)
```python
# verify_parity (L32)
v_init = torch.randn(batch_size, dim, device=device) * 0.1
# verify_parity (L59)
diff_x = (x_py - x_cuda).abs().max().item()
# verify_parity (L60)
diff_v = (v_py - v_cuda).abs().max().item()
# verify_parity (L72)
geo_torus = ToroidalChristoffel(dim).to(device) # Should handle R/r internally
# verify_parity (L82)
diff_x_t = (x_py_t - x_cuda_t).abs().max().item()
# verify_parity (L83)
diff_v_t = (v_py_t - v_cuda_t).abs().max().item()
```

## Archivos CUDA

### gfn\cuda\cuda_kernels.cpp

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 147 | Global | `auto hidden = at::tanh(at::matmul(x, W1.t()) + b1);` |
| 148 | Global | `auto out = at::matmul(hidden, W2.t()) + b2;` |
| 154 | Global | `m.doc() = "GFN CUDA Kernels - High-performance manifold geometry and integration";` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L147)
auto hidden = at::tanh(at::matmul(x, W1.t()) + b1);
// Global (L148)
auto out = at::matmul(hidden, W2.t()) + b2;
// Global (L154)
m.doc() = "GFN CUDA Kernels - High-performance manifold geometry and integration";
```

### gfn\cuda\src\common\device_utils.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 22 | Global | `T wrapped = atan2(sin(x), cos(x));` |
| 24 | Global | `wrapped += static_cast<T>(TWO_PI<T>);` |
| 42 | Global | `x[i] = apply_boundary_device<T>(x[i], topology);` |
| 58 | Global | `T epsilon = static_cast<T>(EPSILON_STRONG<T>) ) { return numerator / (denominator + epsilon);` |
| 69 | Global | `T min_val = static_cast<T>(CURVATURE_CLAMP_MIN<T>), T max_val = static_cast<T>(CURVATURE_CLAMP<T>) ) { return fmin(fmax(value, min_val), max_val);` |
| 81 | Global | `T scale = static_cast<T>(CURVATURE_CLAMP<T>) ) { return scale * tanh(value / scale);` |
| 101 | Global | `result += a[i] * b[i];` |
| 119 | Global | `* Vector addition: c = a + b */ template <typename T> GFN_DEVICE void vector_add( T* c, const T* a, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {` |
| 129 | Global | `c[i] = a[i] + b[i];` |
| 134 | Global | `* Scaled vector addition: c = a + scale * b */ template <typename T> GFN_DEVICE void vector_add_scaled( T* c, const T* a, T scale, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {` |
| 145 | Global | `c[i] = a[i] + scale * b[i];` |
| 150 | Global | `* Vector scaling: b = scale * a */ template <typename T> GFN_DEVICE void vector_scale( T* b, T scale, const T* a, int dim ) { for (int i = 0; i < dim; ++i) {` |
| 160 | Global | `b[i] = scale * a[i];` |
| 165 | Global | `* Copy vector: dst = src */ template <typename T> GFN_DEVICE void vector_copy( T* dst, const T* src, int dim ) { for (int i = 0; i < dim; ++i) {` |
| 179 | Global | `* Zero vector: v = 0 */ template <typename T> GFN_DEVICE void vector_zero( T* v, int dim ) { for (int i = 0; i < dim; ++i) {` |
| 222 | Global | `val += __shfl_down_sync(0xffffffff, val, offset);` |
| 235 | Global | `int wid = threadIdx.x / warpSize;` |
| 242 | Global | `val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : static_cast<T>(0);` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L22)
T wrapped = atan2(sin(x), cos(x));
// Global (L24)
wrapped += static_cast<T>(TWO_PI<T>);
// Global (L42)
x[i] = apply_boundary_device<T>(x[i], topology);
// Global (L58)
T epsilon = static_cast<T>(EPSILON_STRONG<T>) ) { return numerator / (denominator + epsilon);
// Global (L69)
T min_val = static_cast<T>(CURVATURE_CLAMP_MIN<T>), T max_val = static_cast<T>(CURVATURE_CLAMP<T>) ) { return fmin(fmax(value, min_val), max_val);
// Global (L81)
T scale = static_cast<T>(CURVATURE_CLAMP<T>) ) { return scale * tanh(value / scale);
// Global (L101)
result += a[i] * b[i];
// Global (L119)
* Vector addition: c = a + b */ template <typename T> GFN_DEVICE void vector_add( T* c, const T* a, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {
// Global (L129)
c[i] = a[i] + b[i];
// Global (L134)
* Scaled vector addition: c = a + scale * b */ template <typename T> GFN_DEVICE void vector_add_scaled( T* c, const T* a, T scale, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {
// Global (L145)
c[i] = a[i] + scale * b[i];
// Global (L150)
* Vector scaling: b = scale * a */ template <typename T> GFN_DEVICE void vector_scale( T* b, T scale, const T* a, int dim ) { for (int i = 0; i < dim; ++i) {
// Global (L160)
b[i] = scale * a[i];
// Global (L165)
* Copy vector: dst = src */ template <typename T> GFN_DEVICE void vector_copy( T* dst, const T* src, int dim ) { for (int i = 0; i < dim; ++i) {
// Global (L179)
* Zero vector: v = 0 */ template <typename T> GFN_DEVICE void vector_zero( T* v, int dim ) { for (int i = 0; i < dim; ++i) {
// Global (L222)
val += __shfl_down_sync(0xffffffff, val, offset);
// Global (L235)
int wid = threadIdx.x / warpSize;
// Global (L242)
val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : static_cast<T>(0);
```

### gfn\cuda\src\common\integrator_utils.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | warp_reduce_sum_shared | `val += __shfl_down_sync(0xffffffff, val, offset);` |
| 24 | block_reduce_sum_shared | `int wid = threadIdx.x / warpSize;` |
| 32 | block_reduce_sum_shared | `val = (threadIdx.x < blockDim.x / warpSize) ? shared_red[lane] : 0;` |
| 77 | christoffel_distributed_shared | `*gamma_val = 0.0f;` |
| 83 | christoffel_distributed_shared | `scalar_t v_ph = v_shared[tid + 1];` |
| 86 | christoffel_distributed_shared | `scalar_t denom = R + r * c;` |
| 88 | christoffel_distributed_shared | `scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 89 | christoffel_distributed_shared | `*gamma_val = term_th * (v_ph * v_ph) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);` |
| 90 | christoffel_distributed_shared | `} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];` |
| 93 | christoffel_distributed_shared | `scalar_t v_th = v_shared[tid - 1];` |
| 96 | christoffel_distributed_shared | `scalar_t denom = R + r * c;` |
| 98 | christoffel_distributed_shared | `scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 99 | christoffel_distributed_shared | `*gamma_val = 2.0f * term_ph * v_ph * v_th * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);` |
| 105 | christoffel_distributed_shared | `scalar_t prod = U[tid * rank + k] * v_val;` |
| 118 | christoffel_distributed_shared | `scalar_t norm = sqrt(energy);` |
| 119 | christoffel_distributed_shared | `S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));` |
| 126 | christoffel_distributed_shared | `sum_gamma += W[tid * rank + k] * h_shared[k] * h_shared[k] * S_shared * M_shared;` |
| 128 | christoffel_distributed_shared | `*gamma_val = sum_gamma;` |
| 135 | christoffel_distributed_shared | `scalar_t v_dot_gz = v_val * holo_grad_z[tid];` |
| 138 | christoffel_distributed_shared | `scalar_t v_sq = v_val * v_val;` |
| 149 | christoffel_distributed_shared | `scalar_t term_ads = -(1.0f / holo_z) * (2.0f * common_v_dot_gz * v_val - common_v_sq * holo_grad_z[tid]);` |
| 150 | christoffel_distributed_shared | `*gamma_val += term_ads; // Combine curvatures` |
| 157 | christoffel_distributed_shared | `scalar_t f_sq = f_val * f_val;` |
| 158 | christoffel_distributed_shared | `scalar_t head_energy = block_reduce_sum_shared(f_sq) / static_cast<scalar_t>(dim);` |
| 163 | christoffel_distributed_shared | `scalar_t modulator = expf(-thermo_alpha * head_energy / T);` |
| 164 | christoffel_distributed_shared | `*gamma_val *= modulator;` |
| 168 | christoffel_distributed_shared | `*gamma_val = soft_clamp<scalar_t>(*gamma_val, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));` |
| 198 | friction_distributed_shared | `features_shared[dim + tid] = c;` |
| 204 | friction_distributed_shared | `int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;` |
| 207 | friction_distributed_shared | `gate_sum += W_forget[tid * feat_dim + j] * features_shared[j];` |
| 215 | friction_distributed_shared | `gate_sum += W_input[tid * dim + j] * features_shared[j];` |
| 219 | friction_distributed_shared | `scalar_t base_friction = sigmoid(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);` |
| 221 | friction_distributed_shared | `scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 222 | friction_distributed_shared | `*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);` |
| 224 | friction_distributed_shared | `*friction_val = base_friction;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// warp_reduce_sum_shared (L16)
val += __shfl_down_sync(0xffffffff, val, offset);
// block_reduce_sum_shared (L24)
int wid = threadIdx.x / warpSize;
// block_reduce_sum_shared (L32)
val = (threadIdx.x < blockDim.x / warpSize) ? shared_red[lane] : 0;
// christoffel_distributed_shared (L77)
*gamma_val = 0.0f;
// christoffel_distributed_shared (L83)
scalar_t v_ph = v_shared[tid + 1];
// christoffel_distributed_shared (L86)
scalar_t denom = R + r * c;
// christoffel_distributed_shared (L88)
scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// christoffel_distributed_shared (L89)
*gamma_val = term_th * (v_ph * v_ph) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
// christoffel_distributed_shared (L90)
} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];
// christoffel_distributed_shared (L93)
scalar_t v_th = v_shared[tid - 1];
// christoffel_distributed_shared (L96)
scalar_t denom = R + r * c;
// christoffel_distributed_shared (L98)
scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// christoffel_distributed_shared (L99)
*gamma_val = 2.0f * term_ph * v_ph * v_th * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
// christoffel_distributed_shared (L105)
scalar_t prod = U[tid * rank + k] * v_val;
// christoffel_distributed_shared (L118)
scalar_t norm = sqrt(energy);
// christoffel_distributed_shared (L119)
S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// christoffel_distributed_shared (L126)
sum_gamma += W[tid * rank + k] * h_shared[k] * h_shared[k] * S_shared * M_shared;
// christoffel_distributed_shared (L128)
*gamma_val = sum_gamma;
// christoffel_distributed_shared (L135)
scalar_t v_dot_gz = v_val * holo_grad_z[tid];
// christoffel_distributed_shared (L138)
scalar_t v_sq = v_val * v_val;
// christoffel_distributed_shared (L149)
scalar_t term_ads = -(1.0f / holo_z) * (2.0f * common_v_dot_gz * v_val - common_v_sq * holo_grad_z[tid]);
// christoffel_distributed_shared (L150)
*gamma_val += term_ads; // Combine curvatures
// christoffel_distributed_shared (L157)
scalar_t f_sq = f_val * f_val;
// christoffel_distributed_shared (L158)
scalar_t head_energy = block_reduce_sum_shared(f_sq) / static_cast<scalar_t>(dim);
// christoffel_distributed_shared (L163)
scalar_t modulator = expf(-thermo_alpha * head_energy / T);
// christoffel_distributed_shared (L164)
*gamma_val *= modulator;
// christoffel_distributed_shared (L168)
*gamma_val = soft_clamp<scalar_t>(*gamma_val, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
// friction_distributed_shared (L198)
features_shared[dim + tid] = c;
// friction_distributed_shared (L204)
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// friction_distributed_shared (L207)
gate_sum += W_forget[tid * feat_dim + j] * features_shared[j];
// friction_distributed_shared (L215)
gate_sum += W_input[tid * dim + j] * features_shared[j];
// friction_distributed_shared (L219)
scalar_t base_friction = sigmoid(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// friction_distributed_shared (L221)
scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// friction_distributed_shared (L222)
*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
// friction_distributed_shared (L224)
*friction_val = base_friction;
```

### gfn\cuda\src\common\math_utils.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 18 | Global | `* Matrix-vector multiplication: y = A * x * A is [m x n], x is [n], y is [m] */ template <typename T> GFN_DEVICE void matvec( T* y, const T* A, const T* x, int m, int n ) { for (int i = 0; i < m; ++i) {` |
| 32 | Global | `sum += A[i * n + j] * x[j];` |
| 39 | Global | `* Transposed matrix-vector multiplication: y = A^T * x * A is [m x n], x is [m], y is [n] */ template <typename T> GFN_DEVICE void matvec_transpose( T* y, const T* A, const T* x, int m, int n ) { for (int j = 0; j < n; ++j) {` |
| 53 | Global | `sum += A[i * n + j] * x[i];` |
| 60 | Global | `* Outer product: C = a ⊗ b (element-wise) * Result: C[i] = a[i] * b[i] */ template <typename T> GFN_DEVICE void outer_product_elementwise( T* C, const T* a, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {` |
| 71 | Global | `C[i] = a[i] * b[i];` |
| 91 | Global | `*s = std::sin(x);` |
| 92 | Global | `*c = std::cos(x);` |
| 109 | Global | `features[dim + i] = c;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L18)
* Matrix-vector multiplication: y = A * x * A is [m x n], x is [n], y is [m] */ template <typename T> GFN_DEVICE void matvec( T* y, const T* A, const T* x, int m, int n ) { for (int i = 0; i < m; ++i) {
// Global (L32)
sum += A[i * n + j] * x[j];
// Global (L39)
* Transposed matrix-vector multiplication: y = A^T * x * A is [m x n], x is [m], y is [n] */ template <typename T> GFN_DEVICE void matvec_transpose( T* y, const T* A, const T* x, int m, int n ) { for (int j = 0; j < n; ++j) {
// Global (L53)
sum += A[i * n + j] * x[i];
// Global (L60)
* Outer product: C = a ⊗ b (element-wise) * Result: C[i] = a[i] * b[i] */ template <typename T> GFN_DEVICE void outer_product_elementwise( T* C, const T* a, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {
// Global (L71)
C[i] = a[i] * b[i];
// Global (L91)
*s = std::sin(x);
// Global (L92)
*c = std::cos(x);
// Global (L109)
features[dim + i] = c;
```

### gfn\cuda\src\common\types.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 16 | Global | `template<typename T = float> using scalar_t = T;  // Use float precision to match PyTorch float32 and for better performance on consumer GPUs` |
| 34 | Global | `constexpr T CURVATURE_CLAMP = static_cast<T>(3.0);` |
| 36 | Global | `constexpr T CURVATURE_CLAMP_MIN = static_cast<T>(-3.0);` |
| 40 | Global | `constexpr T FRICTION_SCALE = static_cast<T>(0.02);` |
| 42 | Global | `constexpr T DEFAULT_FRICTION = static_cast<T>(0.002);` |
| 46 | Global | `constexpr T PI = static_cast<T>(3.14159265358979323846);` |
| 48 | Global | `constexpr T TWO_PI = static_cast<T>(6.28318530717958647692);` |
| 55 | Global | `constexpr T EPSILON_STANDARD = static_cast<T>(1e-7);` |
| 57 | Global | `constexpr T EPSILON_STRONG = static_cast<T>(1e-7);` |
| 59 | Global | `constexpr T EPSILON_SMOOTH = static_cast<T>(1e-7);` |
| 63 | Global | `constexpr T CLAMP_MIN_WEAK = static_cast<T>(1e-7);` |
| 65 | Global | `constexpr T CLAMP_MIN_STRONG = static_cast<T>(1e-7);` |
| 69 | Global | `constexpr T GATE_BIAS_OPEN = static_cast<T>(1.0);   // sigmoid(1) ≈ 0.73` |
| 71 | Global | `constexpr T GATE_BIAS_CLOSED = static_cast<T>(-3.0); // sigmoid(-3) ≈ 0.05` |
| 75 | Global | `constexpr T TOROIDAL_MAJOR_RADIUS = static_cast<T>(2.0);  // R` |
| 77 | Global | `constexpr T TOROIDAL_MINOR_RADIUS = static_cast<T>(1.0);  // r` |
| 79 | Global | `constexpr T TOROIDAL_CURVATURE_SCALE = static_cast<T>(0.01);` |
| 83 | Global | `constexpr T DEFAULT_PLASTICITY = static_cast<T>(0.02);` |
| 85 | Global | `constexpr T SINGULARITY_THRESHOLD = static_cast<T>(0.5);` |
| 87 | Global | `constexpr T SINGULARITY_GATE_SLOPE = static_cast<T>(0.5);  // REDUCED from 10.0 for stability` |
| 89 | Global | `constexpr T BLACK_HOLE_STRENGTH = static_cast<T>(1.5);` |
| 105 | Global | `constexpr int MAX_THREADS_PER_BLOCK = 1024;` |
| 106 | Global | `constexpr int DEFAULT_BLOCK_SIZE = 256;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L16)
template<typename T = float> using scalar_t = T;  // Use float precision to match PyTorch float32 and for better performance on consumer GPUs
// Global (L34)
constexpr T CURVATURE_CLAMP = static_cast<T>(3.0);
// Global (L36)
constexpr T CURVATURE_CLAMP_MIN = static_cast<T>(-3.0);
// Global (L40)
constexpr T FRICTION_SCALE = static_cast<T>(0.02);
// Global (L42)
constexpr T DEFAULT_FRICTION = static_cast<T>(0.002);
// Global (L46)
constexpr T PI = static_cast<T>(3.14159265358979323846);
// Global (L48)
constexpr T TWO_PI = static_cast<T>(6.28318530717958647692);
// Global (L55)
constexpr T EPSILON_STANDARD = static_cast<T>(1e-7);
// Global (L57)
constexpr T EPSILON_STRONG = static_cast<T>(1e-7);
// Global (L59)
constexpr T EPSILON_SMOOTH = static_cast<T>(1e-7);
// Global (L63)
constexpr T CLAMP_MIN_WEAK = static_cast<T>(1e-7);
// Global (L65)
constexpr T CLAMP_MIN_STRONG = static_cast<T>(1e-7);
// Global (L69)
constexpr T GATE_BIAS_OPEN = static_cast<T>(1.0);   // sigmoid(1) ≈ 0.73
// Global (L71)
constexpr T GATE_BIAS_CLOSED = static_cast<T>(-3.0); // sigmoid(-3) ≈ 0.05
// Global (L75)
constexpr T TOROIDAL_MAJOR_RADIUS = static_cast<T>(2.0);  // R
// Global (L77)
constexpr T TOROIDAL_MINOR_RADIUS = static_cast<T>(1.0);  // r
// Global (L79)
constexpr T TOROIDAL_CURVATURE_SCALE = static_cast<T>(0.01);
// Global (L83)
constexpr T DEFAULT_PLASTICITY = static_cast<T>(0.02);
// Global (L85)
constexpr T SINGULARITY_THRESHOLD = static_cast<T>(0.5);
// Global (L87)
constexpr T SINGULARITY_GATE_SLOPE = static_cast<T>(0.5);  // REDUCED from 10.0 for stability
// Global (L89)
constexpr T BLACK_HOLE_STRENGTH = static_cast<T>(1.5);
// Global (L105)
constexpr int MAX_THREADS_PER_BLOCK = 1024;
// Global (L106)
constexpr int DEFAULT_BLOCK_SIZE = 256;
```

### gfn\cuda\src\geometry\christoffel_impl.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 19 | Global | `* Implements: gamma_sym[i,j] = 0.5 * (gamma[i,j] + gamma[j,i]) * This ensures Gamma^k_ij approx Gamma^k_ji numerically, which is required * for torsion-free connections. * * @param gamma Input/Output Christoffel symbols [dim x dim] * @param dim Dimension of manifold */ template <typename T> GFN_DEVICE void normalize_christoffel_structure(T* gamma, int dim) { for (int i = 0; i < dim; ++i) {` |
| 31 | Global | `T avg = static_cast<T>(0.5) * (gamma[i * dim + j] + gamma[j * dim + i]);` |
| 32 | Global | `gamma[i * dim + j] = avg;` |
| 33 | Global | `gamma[j * dim + i] = avg;` |
| 41 | Global | `* Computes: Γ(v,v) = Σ_r (h_r^2 * W_r) * S * M * where: *   h = U^T * v (projection to rank-R space) *   S = 1 / (1 + \|\|h\|\|)  (stabilization factor) *   M = modulation from plasticity and singularities * * @param v Velocity vector [dim] * @param U Low-rank matrix U [dim x rank] * @param W Low-rank matrix W [dim x rank] * @param x Position vector [dim] (optional, for friction/singularities) * @param V_w Potential weights [dim] (optional, for singularities) * @param dim Dimension of manifold * @param rank Rank of decomposition * @param plasticity Plasticity coefficient (energy-dependent curvature) * @param sing_thresh Singularity threshold * @param sing_strength Singularity strength multiplier * @param topology Topology type (EUCLIDEAN or TORUS) * @param R Toroidal major radius * @param r Toroidal minor radius * @param gamma Output Christoffel force [dim] */ template <typename T> GFN_DEVICE void christoffel_device( const T* v, const T* U, const T* W, const T* x, const T* V_w, int dim, int rank, T plasticity, T sing_thresh, T sing_strength, Topology topology, T R, T r, T* gamma ) { if (topology == Topology::TORUS && x != nullptr && V_w == nullptr) { for (int i = 0; i < dim; ++i) gamma[i] = static_cast<T>(0);` |
| 84 | Global | `T v_ph = v[i + 1];` |
| 85 | Global | `T denom = fmax(R + r * cos(th), static_cast<T>(CLAMP_MIN_STRONG<T>));` |
| 86 | Global | `T term_th = denom * sin(th) / (r + static_cast<T>(EPSILON_SMOOTH<T>));` |
| 87 | Global | `gamma[i] = term_th * (v_ph * v_ph);` |
| 88 | Global | `T term_ph = -(r * sin(th)) / (denom + static_cast<T>(EPSILON_SMOOTH<T>));` |
| 89 | Global | `gamma[i + 1] = static_cast<T>(2) * term_ph * v_ph * v_th;` |
| 92 | Global | `gamma[i] = soft_clamp<T>(gamma[i] * static_cast<T>(TOROIDAL_CURVATURE_SCALE<T>), static_cast<T>(CURVATURE_CLAMP<T>));` |
| 101 | Global | `sum += U[j * rank + i] * v[j];` |
| 107 | Global | `energy += h[i] * h[i];` |
| 112 | Global | `T norm_val = sqrt(energy);` |
| 114 | Global | `T S = static_cast<T>(1) / (static_cast<T>(1) + norm_val + static_cast<T>(EPSILON_STRONG<T>));` |
| 119 | Global | `v_energy += v[i] * v[i];` |
| 121 | Global | `v_energy /= static_cast<T>(dim);` |
| 123 | Global | `M *= (static_cast<T>(1) + plasticity * static_cast<T>(0.1) * tanh(v_energy));` |
| 129 | Global | `pot += sin(x[i]) * V_w[i];` |
| 133 | Global | `pot += x[i] * V_w[i];` |
| 137 | Global | `T soft_m = sigmoid<T>(static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * (gate - sing_thresh));` |
| 138 | Global | `M *= (static_cast<T>(1) + (sing_strength - static_cast<T>(1)) * soft_m);` |
| 141 | Global | `h_sq[i] = h[i] * h[i] * S * M;` |
| 146 | Global | `sum += W[i * rank + j] * h_sq[j];` |
| 163 | Global | `* μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE * * @param x Position vector [dim] * @param force External force [dim] (optional) * @param W_forget Forget gate weights [dim x feature_dim] * @param b_forget Forget gate bias [dim] * @param W_input Input gate weights [dim x dim] (optional) * @param dim Dimension * @param topology Topology type * @param velocity_friction_scale Velocity friction scaling factor * @param v_norm_val Pre-computed velocity norm (optional, if available) * @param friction Output friction coefficients [dim] */ template <typename T> GFN_DEVICE void compute_friction( const T* x, const T* force, const T* W_forget, const T* b_forget, const T* W_input, int dim, Topology topology, T velocity_friction_scale, T v_norm_val, T* friction ) { T features[128]; // Max 2*dim for Fourier features` |
| 196 | Global | `feature_dim = 2 * dim;` |
| 206 | Global | `gate_val += W_forget[i * feature_dim + j] * features[j];` |
| 212 | Global | `gate_val += W_input[i * dim + j] * force[j];` |
| 216 | Global | `T base_friction = sigmoid<T>(gate_val) * static_cast<T>(FRICTION_SCALE<T>);` |
| 221 | Global | `T v_scale = v_norm_val / (sqrt(static_cast<T>(dim)) + static_cast<T>(EPSILON_SMOOTH<T>));` |
| 222 | Global | `friction[i] = base_friction * (static_cast<T>(1) + velocity_friction_scale * v_scale);` |
| 275 | Global | `v_norm_val = sqrt(v_norm_val);` |
| 285 | Global | `output[i] = gamma[i] + friction[i] * v[i];` |
| 337 | Global | `v_norm_val = sqrt(v_norm_val);` |
| 392 | Global | `T* grad_V_w = nullptr ) { T h[64];` |
| 404 | Global | `sum += U[j * rank + i] * v[j];` |
| 407 | Global | `h_energy += sum * sum;` |
| 410 | Global | `h_energy /= static_cast<T>(rank);` |
| 413 | Global | `T norm_val = sqrt(h_energy);` |
| 415 | Global | `T S = static_cast<T>(1) / (static_cast<T>(1) + norm_val + static_cast<T>(EPSILON_STRONG<T>));` |
| 422 | Global | `v_energy /= static_cast<T>(dim);` |
| 423 | Global | `M_plas = (static_cast<T>(1) + plasticity * static_cast<T>(0.1) * tanh(v_energy));` |
| 426 | Global | `T M_sing = static_cast<T>(1);` |
| 436 | Global | `soft_m = sigmoid<T>(static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * (gate - sing_thresh));` |
| 437 | Global | `M_sing = (static_cast<T>(1) + (sing_strength - static_cast<T>(1)) * soft_m);` |
| 439 | Global | `T M = M_plas * M_sing;` |
| 445 | Global | `T t = gamma[i] / static_cast<T>(CURVATURE_CLAMP<T>);` |
| 446 | Global | `grad_raw[i] = grad_out[i] * (static_cast<T>(1) - t * t);` |
| 452 | Global | `T q_base = h[j] * h[j] * S * M;` |
| 454 | Global | `grad_W[i * rank + j] += grad_raw[i] * q_base;` |
| 455 | Global | `grad_q[j] += W[i * rank + j] * grad_raw[i];` |
| 462 | Global | `sum_grad_q_h_sq += grad_q[i] * h[i] * h[i];` |
| 467 | Global | `T S_sq_M_norm = (norm_val > EPSILON_STANDARD<T> && rank > 0) ? (M * S * S / (norm_val * static_cast<T>(rank))) : static_cast<T>(0);` |
| 469 | Global | `T two_S_M = static_cast<T>(2) * S * M;` |
| 472 | Global | `grad_h[i] = grad_q[i] * h[i] * two_S_M - sum_grad_q_h_sq * S_sq_M_norm * h[i];` |
| 478 | Global | `grad_U[i * rank + j] += v[i] * grad_h[j];` |
| 479 | Global | `grad_v[i] += U[i * rank + j] * grad_h[j];` |
| 486 | Global | `T dL_dM_plas = sum_grad_q_h_sq * S * M_sing;` |
| 488 | Global | `T sech_sq = static_cast<T>(1) - tanh_v * tanh_v;` |
| 489 | Global | `T factor = dL_dM_plas * (plasticity * static_cast<T>(0.1)) * sech_sq * (static_cast<T>(2) / static_cast<T>(dim));` |
| 491 | Global | `grad_v[i] += factor * v[i];` |
| 497 | Global | `T dL_dM_sing = sum_grad_q_h_sq * S * M_plas;` |
| 498 | Global | `T dM_dsoft = (sing_strength - static_cast<T>(1));` |
| 499 | Global | `T dsoft_dgate = static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * soft_m * (static_cast<T>(1) - soft_m);` |
| 500 | Global | `T dgate_dpot = gate * (static_cast<T>(1) - gate);` |
| 501 | Global | `T factor = dL_dM_sing * dM_dsoft * dsoft_dgate * dgate_dpot;` |
| 505 | Global | `T dpot_dxi = (topology == Topology::TORUS) ? cos(x[i]) * V_w[i] : V_w[i];` |
| 506 | Global | `grad_x[i] += factor * dpot_dxi;` |
| 509 | Global | `T dpot_dVwi = (topology == Topology::TORUS) ? sin(x[i]) : x[i];` |
| 510 | Global | `grad_V_w[i] += factor * dpot_dVwi;` |
| 538 | Global | `int feature_dim = (topology == Topology::TORUS) ? 2 * dim : dim;` |
| 551 | Global | `z += W_forget[i * feature_dim + j] * features[j];` |
| 556 | Global | `z += W_input[i * dim + j] * force[j];` |
| 561 | Global | `T dz = grad_out[i] * static_cast<T>(FRICTION_SCALE<T>) * s * (static_cast<T>(1) - s);` |
| 564 | Global | `grad_b_forget[i] += dz;` |
| 566 | Global | `grad_W_forget[i * feature_dim + j] += dz * features[j];` |
| 572 | Global | `grad_W_input[i * dim + j] += dz * force[j];` |
| 574 | Global | `grad_force[j] += dz * W_input[i * dim + j];` |
| 582 | Global | `T d_sin = W_forget[i * feature_dim + j] * dz;` |
| 583 | Global | `T d_cos = W_forget[i * feature_dim + (dim + j)] * dz;` |
| 584 | Global | `grad_x[j] += d_sin * cos(x[j]) - d_cos * sin(x[j]);` |
| 588 | Global | `grad_x[j] += W_forget[i * feature_dim + j] * dz;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L19)
* Implements: gamma_sym[i,j] = 0.5 * (gamma[i,j] + gamma[j,i]) * This ensures Gamma^k_ij approx Gamma^k_ji numerically, which is required * for torsion-free connections. * * @param gamma Input/Output Christoffel symbols [dim x dim] * @param dim Dimension of manifold */ template <typename T> GFN_DEVICE void normalize_christoffel_structure(T* gamma, int dim) { for (int i = 0; i < dim; ++i) {
// Global (L31)
T avg = static_cast<T>(0.5) * (gamma[i * dim + j] + gamma[j * dim + i]);
// Global (L32)
gamma[i * dim + j] = avg;
// Global (L33)
gamma[j * dim + i] = avg;
// Global (L41)
* Computes: Γ(v,v) = Σ_r (h_r^2 * W_r) * S * M * where: *   h = U^T * v (projection to rank-R space) *   S = 1 / (1 + ||h||)  (stabilization factor) *   M = modulation from plasticity and singularities * * @param v Velocity vector [dim] * @param U Low-rank matrix U [dim x rank] * @param W Low-rank matrix W [dim x rank] * @param x Position vector [dim] (optional, for friction/singularities) * @param V_w Potential weights [dim] (optional, for singularities) * @param dim Dimension of manifold * @param rank Rank of decomposition * @param plasticity Plasticity coefficient (energy-dependent curvature) * @param sing_thresh Singularity threshold * @param sing_strength Singularity strength multiplier * @param topology Topology type (EUCLIDEAN or TORUS) * @param R Toroidal major radius * @param r Toroidal minor radius * @param gamma Output Christoffel force [dim] */ template <typename T> GFN_DEVICE void christoffel_device( const T* v, const T* U, const T* W, const T* x, const T* V_w, int dim, int rank, T plasticity, T sing_thresh, T sing_strength, Topology topology, T R, T r, T* gamma ) { if (topology == Topology::TORUS && x != nullptr && V_w == nullptr) { for (int i = 0; i < dim; ++i) gamma[i] = static_cast<T>(0);
// Global (L84)
T v_ph = v[i + 1];
// Global (L85)
T denom = fmax(R + r * cos(th), static_cast<T>(CLAMP_MIN_STRONG<T>));
// Global (L86)
T term_th = denom * sin(th) / (r + static_cast<T>(EPSILON_SMOOTH<T>));
// Global (L87)
gamma[i] = term_th * (v_ph * v_ph);
// Global (L88)
T term_ph = -(r * sin(th)) / (denom + static_cast<T>(EPSILON_SMOOTH<T>));
// Global (L89)
gamma[i + 1] = static_cast<T>(2) * term_ph * v_ph * v_th;
// Global (L92)
gamma[i] = soft_clamp<T>(gamma[i] * static_cast<T>(TOROIDAL_CURVATURE_SCALE<T>), static_cast<T>(CURVATURE_CLAMP<T>));
// Global (L101)
sum += U[j * rank + i] * v[j];
// Global (L107)
energy += h[i] * h[i];
// Global (L112)
T norm_val = sqrt(energy);
// Global (L114)
T S = static_cast<T>(1) / (static_cast<T>(1) + norm_val + static_cast<T>(EPSILON_STRONG<T>));
// Global (L119)
v_energy += v[i] * v[i];
// Global (L121)
v_energy /= static_cast<T>(dim);
// Global (L123)
M *= (static_cast<T>(1) + plasticity * static_cast<T>(0.1) * tanh(v_energy));
// Global (L129)
pot += sin(x[i]) * V_w[i];
// Global (L133)
pot += x[i] * V_w[i];
// Global (L137)
T soft_m = sigmoid<T>(static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * (gate - sing_thresh));
// Global (L138)
M *= (static_cast<T>(1) + (sing_strength - static_cast<T>(1)) * soft_m);
// Global (L141)
h_sq[i] = h[i] * h[i] * S * M;
// Global (L146)
sum += W[i * rank + j] * h_sq[j];
// Global (L163)
* μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE * * @param x Position vector [dim] * @param force External force [dim] (optional) * @param W_forget Forget gate weights [dim x feature_dim] * @param b_forget Forget gate bias [dim] * @param W_input Input gate weights [dim x dim] (optional) * @param dim Dimension * @param topology Topology type * @param velocity_friction_scale Velocity friction scaling factor * @param v_norm_val Pre-computed velocity norm (optional, if available) * @param friction Output friction coefficients [dim] */ template <typename T> GFN_DEVICE void compute_friction( const T* x, const T* force, const T* W_forget, const T* b_forget, const T* W_input, int dim, Topology topology, T velocity_friction_scale, T v_norm_val, T* friction ) { T features[128]; // Max 2*dim for Fourier features
// Global (L196)
feature_dim = 2 * dim;
// Global (L206)
gate_val += W_forget[i * feature_dim + j] * features[j];
// Global (L212)
gate_val += W_input[i * dim + j] * force[j];
// Global (L216)
T base_friction = sigmoid<T>(gate_val) * static_cast<T>(FRICTION_SCALE<T>);
// Global (L221)
T v_scale = v_norm_val / (sqrt(static_cast<T>(dim)) + static_cast<T>(EPSILON_SMOOTH<T>));
// Global (L222)
friction[i] = base_friction * (static_cast<T>(1) + velocity_friction_scale * v_scale);
// Global (L275)
v_norm_val = sqrt(v_norm_val);
// Global (L285)
output[i] = gamma[i] + friction[i] * v[i];
// Global (L337)
v_norm_val = sqrt(v_norm_val);
// Global (L392)
T* grad_V_w = nullptr ) { T h[64];
// Global (L404)
sum += U[j * rank + i] * v[j];
// Global (L407)
h_energy += sum * sum;
// Global (L410)
h_energy /= static_cast<T>(rank);
// Global (L413)
T norm_val = sqrt(h_energy);
// Global (L415)
T S = static_cast<T>(1) / (static_cast<T>(1) + norm_val + static_cast<T>(EPSILON_STRONG<T>));
// Global (L422)
v_energy /= static_cast<T>(dim);
// Global (L423)
M_plas = (static_cast<T>(1) + plasticity * static_cast<T>(0.1) * tanh(v_energy));
// Global (L426)
T M_sing = static_cast<T>(1);
// Global (L436)
soft_m = sigmoid<T>(static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * (gate - sing_thresh));
// Global (L437)
M_sing = (static_cast<T>(1) + (sing_strength - static_cast<T>(1)) * soft_m);
// Global (L439)
T M = M_plas * M_sing;
// Global (L445)
T t = gamma[i] / static_cast<T>(CURVATURE_CLAMP<T>);
// Global (L446)
grad_raw[i] = grad_out[i] * (static_cast<T>(1) - t * t);
// Global (L452)
T q_base = h[j] * h[j] * S * M;
// Global (L454)
grad_W[i * rank + j] += grad_raw[i] * q_base;
// Global (L455)
grad_q[j] += W[i * rank + j] * grad_raw[i];
// Global (L462)
sum_grad_q_h_sq += grad_q[i] * h[i] * h[i];
// Global (L467)
T S_sq_M_norm = (norm_val > EPSILON_STANDARD<T> && rank > 0) ? (M * S * S / (norm_val * static_cast<T>(rank))) : static_cast<T>(0);
// Global (L469)
T two_S_M = static_cast<T>(2) * S * M;
// Global (L472)
grad_h[i] = grad_q[i] * h[i] * two_S_M - sum_grad_q_h_sq * S_sq_M_norm * h[i];
// Global (L478)
grad_U[i * rank + j] += v[i] * grad_h[j];
// Global (L479)
grad_v[i] += U[i * rank + j] * grad_h[j];
// Global (L486)
T dL_dM_plas = sum_grad_q_h_sq * S * M_sing;
// Global (L488)
T sech_sq = static_cast<T>(1) - tanh_v * tanh_v;
// Global (L489)
T factor = dL_dM_plas * (plasticity * static_cast<T>(0.1)) * sech_sq * (static_cast<T>(2) / static_cast<T>(dim));
// Global (L491)
grad_v[i] += factor * v[i];
// Global (L497)
T dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
// Global (L498)
T dM_dsoft = (sing_strength - static_cast<T>(1));
// Global (L499)
T dsoft_dgate = static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * soft_m * (static_cast<T>(1) - soft_m);
// Global (L500)
T dgate_dpot = gate * (static_cast<T>(1) - gate);
// Global (L501)
T factor = dL_dM_sing * dM_dsoft * dsoft_dgate * dgate_dpot;
// Global (L505)
T dpot_dxi = (topology == Topology::TORUS) ? cos(x[i]) * V_w[i] : V_w[i];
// Global (L506)
grad_x[i] += factor * dpot_dxi;
// Global (L509)
T dpot_dVwi = (topology == Topology::TORUS) ? sin(x[i]) : x[i];
// Global (L510)
grad_V_w[i] += factor * dpot_dVwi;
// Global (L538)
int feature_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// Global (L551)
z += W_forget[i * feature_dim + j] * features[j];
// Global (L556)
z += W_input[i * dim + j] * force[j];
// Global (L561)
T dz = grad_out[i] * static_cast<T>(FRICTION_SCALE<T>) * s * (static_cast<T>(1) - s);
// Global (L564)
grad_b_forget[i] += dz;
// Global (L566)
grad_W_forget[i * feature_dim + j] += dz * features[j];
// Global (L572)
grad_W_input[i * dim + j] += dz * force[j];
// Global (L574)
grad_force[j] += dz * W_input[i * dim + j];
// Global (L582)
T d_sin = W_forget[i * feature_dim + j] * dz;
// Global (L583)
T d_cos = W_forget[i * feature_dim + (dim + j)] * dz;
// Global (L584)
grad_x[j] += d_sin * cos(x[j]) - d_cos * sin(x[j]);
// Global (L588)
grad_x[j] += W_forget[i * feature_dim + j] * dz;
```

### gfn\cuda\src\geometry\geometry_library.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 34 | Global | `*gamma_val = static_cast<scalar_t>(0);` |
| 35 | Global | `v_shared[tid] = *s.v;` |
| 40 | Global | `scalar_t v_ph = v_shared[tid + 1];` |
| 44 | Global | `scalar_t denom = p.torus_R + p.torus_r * cos_th;` |
| 47 | Global | `scalar_t term_th = denom * sin_th / (p.torus_r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 48 | Global | `*gamma_val = term_th * (v_ph * v_ph) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);` |
| 49 | Global | `} else if (tid % 2 != 0) { scalar_t th = s.x[tid - 1];` |
| 51 | Global | `scalar_t v_ph = *s.v;` |
| 52 | Global | `scalar_t v_th = v_shared[tid - 1];` |
| 56 | Global | `scalar_t denom = p.torus_R + p.torus_r * cos_th;` |
| 59 | Global | `scalar_t term_ph = -(p.torus_r * sin_th) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 60 | Global | `*gamma_val = static_cast<scalar_t>(2) * term_ph * v_ph * v_th * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);` |
| 65 | Global | `scalar_t prod = p.U[tid * p.rank + k] * (*s.v);` |
| 76 | Global | `S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + sqrt(h_energy) + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));` |
| 82 | Global | `sum_gamma += p.W[tid * p.rank + k] * h_shared[k] * h_shared[k] * S_shared;` |
| 84 | Global | `*gamma_val = sum_gamma;` |
| 93 | Global | `scalar_t v_sq = (*s.v) * (*s.v);` |
| 94 | Global | `scalar_t total_energy = block_reduce_sum_shared(v_sq) / static_cast<scalar_t>(dim);` |
| 95 | Global | `M += p.plasticity * static_cast<scalar_t>(0.1) * tanh(total_energy);` |
| 105 | Global | `scalar_t w_sin = p.V_w[tid];` |
| 106 | Global | `scalar_t w_cos = p.V_w[dim + tid];` |
| 107 | Global | `pot_term = sin_th * w_sin + cos_th * w_cos;` |
| 109 | Global | `pot_term = s.x[tid] * p.V_w[tid];` |
| 121 | Global | `scalar_t soft_m = sigmoid(slope * (gate - p.sing_thresh));` |
| 123 | Global | `M *= (static_cast<scalar_t>(1.0) + (p.sing_strength - static_cast<scalar_t>(1.0)) * soft_m);` |
| 126 | Global | `*gamma_val *= M;` |
| 130 | Global | `scalar_t v_dot_gz = (*s.v) * p.holo_grad_z[tid];` |
| 132 | Global | `scalar_t v_sq_sum = block_reduce_sum_shared((*s.v) * (*s.v));` |
| 142 | Global | `scalar_t ads = -(static_cast<scalar_t>(1) / local_holo_z) * (static_cast<scalar_t>(2) * common_v_dot_gz * (*s.v) - common_v_sq * p.holo_grad_z[tid]);` |
| 143 | Global | `*gamma_val += ads;` |
| 148 | Global | `scalar_t head_energy = block_reduce_sum_shared(s.f_ext * s.f_ext) / static_cast<scalar_t>(dim);` |
| 150 | Global | `scalar_t modulator = exp(-p.thermo_alpha * head_energy / T);` |
| 151 | Global | `*gamma_val *= modulator;` |
| 155 | Global | `*gamma_val = soft_clamp<scalar_t>(*gamma_val, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L34)
*gamma_val = static_cast<scalar_t>(0);
// Global (L35)
v_shared[tid] = *s.v;
// Global (L40)
scalar_t v_ph = v_shared[tid + 1];
// Global (L44)
scalar_t denom = p.torus_R + p.torus_r * cos_th;
// Global (L47)
scalar_t term_th = denom * sin_th / (p.torus_r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// Global (L48)
*gamma_val = term_th * (v_ph * v_ph) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
// Global (L49)
} else if (tid % 2 != 0) { scalar_t th = s.x[tid - 1];
// Global (L51)
scalar_t v_ph = *s.v;
// Global (L52)
scalar_t v_th = v_shared[tid - 1];
// Global (L56)
scalar_t denom = p.torus_R + p.torus_r * cos_th;
// Global (L59)
scalar_t term_ph = -(p.torus_r * sin_th) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// Global (L60)
*gamma_val = static_cast<scalar_t>(2) * term_ph * v_ph * v_th * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
// Global (L65)
scalar_t prod = p.U[tid * p.rank + k] * (*s.v);
// Global (L76)
S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + sqrt(h_energy) + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// Global (L82)
sum_gamma += p.W[tid * p.rank + k] * h_shared[k] * h_shared[k] * S_shared;
// Global (L84)
*gamma_val = sum_gamma;
// Global (L93)
scalar_t v_sq = (*s.v) * (*s.v);
// Global (L94)
scalar_t total_energy = block_reduce_sum_shared(v_sq) / static_cast<scalar_t>(dim);
// Global (L95)
M += p.plasticity * static_cast<scalar_t>(0.1) * tanh(total_energy);
// Global (L105)
scalar_t w_sin = p.V_w[tid];
// Global (L106)
scalar_t w_cos = p.V_w[dim + tid];
// Global (L107)
pot_term = sin_th * w_sin + cos_th * w_cos;
// Global (L109)
pot_term = s.x[tid] * p.V_w[tid];
// Global (L121)
scalar_t soft_m = sigmoid(slope * (gate - p.sing_thresh));
// Global (L123)
M *= (static_cast<scalar_t>(1.0) + (p.sing_strength - static_cast<scalar_t>(1.0)) * soft_m);
// Global (L126)
*gamma_val *= M;
// Global (L130)
scalar_t v_dot_gz = (*s.v) * p.holo_grad_z[tid];
// Global (L132)
scalar_t v_sq_sum = block_reduce_sum_shared((*s.v) * (*s.v));
// Global (L142)
scalar_t ads = -(static_cast<scalar_t>(1) / local_holo_z) * (static_cast<scalar_t>(2) * common_v_dot_gz * (*s.v) - common_v_sq * p.holo_grad_z[tid]);
// Global (L143)
*gamma_val += ads;
// Global (L148)
scalar_t head_energy = block_reduce_sum_shared(s.f_ext * s.f_ext) / static_cast<scalar_t>(dim);
// Global (L150)
scalar_t modulator = exp(-p.thermo_alpha * head_energy / T);
// Global (L151)
*gamma_val *= modulator;
// Global (L155)
*gamma_val = soft_clamp<scalar_t>(*gamma_val, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
```

### gfn\cuda\src\geometry\lowrank_christoffel.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 31 | lowrank_christoffel_kernel | `int idx = blockIdx.x * blockDim.x + threadIdx.x;` |
| 36 | lowrank_christoffel_kernel | `const scalar_t* v_ptr = v + idx * dim;` |
| 37 | lowrank_christoffel_kernel | `const scalar_t* x_ptr = (x != nullptr) ? (x + idx * dim) : nullptr;` |
| 38 | lowrank_christoffel_kernel | `scalar_t* gamma_ptr = gamma + idx * dim;` |
| 40 | lowrank_christoffel_kernel | `Topology topology = static_cast<Topology>(topology_id);` |
| 79 | lowrank_christoffel_friction_kernel | `int idx = blockIdx.x * blockDim.x + threadIdx.x;` |
| 84 | lowrank_christoffel_friction_kernel | `const scalar_t* v_ptr = v + idx * dim;` |
| 85 | lowrank_christoffel_friction_kernel | `const scalar_t* x_ptr = x + idx * dim;` |
| 86 | lowrank_christoffel_friction_kernel | `const scalar_t* force_ptr = (force != nullptr) ? (force + idx * dim) : nullptr;` |
| 87 | lowrank_christoffel_friction_kernel | `scalar_t* output_ptr = output + idx * dim;` |
| 89 | lowrank_christoffel_friction_kernel | `Topology topology = static_cast<Topology>(topology_id);` |
| 129 | lowrank_christoffel_friction_kernel | `int blocks = (batch_size + threads - 1) / threads;` |
| 132 | lowrank_christoffel_friction_kernel | `const scalar_t* x_ptr = (x.numel() > 0) ? x.data_ptr<scalar_t>() : nullptr;` |
| 133 | lowrank_christoffel_friction_kernel | `const scalar_t* V_w_ptr = (V_w.numel() > 0) ? V_w.data_ptr<scalar_t>() : nullptr;` |
| 184 | lowrank_christoffel_friction_kernel | `int blocks = (batch_size + threads - 1) / threads;` |
| 187 | lowrank_christoffel_friction_kernel | `const scalar_t* V_w_ptr = (V_w.numel() > 0) ? V_w.data_ptr<scalar_t>() : nullptr;` |
| 188 | lowrank_christoffel_friction_kernel | `const scalar_t* force_ptr = (force.numel() > 0) ? force.data_ptr<scalar_t>() : nullptr;` |
| 189 | lowrank_christoffel_friction_kernel | `const scalar_t* W_input_ptr = (W_input.numel() > 0) ? W_input.data_ptr<scalar_t>() : nullptr;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// lowrank_christoffel_kernel (L31)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// lowrank_christoffel_kernel (L36)
const scalar_t* v_ptr = v + idx * dim;
// lowrank_christoffel_kernel (L37)
const scalar_t* x_ptr = (x != nullptr) ? (x + idx * dim) : nullptr;
// lowrank_christoffel_kernel (L38)
scalar_t* gamma_ptr = gamma + idx * dim;
// lowrank_christoffel_kernel (L40)
Topology topology = static_cast<Topology>(topology_id);
// lowrank_christoffel_friction_kernel (L79)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// lowrank_christoffel_friction_kernel (L84)
const scalar_t* v_ptr = v + idx * dim;
// lowrank_christoffel_friction_kernel (L85)
const scalar_t* x_ptr = x + idx * dim;
// lowrank_christoffel_friction_kernel (L86)
const scalar_t* force_ptr = (force != nullptr) ? (force + idx * dim) : nullptr;
// lowrank_christoffel_friction_kernel (L87)
scalar_t* output_ptr = output + idx * dim;
// lowrank_christoffel_friction_kernel (L89)
Topology topology = static_cast<Topology>(topology_id);
// lowrank_christoffel_friction_kernel (L129)
int blocks = (batch_size + threads - 1) / threads;
// lowrank_christoffel_friction_kernel (L132)
const scalar_t* x_ptr = (x.numel() > 0) ? x.data_ptr<scalar_t>() : nullptr;
// lowrank_christoffel_friction_kernel (L133)
const scalar_t* V_w_ptr = (V_w.numel() > 0) ? V_w.data_ptr<scalar_t>() : nullptr;
// lowrank_christoffel_friction_kernel (L184)
int blocks = (batch_size + threads - 1) / threads;
// lowrank_christoffel_friction_kernel (L187)
const scalar_t* V_w_ptr = (V_w.numel() > 0) ? V_w.data_ptr<scalar_t>() : nullptr;
// lowrank_christoffel_friction_kernel (L188)
const scalar_t* force_ptr = (force.numel() > 0) ? force.data_ptr<scalar_t>() : nullptr;
// lowrank_christoffel_friction_kernel (L189)
const scalar_t* W_input_ptr = (W_input.numel() > 0) ? W_input.data_ptr<scalar_t>() : nullptr;
```

### gfn\cuda\src\geometry\lowrank_christoffel_backward.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 33 | christoffel_backward_kernel | `int b = blockIdx.x * blockDim.x + threadIdx.x;` |
| 36 | christoffel_backward_kernel | `const scalar_t* grad_out_b = grad_out + b * dim;` |
| 37 | christoffel_backward_kernel | `const scalar_t* gamma_b = gamma + b * dim;` |
| 38 | christoffel_backward_kernel | `const scalar_t* v_b = v + b * dim;` |
| 39 | christoffel_backward_kernel | `const scalar_t* x_b = (x != nullptr) ? (x + b * dim) : nullptr;` |
| 41 | christoffel_backward_kernel | `scalar_t* g_v_b = grad_v + b * dim;` |
| 42 | christoffel_backward_kernel | `scalar_t* g_x_b = (grad_x != nullptr) ? (grad_x + b * dim) : nullptr;` |
| 51 | christoffel_backward_kernel | `h_energy_acc += sum * sum;` |
| 55 | christoffel_backward_kernel | `h_energy /= static_cast<scalar_t>(rank);` |
| 57 | christoffel_backward_kernel | `scalar_t norm = sqrt(h_energy);` |
| 58 | christoffel_backward_kernel | `scalar_t S = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));` |
| 65 | christoffel_backward_kernel | `v_e /= static_cast<scalar_t>(dim);` |
| 67 | christoffel_backward_kernel | `M_plas = (static_cast<scalar_t>(1) + plasticity * static_cast<scalar_t>(0.1) * tanh_v_e);` |
| 70 | christoffel_backward_kernel | `scalar_t M_sing = static_cast<scalar_t>(1);` |
| 74 | christoffel_backward_kernel | `else { for (int i = 0; i < dim; ++i) pot += x_b[i] * V_w[i]; }` |
| 76 | christoffel_backward_kernel | `soft_m = sigmoid<scalar_t>(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));` |
| 77 | christoffel_backward_kernel | `M_sing = (static_cast<scalar_t>(1) + (sing_strength - static_cast<scalar_t>(1)) * soft_m);` |
| 79 | christoffel_backward_kernel | `scalar_t M = M_plas * M_sing;` |
| 85 | christoffel_backward_kernel | `scalar_t t = gamma_b[i] / static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>);` |
| 86 | christoffel_backward_kernel | `scalar_t grad_raw_i = grad_out_b[i] * (static_cast<scalar_t>(1) - t * t);` |
| 88 | christoffel_backward_kernel | `scalar_t q_base = h[j] * h[j] * S * M;` |
| 90 | christoffel_backward_kernel | `grad_q[j] += W[i * rank + j] * grad_raw_i;` |
| 103 | christoffel_backward_kernel | `S_sq_M_norm = M * S * S / (norm * static_cast<scalar_t>(rank));` |
| 107 | christoffel_backward_kernel | `scalar_t dL_dM_plas = sum_grad_q_h_sq * S * M_sing;` |
| 108 | christoffel_backward_kernel | `scalar_t plas_scale = plasticity * static_cast<scalar_t>(0.1);` |
| 109 | christoffel_backward_kernel | `scalar_t dM_plas_dv_scale = plas_scale * (static_cast<scalar_t>(1) - tanh_v_e * tanh_v_e) * static_cast<scalar_t>(2) / static_cast<scalar_t>(dim);` |
| 114 | christoffel_backward_kernel | `g_v_b[i] += U[i * rank + j] * grad_h[j];` |
| 117 | christoffel_backward_kernel | `g_v_b[i] += dL_dM_plas * dM_plas_dv_scale * v_b[i];` |
| 122 | christoffel_backward_kernel | `scalar_t dL_dM_sing = sum_grad_q_h_sq * S * M_plas;` |
| 123 | christoffel_backward_kernel | `scalar_t factor = dL_dM_sing * (sing_strength - static_cast<scalar_t>(1)) * static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * soft_m * (static_cast<scalar_t>(1) - soft_m) * gate * (static_cast<scalar_t>(1) - gate);` |
| 131 | christoffel_backward_kernel | `scalar_t feature = (topology == Topology::TORUS) ? (x_b ? sin(x_b[i]) : static_cast<scalar_t>(0)) : (x_b ? x_b[i] : static_cast<scalar_t>(0));` |
| 139 | christoffel_backward_kernel | `g_x_b[i] = factor * ((topology == Topology::TORUS) ? cos(x_b[i]) * V_w[i] : V_w[i]);` |
| 175 | christoffel_backward_kernel | `int blocks = (batch_size + threads - 1) / threads;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// christoffel_backward_kernel (L33)
int b = blockIdx.x * blockDim.x + threadIdx.x;
// christoffel_backward_kernel (L36)
const scalar_t* grad_out_b = grad_out + b * dim;
// christoffel_backward_kernel (L37)
const scalar_t* gamma_b = gamma + b * dim;
// christoffel_backward_kernel (L38)
const scalar_t* v_b = v + b * dim;
// christoffel_backward_kernel (L39)
const scalar_t* x_b = (x != nullptr) ? (x + b * dim) : nullptr;
// christoffel_backward_kernel (L41)
scalar_t* g_v_b = grad_v + b * dim;
// christoffel_backward_kernel (L42)
scalar_t* g_x_b = (grad_x != nullptr) ? (grad_x + b * dim) : nullptr;
// christoffel_backward_kernel (L51)
h_energy_acc += sum * sum;
// christoffel_backward_kernel (L55)
h_energy /= static_cast<scalar_t>(rank);
// christoffel_backward_kernel (L57)
scalar_t norm = sqrt(h_energy);
// christoffel_backward_kernel (L58)
scalar_t S = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// christoffel_backward_kernel (L65)
v_e /= static_cast<scalar_t>(dim);
// christoffel_backward_kernel (L67)
M_plas = (static_cast<scalar_t>(1) + plasticity * static_cast<scalar_t>(0.1) * tanh_v_e);
// christoffel_backward_kernel (L70)
scalar_t M_sing = static_cast<scalar_t>(1);
// christoffel_backward_kernel (L74)
else { for (int i = 0; i < dim; ++i) pot += x_b[i] * V_w[i]; }
// christoffel_backward_kernel (L76)
soft_m = sigmoid<scalar_t>(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));
// christoffel_backward_kernel (L77)
M_sing = (static_cast<scalar_t>(1) + (sing_strength - static_cast<scalar_t>(1)) * soft_m);
// christoffel_backward_kernel (L79)
scalar_t M = M_plas * M_sing;
// christoffel_backward_kernel (L85)
scalar_t t = gamma_b[i] / static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>);
// christoffel_backward_kernel (L86)
scalar_t grad_raw_i = grad_out_b[i] * (static_cast<scalar_t>(1) - t * t);
// christoffel_backward_kernel (L88)
scalar_t q_base = h[j] * h[j] * S * M;
// christoffel_backward_kernel (L90)
grad_q[j] += W[i * rank + j] * grad_raw_i;
// christoffel_backward_kernel (L103)
S_sq_M_norm = M * S * S / (norm * static_cast<scalar_t>(rank));
// christoffel_backward_kernel (L107)
scalar_t dL_dM_plas = sum_grad_q_h_sq * S * M_sing;
// christoffel_backward_kernel (L108)
scalar_t plas_scale = plasticity * static_cast<scalar_t>(0.1);
// christoffel_backward_kernel (L109)
scalar_t dM_plas_dv_scale = plas_scale * (static_cast<scalar_t>(1) - tanh_v_e * tanh_v_e) * static_cast<scalar_t>(2) / static_cast<scalar_t>(dim);
// christoffel_backward_kernel (L114)
g_v_b[i] += U[i * rank + j] * grad_h[j];
// christoffel_backward_kernel (L117)
g_v_b[i] += dL_dM_plas * dM_plas_dv_scale * v_b[i];
// christoffel_backward_kernel (L122)
scalar_t dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
// christoffel_backward_kernel (L123)
scalar_t factor = dL_dM_sing * (sing_strength - static_cast<scalar_t>(1)) * static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * soft_m * (static_cast<scalar_t>(1) - soft_m) * gate * (static_cast<scalar_t>(1) - gate);
// christoffel_backward_kernel (L131)
scalar_t feature = (topology == Topology::TORUS) ? (x_b ? sin(x_b[i]) : static_cast<scalar_t>(0)) : (x_b ? x_b[i] : static_cast<scalar_t>(0));
// christoffel_backward_kernel (L139)
g_x_b[i] = factor * ((topology == Topology::TORUS) ? cos(x_b[i]) * V_w[i] : V_w[i]);
// christoffel_backward_kernel (L175)
int blocks = (batch_size + threads - 1) / threads;
```

### gfn\cuda\src\geometry\lowrank_christoffel_friction_backward.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 46 | lowrank_christoffel_friction_backward_kernel | `int b = blockIdx.x * blockDim.x + threadIdx.x;` |
| 49 | lowrank_christoffel_friction_backward_kernel | `const scalar_t* grad_out_b = grad_out + b * dim;` |
| 50 | lowrank_christoffel_friction_backward_kernel | `const scalar_t* v_b = v + b * dim;` |
| 51 | lowrank_christoffel_friction_backward_kernel | `const scalar_t* x_b = x + b * dim;` |
| 52 | lowrank_christoffel_friction_backward_kernel | `const scalar_t* force_b = (force != nullptr) ? (force + b * dim) : nullptr;` |
| 60 | lowrank_christoffel_friction_backward_kernel | `scalar_t t = gamma_b[i] / static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>);` |
| 61 | lowrank_christoffel_friction_backward_kernel | `grad_pre[i] = grad_out_b[i] * (static_cast<scalar_t>(1) - t * t);` |
| 65 | lowrank_christoffel_friction_backward_kernel | `scalar_t* g_v_b = grad_v + b * dim;` |
| 66 | lowrank_christoffel_friction_backward_kernel | `scalar_t* g_x_b = grad_x + b * dim;` |
| 67 | lowrank_christoffel_friction_backward_kernel | `scalar_t* g_f_b = (grad_force != nullptr) ? (grad_force + b * dim) : nullptr;` |
| 74 | lowrank_christoffel_friction_backward_kernel | `grad_mu[i] = grad_out_b[i] * v_b[i];` |
| 78 | lowrank_christoffel_friction_backward_kernel | `int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;` |
| 87 | lowrank_christoffel_friction_backward_kernel | `v_norm = sqrt(v_norm);` |
| 99 | lowrank_christoffel_friction_backward_kernel | `scalar_t mu_base = s * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);` |
| 106 | lowrank_christoffel_friction_backward_kernel | `scalar_t scale_factor = static_cast<scalar_t>(1) + velocity_friction_scale * v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 107 | lowrank_christoffel_friction_backward_kernel | `dL_dmu_base *= scale_factor;` |
| 110 | lowrank_christoffel_friction_backward_kernel | `scalar_t dz = dL_dmu_base * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>) * s * (static_cast<scalar_t>(1) - s);` |
| 123 | lowrank_christoffel_friction_backward_kernel | `scalar_t d_sin = W_forget[i * feat_dim + j] * dz;` |
| 124 | lowrank_christoffel_friction_backward_kernel | `scalar_t d_cos = W_forget[i * feat_dim + (dim + j)] * dz;` |
| 125 | lowrank_christoffel_friction_backward_kernel | `g_x_b[j] += d_sin * cos(x_b[j]) - d_cos * sin(x_b[j]);` |
| 140 | lowrank_christoffel_friction_backward_kernel | `h_energy += sum * sum;` |
| 143 | lowrank_christoffel_friction_backward_kernel | `h_energy /= static_cast<scalar_t>(rank);` |
| 145 | lowrank_christoffel_friction_backward_kernel | `scalar_t norm_h = sqrt(h_energy);` |
| 146 | lowrank_christoffel_friction_backward_kernel | `scalar_t S = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm_h + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));` |
| 152 | lowrank_christoffel_friction_backward_kernel | `v_e /= static_cast<scalar_t>(dim);` |
| 153 | lowrank_christoffel_friction_backward_kernel | `M_plas = (static_cast<scalar_t>(1) + plasticity * static_cast<scalar_t>(0.1) * tanh(v_e));` |
| 156 | lowrank_christoffel_friction_backward_kernel | `scalar_t M_sing = static_cast<scalar_t>(1);` |
| 160 | lowrank_christoffel_friction_backward_kernel | `else { for (int i = 0; i < dim; ++i) pot += x_b[i] * V_w[i]; }` |
| 162 | lowrank_christoffel_friction_backward_kernel | `soft_m = sigmoid<scalar_t>(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));` |
| 163 | lowrank_christoffel_friction_backward_kernel | `M_sing = (static_cast<scalar_t>(1) + (sing_strength - static_cast<scalar_t>(1)) * soft_m);` |
| 165 | lowrank_christoffel_friction_backward_kernel | `scalar_t M = M_plas * M_sing;` |
| 169 | lowrank_christoffel_friction_backward_kernel | `scalar_t q_base = h[j] * h[j] * S * M;` |
| 172 | lowrank_christoffel_friction_backward_kernel | `grad_q[j] += W[i * rank + j] * grad_pre[i];` |
| 178 | lowrank_christoffel_friction_backward_kernel | `scalar_t S_sq_M_norm = (norm_h > EPSILON_STANDARD<scalar_t>) ? (M * S * S / norm_h) : static_cast<scalar_t>(0);` |
| 184 | lowrank_christoffel_friction_backward_kernel | `g_v_b[i] += U[i * rank + j] * grad_h[j];` |
| 195 | lowrank_christoffel_friction_backward_kernel | `scalar_t v_scale_grad_factor = velocity_friction_scale / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 212 | lowrank_christoffel_friction_backward_kernel | `scalar_t scale_term = 1.0f + v_scale_grad_factor * v_norm;` |
| 213 | lowrank_christoffel_friction_backward_kernel | `scalar_t mu_base_i = mu_b[i] / scale_term;` |
| 222 | lowrank_christoffel_friction_backward_kernel | `scalar_t scale_term = static_cast<scalar_t>(1) + v_scale_grad_factor * v_norm;` |
| 224 | lowrank_christoffel_friction_backward_kernel | `scalar_t mu_base_k = mu_b[k] / scale_term;` |
| 225 | lowrank_christoffel_friction_backward_kernel | `common_sum += grad_out_b[k] * v_b[k] * mu_base_k;` |
| 228 | lowrank_christoffel_friction_backward_kernel | `scalar_t factor = common_sum * v_scale_grad_factor / v_norm;` |
| 230 | lowrank_christoffel_friction_backward_kernel | `g_v_b[j] += factor * v_b[j];` |
| 262 | lowrank_christoffel_friction_backward_kernel | `const int rank = U.size(-1);` |
| 277 | lowrank_christoffel_friction_backward_kernel | `int threads = 128; // Reduced threads to increase register availability` |
| 278 | lowrank_christoffel_friction_backward_kernel | `int blocks = (batch_size + threads - 1) / threads;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// lowrank_christoffel_friction_backward_kernel (L46)
int b = blockIdx.x * blockDim.x + threadIdx.x;
// lowrank_christoffel_friction_backward_kernel (L49)
const scalar_t* grad_out_b = grad_out + b * dim;
// lowrank_christoffel_friction_backward_kernel (L50)
const scalar_t* v_b = v + b * dim;
// lowrank_christoffel_friction_backward_kernel (L51)
const scalar_t* x_b = x + b * dim;
// lowrank_christoffel_friction_backward_kernel (L52)
const scalar_t* force_b = (force != nullptr) ? (force + b * dim) : nullptr;
// lowrank_christoffel_friction_backward_kernel (L60)
scalar_t t = gamma_b[i] / static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>);
// lowrank_christoffel_friction_backward_kernel (L61)
grad_pre[i] = grad_out_b[i] * (static_cast<scalar_t>(1) - t * t);
// lowrank_christoffel_friction_backward_kernel (L65)
scalar_t* g_v_b = grad_v + b * dim;
// lowrank_christoffel_friction_backward_kernel (L66)
scalar_t* g_x_b = grad_x + b * dim;
// lowrank_christoffel_friction_backward_kernel (L67)
scalar_t* g_f_b = (grad_force != nullptr) ? (grad_force + b * dim) : nullptr;
// lowrank_christoffel_friction_backward_kernel (L74)
grad_mu[i] = grad_out_b[i] * v_b[i];
// lowrank_christoffel_friction_backward_kernel (L78)
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// lowrank_christoffel_friction_backward_kernel (L87)
v_norm = sqrt(v_norm);
// lowrank_christoffel_friction_backward_kernel (L99)
scalar_t mu_base = s * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// lowrank_christoffel_friction_backward_kernel (L106)
scalar_t scale_factor = static_cast<scalar_t>(1) + velocity_friction_scale * v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// lowrank_christoffel_friction_backward_kernel (L107)
dL_dmu_base *= scale_factor;
// lowrank_christoffel_friction_backward_kernel (L110)
scalar_t dz = dL_dmu_base * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>) * s * (static_cast<scalar_t>(1) - s);
// lowrank_christoffel_friction_backward_kernel (L123)
scalar_t d_sin = W_forget[i * feat_dim + j] * dz;
// lowrank_christoffel_friction_backward_kernel (L124)
scalar_t d_cos = W_forget[i * feat_dim + (dim + j)] * dz;
// lowrank_christoffel_friction_backward_kernel (L125)
g_x_b[j] += d_sin * cos(x_b[j]) - d_cos * sin(x_b[j]);
// lowrank_christoffel_friction_backward_kernel (L140)
h_energy += sum * sum;
// lowrank_christoffel_friction_backward_kernel (L143)
h_energy /= static_cast<scalar_t>(rank);
// lowrank_christoffel_friction_backward_kernel (L145)
scalar_t norm_h = sqrt(h_energy);
// lowrank_christoffel_friction_backward_kernel (L146)
scalar_t S = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm_h + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// lowrank_christoffel_friction_backward_kernel (L152)
v_e /= static_cast<scalar_t>(dim);
// lowrank_christoffel_friction_backward_kernel (L153)
M_plas = (static_cast<scalar_t>(1) + plasticity * static_cast<scalar_t>(0.1) * tanh(v_e));
// lowrank_christoffel_friction_backward_kernel (L156)
scalar_t M_sing = static_cast<scalar_t>(1);
// lowrank_christoffel_friction_backward_kernel (L160)
else { for (int i = 0; i < dim; ++i) pot += x_b[i] * V_w[i]; }
// lowrank_christoffel_friction_backward_kernel (L162)
soft_m = sigmoid<scalar_t>(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));
// lowrank_christoffel_friction_backward_kernel (L163)
M_sing = (static_cast<scalar_t>(1) + (sing_strength - static_cast<scalar_t>(1)) * soft_m);
// lowrank_christoffel_friction_backward_kernel (L165)
scalar_t M = M_plas * M_sing;
// lowrank_christoffel_friction_backward_kernel (L169)
scalar_t q_base = h[j] * h[j] * S * M;
// lowrank_christoffel_friction_backward_kernel (L172)
grad_q[j] += W[i * rank + j] * grad_pre[i];
// lowrank_christoffel_friction_backward_kernel (L178)
scalar_t S_sq_M_norm = (norm_h > EPSILON_STANDARD<scalar_t>) ? (M * S * S / norm_h) : static_cast<scalar_t>(0);
// lowrank_christoffel_friction_backward_kernel (L184)
g_v_b[i] += U[i * rank + j] * grad_h[j];
// lowrank_christoffel_friction_backward_kernel (L195)
scalar_t v_scale_grad_factor = velocity_friction_scale / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// lowrank_christoffel_friction_backward_kernel (L212)
scalar_t scale_term = 1.0f + v_scale_grad_factor * v_norm;
// lowrank_christoffel_friction_backward_kernel (L213)
scalar_t mu_base_i = mu_b[i] / scale_term;
// lowrank_christoffel_friction_backward_kernel (L222)
scalar_t scale_term = static_cast<scalar_t>(1) + v_scale_grad_factor * v_norm;
// lowrank_christoffel_friction_backward_kernel (L224)
scalar_t mu_base_k = mu_b[k] / scale_term;
// lowrank_christoffel_friction_backward_kernel (L225)
common_sum += grad_out_b[k] * v_b[k] * mu_base_k;
// lowrank_christoffel_friction_backward_kernel (L228)
scalar_t factor = common_sum * v_scale_grad_factor / v_norm;
// lowrank_christoffel_friction_backward_kernel (L230)
g_v_b[j] += factor * v_b[j];
// lowrank_christoffel_friction_backward_kernel (L262)
const int rank = U.size(-1);
// lowrank_christoffel_friction_backward_kernel (L277)
int threads = 128; // Reduced threads to increase register availability
// lowrank_christoffel_friction_backward_kernel (L278)
int blocks = (batch_size + threads - 1) / threads;
```

### gfn\cuda\src\integrators\recurrent_manifold_fused.cpp

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 3 | Global | `* ===================================================== * * BUG-7 FIX (2026-02-11): Added energy normalization, soft clamping, * constant friction damping, and velocity saturation. * * WARNING: This is NOT a true CUDA kernel — it uses ATen C++ ops. * It is kept as a fast inference-only path. For training with gradients, * the Python autograd fallback handles the backward pass correctly. * * Missing features vs. full Python: * - No learned friction gates (uses constant DEFAULT_FRICTION) * - No boundary conditions (assumes Euclidean topology) * - No plasticity or singularity amplification * - No hysteresis/ghost forces */ #include <torch/extension.h> #include <vector> #include <iostream> static constexpr double CURVATURE_CLAMP = 3.0;` |
| 25 | Global | `static constexpr double EPSILON_STANDARD = 1e-7;` |
| 26 | Global | `static constexpr double DEFAULT_FRICTION = 0.002;` |
| 70 | Global | `const auto head_dim = D / H;` |
| 71 | Global | `const auto L = U_stack.size(0) / H;` |
| 72 | Global | `const auto dt_eff = dt * dt_scale;` |
| 83 | Global | `int64_t s = h * head_dim;` |
| 84 | Global | `int64_t e = s + head_dim;` |
| 88 | Global | `auto U_h = U_stack.index({l * H + h});` |
| 89 | Global | `auto W_h = W_stack.index({l * H + h});` |
| 93 | Global | `auto h_sq = h_vec * h_vec;` |
| 96 | Global | `auto energy = h_sq.mean(-1, /*keepdim=*/true);` |
| 97 | Global | `auto S = 1.0 / (1.0 + energy.sqrt() + EPSILON_STANDARD);` |
| 100 | Global | `auto gamma = at::matmul(h_sq * S, W_h.t());` |
| 101 | Global | `gamma = gamma.clamp(-CURVATURE_CLAMP, CURVATURE_CLAMP);` |
| 104 | Global | `auto v_new = v_h + f_h * dt_eff - gamma * dt_eff - v_h * (DEFAULT_FRICTION * dt_eff);` |
| 106 | Global | `auto x_new = x_h + v_new.mul(dt_eff);` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L3)
* ===================================================== * * BUG-7 FIX (2026-02-11): Added energy normalization, soft clamping, * constant friction damping, and velocity saturation. * * WARNING: This is NOT a true CUDA kernel — it uses ATen C++ ops. * It is kept as a fast inference-only path. For training with gradients, * the Python autograd fallback handles the backward pass correctly. * * Missing features vs. full Python: * - No learned friction gates (uses constant DEFAULT_FRICTION) * - No boundary conditions (assumes Euclidean topology) * - No plasticity or singularity amplification * - No hysteresis/ghost forces */ #include <torch/extension.h> #include <vector> #include <iostream> static constexpr double CURVATURE_CLAMP = 3.0;
// Global (L25)
static constexpr double EPSILON_STANDARD = 1e-7;
// Global (L26)
static constexpr double DEFAULT_FRICTION = 0.002;
// Global (L70)
const auto head_dim = D / H;
// Global (L71)
const auto L = U_stack.size(0) / H;
// Global (L72)
const auto dt_eff = dt * dt_scale;
// Global (L83)
int64_t s = h * head_dim;
// Global (L84)
int64_t e = s + head_dim;
// Global (L88)
auto U_h = U_stack.index({l * H + h});
// Global (L89)
auto W_h = W_stack.index({l * H + h});
// Global (L93)
auto h_sq = h_vec * h_vec;
// Global (L96)
auto energy = h_sq.mean(-1, /*keepdim=*/true);
// Global (L97)
auto S = 1.0 / (1.0 + energy.sqrt() + EPSILON_STANDARD);
// Global (L100)
auto gamma = at::matmul(h_sq * S, W_h.t());
// Global (L101)
gamma = gamma.clamp(-CURVATURE_CLAMP, CURVATURE_CLAMP);
// Global (L104)
auto v_new = v_h + f_h * dt_eff - gamma * dt_eff - v_h * (DEFAULT_FRICTION * dt_eff);
// Global (L106)
auto x_new = x_h + v_new.mul(dt_eff);
```

### gfn\cuda\src\integrators\runge_kutta\heun_backward.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 38 | heun_backward_kernel | `int idx = blockIdx.x * blockDim.x + threadIdx.x;` |
| 41 | heun_backward_kernel | `Topology topology = static_cast<Topology>(topology_id);` |
| 42 | heun_backward_kernel | `scalar_t effective_dt = dt * dt_scale;` |
| 43 | heun_backward_kernel | `scalar_t h_half = static_cast<scalar_t>(0.5) * effective_dt;` |
| 48 | heun_backward_kernel | `lx[i] = grad_x_out[idx * dim + i];` |
| 49 | heun_backward_kernel | `lv[i] = grad_v_out[idx * dim + i];` |
| 52 | heun_backward_kernel | `scalar_t* gU_b = grad_U + idx * dim * rank;` |
| 53 | heun_backward_kernel | `scalar_t* gW_b = grad_W + idx * dim * rank;` |
| 54 | heun_backward_kernel | `scalar_t* gf_b = grad_force + idx * dim;` |
| 60 | heun_backward_kernel | `const scalar_t* x_n = traj_x + idx * (steps + 1) * dim + step * dim;` |
| 61 | heun_backward_kernel | `const scalar_t* v_n = traj_v + idx * (steps + 1) * dim + step * dim;` |
| 62 | heun_backward_kernel | `const scalar_t* acc1 = traj_acc1 + idx * steps * dim + step * dim;` |
| 67 | heun_backward_kernel | `x_pred[i] = x_n[i] + effective_dt * v_n[i];` |
| 68 | heun_backward_kernel | `v_pred[i] = v_n[i] + effective_dt * acc1[i];` |
| 77 | heun_backward_kernel | `l_v_n[i] = lx[i] * h_half + lv[i];` |
| 78 | heun_backward_kernel | `l_v_pred[i] = lx[i] * h_half;` |
| 79 | heun_backward_kernel | `l_acc1[i] = lv[i] * h_half;` |
| 80 | heun_backward_kernel | `l_acc2[i] = lv[i] * h_half;` |
| 90 | heun_backward_kernel | `l_gamma2[i] = -l_acc2[i]; // acc2 = F - gamma2` |
| 91 | heun_backward_kernel | `gf_b[i] += l_acc2[i];` |
| 99 | heun_backward_kernel | `l_v_pred[i] += gv_c[i];` |
| 108 | heun_backward_kernel | `l_x_n[i] += gx_c[i]; // from S2 through x_pred` |
| 109 | heun_backward_kernel | `l_v_n[i] += l_v_pred[i] + effective_dt * gx_c[i];` |
| 110 | heun_backward_kernel | `l_acc1[i] += l_v_pred[i] * effective_dt;` |
| 119 | heun_backward_kernel | `l_gamma1[i] = -l_acc1[i];` |
| 120 | heun_backward_kernel | `gf_b[i] += l_acc1[i];` |
| 127 | heun_backward_kernel | `lx[i] = l_x_n[i] + gx_c[i];` |
| 128 | heun_backward_kernel | `lv[i] = l_v_n[i] + gv_c[i];` |
| 133 | heun_backward_kernel | `grad_x_in[idx * dim + i] = lx[i];` |
| 134 | heun_backward_kernel | `grad_v_in[idx * dim + i] = lv[i];` |
| 148 | heun_forward_traj_kernel | `int idx = blockIdx.x * blockDim.x + threadIdx.x;` |
| 154 | heun_forward_traj_kernel | `Topology topology = static_cast<Topology>(topology_id);` |
| 155 | heun_forward_traj_kernel | `scalar_t effective_dt = dt * dt_scale;` |
| 156 | heun_forward_traj_kernel | `const scalar_t* f_ptr = force + idx * dim;` |
| 160 | heun_forward_traj_kernel | `traj_x[idx * (steps + 1) * dim + step * dim + i] = cx[i];` |
| 161 | heun_forward_traj_kernel | `traj_v[idx * (steps + 1) * dim + step * dim + i] = cv[i];` |
| 166 | heun_forward_traj_kernel | `acc1[i] = f_ptr[i] - gamma[i];` |
| 167 | heun_forward_traj_kernel | `traj_acc1[idx * steps * dim + step * dim + i] = acc1[i];` |
| 168 | heun_forward_traj_kernel | `x_pred[i] = cx[i] + effective_dt * cv[i];` |
| 169 | heun_forward_traj_kernel | `v_pred[i] = cv[i] + effective_dt * acc1[i];` |
| 175 | heun_forward_traj_kernel | `acc2[i] = f_ptr[i] - gamma[i];` |
| 176 | heun_forward_traj_kernel | `cx[i] += (effective_dt / static_cast<scalar_t>(2.0)) * (cv[i] + v_pred[i]);` |
| 177 | heun_forward_traj_kernel | `cv[i] += (effective_dt / static_cast<scalar_t>(2.0)) * (acc1[i] + acc2[i]);` |
| 182 | heun_forward_traj_kernel | `traj_x[idx * (steps + 1) * dim + steps * dim + i] = cx[i];` |
| 183 | heun_forward_traj_kernel | `traj_v[idx * (steps + 1) * dim + steps * dim + i] = cv[i];` |
| 203 | heun_forward_traj_kernel | `auto traj_x = torch::empty({batch_size, steps + 1, dim}, options);` |
| 204 | heun_forward_traj_kernel | `auto traj_v = torch::empty({batch_size, steps + 1, dim}, options);` |
| 214 | heun_forward_traj_kernel | `int blocks = (batch_size + threads - 1) / threads;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// heun_backward_kernel (L38)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// heun_backward_kernel (L41)
Topology topology = static_cast<Topology>(topology_id);
// heun_backward_kernel (L42)
scalar_t effective_dt = dt * dt_scale;
// heun_backward_kernel (L43)
scalar_t h_half = static_cast<scalar_t>(0.5) * effective_dt;
// heun_backward_kernel (L48)
lx[i] = grad_x_out[idx * dim + i];
// heun_backward_kernel (L49)
lv[i] = grad_v_out[idx * dim + i];
// heun_backward_kernel (L52)
scalar_t* gU_b = grad_U + idx * dim * rank;
// heun_backward_kernel (L53)
scalar_t* gW_b = grad_W + idx * dim * rank;
// heun_backward_kernel (L54)
scalar_t* gf_b = grad_force + idx * dim;
// heun_backward_kernel (L60)
const scalar_t* x_n = traj_x + idx * (steps + 1) * dim + step * dim;
// heun_backward_kernel (L61)
const scalar_t* v_n = traj_v + idx * (steps + 1) * dim + step * dim;
// heun_backward_kernel (L62)
const scalar_t* acc1 = traj_acc1 + idx * steps * dim + step * dim;
// heun_backward_kernel (L67)
x_pred[i] = x_n[i] + effective_dt * v_n[i];
// heun_backward_kernel (L68)
v_pred[i] = v_n[i] + effective_dt * acc1[i];
// heun_backward_kernel (L77)
l_v_n[i] = lx[i] * h_half + lv[i];
// heun_backward_kernel (L78)
l_v_pred[i] = lx[i] * h_half;
// heun_backward_kernel (L79)
l_acc1[i] = lv[i] * h_half;
// heun_backward_kernel (L80)
l_acc2[i] = lv[i] * h_half;
// heun_backward_kernel (L90)
l_gamma2[i] = -l_acc2[i]; // acc2 = F - gamma2
// heun_backward_kernel (L91)
gf_b[i] += l_acc2[i];
// heun_backward_kernel (L99)
l_v_pred[i] += gv_c[i];
// heun_backward_kernel (L108)
l_x_n[i] += gx_c[i]; // from S2 through x_pred
// heun_backward_kernel (L109)
l_v_n[i] += l_v_pred[i] + effective_dt * gx_c[i];
// heun_backward_kernel (L110)
l_acc1[i] += l_v_pred[i] * effective_dt;
// heun_backward_kernel (L119)
l_gamma1[i] = -l_acc1[i];
// heun_backward_kernel (L120)
gf_b[i] += l_acc1[i];
// heun_backward_kernel (L127)
lx[i] = l_x_n[i] + gx_c[i];
// heun_backward_kernel (L128)
lv[i] = l_v_n[i] + gv_c[i];
// heun_backward_kernel (L133)
grad_x_in[idx * dim + i] = lx[i];
// heun_backward_kernel (L134)
grad_v_in[idx * dim + i] = lv[i];
// heun_forward_traj_kernel (L148)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// heun_forward_traj_kernel (L154)
Topology topology = static_cast<Topology>(topology_id);
// heun_forward_traj_kernel (L155)
scalar_t effective_dt = dt * dt_scale;
// heun_forward_traj_kernel (L156)
const scalar_t* f_ptr = force + idx * dim;
// heun_forward_traj_kernel (L160)
traj_x[idx * (steps + 1) * dim + step * dim + i] = cx[i];
// heun_forward_traj_kernel (L161)
traj_v[idx * (steps + 1) * dim + step * dim + i] = cv[i];
// heun_forward_traj_kernel (L166)
acc1[i] = f_ptr[i] - gamma[i];
// heun_forward_traj_kernel (L167)
traj_acc1[idx * steps * dim + step * dim + i] = acc1[i];
// heun_forward_traj_kernel (L168)
x_pred[i] = cx[i] + effective_dt * cv[i];
// heun_forward_traj_kernel (L169)
v_pred[i] = cv[i] + effective_dt * acc1[i];
// heun_forward_traj_kernel (L175)
acc2[i] = f_ptr[i] - gamma[i];
// heun_forward_traj_kernel (L176)
cx[i] += (effective_dt / static_cast<scalar_t>(2.0)) * (cv[i] + v_pred[i]);
// heun_forward_traj_kernel (L177)
cv[i] += (effective_dt / static_cast<scalar_t>(2.0)) * (acc1[i] + acc2[i]);
// heun_forward_traj_kernel (L182)
traj_x[idx * (steps + 1) * dim + steps * dim + i] = cx[i];
// heun_forward_traj_kernel (L183)
traj_v[idx * (steps + 1) * dim + steps * dim + i] = cv[i];
// heun_forward_traj_kernel (L203)
auto traj_x = torch::empty({batch_size, steps + 1, dim}, options);
// heun_forward_traj_kernel (L204)
auto traj_v = torch::empty({batch_size, steps + 1, dim}, options);
// heun_forward_traj_kernel (L214)
int blocks = (batch_size + threads - 1) / threads;
```

### gfn\cuda\src\integrators\runge_kutta\heun_fused.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 3 | Global | `* ==================================== * * Block-parallel Heun integrator for manifold dynamics. * 1 block = 1 batch item, blockDim.x = dim. * * FIX (2026-02-11): Rewritten from single-thread-per-batch to block-parallel. *   BUG-1: Old kernel used 1 thread per batch with scalar_t[64] stacks, *          limiting dim<=64 and running without dimension parallelism. *   BUG-2: Host wrapper used hardcoded scalar_t instead of AT_DISPATCH. *   BUG-5: Added W_input parameter for full friction gate parity with Python. * * Integration scheme (Heun / RK2 Predictor-Corrector): *   1. Compute acceleration: a1 = F - Γ(v, x) - μ(x, F) * v *   2. Euler predictor: x_pred = x + dt*v, v_pred = v + dt*a1 *   3. Compute acceleration at predicted state: a2 = F - Γ(v_pred, x_pred) - μ(x_pred, F)*v_pred *   4. Corrector (average): x_new = x + dt/2*(v + v_pred), v_new = v + dt/2*(a1 + a2) */ #include "../../geometry/christoffel_impl.cuh" #include <torch/extension.h> #include <cuda.h> #include <cuda_runtime.h> namespace gfn { namespace cuda { template <typename scalar_t> __device__ inline scalar_t warp_reduce_sum_heun(scalar_t val) { for (int offset = warpSize/2; offset > 0; offset /= 2)` |
| 36 | Global | `val += __shfl_down_sync(0xffffffff, val, offset);` |
| 44 | block_reduce_sum_heun | `int wid = threadIdx.x / warpSize;` |
| 51 | block_reduce_sum_heun | `val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;` |
| 84 | christoffel_distributed_heun | `*gamma_val = 0.0f;` |
| 91 | christoffel_distributed_heun | `scalar_t v_ph = v_shared[tid + 1];` |
| 95 | christoffel_distributed_heun | `scalar_t denom = R + r * c;` |
| 97 | christoffel_distributed_heun | `scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 98 | christoffel_distributed_heun | `scalar_t g0 = term_th * (v_ph * v_ph);` |
| 100 | christoffel_distributed_heun | `*gamma_val = soft_clamp<scalar_t>( static_cast<scalar_t>(g0) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );` |
| 104 | christoffel_distributed_heun | `} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];` |
| 107 | christoffel_distributed_heun | `scalar_t v_th = v_shared[tid - 1];` |
| 111 | christoffel_distributed_heun | `scalar_t denom = R + r * c;` |
| 113 | christoffel_distributed_heun | `scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 114 | christoffel_distributed_heun | `scalar_t g1 = 2.0f * term_ph * v_ph * v_th;` |
| 116 | christoffel_distributed_heun | `*gamma_val = soft_clamp<scalar_t>( static_cast<scalar_t>(g1) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );` |
| 126 | christoffel_distributed_heun | `scalar_t u_val = U[tid * rank + k];` |
| 127 | christoffel_distributed_heun | `scalar_t prod = u_val * v_val;` |
| 146 | christoffel_distributed_heun | `scalar_t norm = sqrt(energy);` |
| 147 | christoffel_distributed_heun | `S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));` |
| 156 | christoffel_distributed_heun | `scalar_t h_sq = h_val * h_val * S_shared * M_shared;` |
| 157 | christoffel_distributed_heun | `sum_gamma += W[tid * rank + k] * h_sq;` |
| 160 | christoffel_distributed_heun | `*gamma_val = soft_clamp<scalar_t>(static_cast<scalar_t>(sum_gamma), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));` |
| 189 | friction_distributed_heun | `features_shared[dim + tid] = c;` |
| 195 | friction_distributed_heun | `int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;` |
| 200 | friction_distributed_heun | `gate_sum += W_forget[tid * feat_dim + j] * features_shared[j];` |
| 213 | friction_distributed_heun | `input_sum += W_input[tid * dim + j] * features_shared[j];` |
| 215 | friction_distributed_heun | `gate_sum += input_sum;` |
| 218 | friction_distributed_heun | `scalar_t base_friction = sigmoid(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);` |
| 222 | friction_distributed_heun | `scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 223 | friction_distributed_heun | `*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);` |
| 225 | friction_distributed_heun | `*friction_val = base_friction;` |
| 281 | heun_fused_kernel | `scalar_t* h_shared = reinterpret_cast<scalar_t*>(shared_mem);` |
| 282 | heun_fused_kernel | `scalar_t* features_shared = h_shared + rank;` |
| 283 | heun_fused_kernel | `scalar_t* x_shared = features_shared + (2 * dim);` |
| 284 | heun_fused_kernel | `scalar_t* v_shared = x_shared + dim;` |
| 287 | heun_fused_kernel | `scalar_t curr_x = x_in[bid * dim + tid];` |
| 288 | heun_fused_kernel | `scalar_t curr_v = v_in[bid * dim + tid];` |
| 289 | heun_fused_kernel | `scalar_t f_ext = force[bid * dim + tid];` |
| 293 | heun_fused_kernel | `hyst_val = hysteresis_state[bid * dim + tid];` |
| 296 | heun_fused_kernel | `Topology topology = static_cast<Topology>(topology_id);` |
| 297 | heun_fused_kernel | `scalar_t effective_dt = dt * dt_scale;` |
| 310 | heun_fused_kernel | `scalar_t* hyst_shared_buf = features_shared; // Reuse features buf` |
| 316 | heun_fused_kernel | `sum += hyst_shared_buf[j] * hyst_readout_w[tid * dim + j];` |
| 326 | heun_fused_kernel | `scalar_t v_sq = curr_v * curr_v;` |
| 328 | heun_fused_kernel | `scalar_t v_norm = sqrt(v_sum);` |
| 347 | heun_fused_kernel | `scalar_t acc1 = f_ext + f_ghost - gamma1 - friction * curr_v;` |
| 350 | heun_fused_kernel | `scalar_t v_pred = curr_v + effective_dt * acc1;` |
| 351 | heun_fused_kernel | `scalar_t x_pred = curr_x + effective_dt * curr_v;` |
| 352 | heun_fused_kernel | `x_pred = apply_boundary_device(x_pred, topology);` |
| 367 | heun_fused_kernel | `scalar_t v_sq = v_pred * v_pred;` |
| 369 | heun_fused_kernel | `scalar_t v_norm = sqrt(v_sum);` |
| 388 | heun_fused_kernel | `scalar_t acc2 = f_ext + f_ghost - gamma2 - friction2 * v_pred;` |
| 391 | heun_fused_kernel | `curr_x += (effective_dt / 2.0f) * (curr_v + v_pred);` |
| 392 | heun_fused_kernel | `curr_v += (effective_dt / 2.0f) * (acc1 + acc2);` |
| 395 | heun_fused_kernel | `curr_x = apply_boundary_device(curr_x, topology);` |
| 399 | heun_fused_kernel | `scalar_t* input_shared = features_shared; // Reuse` |
| 404 | heun_fused_kernel | `input_shared[dim + tid] = c;` |
| 410 | heun_fused_kernel | `int offset = (topology == Topology::TORUS) ? 2*dim : dim;` |
| 411 | heun_fused_kernel | `input_shared[offset + tid] = curr_v;` |
| 416 | heun_fused_kernel | `sum += input_shared[j] * hyst_update_w[tid * hyst_in_dim + j];` |
| 419 | heun_fused_kernel | `hyst_val = hyst_val * hyst_decay + tanhf(sum);` |
| 424 | heun_fused_kernel | `x_out[bid * dim + tid] = curr_x;` |
| 425 | heun_fused_kernel | `v_out[bid * dim + tid] = curr_v;` |
| 428 | heun_fused_kernel | `hysteresis_state[bid * dim + tid] = hyst_val;` |
| 485 | heun_fused_kernel | `size_t shared_mem_size = (rank + 4 * dim) * x.element_size();` |
| 490 | heun_fused_kernel | `const scalar_t* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<scalar_t>() : nullptr;` |
| 491 | heun_fused_kernel | `const scalar_t* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<scalar_t>() : nullptr;` |
| 492 | heun_fused_kernel | `const scalar_t* W_input_ptr = (W_input.numel() > 0) ? W_input.data_ptr<scalar_t>() : nullptr;` |
| 494 | heun_fused_kernel | `scalar_t* hyst_state_ptr = nullptr;` |
| 495 | heun_fused_kernel | `const scalar_t* h_up_w_ptr = nullptr;` |
| 496 | heun_fused_kernel | `const scalar_t* h_up_b_ptr = nullptr;` |
| 497 | heun_fused_kernel | `const scalar_t* h_rd_w_ptr = nullptr;` |
| 498 | heun_fused_kernel | `const scalar_t* h_rd_b_ptr = nullptr;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L3)
* ==================================== * * Block-parallel Heun integrator for manifold dynamics. * 1 block = 1 batch item, blockDim.x = dim. * * FIX (2026-02-11): Rewritten from single-thread-per-batch to block-parallel. *   BUG-1: Old kernel used 1 thread per batch with scalar_t[64] stacks, *          limiting dim<=64 and running without dimension parallelism. *   BUG-2: Host wrapper used hardcoded scalar_t instead of AT_DISPATCH. *   BUG-5: Added W_input parameter for full friction gate parity with Python. * * Integration scheme (Heun / RK2 Predictor-Corrector): *   1. Compute acceleration: a1 = F - Γ(v, x) - μ(x, F) * v *   2. Euler predictor: x_pred = x + dt*v, v_pred = v + dt*a1 *   3. Compute acceleration at predicted state: a2 = F - Γ(v_pred, x_pred) - μ(x_pred, F)*v_pred *   4. Corrector (average): x_new = x + dt/2*(v + v_pred), v_new = v + dt/2*(a1 + a2) */ #include "../../geometry/christoffel_impl.cuh" #include <torch/extension.h> #include <cuda.h> #include <cuda_runtime.h> namespace gfn { namespace cuda { template <typename scalar_t> __device__ inline scalar_t warp_reduce_sum_heun(scalar_t val) { for (int offset = warpSize/2; offset > 0; offset /= 2)
// Global (L36)
val += __shfl_down_sync(0xffffffff, val, offset);
// block_reduce_sum_heun (L44)
int wid = threadIdx.x / warpSize;
// block_reduce_sum_heun (L51)
val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
// christoffel_distributed_heun (L84)
*gamma_val = 0.0f;
// christoffel_distributed_heun (L91)
scalar_t v_ph = v_shared[tid + 1];
// christoffel_distributed_heun (L95)
scalar_t denom = R + r * c;
// christoffel_distributed_heun (L97)
scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// christoffel_distributed_heun (L98)
scalar_t g0 = term_th * (v_ph * v_ph);
// christoffel_distributed_heun (L100)
*gamma_val = soft_clamp<scalar_t>( static_cast<scalar_t>(g0) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );
// christoffel_distributed_heun (L104)
} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];
// christoffel_distributed_heun (L107)
scalar_t v_th = v_shared[tid - 1];
// christoffel_distributed_heun (L111)
scalar_t denom = R + r * c;
// christoffel_distributed_heun (L113)
scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// christoffel_distributed_heun (L114)
scalar_t g1 = 2.0f * term_ph * v_ph * v_th;
// christoffel_distributed_heun (L116)
*gamma_val = soft_clamp<scalar_t>( static_cast<scalar_t>(g1) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );
// christoffel_distributed_heun (L126)
scalar_t u_val = U[tid * rank + k];
// christoffel_distributed_heun (L127)
scalar_t prod = u_val * v_val;
// christoffel_distributed_heun (L146)
scalar_t norm = sqrt(energy);
// christoffel_distributed_heun (L147)
S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// christoffel_distributed_heun (L156)
scalar_t h_sq = h_val * h_val * S_shared * M_shared;
// christoffel_distributed_heun (L157)
sum_gamma += W[tid * rank + k] * h_sq;
// christoffel_distributed_heun (L160)
*gamma_val = soft_clamp<scalar_t>(static_cast<scalar_t>(sum_gamma), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
// friction_distributed_heun (L189)
features_shared[dim + tid] = c;
// friction_distributed_heun (L195)
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// friction_distributed_heun (L200)
gate_sum += W_forget[tid * feat_dim + j] * features_shared[j];
// friction_distributed_heun (L213)
input_sum += W_input[tid * dim + j] * features_shared[j];
// friction_distributed_heun (L215)
gate_sum += input_sum;
// friction_distributed_heun (L218)
scalar_t base_friction = sigmoid(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// friction_distributed_heun (L222)
scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// friction_distributed_heun (L223)
*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
// friction_distributed_heun (L225)
*friction_val = base_friction;
// heun_fused_kernel (L281)
scalar_t* h_shared = reinterpret_cast<scalar_t*>(shared_mem);
// heun_fused_kernel (L282)
scalar_t* features_shared = h_shared + rank;
// heun_fused_kernel (L283)
scalar_t* x_shared = features_shared + (2 * dim);
// heun_fused_kernel (L284)
scalar_t* v_shared = x_shared + dim;
// heun_fused_kernel (L287)
scalar_t curr_x = x_in[bid * dim + tid];
// heun_fused_kernel (L288)
scalar_t curr_v = v_in[bid * dim + tid];
// heun_fused_kernel (L289)
scalar_t f_ext = force[bid * dim + tid];
// heun_fused_kernel (L293)
hyst_val = hysteresis_state[bid * dim + tid];
// heun_fused_kernel (L296)
Topology topology = static_cast<Topology>(topology_id);
// heun_fused_kernel (L297)
scalar_t effective_dt = dt * dt_scale;
// heun_fused_kernel (L310)
scalar_t* hyst_shared_buf = features_shared; // Reuse features buf
// heun_fused_kernel (L316)
sum += hyst_shared_buf[j] * hyst_readout_w[tid * dim + j];
// heun_fused_kernel (L326)
scalar_t v_sq = curr_v * curr_v;
// heun_fused_kernel (L328)
scalar_t v_norm = sqrt(v_sum);
// heun_fused_kernel (L347)
scalar_t acc1 = f_ext + f_ghost - gamma1 - friction * curr_v;
// heun_fused_kernel (L350)
scalar_t v_pred = curr_v + effective_dt * acc1;
// heun_fused_kernel (L351)
scalar_t x_pred = curr_x + effective_dt * curr_v;
// heun_fused_kernel (L352)
x_pred = apply_boundary_device(x_pred, topology);
// heun_fused_kernel (L367)
scalar_t v_sq = v_pred * v_pred;
// heun_fused_kernel (L369)
scalar_t v_norm = sqrt(v_sum);
// heun_fused_kernel (L388)
scalar_t acc2 = f_ext + f_ghost - gamma2 - friction2 * v_pred;
// heun_fused_kernel (L391)
curr_x += (effective_dt / 2.0f) * (curr_v + v_pred);
// heun_fused_kernel (L392)
curr_v += (effective_dt / 2.0f) * (acc1 + acc2);
// heun_fused_kernel (L395)
curr_x = apply_boundary_device(curr_x, topology);
// heun_fused_kernel (L399)
scalar_t* input_shared = features_shared; // Reuse
// heun_fused_kernel (L404)
input_shared[dim + tid] = c;
// heun_fused_kernel (L410)
int offset = (topology == Topology::TORUS) ? 2*dim : dim;
// heun_fused_kernel (L411)
input_shared[offset + tid] = curr_v;
// heun_fused_kernel (L416)
sum += input_shared[j] * hyst_update_w[tid * hyst_in_dim + j];
// heun_fused_kernel (L419)
hyst_val = hyst_val * hyst_decay + tanhf(sum);
// heun_fused_kernel (L424)
x_out[bid * dim + tid] = curr_x;
// heun_fused_kernel (L425)
v_out[bid * dim + tid] = curr_v;
// heun_fused_kernel (L428)
hysteresis_state[bid * dim + tid] = hyst_val;
// heun_fused_kernel (L485)
size_t shared_mem_size = (rank + 4 * dim) * x.element_size();
// heun_fused_kernel (L490)
const scalar_t* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<scalar_t>() : nullptr;
// heun_fused_kernel (L491)
const scalar_t* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<scalar_t>() : nullptr;
// heun_fused_kernel (L492)
const scalar_t* W_input_ptr = (W_input.numel() > 0) ? W_input.data_ptr<scalar_t>() : nullptr;
// heun_fused_kernel (L494)
scalar_t* hyst_state_ptr = nullptr;
// heun_fused_kernel (L495)
const scalar_t* h_up_w_ptr = nullptr;
// heun_fused_kernel (L496)
const scalar_t* h_up_b_ptr = nullptr;
// heun_fused_kernel (L497)
const scalar_t* h_rd_w_ptr = nullptr;
// heun_fused_kernel (L498)
const scalar_t* h_rd_b_ptr = nullptr;
```

### gfn\cuda\src\integrators\symplectic\leapfrog_backward.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 67 | leapfrog_backward_kernel | `int idx = blockIdx.x * blockDim.x + threadIdx.x;` |
| 70 | leapfrog_backward_kernel | `Topology topology = static_cast<Topology>(topology_id);` |
| 71 | leapfrog_backward_kernel | `scalar_t effective_dt = dt * dt_scale;` |
| 72 | leapfrog_backward_kernel | `scalar_t h = static_cast<scalar_t>(0.5) * effective_dt;` |
| 78 | leapfrog_backward_kernel | `lx[i] = grad_x_out[idx * dim + i];` |
| 79 | leapfrog_backward_kernel | `lv[i] = grad_v_out[idx * dim + i];` |
| 83 | leapfrog_backward_kernel | `scalar_t* gU_b = grad_U + idx * dim * rank;` |
| 84 | leapfrog_backward_kernel | `scalar_t* gW_b = grad_W + idx * dim * rank;` |
| 85 | leapfrog_backward_kernel | `int f_dim = (topology == Topology::TORUS) ? 2 * dim : dim;` |
| 86 | leapfrog_backward_kernel | `scalar_t* gWf_b = grad_W_forget + idx * dim * f_dim;` |
| 87 | leapfrog_backward_kernel | `scalar_t* gbf_b = grad_b_forget + idx * dim;` |
| 88 | leapfrog_backward_kernel | `scalar_t* gf_b = grad_force + idx * dim;` |
| 89 | leapfrog_backward_kernel | `scalar_t* gWinput_b = (W_input != nullptr) ? (grad_W_input + idx * dim * dim) : nullptr;` |
| 90 | leapfrog_backward_kernel | `scalar_t* gVw_b = (V_w != nullptr) ? (grad_V_w + idx * dim) : nullptr;` |
| 93 | leapfrog_backward_kernel | `scalar_t* gHupdate_w_b = hyst_enabled ? (grad_hyst_update_w + idx * dim * hyst_in_dim) : nullptr;` |
| 94 | leapfrog_backward_kernel | `scalar_t* gHupdate_b_b = hyst_enabled ? (grad_hyst_update_b + idx * dim) : nullptr;` |
| 95 | leapfrog_backward_kernel | `scalar_t* gHreadout_w_b = hyst_enabled ? (grad_hyst_readout_w + idx * dim * dim) : nullptr;` |
| 96 | leapfrog_backward_kernel | `scalar_t* gHreadout_b_b = hyst_enabled ? (grad_hyst_readout_b + idx * dim) : nullptr;` |
| 98 | leapfrog_backward_kernel | `const scalar_t* f_ptr = force + idx * dim;` |
| 122 | leapfrog_backward_kernel | `const scalar_t* x_n = traj_x + idx * (steps + 1) * dim + step * dim;` |
| 123 | leapfrog_backward_kernel | `const scalar_t* v_n = traj_v + idx * steps * 2 * dim + (step * 2 + 0) * dim;` |
| 124 | leapfrog_backward_kernel | `const scalar_t* v_mid = traj_v + idx * steps * 2 * dim + (step * 2 + 1) * dim;` |
| 125 | leapfrog_backward_kernel | `const scalar_t* x_next = traj_x + idx * (steps + 1) * dim + (step + 1) * dim;` |
| 133 | leapfrog_backward_kernel | `v_mid_norm = sqrt(v_mid_norm);` |
| 145 | leapfrog_backward_kernel | `scalar_t den = static_cast<scalar_t>(1) + h * mu_next[i];` |
| 146 | leapfrog_backward_kernel | `l_v_mid[i] = lv[i] / den;` |
| 147 | leapfrog_backward_kernel | `l_mu_next[i] = -h * lv[i] * ((v_mid[i] + h * (f_ptr[i] - gamma_mid[i])) / (den * den));` |
| 148 | leapfrog_backward_kernel | `l_gamma_mid[i] = -h * lv[i] / den;` |
| 149 | leapfrog_backward_kernel | `gf_b[i] += h * lv[i] / den;` |
| 159 | leapfrog_backward_kernel | `scalar_t norm_eps = sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>);` |
| 160 | leapfrog_backward_kernel | `v_scale_term = static_cast<scalar_t>(1) + velocity_friction_scale * v_mid_norm / norm_eps;` |
| 162 | leapfrog_backward_kernel | `v_scale_norm_factor = velocity_friction_scale / norm_eps;` |
| 167 | leapfrog_backward_kernel | `l_mu_base[i] = l_mu_next[i] * v_scale_term;` |
| 173 | leapfrog_backward_kernel | `scalar_t mu_base_i = mu_next[i] / v_scale_term;` |
| 174 | leapfrog_backward_kernel | `sum_factor += l_mu_next[i] * mu_base_i;` |
| 176 | leapfrog_backward_kernel | `sum_factor *= (v_scale_norm_factor / v_mid_norm);` |
| 179 | leapfrog_backward_kernel | `l_v_mid[j] += sum_factor * v_mid[j];` |
| 192 | leapfrog_backward_kernel | `l_v_mid[i] += gv_c[i];` |
| 193 | leapfrog_backward_kernel | `lx[i] += gx_c[i];` |
| 199 | leapfrog_backward_kernel | `l_v_mid[i] += effective_dt * lx[i];` |
| 208 | leapfrog_backward_kernel | `v_n_norm = sqrt(v_n_norm);` |
| 220 | leapfrog_backward_kernel | `scalar_t den = static_cast<scalar_t>(1) + h * mu_n[i];` |
| 221 | leapfrog_backward_kernel | `l_v_n[i] = l_v_mid[i] / den;` |
| 222 | leapfrog_backward_kernel | `l_mu_n[i] = -h * l_v_mid[i] * ((v_n[i] + h * (f_ptr[i] - gamma_n[i])) / (den * den));` |
| 223 | leapfrog_backward_kernel | `l_gamma_n[i] = -h * l_v_mid[i] / den;` |
| 224 | leapfrog_backward_kernel | `gf_b[i] += h * l_v_mid[i] / den;` |
| 234 | leapfrog_backward_kernel | `scalar_t norm_eps = sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>);` |
| 235 | leapfrog_backward_kernel | `v_scale_term = static_cast<scalar_t>(1) + velocity_friction_scale * v_n_norm / norm_eps;` |
| 237 | leapfrog_backward_kernel | `v_scale_norm_factor = velocity_friction_scale / norm_eps;` |
| 242 | leapfrog_backward_kernel | `l_mu_base[i] = l_mu_n[i] * v_scale_term;` |
| 248 | leapfrog_backward_kernel | `scalar_t mu_base_i = mu_n[i] / v_scale_term;` |
| 249 | leapfrog_backward_kernel | `sum_factor += l_mu_n[i] * mu_base_i;` |
| 251 | leapfrog_backward_kernel | `sum_factor *= (v_scale_norm_factor / v_n_norm);` |
| 254 | leapfrog_backward_kernel | `l_v_n[j] += sum_factor * v_n[j];` |
| 266 | leapfrog_backward_kernel | `lv[i] = l_v_n[i] + gv_c[i];` |
| 267 | leapfrog_backward_kernel | `lx[i] += gx_c[i];` |
| 283 | leapfrog_backward_kernel | `const scalar_t* x_step = traj_x + idx * (steps + 1) * dim + step * dim;` |
| 284 | leapfrog_backward_kernel | `const scalar_t* v_step = traj_v + idx * steps * 2 * dim + (step * 2 + 1) * dim;  // v after KICK2` |
| 285 | leapfrog_backward_kernel | `const scalar_t* h_prev = traj_h + idx * (steps + 1) * dim + step * dim;` |
| 286 | leapfrog_backward_kernel | `const scalar_t* h_curr = traj_h + idx * (steps + 1) * dim + (step + 1) * dim;` |
| 304 | leapfrog_backward_kernel | `sum[i] += sinf(x_step[j]) * hyst_update_w[i * hyst_in_dim + j];` |
| 305 | leapfrog_backward_kernel | `sum[i] += cosf(x_step[j]) * hyst_update_w[i * hyst_in_dim + (dim + j)];` |
| 306 | leapfrog_backward_kernel | `sum[i] += v_step[j] * hyst_update_w[i * hyst_in_dim + (2*dim + j)];` |
| 310 | leapfrog_backward_kernel | `sum[i] += x_step[j] * hyst_update_w[i * hyst_in_dim + j];` |
| 311 | leapfrog_backward_kernel | `sum[i] += v_step[j] * hyst_update_w[i * hyst_in_dim + (dim + j)];` |
| 316 | leapfrog_backward_kernel | `tanh_grad[i] = static_cast<scalar_t>(1) - tanh_val[i] * tanh_val[i];  // sech²(sum) = 1 - tanh²(sum)` |
| 322 | leapfrog_backward_kernel | `lsum[i] = lh[i] * tanh_grad[i];` |
| 329 | leapfrog_backward_kernel | `gHupdate_b_b[i] += lsum[i];` |
| 334 | leapfrog_backward_kernel | `gHupdate_w_b[i * hyst_in_dim + j] += lsum[i] * sin(x_step[j]);` |
| 335 | leapfrog_backward_kernel | `gHupdate_w_b[i * hyst_in_dim + (dim + j)] += lsum[i] * cos(x_step[j]);` |
| 336 | leapfrog_backward_kernel | `gHupdate_w_b[i * hyst_in_dim + (2*dim + j)] += lsum[i] * v_step[j];` |
| 340 | leapfrog_backward_kernel | `gHupdate_w_b[i * hyst_in_dim + j] += lsum[i] * x_step[j];` |
| 341 | leapfrog_backward_kernel | `gHupdate_w_b[i * hyst_in_dim + (dim + j)] += lsum[i] * v_step[j];` |
| 350 | leapfrog_backward_kernel | `gHreadout_b_b[i] += static_cast<scalar_t>(0);  // Placeholder for now` |
| 352 | leapfrog_backward_kernel | `gHreadout_w_b[i * dim + j] += static_cast<scalar_t>(0);  // Placeholder` |
| 360 | leapfrog_backward_kernel | `lh[i] = lh[i] * hyst_decay;` |
| 367 | leapfrog_backward_kernel | `grad_x_in[idx * dim + i] = lx[i];` |
| 368 | leapfrog_backward_kernel | `grad_v_in[idx * dim + i] = lv[i];` |
| 401 | leapfrog_forward_traj_kernel | `int idx = blockIdx.x * blockDim.x + threadIdx.x;` |
| 408 | leapfrog_forward_traj_kernel | `cx[i] = x_in[idx * dim + i];` |
| 409 | leapfrog_forward_traj_kernel | `cv[i] = v_in[idx * dim + i];` |
| 410 | leapfrog_forward_traj_kernel | `hyst_local[i] = hyst_enabled && hysteresis_state_in ? hysteresis_state_in[idx * dim + i] : static_cast<scalar_t>(0);` |
| 413 | leapfrog_forward_traj_kernel | `Topology topology = static_cast<Topology>(topology_id);` |
| 414 | leapfrog_forward_traj_kernel | `scalar_t effective_dt = dt * dt_scale;` |
| 415 | leapfrog_forward_traj_kernel | `scalar_t h = static_cast<scalar_t>(0.5) * effective_dt;` |
| 416 | leapfrog_forward_traj_kernel | `const scalar_t* f_ptr = force + idx * dim;` |
| 421 | leapfrog_forward_traj_kernel | `traj_h[idx * (steps + 1) * dim + i] = hyst_local[i];` |
| 428 | leapfrog_forward_traj_kernel | `traj_x[idx * (steps + 1) * dim + step * dim + i] = cx[i];` |
| 429 | leapfrog_forward_traj_kernel | `traj_v[idx * steps * 2 * dim + (step * 2 + 0) * dim + i] = cv[i];` |
| 440 | leapfrog_forward_traj_kernel | `sum += hyst_readout_w[i * dim + j] * hyst_local[j];` |
| 450 | leapfrog_forward_traj_kernel | `cv_norm = sqrt(cv_norm);` |
| 460 | leapfrog_forward_traj_kernel | `cv[i] = (cv[i] + h * (f_ptr[i] + f_ghost[i] - gamma[i])) / (static_cast<scalar_t>(1) + h * friction[i]);` |
| 461 | leapfrog_forward_traj_kernel | `traj_v[idx * steps * 2 * dim + (step * 2 + 1) * dim + i] = cv[i]; // Store v_mid` |
| 471 | leapfrog_forward_traj_kernel | `cv_norm = sqrt(cv_norm);` |
| 487 | leapfrog_forward_traj_kernel | `sum += hyst_readout_w[i * dim + j] * hyst_local[j];` |
| 494 | leapfrog_forward_traj_kernel | `cv[i] = (cv[i] + h * (f_ptr[i] + f_ghost[i] - gamma[i])) / (static_cast<scalar_t>(1) + h * friction[i]);` |
| 504 | leapfrog_forward_traj_kernel | `sum += sin(cx[j]) * hyst_update_w[i * hyst_in_dim + j];` |
| 505 | leapfrog_forward_traj_kernel | `sum += cos(cx[j]) * hyst_update_w[i * hyst_in_dim + (dim + j)];` |
| 506 | leapfrog_forward_traj_kernel | `sum += cv[j] * hyst_update_w[i * hyst_in_dim + (2*dim + j)];` |
| 510 | leapfrog_forward_traj_kernel | `sum += cx[j] * hyst_update_w[i * hyst_in_dim + j];` |
| 511 | leapfrog_forward_traj_kernel | `sum += cv[j] * hyst_update_w[i * hyst_in_dim + (dim + j)];` |
| 515 | leapfrog_forward_traj_kernel | `hyst_local[i] = hyst_local[i] * hyst_decay + tanh(sum);` |
| 522 | leapfrog_forward_traj_kernel | `traj_h[idx * (steps + 1) * dim + (step + 1) * dim + i] = hyst_local[i];` |
| 552 | leapfrog_forward_traj_kernel | `int f_dim = (topology == 1) ? 2 * dim : dim;` |
| 553 | leapfrog_forward_traj_kernel | `int hyst_in_dim = (topology == 1) ? 3 * dim : 2 * dim;` |
| 556 | leapfrog_forward_traj_kernel | `auto traj_x = torch::empty({batch_size, steps + 1, dim}, options);` |
| 557 | leapfrog_forward_traj_kernel | `auto traj_v = torch::empty({batch_size, steps, 2, dim}, options); // Stores v_n and v_mid` |
| 558 | leapfrog_forward_traj_kernel | `auto traj_h = hyst_enabled ? torch::empty({batch_size, steps + 1, dim}, options) : torch::empty({0}, options);  // AUDIT FIX` |
| 577 | leapfrog_forward_traj_kernel | `int blocks = (batch_size + threads - 1) / threads;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// leapfrog_backward_kernel (L67)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// leapfrog_backward_kernel (L70)
Topology topology = static_cast<Topology>(topology_id);
// leapfrog_backward_kernel (L71)
scalar_t effective_dt = dt * dt_scale;
// leapfrog_backward_kernel (L72)
scalar_t h = static_cast<scalar_t>(0.5) * effective_dt;
// leapfrog_backward_kernel (L78)
lx[i] = grad_x_out[idx * dim + i];
// leapfrog_backward_kernel (L79)
lv[i] = grad_v_out[idx * dim + i];
// leapfrog_backward_kernel (L83)
scalar_t* gU_b = grad_U + idx * dim * rank;
// leapfrog_backward_kernel (L84)
scalar_t* gW_b = grad_W + idx * dim * rank;
// leapfrog_backward_kernel (L85)
int f_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// leapfrog_backward_kernel (L86)
scalar_t* gWf_b = grad_W_forget + idx * dim * f_dim;
// leapfrog_backward_kernel (L87)
scalar_t* gbf_b = grad_b_forget + idx * dim;
// leapfrog_backward_kernel (L88)
scalar_t* gf_b = grad_force + idx * dim;
// leapfrog_backward_kernel (L89)
scalar_t* gWinput_b = (W_input != nullptr) ? (grad_W_input + idx * dim * dim) : nullptr;
// leapfrog_backward_kernel (L90)
scalar_t* gVw_b = (V_w != nullptr) ? (grad_V_w + idx * dim) : nullptr;
// leapfrog_backward_kernel (L93)
scalar_t* gHupdate_w_b = hyst_enabled ? (grad_hyst_update_w + idx * dim * hyst_in_dim) : nullptr;
// leapfrog_backward_kernel (L94)
scalar_t* gHupdate_b_b = hyst_enabled ? (grad_hyst_update_b + idx * dim) : nullptr;
// leapfrog_backward_kernel (L95)
scalar_t* gHreadout_w_b = hyst_enabled ? (grad_hyst_readout_w + idx * dim * dim) : nullptr;
// leapfrog_backward_kernel (L96)
scalar_t* gHreadout_b_b = hyst_enabled ? (grad_hyst_readout_b + idx * dim) : nullptr;
// leapfrog_backward_kernel (L98)
const scalar_t* f_ptr = force + idx * dim;
// leapfrog_backward_kernel (L122)
const scalar_t* x_n = traj_x + idx * (steps + 1) * dim + step * dim;
// leapfrog_backward_kernel (L123)
const scalar_t* v_n = traj_v + idx * steps * 2 * dim + (step * 2 + 0) * dim;
// leapfrog_backward_kernel (L124)
const scalar_t* v_mid = traj_v + idx * steps * 2 * dim + (step * 2 + 1) * dim;
// leapfrog_backward_kernel (L125)
const scalar_t* x_next = traj_x + idx * (steps + 1) * dim + (step + 1) * dim;
// leapfrog_backward_kernel (L133)
v_mid_norm = sqrt(v_mid_norm);
// leapfrog_backward_kernel (L145)
scalar_t den = static_cast<scalar_t>(1) + h * mu_next[i];
// leapfrog_backward_kernel (L146)
l_v_mid[i] = lv[i] / den;
// leapfrog_backward_kernel (L147)
l_mu_next[i] = -h * lv[i] * ((v_mid[i] + h * (f_ptr[i] - gamma_mid[i])) / (den * den));
// leapfrog_backward_kernel (L148)
l_gamma_mid[i] = -h * lv[i] / den;
// leapfrog_backward_kernel (L149)
gf_b[i] += h * lv[i] / den;
// leapfrog_backward_kernel (L159)
scalar_t norm_eps = sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>);
// leapfrog_backward_kernel (L160)
v_scale_term = static_cast<scalar_t>(1) + velocity_friction_scale * v_mid_norm / norm_eps;
// leapfrog_backward_kernel (L162)
v_scale_norm_factor = velocity_friction_scale / norm_eps;
// leapfrog_backward_kernel (L167)
l_mu_base[i] = l_mu_next[i] * v_scale_term;
// leapfrog_backward_kernel (L173)
scalar_t mu_base_i = mu_next[i] / v_scale_term;
// leapfrog_backward_kernel (L174)
sum_factor += l_mu_next[i] * mu_base_i;
// leapfrog_backward_kernel (L176)
sum_factor *= (v_scale_norm_factor / v_mid_norm);
// leapfrog_backward_kernel (L179)
l_v_mid[j] += sum_factor * v_mid[j];
// leapfrog_backward_kernel (L192)
l_v_mid[i] += gv_c[i];
// leapfrog_backward_kernel (L193)
lx[i] += gx_c[i];
// leapfrog_backward_kernel (L199)
l_v_mid[i] += effective_dt * lx[i];
// leapfrog_backward_kernel (L208)
v_n_norm = sqrt(v_n_norm);
// leapfrog_backward_kernel (L220)
scalar_t den = static_cast<scalar_t>(1) + h * mu_n[i];
// leapfrog_backward_kernel (L221)
l_v_n[i] = l_v_mid[i] / den;
// leapfrog_backward_kernel (L222)
l_mu_n[i] = -h * l_v_mid[i] * ((v_n[i] + h * (f_ptr[i] - gamma_n[i])) / (den * den));
// leapfrog_backward_kernel (L223)
l_gamma_n[i] = -h * l_v_mid[i] / den;
// leapfrog_backward_kernel (L224)
gf_b[i] += h * l_v_mid[i] / den;
// leapfrog_backward_kernel (L234)
scalar_t norm_eps = sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>);
// leapfrog_backward_kernel (L235)
v_scale_term = static_cast<scalar_t>(1) + velocity_friction_scale * v_n_norm / norm_eps;
// leapfrog_backward_kernel (L237)
v_scale_norm_factor = velocity_friction_scale / norm_eps;
// leapfrog_backward_kernel (L242)
l_mu_base[i] = l_mu_n[i] * v_scale_term;
// leapfrog_backward_kernel (L248)
scalar_t mu_base_i = mu_n[i] / v_scale_term;
// leapfrog_backward_kernel (L249)
sum_factor += l_mu_n[i] * mu_base_i;
// leapfrog_backward_kernel (L251)
sum_factor *= (v_scale_norm_factor / v_n_norm);
// leapfrog_backward_kernel (L254)
l_v_n[j] += sum_factor * v_n[j];
// leapfrog_backward_kernel (L266)
lv[i] = l_v_n[i] + gv_c[i];
// leapfrog_backward_kernel (L267)
lx[i] += gx_c[i];
// leapfrog_backward_kernel (L283)
const scalar_t* x_step = traj_x + idx * (steps + 1) * dim + step * dim;
// leapfrog_backward_kernel (L284)
const scalar_t* v_step = traj_v + idx * steps * 2 * dim + (step * 2 + 1) * dim;  // v after KICK2
// leapfrog_backward_kernel (L285)
const scalar_t* h_prev = traj_h + idx * (steps + 1) * dim + step * dim;
// leapfrog_backward_kernel (L286)
const scalar_t* h_curr = traj_h + idx * (steps + 1) * dim + (step + 1) * dim;
// leapfrog_backward_kernel (L304)
sum[i] += sinf(x_step[j]) * hyst_update_w[i * hyst_in_dim + j];
// leapfrog_backward_kernel (L305)
sum[i] += cosf(x_step[j]) * hyst_update_w[i * hyst_in_dim + (dim + j)];
// leapfrog_backward_kernel (L306)
sum[i] += v_step[j] * hyst_update_w[i * hyst_in_dim + (2*dim + j)];
// leapfrog_backward_kernel (L310)
sum[i] += x_step[j] * hyst_update_w[i * hyst_in_dim + j];
// leapfrog_backward_kernel (L311)
sum[i] += v_step[j] * hyst_update_w[i * hyst_in_dim + (dim + j)];
// leapfrog_backward_kernel (L316)
tanh_grad[i] = static_cast<scalar_t>(1) - tanh_val[i] * tanh_val[i];  // sech²(sum) = 1 - tanh²(sum)
// leapfrog_backward_kernel (L322)
lsum[i] = lh[i] * tanh_grad[i];
// leapfrog_backward_kernel (L329)
gHupdate_b_b[i] += lsum[i];
// leapfrog_backward_kernel (L334)
gHupdate_w_b[i * hyst_in_dim + j] += lsum[i] * sin(x_step[j]);
// leapfrog_backward_kernel (L335)
gHupdate_w_b[i * hyst_in_dim + (dim + j)] += lsum[i] * cos(x_step[j]);
// leapfrog_backward_kernel (L336)
gHupdate_w_b[i * hyst_in_dim + (2*dim + j)] += lsum[i] * v_step[j];
// leapfrog_backward_kernel (L340)
gHupdate_w_b[i * hyst_in_dim + j] += lsum[i] * x_step[j];
// leapfrog_backward_kernel (L341)
gHupdate_w_b[i * hyst_in_dim + (dim + j)] += lsum[i] * v_step[j];
// leapfrog_backward_kernel (L350)
gHreadout_b_b[i] += static_cast<scalar_t>(0);  // Placeholder for now
// leapfrog_backward_kernel (L352)
gHreadout_w_b[i * dim + j] += static_cast<scalar_t>(0);  // Placeholder
// leapfrog_backward_kernel (L360)
lh[i] = lh[i] * hyst_decay;
// leapfrog_backward_kernel (L367)
grad_x_in[idx * dim + i] = lx[i];
// leapfrog_backward_kernel (L368)
grad_v_in[idx * dim + i] = lv[i];
// leapfrog_forward_traj_kernel (L401)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// leapfrog_forward_traj_kernel (L408)
cx[i] = x_in[idx * dim + i];
// leapfrog_forward_traj_kernel (L409)
cv[i] = v_in[idx * dim + i];
// leapfrog_forward_traj_kernel (L410)
hyst_local[i] = hyst_enabled && hysteresis_state_in ? hysteresis_state_in[idx * dim + i] : static_cast<scalar_t>(0);
// leapfrog_forward_traj_kernel (L413)
Topology topology = static_cast<Topology>(topology_id);
// leapfrog_forward_traj_kernel (L414)
scalar_t effective_dt = dt * dt_scale;
// leapfrog_forward_traj_kernel (L415)
scalar_t h = static_cast<scalar_t>(0.5) * effective_dt;
// leapfrog_forward_traj_kernel (L416)
const scalar_t* f_ptr = force + idx * dim;
// leapfrog_forward_traj_kernel (L421)
traj_h[idx * (steps + 1) * dim + i] = hyst_local[i];
// leapfrog_forward_traj_kernel (L428)
traj_x[idx * (steps + 1) * dim + step * dim + i] = cx[i];
// leapfrog_forward_traj_kernel (L429)
traj_v[idx * steps * 2 * dim + (step * 2 + 0) * dim + i] = cv[i];
// leapfrog_forward_traj_kernel (L440)
sum += hyst_readout_w[i * dim + j] * hyst_local[j];
// leapfrog_forward_traj_kernel (L450)
cv_norm = sqrt(cv_norm);
// leapfrog_forward_traj_kernel (L460)
cv[i] = (cv[i] + h * (f_ptr[i] + f_ghost[i] - gamma[i])) / (static_cast<scalar_t>(1) + h * friction[i]);
// leapfrog_forward_traj_kernel (L461)
traj_v[idx * steps * 2 * dim + (step * 2 + 1) * dim + i] = cv[i]; // Store v_mid
// leapfrog_forward_traj_kernel (L471)
cv_norm = sqrt(cv_norm);
// leapfrog_forward_traj_kernel (L487)
sum += hyst_readout_w[i * dim + j] * hyst_local[j];
// leapfrog_forward_traj_kernel (L494)
cv[i] = (cv[i] + h * (f_ptr[i] + f_ghost[i] - gamma[i])) / (static_cast<scalar_t>(1) + h * friction[i]);
// leapfrog_forward_traj_kernel (L504)
sum += sin(cx[j]) * hyst_update_w[i * hyst_in_dim + j];
// leapfrog_forward_traj_kernel (L505)
sum += cos(cx[j]) * hyst_update_w[i * hyst_in_dim + (dim + j)];
// leapfrog_forward_traj_kernel (L506)
sum += cv[j] * hyst_update_w[i * hyst_in_dim + (2*dim + j)];
// leapfrog_forward_traj_kernel (L510)
sum += cx[j] * hyst_update_w[i * hyst_in_dim + j];
// leapfrog_forward_traj_kernel (L511)
sum += cv[j] * hyst_update_w[i * hyst_in_dim + (dim + j)];
// leapfrog_forward_traj_kernel (L515)
hyst_local[i] = hyst_local[i] * hyst_decay + tanh(sum);
// leapfrog_forward_traj_kernel (L522)
traj_h[idx * (steps + 1) * dim + (step + 1) * dim + i] = hyst_local[i];
// leapfrog_forward_traj_kernel (L552)
int f_dim = (topology == 1) ? 2 * dim : dim;
// leapfrog_forward_traj_kernel (L553)
int hyst_in_dim = (topology == 1) ? 3 * dim : 2 * dim;
// leapfrog_forward_traj_kernel (L556)
auto traj_x = torch::empty({batch_size, steps + 1, dim}, options);
// leapfrog_forward_traj_kernel (L557)
auto traj_v = torch::empty({batch_size, steps, 2, dim}, options); // Stores v_n and v_mid
// leapfrog_forward_traj_kernel (L558)
auto traj_h = hyst_enabled ? torch::empty({batch_size, steps + 1, dim}, options) : torch::empty({0}, options);  // AUDIT FIX
// leapfrog_forward_traj_kernel (L577)
int blocks = (batch_size + threads - 1) / threads;
```

### gfn\cuda\src\integrators\symplectic\leapfrog_fused.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 34 | christoffel_distributed | `const scalar_t* V_w = nullptr // [dim] (Optional singularity vector) ) { int tid = threadIdx.x;` |
| 41 | christoffel_distributed | `*gamma_val = 0.0f;` |
| 51 | christoffel_distributed | `scalar_t v_ph = v_shared[tid + 1]; // Safe: reads from shared mem` |
| 55 | christoffel_distributed | `scalar_t denom = R + r * c;` |
| 57 | christoffel_distributed | `scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 58 | christoffel_distributed | `scalar_t g0 = term_th * (v_ph * v_ph);` |
| 60 | christoffel_distributed | `*gamma_val = soft_clamp<scalar_t>( g0 * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );` |
| 64 | christoffel_distributed | `} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];` |
| 67 | christoffel_distributed | `scalar_t v_th = v_shared[tid - 1]; // Safe: reads from shared mem` |
| 71 | christoffel_distributed | `scalar_t denom = R + r * c;` |
| 73 | christoffel_distributed | `scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 74 | christoffel_distributed | `scalar_t g1 = 2.0f * term_ph * v_ph * v_th;` |
| 76 | christoffel_distributed | `*gamma_val = soft_clamp<scalar_t>( g1 * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );` |
| 90 | christoffel_distributed | `scalar_t u_val = U[tid * rank + k];` |
| 91 | christoffel_distributed | `scalar_t prod = u_val * v_val;` |
| 106 | christoffel_distributed | `v_sum = block_reduce_sum(v_val * v_val);` |
| 116 | christoffel_distributed | `pot_val = s * V_w[tid];` |
| 118 | christoffel_distributed | `pot_val = x_shared[tid] * V_w[tid];` |
| 132 | christoffel_distributed | `scalar_t norm = sqrt(energy);` |
| 133 | christoffel_distributed | `S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));` |
| 139 | christoffel_distributed | `scalar_t v_energy = v_sum / static_cast<scalar_t>(dim);` |
| 140 | christoffel_distributed | `M *= (1.0f + plasticity * 0.1f * tanh(v_energy));` |
| 146 | christoffel_distributed | `scalar_t soft_m = sigmoid(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));` |
| 147 | christoffel_distributed | `M *= (1.0f + (sing_strength - 1.0f) * soft_m);` |
| 159 | christoffel_distributed | `scalar_t h_sq = h_val * h_val * S_shared * M_shared;` |
| 160 | christoffel_distributed | `sum_gamma += W[tid * rank + k] * h_sq;` |
| 163 | christoffel_distributed | `*gamma_val = soft_clamp<scalar_t>(sum_gamma, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));` |
| 191 | friction_distributed | `features_shared[dim + tid] = c;` |
| 197 | friction_distributed | `int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;` |
| 203 | friction_distributed | `sum += W_forget[tid * feat_dim + j] * features_shared[j];` |
| 215 | friction_distributed | `sum += W_input[tid * dim + j] * features_shared[j];` |
| 219 | friction_distributed | `scalar_t base_friction = sigmoid(sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);` |
| 223 | friction_distributed | `scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 224 | friction_distributed | `*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);` |
| 226 | friction_distributed | `*friction_val = base_friction;` |
| 286 | leapfrog_fused_kernel | `scalar_t* h_shared = reinterpret_cast<scalar_t*>(shared_mem);` |
| 287 | leapfrog_fused_kernel | `scalar_t* features_shared = h_shared + rank;` |
| 288 | leapfrog_fused_kernel | `scalar_t* x_shared = features_shared + (2 * dim);` |
| 289 | leapfrog_fused_kernel | `scalar_t* v_shared = x_shared + dim;` |
| 292 | leapfrog_fused_kernel | `scalar_t curr_x = x_in[bid * dim + tid];` |
| 293 | leapfrog_fused_kernel | `scalar_t curr_v = v_in[bid * dim + tid];` |
| 294 | leapfrog_fused_kernel | `scalar_t f_ext = force[bid * dim + tid];` |
| 298 | leapfrog_fused_kernel | `hyst_val = hysteresis_state[bid * dim + tid];` |
| 301 | leapfrog_fused_kernel | `Topology topology = static_cast<Topology>(topology_id);` |
| 302 | leapfrog_fused_kernel | `scalar_t effective_dt = dt * dt_scale;` |
| 303 | leapfrog_fused_kernel | `scalar_t step_h = 0.5f * effective_dt;` |
| 314 | leapfrog_fused_kernel | `scalar_t* hyst_shared_buf = features_shared;` |
| 320 | leapfrog_fused_kernel | `sum += hyst_shared_buf[j] * hyst_readout_w[tid * dim + j];` |
| 330 | leapfrog_fused_kernel | `scalar_t v_sq = curr_v * curr_v;` |
| 332 | leapfrog_fused_kernel | `scalar_t v_norm = sqrt(v_sum);` |
| 351 | leapfrog_fused_kernel | `scalar_t total_force = f_ext + f_ghost;` |
| 352 | leapfrog_fused_kernel | `scalar_t num = curr_v + step_h * (total_force - gamma);` |
| 353 | leapfrog_fused_kernel | `scalar_t den = 1.0f + step_h * friction + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>);` |
| 357 | leapfrog_fused_kernel | `curr_x += effective_dt * curr_v;` |
| 358 | leapfrog_fused_kernel | `curr_x = apply_boundary_device(curr_x, topology);` |
| 367 | leapfrog_fused_kernel | `scalar_t v_sq = curr_v * curr_v;` |
| 369 | leapfrog_fused_kernel | `scalar_t v_norm = sqrt(v_sum);` |
| 389 | leapfrog_fused_kernel | `scalar_t* hyst_shared_buf2 = features_shared;` |
| 395 | leapfrog_fused_kernel | `sum2 += hyst_shared_buf2[j] * hyst_readout_w[tid * dim + j];` |
| 397 | leapfrog_fused_kernel | `total_force2 += sum2;` |
| 400 | leapfrog_fused_kernel | `num = curr_v + step_h * (total_force2 - gamma);` |
| 402 | leapfrog_fused_kernel | `den = 1.0f + step_h * friction + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>);` |
| 407 | leapfrog_fused_kernel | `scalar_t* input_shared = features_shared; // reuse` |
| 412 | leapfrog_fused_kernel | `input_shared[dim + tid] = c;` |
| 419 | leapfrog_fused_kernel | `input_shared[2*dim + tid] = curr_v;` |
| 421 | leapfrog_fused_kernel | `input_shared[dim + tid] = curr_v;` |
| 427 | leapfrog_fused_kernel | `sum += input_shared[j] * hyst_update_w[tid * hyst_in_dim + j];` |
| 430 | leapfrog_fused_kernel | `hyst_val = hyst_val * hyst_decay + tanhf(sum);` |
| 435 | leapfrog_fused_kernel | `x_out[bid * dim + tid] = curr_x;` |
| 436 | leapfrog_fused_kernel | `v_out[bid * dim + tid] = curr_v;` |
| 439 | leapfrog_fused_kernel | `hysteresis_state[bid * dim + tid] = hyst_val;` |
| 485 | leapfrog_fused_kernel | `const void* W_forget_ptr = nullptr;` |
| 486 | leapfrog_fused_kernel | `const void* b_forget_ptr = nullptr;` |
| 490 | leapfrog_fused_kernel | `const void* W_input_ptr = nullptr;` |
| 493 | leapfrog_fused_kernel | `const void* V_w_ptr = nullptr;` |
| 496 | leapfrog_fused_kernel | `void* hyst_state_ptr = nullptr;` |
| 497 | leapfrog_fused_kernel | `const void* hyst_up_w_ptr = nullptr;` |
| 498 | leapfrog_fused_kernel | `const void* hyst_up_b_ptr = nullptr;` |
| 499 | leapfrog_fused_kernel | `const void* hyst_read_w_ptr = nullptr;` |
| 500 | leapfrog_fused_kernel | `const void* hyst_read_b_ptr = nullptr;` |
| 524 | leapfrog_fused_kernel | `size_t shared_mem_size = (rank + 4 * dim) * x.element_size();` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// christoffel_distributed (L34)
const scalar_t* V_w = nullptr // [dim] (Optional singularity vector) ) { int tid = threadIdx.x;
// christoffel_distributed (L41)
*gamma_val = 0.0f;
// christoffel_distributed (L51)
scalar_t v_ph = v_shared[tid + 1]; // Safe: reads from shared mem
// christoffel_distributed (L55)
scalar_t denom = R + r * c;
// christoffel_distributed (L57)
scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// christoffel_distributed (L58)
scalar_t g0 = term_th * (v_ph * v_ph);
// christoffel_distributed (L60)
*gamma_val = soft_clamp<scalar_t>( g0 * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );
// christoffel_distributed (L64)
} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];
// christoffel_distributed (L67)
scalar_t v_th = v_shared[tid - 1]; // Safe: reads from shared mem
// christoffel_distributed (L71)
scalar_t denom = R + r * c;
// christoffel_distributed (L73)
scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// christoffel_distributed (L74)
scalar_t g1 = 2.0f * term_ph * v_ph * v_th;
// christoffel_distributed (L76)
*gamma_val = soft_clamp<scalar_t>( g1 * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );
// christoffel_distributed (L90)
scalar_t u_val = U[tid * rank + k];
// christoffel_distributed (L91)
scalar_t prod = u_val * v_val;
// christoffel_distributed (L106)
v_sum = block_reduce_sum(v_val * v_val);
// christoffel_distributed (L116)
pot_val = s * V_w[tid];
// christoffel_distributed (L118)
pot_val = x_shared[tid] * V_w[tid];
// christoffel_distributed (L132)
scalar_t norm = sqrt(energy);
// christoffel_distributed (L133)
S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// christoffel_distributed (L139)
scalar_t v_energy = v_sum / static_cast<scalar_t>(dim);
// christoffel_distributed (L140)
M *= (1.0f + plasticity * 0.1f * tanh(v_energy));
// christoffel_distributed (L146)
scalar_t soft_m = sigmoid(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));
// christoffel_distributed (L147)
M *= (1.0f + (sing_strength - 1.0f) * soft_m);
// christoffel_distributed (L159)
scalar_t h_sq = h_val * h_val * S_shared * M_shared;
// christoffel_distributed (L160)
sum_gamma += W[tid * rank + k] * h_sq;
// christoffel_distributed (L163)
*gamma_val = soft_clamp<scalar_t>(sum_gamma, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
// friction_distributed (L191)
features_shared[dim + tid] = c;
// friction_distributed (L197)
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// friction_distributed (L203)
sum += W_forget[tid * feat_dim + j] * features_shared[j];
// friction_distributed (L215)
sum += W_input[tid * dim + j] * features_shared[j];
// friction_distributed (L219)
scalar_t base_friction = sigmoid(sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// friction_distributed (L223)
scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// friction_distributed (L224)
*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
// friction_distributed (L226)
*friction_val = base_friction;
// leapfrog_fused_kernel (L286)
scalar_t* h_shared = reinterpret_cast<scalar_t*>(shared_mem);
// leapfrog_fused_kernel (L287)
scalar_t* features_shared = h_shared + rank;
// leapfrog_fused_kernel (L288)
scalar_t* x_shared = features_shared + (2 * dim);
// leapfrog_fused_kernel (L289)
scalar_t* v_shared = x_shared + dim;
// leapfrog_fused_kernel (L292)
scalar_t curr_x = x_in[bid * dim + tid];
// leapfrog_fused_kernel (L293)
scalar_t curr_v = v_in[bid * dim + tid];
// leapfrog_fused_kernel (L294)
scalar_t f_ext = force[bid * dim + tid];
// leapfrog_fused_kernel (L298)
hyst_val = hysteresis_state[bid * dim + tid];
// leapfrog_fused_kernel (L301)
Topology topology = static_cast<Topology>(topology_id);
// leapfrog_fused_kernel (L302)
scalar_t effective_dt = dt * dt_scale;
// leapfrog_fused_kernel (L303)
scalar_t step_h = 0.5f * effective_dt;
// leapfrog_fused_kernel (L314)
scalar_t* hyst_shared_buf = features_shared;
// leapfrog_fused_kernel (L320)
sum += hyst_shared_buf[j] * hyst_readout_w[tid * dim + j];
// leapfrog_fused_kernel (L330)
scalar_t v_sq = curr_v * curr_v;
// leapfrog_fused_kernel (L332)
scalar_t v_norm = sqrt(v_sum);
// leapfrog_fused_kernel (L351)
scalar_t total_force = f_ext + f_ghost;
// leapfrog_fused_kernel (L352)
scalar_t num = curr_v + step_h * (total_force - gamma);
// leapfrog_fused_kernel (L353)
scalar_t den = 1.0f + step_h * friction + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>);
// leapfrog_fused_kernel (L357)
curr_x += effective_dt * curr_v;
// leapfrog_fused_kernel (L358)
curr_x = apply_boundary_device(curr_x, topology);
// leapfrog_fused_kernel (L367)
scalar_t v_sq = curr_v * curr_v;
// leapfrog_fused_kernel (L369)
scalar_t v_norm = sqrt(v_sum);
// leapfrog_fused_kernel (L389)
scalar_t* hyst_shared_buf2 = features_shared;
// leapfrog_fused_kernel (L395)
sum2 += hyst_shared_buf2[j] * hyst_readout_w[tid * dim + j];
// leapfrog_fused_kernel (L397)
total_force2 += sum2;
// leapfrog_fused_kernel (L400)
num = curr_v + step_h * (total_force2 - gamma);
// leapfrog_fused_kernel (L402)
den = 1.0f + step_h * friction + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>);
// leapfrog_fused_kernel (L407)
scalar_t* input_shared = features_shared; // reuse
// leapfrog_fused_kernel (L412)
input_shared[dim + tid] = c;
// leapfrog_fused_kernel (L419)
input_shared[2*dim + tid] = curr_v;
// leapfrog_fused_kernel (L421)
input_shared[dim + tid] = curr_v;
// leapfrog_fused_kernel (L427)
sum += input_shared[j] * hyst_update_w[tid * hyst_in_dim + j];
// leapfrog_fused_kernel (L430)
hyst_val = hyst_val * hyst_decay + tanhf(sum);
// leapfrog_fused_kernel (L435)
x_out[bid * dim + tid] = curr_x;
// leapfrog_fused_kernel (L436)
v_out[bid * dim + tid] = curr_v;
// leapfrog_fused_kernel (L439)
hysteresis_state[bid * dim + tid] = hyst_val;
// leapfrog_fused_kernel (L485)
const void* W_forget_ptr = nullptr;
// leapfrog_fused_kernel (L486)
const void* b_forget_ptr = nullptr;
// leapfrog_fused_kernel (L490)
const void* W_input_ptr = nullptr;
// leapfrog_fused_kernel (L493)
const void* V_w_ptr = nullptr;
// leapfrog_fused_kernel (L496)
void* hyst_state_ptr = nullptr;
// leapfrog_fused_kernel (L497)
const void* hyst_up_w_ptr = nullptr;
// leapfrog_fused_kernel (L498)
const void* hyst_up_b_ptr = nullptr;
// leapfrog_fused_kernel (L499)
const void* hyst_read_w_ptr = nullptr;
// leapfrog_fused_kernel (L500)
const void* hyst_read_b_ptr = nullptr;
// leapfrog_fused_kernel (L524)
size_t shared_mem_size = (rank + 4 * dim) * x.element_size();
```

### gfn\cuda\src\integrators\toroidal\toroidal_christoffel_fused.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 3 | Global | `* ================================== * * Dedicated CUDA kernel for computing Christoffel symbols on toroidal manifolds. * This kernel implements the metric-derived connection for torus topology. * * AUDIT FIX (2026-02-06): Component 2 - Toroidal Geometry in CUDA Fused Mode * * Problem: fusion.py was passing dummy zero tensors instead of computing * actual toroidal Christoffel symbols, causing complete loss of curvature. * * Solution: Dedicated kernel that computes toroidal Christoffel from metric: *   ds² = (R + r*cos(θ))² dφ² + r² dθ² * * Christoffel symbols (non-zero components): *   Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r *   Γ^φ_θφ = Γ^φ_φθ = -r*sin(θ) / (R + r*cos(θ)) * * Author: MiniMax Agent (Audit Implementation) * Date: 2026-02-06 * References: *   - technical_analysis.md: Lines 55-72 *   - implementation_plan.md: Component 2 */ #include <ATen/ATen.h> #include <c10/util/Exception.h> #include <cuda.h> #include <cuda_runtime.h> #include "../../common/types.cuh" #include "../../common/device_utils.cuh" #include "../../common/math_utils.cuh" #include "../../geometry/christoffel_impl.cuh" namespace gfn { namespace cuda { /** * @brief Compute toroidal Christoffel symbols for a single pair (θ, φ) * * For toroidal manifold with metric: *   ds² = (R + r*cos(θ))² dφ² + r² dθ² * * Non-zero Christoffel symbols: *   Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r *   Γ^φ_θφ = Γ^φ_φθ = -r*sin(θ) / (R + r*cos(θ)) * * @param theta Position angle θ (poloidal) * @param phi Position angle φ (toroidal) - not used in computation but for API consistency * @param v_theta Velocity component v^θ * @param v_phi Velocity component v^φ * @param R Major radius of torus * @param r Minor radius of torus * @param gamma_theta Output: Christoffel force component Γ(v,v)^θ * @param gamma_phi Output: Christoffel force component Γ(v,v)^φ */ GFN_DEVICE void toroidal_christoffel_pair( float theta, float phi,  // Unused but kept for API consistency float v_theta, float v_phi, float R, float r, float* gamma_theta, float* gamma_phi ) { float sin_theta = sinf(theta);` |
| 74 | Global | `float cos_theta = cosf(theta);` |
| 77 | Global | `float denom = fmaxf(R + r * cos_theta, 1e-6f);` |
| 81 | Global | `float term_theta = denom * sin_theta / (r + 1e-6f);` |
| 82 | Global | `*gamma_theta = term_theta * (v_phi * v_phi);` |
| 86 | Global | `float term_phi = -(r * sin_theta) / (denom + 1e-6f);` |
| 87 | Global | `*gamma_phi = 2.0f * term_phi * v_theta * v_phi;` |
| 119 | Global | `float phi = x[i + 1];` |
| 121 | Global | `float v_phi = v[i + 1];` |
| 131 | Global | `gamma[i + 1] = g_phi;` |
| 136 | Global | `gamma[i] = fminf(10.0f, fmaxf(-10.0f, gamma[i]));` |
| 153 | Global | `*   3. KICK 1: v_half = (v + h*(F - Γ)) / (1 + h*μ) *   4. DRIFT: x_new = x + dt * v_half *   5. Apply toroidal boundary: x ∈ [0, 2π) *   6. Recompute μ(x_new) and Γ(v_half, v_half) *   7. KICK 2: v_new = (v_half + h*(F - Γ)) / (1 + h*μ) * * @param x Initial position [batch, dim] * @param v Initial velocity [batch, dim] * @param f Force sequence [batch, seq_len, dim] * @param R Major radius * @param r Minor radius * @param dt Time step * @param batch Batch size * @param seq_len Sequence length * @param dim Dimension * @param x_out Output positions [batch, seq_len, dim] * @param v_out Output velocities [batch, seq_len, dim] */ GFN_GLOBAL void toroidal_leapfrog_fused_kernel( const float* x, const float* v, const float* f, const float* W_forget,   // [dim, feat_dim] or nullptr for DEFAULT_FRICTION const float* b_forget,   // [dim] or nullptr float R, float r, float dt, int batch, int seq_len, int dim, float* x_out, float* v_out ) { int tid = blockIdx.x * blockDim.x + threadIdx.x;` |
| 190 | Global | `float curr_x[256];  // Max dim = 256` |
| 197 | Global | `curr_x[i] = x[tid * dim + i];` |
| 198 | Global | `curr_v[i] = v[tid * dim + i];` |
| 203 | Global | `const float* f_ptr = &f[tid * seq_len * dim + t * dim];` |
| 204 | Global | `float h = dt * 0.5f;  // Half time step` |
| 209 | Global | `v_half[i] = curr_v[i] + h * f_ptr[i];` |
| 214 | Global | `curr_x[i] += effective_dt * v_half[i];` |
| 219 | Global | `curr_x[i] = atan2f(sinf(curr_x[i]), cosf(curr_x[i]));` |
| 227 | Global | `curr_v[i] = v_half[i] + h * (f_ptr[i] - gamma[i]);` |
| 232 | Global | `x_out[tid * seq_len * dim + t * dim + i] = curr_x[i];` |
| 233 | Global | `v_out[tid * seq_len * dim + t * dim + i] = curr_v[i];` |
| 277 | Global | `int grid_size = (batch + block_size - 1) / block_size;` |
| 332 | Global | `const float* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<float>() : nullptr;` |
| 333 | Global | `const float* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<float>() : nullptr;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L3)
* ================================== * * Dedicated CUDA kernel for computing Christoffel symbols on toroidal manifolds. * This kernel implements the metric-derived connection for torus topology. * * AUDIT FIX (2026-02-06): Component 2 - Toroidal Geometry in CUDA Fused Mode * * Problem: fusion.py was passing dummy zero tensors instead of computing * actual toroidal Christoffel symbols, causing complete loss of curvature. * * Solution: Dedicated kernel that computes toroidal Christoffel from metric: *   ds² = (R + r*cos(θ))² dφ² + r² dθ² * * Christoffel symbols (non-zero components): *   Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r *   Γ^φ_θφ = Γ^φ_φθ = -r*sin(θ) / (R + r*cos(θ)) * * Author: MiniMax Agent (Audit Implementation) * Date: 2026-02-06 * References: *   - technical_analysis.md: Lines 55-72 *   - implementation_plan.md: Component 2 */ #include <ATen/ATen.h> #include <c10/util/Exception.h> #include <cuda.h> #include <cuda_runtime.h> #include "../../common/types.cuh" #include "../../common/device_utils.cuh" #include "../../common/math_utils.cuh" #include "../../geometry/christoffel_impl.cuh" namespace gfn { namespace cuda { /** * @brief Compute toroidal Christoffel symbols for a single pair (θ, φ) * * For toroidal manifold with metric: *   ds² = (R + r*cos(θ))² dφ² + r² dθ² * * Non-zero Christoffel symbols: *   Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r *   Γ^φ_θφ = Γ^φ_φθ = -r*sin(θ) / (R + r*cos(θ)) * * @param theta Position angle θ (poloidal) * @param phi Position angle φ (toroidal) - not used in computation but for API consistency * @param v_theta Velocity component v^θ * @param v_phi Velocity component v^φ * @param R Major radius of torus * @param r Minor radius of torus * @param gamma_theta Output: Christoffel force component Γ(v,v)^θ * @param gamma_phi Output: Christoffel force component Γ(v,v)^φ */ GFN_DEVICE void toroidal_christoffel_pair( float theta, float phi,  // Unused but kept for API consistency float v_theta, float v_phi, float R, float r, float* gamma_theta, float* gamma_phi ) { float sin_theta = sinf(theta);
// Global (L74)
float cos_theta = cosf(theta);
// Global (L77)
float denom = fmaxf(R + r * cos_theta, 1e-6f);
// Global (L81)
float term_theta = denom * sin_theta / (r + 1e-6f);
// Global (L82)
*gamma_theta = term_theta * (v_phi * v_phi);
// Global (L86)
float term_phi = -(r * sin_theta) / (denom + 1e-6f);
// Global (L87)
*gamma_phi = 2.0f * term_phi * v_theta * v_phi;
// Global (L119)
float phi = x[i + 1];
// Global (L121)
float v_phi = v[i + 1];
// Global (L131)
gamma[i + 1] = g_phi;
// Global (L136)
gamma[i] = fminf(10.0f, fmaxf(-10.0f, gamma[i]));
// Global (L153)
*   3. KICK 1: v_half = (v + h*(F - Γ)) / (1 + h*μ) *   4. DRIFT: x_new = x + dt * v_half *   5. Apply toroidal boundary: x ∈ [0, 2π) *   6. Recompute μ(x_new) and Γ(v_half, v_half) *   7. KICK 2: v_new = (v_half + h*(F - Γ)) / (1 + h*μ) * * @param x Initial position [batch, dim] * @param v Initial velocity [batch, dim] * @param f Force sequence [batch, seq_len, dim] * @param R Major radius * @param r Minor radius * @param dt Time step * @param batch Batch size * @param seq_len Sequence length * @param dim Dimension * @param x_out Output positions [batch, seq_len, dim] * @param v_out Output velocities [batch, seq_len, dim] */ GFN_GLOBAL void toroidal_leapfrog_fused_kernel( const float* x, const float* v, const float* f, const float* W_forget,   // [dim, feat_dim] or nullptr for DEFAULT_FRICTION const float* b_forget,   // [dim] or nullptr float R, float r, float dt, int batch, int seq_len, int dim, float* x_out, float* v_out ) { int tid = blockIdx.x * blockDim.x + threadIdx.x;
// Global (L190)
float curr_x[256];  // Max dim = 256
// Global (L197)
curr_x[i] = x[tid * dim + i];
// Global (L198)
curr_v[i] = v[tid * dim + i];
// Global (L203)
const float* f_ptr = &f[tid * seq_len * dim + t * dim];
// Global (L204)
float h = dt * 0.5f;  // Half time step
// Global (L209)
v_half[i] = curr_v[i] + h * f_ptr[i];
// Global (L214)
curr_x[i] += effective_dt * v_half[i];
// Global (L219)
curr_x[i] = atan2f(sinf(curr_x[i]), cosf(curr_x[i]));
// Global (L227)
curr_v[i] = v_half[i] + h * (f_ptr[i] - gamma[i]);
// Global (L232)
x_out[tid * seq_len * dim + t * dim + i] = curr_x[i];
// Global (L233)
v_out[tid * seq_len * dim + t * dim + i] = curr_v[i];
// Global (L277)
int grid_size = (batch + block_size - 1) / block_size;
// Global (L332)
const float* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<float>() : nullptr;
// Global (L333)
const float* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<float>() : nullptr;
```

### gfn\cuda\src\integrators\toroidal\toroidal_christoffel_fused.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 5 | Global | `* ======================================== * * Header file for toroidal-specific CUDA kernels. * Provides interface for Python bindings. */ #ifndef GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH #define GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH #include "../../common/types.cuh" #include <cuda_runtime.h> namespace gfn { namespace cuda { /** * @brief Launch toroidal leapfrog fused kernel * * Performs full sequence integration using metric-derived * Christoffel symbols for toroidal topology. * * @param x Initial positions [batch, dim] * @param v Initial velocities [batch, dim] * @param f Force sequence [batch, seq_len, dim] * @param R Major radius of torus * @param r Minor radius of torus * @param dt Time step * @param batch Batch size * @param seq_len Sequence length * @param dim Dimension (should be even for angle pairs) * @param x_out Output positions [batch, seq_len, dim] * @param v_out Output velocities [batch, seq_len, dim] * @param stream CUDA stream (optional, default=0) */ void launch_toroidal_leapfrog_fused( const float* x, const float* v, const float* f, const float* W_forget, const float* b_forget, float R, float r, float dt, int batch, int seq_len, int dim, float* x_out, float* v_out, cudaStream_t stream = 0 );` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L5)
* ======================================== * * Header file for toroidal-specific CUDA kernels. * Provides interface for Python bindings. */ #ifndef GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH #define GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH #include "../../common/types.cuh" #include <cuda_runtime.h> namespace gfn { namespace cuda { /** * @brief Launch toroidal leapfrog fused kernel * * Performs full sequence integration using metric-derived * Christoffel symbols for toroidal topology. * * @param x Initial positions [batch, dim] * @param v Initial velocities [batch, dim] * @param f Force sequence [batch, seq_len, dim] * @param R Major radius of torus * @param r Minor radius of torus * @param dt Time step * @param batch Batch size * @param seq_len Sequence length * @param dim Dimension (should be even for angle pairs) * @param x_out Output positions [batch, seq_len, dim] * @param v_out Output velocities [batch, seq_len, dim] * @param stream CUDA stream (optional, default=0) */ void launch_toroidal_leapfrog_fused( const float* x, const float* v, const float* f, const float* W_forget, const float* b_forget, float R, float r, float dt, int batch, int seq_len, int dim, float* x_out, float* v_out, cudaStream_t stream = 0 );
```

### gfn\cuda\src\integrators\unified_mlayer.cu

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 37 | Global | `int head_dim = total_dim / num_heads;` |
| 49 | Global | `size_t shared_mem_size = (rank + 6 * head_dim) * x.element_size();` |
| 57 | Global | `geo_p.topology = static_cast<Topology>(topology);` |
| 61 | Global | `geo_p.sing_thresh = static_cast<scalar_t>(sing_thresh);` |
| 62 | Global | `geo_p.sing_strength = static_cast<scalar_t>(sing_strength);` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L37)
int head_dim = total_dim / num_heads;
// Global (L49)
size_t shared_mem_size = (rank + 6 * head_dim) * x.element_size();
// Global (L57)
geo_p.topology = static_cast<Topology>(topology);
// Global (L61)
geo_p.sing_thresh = static_cast<scalar_t>(sing_thresh);
// Global (L62)
geo_p.sing_strength = static_cast<scalar_t>(sing_strength);
```

### gfn\cuda\src\integrators\universal_integrator.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 36 | universal_mlayer_kernel | `int bid = blockIdx.x; // batch index` |
| 37 | universal_mlayer_kernel | `int hid = blockIdx.y; // head index` |
| 42 | universal_mlayer_kernel | `int head_offset = hid * head_dim;` |
| 45 | universal_mlayer_kernel | `const scalar_t* x_ptr = x_init + bid * total_dim + head_offset;` |
| 46 | universal_mlayer_kernel | `const scalar_t* v_ptr = v_init + bid * total_dim + head_offset;` |
| 50 | universal_mlayer_kernel | `scalar_t* h_shared = (scalar_t*)shared_buf;` |
| 51 | universal_mlayer_kernel | `scalar_t* x_shared = h_shared + geo_p.rank;` |
| 52 | universal_mlayer_kernel | `scalar_t* v_shared = x_shared + head_dim;` |
| 53 | universal_mlayer_kernel | `scalar_t* f_shared = v_shared + head_dim;` |
| 54 | universal_mlayer_kernel | `scalar_t* feat_shared = f_shared + head_dim; // For friction [2*head_dim]` |
| 59 | universal_mlayer_kernel | `scalar_t curr_h = (phys_p.hyst_enabled && phys_p.hysteresis_settings) ? phys_p.hysteresis_settings[bid * total_dim + head_offset + tid] : 0.0f;` |
| 66 | universal_mlayer_kernel | `const scalar_t dt_eff = phys_p.dt * phys_p.dt_scales[hid];` |
| 67 | universal_mlayer_kernel | `const scalar_t local_holo_z = (geo_p.holo_z_ptr) ? geo_p.holo_z_ptr[bid * num_heads + hid] : 0.0f;` |
| 70 | universal_mlayer_kernel | `head_geo_p.U = geo_p.U + hid * head_dim * geo_p.rank;` |
| 71 | universal_mlayer_kernel | `head_geo_p.W = geo_p.W + hid * head_dim * geo_p.rank;` |
| 73 | universal_mlayer_kernel | `head_geo_p.holo_grad_z = geo_p.holo_grad_z + (bid * num_heads * head_dim + hid * head_dim);` |
| 77 | universal_mlayer_kernel | `int feat_dim = (geo_p.topology == Topology::TORUS) ? 2 * head_dim : head_dim;` |
| 81 | universal_mlayer_kernel | `head_geo_p.V_w = geo_p.V_w + hid * feat_dim;` |
| 84 | universal_mlayer_kernel | `head_phys_p.W_forget = phys_p.W_forget + hid * head_dim * feat_dim;` |
| 85 | universal_mlayer_kernel | `head_phys_p.b_forget = phys_p.b_forget + hid * head_dim;` |
| 87 | universal_mlayer_kernel | `head_phys_p.W_input = phys_p.W_input + hid * head_dim * head_dim;` |
| 92 | universal_mlayer_kernel | `int x_feat_dim = (geo_p.topology == Topology::TORUS) ? 2 * head_dim : head_dim;` |
| 93 | universal_mlayer_kernel | `int hyst_in = x_feat_dim + head_dim;` |
| 94 | universal_mlayer_kernel | `head_phys_p.hyst_up_w = phys_p.hyst_up_w + hid * head_dim * hyst_in;` |
| 95 | universal_mlayer_kernel | `head_phys_p.hyst_up_b = phys_p.hyst_up_b + hid * head_dim;` |
| 96 | universal_mlayer_kernel | `head_phys_p.hyst_rd_w = phys_p.hyst_rd_w + hid * head_dim * head_dim;` |
| 97 | universal_mlayer_kernel | `head_phys_p.hyst_rd_b = phys_p.hyst_rd_b + hid * head_dim;` |
| 102 | universal_mlayer_kernel | `scalar_t f_ext = forces[(bid * seq_len + t) * total_dim + head_offset + tid];` |
| 114 | universal_mlayer_kernel | `scalar_t v_sq = (*state.v) * (*state.v);` |
| 115 | universal_mlayer_kernel | `scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));` |
| 122 | universal_mlayer_kernel | `scalar_t h = static_cast<scalar_t>(0.5) * dt_eff;` |
| 123 | universal_mlayer_kernel | `scalar_t v_half = (curr_v + h * (f_ext + ghost_f - gamma)) / (static_cast<scalar_t>(1) + h * mu + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));` |
| 128 | universal_mlayer_kernel | `curr_x = apply_boundary_device(curr_x + dt_eff * v_half, head_geo_p.topology);` |
| 129 | universal_mlayer_kernel | `x_shared[tid] = curr_x; // Update shared for next kick` |
| 133 | universal_mlayer_kernel | `state.v = &v_half; // Use half-velocity for christoffel` |
| 135 | universal_mlayer_kernel | `v_sq = v_half * v_half;` |
| 136 | universal_mlayer_kernel | `v_norm = sqrt(block_reduce_sum_shared(v_sq));` |
| 141 | universal_mlayer_kernel | `curr_v = (v_half + h * (f_ext + ghost_f - gamma)) / (static_cast<scalar_t>(1) + h * mu + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));` |
| 145 | universal_mlayer_kernel | `else if (Method == IntegrationMethod::HEUN) { scalar_t v_sq = (*state.v) * (*state.v);` |
| 148 | universal_mlayer_kernel | `scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));` |
| 154 | universal_mlayer_kernel | `scalar_t a1 = f_ext + ghost_f - gamma - mu * curr_v;` |
| 155 | universal_mlayer_kernel | `scalar_t x_inter = apply_boundary_device(curr_x + dt_eff * curr_v, head_geo_p.topology);` |
| 156 | universal_mlayer_kernel | `scalar_t v_inter = curr_v + dt_eff * a1;` |
| 163 | universal_mlayer_kernel | `v_sq = v_inter * v_inter;` |
| 164 | universal_mlayer_kernel | `v_norm = sqrt(block_reduce_sum_shared(v_sq));` |
| 169 | universal_mlayer_kernel | `scalar_t a2 = f_ext + ghost_f - gamma - mu * v_inter;` |
| 171 | universal_mlayer_kernel | `curr_x = x_inter; // Heun typically uses x_predictor` |
| 172 | universal_mlayer_kernel | `curr_v = curr_v + 0.5f * dt_eff * (a1 + a2);` |
| 174 | universal_mlayer_kernel | `else if (Method == IntegrationMethod::EULER) { scalar_t v_sq = (*state.v) * (*state.v);` |
| 176 | universal_mlayer_kernel | `scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));` |
| 184 | universal_mlayer_kernel | `scalar_t a = f_ext + ghost_f - gamma - mu * curr_v;` |
| 186 | universal_mlayer_kernel | `curr_x = apply_boundary_device(curr_x + dt_eff * curr_v, head_geo_p.topology);` |
| 187 | universal_mlayer_kernel | `curr_v = curr_v + dt_eff * a;` |
| 192 | universal_mlayer_kernel | `x_seq[(bid * seq_len + t) * total_dim + head_offset + tid] = curr_x;` |
| 196 | universal_mlayer_kernel | `x_final[bid * total_dim + head_offset + tid] = curr_x;` |
| 197 | universal_mlayer_kernel | `v_final[bid * total_dim + head_offset + tid] = curr_v;` |
| 199 | universal_mlayer_kernel | `phys_p.hysteresis_settings[bid * total_dim + head_offset + tid] = curr_h;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// universal_mlayer_kernel (L36)
int bid = blockIdx.x; // batch index
// universal_mlayer_kernel (L37)
int hid = blockIdx.y; // head index
// universal_mlayer_kernel (L42)
int head_offset = hid * head_dim;
// universal_mlayer_kernel (L45)
const scalar_t* x_ptr = x_init + bid * total_dim + head_offset;
// universal_mlayer_kernel (L46)
const scalar_t* v_ptr = v_init + bid * total_dim + head_offset;
// universal_mlayer_kernel (L50)
scalar_t* h_shared = (scalar_t*)shared_buf;
// universal_mlayer_kernel (L51)
scalar_t* x_shared = h_shared + geo_p.rank;
// universal_mlayer_kernel (L52)
scalar_t* v_shared = x_shared + head_dim;
// universal_mlayer_kernel (L53)
scalar_t* f_shared = v_shared + head_dim;
// universal_mlayer_kernel (L54)
scalar_t* feat_shared = f_shared + head_dim; // For friction [2*head_dim]
// universal_mlayer_kernel (L59)
scalar_t curr_h = (phys_p.hyst_enabled && phys_p.hysteresis_settings) ? phys_p.hysteresis_settings[bid * total_dim + head_offset + tid] : 0.0f;
// universal_mlayer_kernel (L66)
const scalar_t dt_eff = phys_p.dt * phys_p.dt_scales[hid];
// universal_mlayer_kernel (L67)
const scalar_t local_holo_z = (geo_p.holo_z_ptr) ? geo_p.holo_z_ptr[bid * num_heads + hid] : 0.0f;
// universal_mlayer_kernel (L70)
head_geo_p.U = geo_p.U + hid * head_dim * geo_p.rank;
// universal_mlayer_kernel (L71)
head_geo_p.W = geo_p.W + hid * head_dim * geo_p.rank;
// universal_mlayer_kernel (L73)
head_geo_p.holo_grad_z = geo_p.holo_grad_z + (bid * num_heads * head_dim + hid * head_dim);
// universal_mlayer_kernel (L77)
int feat_dim = (geo_p.topology == Topology::TORUS) ? 2 * head_dim : head_dim;
// universal_mlayer_kernel (L81)
head_geo_p.V_w = geo_p.V_w + hid * feat_dim;
// universal_mlayer_kernel (L84)
head_phys_p.W_forget = phys_p.W_forget + hid * head_dim * feat_dim;
// universal_mlayer_kernel (L85)
head_phys_p.b_forget = phys_p.b_forget + hid * head_dim;
// universal_mlayer_kernel (L87)
head_phys_p.W_input = phys_p.W_input + hid * head_dim * head_dim;
// universal_mlayer_kernel (L92)
int x_feat_dim = (geo_p.topology == Topology::TORUS) ? 2 * head_dim : head_dim;
// universal_mlayer_kernel (L93)
int hyst_in = x_feat_dim + head_dim;
// universal_mlayer_kernel (L94)
head_phys_p.hyst_up_w = phys_p.hyst_up_w + hid * head_dim * hyst_in;
// universal_mlayer_kernel (L95)
head_phys_p.hyst_up_b = phys_p.hyst_up_b + hid * head_dim;
// universal_mlayer_kernel (L96)
head_phys_p.hyst_rd_w = phys_p.hyst_rd_w + hid * head_dim * head_dim;
// universal_mlayer_kernel (L97)
head_phys_p.hyst_rd_b = phys_p.hyst_rd_b + hid * head_dim;
// universal_mlayer_kernel (L102)
scalar_t f_ext = forces[(bid * seq_len + t) * total_dim + head_offset + tid];
// universal_mlayer_kernel (L114)
scalar_t v_sq = (*state.v) * (*state.v);
// universal_mlayer_kernel (L115)
scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));
// universal_mlayer_kernel (L122)
scalar_t h = static_cast<scalar_t>(0.5) * dt_eff;
// universal_mlayer_kernel (L123)
scalar_t v_half = (curr_v + h * (f_ext + ghost_f - gamma)) / (static_cast<scalar_t>(1) + h * mu + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// universal_mlayer_kernel (L128)
curr_x = apply_boundary_device(curr_x + dt_eff * v_half, head_geo_p.topology);
// universal_mlayer_kernel (L129)
x_shared[tid] = curr_x; // Update shared for next kick
// universal_mlayer_kernel (L133)
state.v = &v_half; // Use half-velocity for christoffel
// universal_mlayer_kernel (L135)
v_sq = v_half * v_half;
// universal_mlayer_kernel (L136)
v_norm = sqrt(block_reduce_sum_shared(v_sq));
// universal_mlayer_kernel (L141)
curr_v = (v_half + h * (f_ext + ghost_f - gamma)) / (static_cast<scalar_t>(1) + h * mu + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// universal_mlayer_kernel (L145)
else if (Method == IntegrationMethod::HEUN) { scalar_t v_sq = (*state.v) * (*state.v);
// universal_mlayer_kernel (L148)
scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));
// universal_mlayer_kernel (L154)
scalar_t a1 = f_ext + ghost_f - gamma - mu * curr_v;
// universal_mlayer_kernel (L155)
scalar_t x_inter = apply_boundary_device(curr_x + dt_eff * curr_v, head_geo_p.topology);
// universal_mlayer_kernel (L156)
scalar_t v_inter = curr_v + dt_eff * a1;
// universal_mlayer_kernel (L163)
v_sq = v_inter * v_inter;
// universal_mlayer_kernel (L164)
v_norm = sqrt(block_reduce_sum_shared(v_sq));
// universal_mlayer_kernel (L169)
scalar_t a2 = f_ext + ghost_f - gamma - mu * v_inter;
// universal_mlayer_kernel (L171)
curr_x = x_inter; // Heun typically uses x_predictor
// universal_mlayer_kernel (L172)
curr_v = curr_v + 0.5f * dt_eff * (a1 + a2);
// universal_mlayer_kernel (L174)
else if (Method == IntegrationMethod::EULER) { scalar_t v_sq = (*state.v) * (*state.v);
// universal_mlayer_kernel (L176)
scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));
// universal_mlayer_kernel (L184)
scalar_t a = f_ext + ghost_f - gamma - mu * curr_v;
// universal_mlayer_kernel (L186)
curr_x = apply_boundary_device(curr_x + dt_eff * curr_v, head_geo_p.topology);
// universal_mlayer_kernel (L187)
curr_v = curr_v + dt_eff * a;
// universal_mlayer_kernel (L192)
x_seq[(bid * seq_len + t) * total_dim + head_offset + tid] = curr_x;
// universal_mlayer_kernel (L196)
x_final[bid * total_dim + head_offset + tid] = curr_x;
// universal_mlayer_kernel (L197)
v_final[bid * total_dim + head_offset + tid] = curr_v;
// universal_mlayer_kernel (L199)
phys_p.hysteresis_settings[bid * total_dim + head_offset + tid] = curr_h;
```

### gfn\cuda\src\physics\physics_library.cuh

| Línea | Contexto | Fórmula |
| :--- | :--- | :--- |
| 13 | Global | `* μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE */ template <typename scalar_t> GFN_DEVICE void compute_friction_distributed( const PhysicsParams<scalar_t>& p, const MLayerState<scalar_t>& s, int dim, Topology topology, scalar_t v_norm, scalar_t* friction_val, scalar_t* features_shared // Buffer for [2*dim] ) { int tid = threadIdx.x;` |
| 32 | Global | `features_shared[tid] = sin_th;` |
| 33 | Global | `features_shared[dim + tid] = cos_th;` |
| 39 | Global | `int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;` |
| 44 | Global | `gate_sum += p.W_forget[tid * feat_dim + j] * features_shared[j];` |
| 53 | Global | `gate_sum += p.W_input[tid * dim + j] * features_shared[j];` |
| 57 | Global | `scalar_t base_mu = sigmoid<scalar_t>(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);` |
| 61 | Global | `scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));` |
| 62 | Global | `*friction_val = base_mu * (static_cast<scalar_t>(1) + p.v_fric_scale * v_scale);` |
| 64 | Global | `*friction_val = base_mu;` |
| 81 | Global | `*ghost_force_val = static_cast<scalar_t>(0);` |
| 91 | Global | `features_shared[tid] = sin_th;` |
| 92 | Global | `features_shared[dim + tid] = cos_th;` |
| 98 | Global | `int x_feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;` |
| 99 | Global | `features_shared[x_feat_dim + tid] = s.v[tid];` |
| 104 | Global | `int total_in = x_feat_dim + dim;` |
| 106 | Global | `up_sum += p.hyst_up_w[tid * total_in + j] * features_shared[j];` |
| 110 | Global | `*s.h = (*s.h) * (static_cast<scalar_t>(1) - p.hyst_decay) + tanh(up_sum) * p.hyst_decay;` |
| 114 | Global | `features_shared[tid] = *s.h;` |
| 119 | Global | `rd_sum += p.hyst_rd_w[tid * dim + j] * features_shared[j];` |
| 121 | Global | `*ghost_force_val = rd_sum;` |

#### Fórmulas Listas para Usar (CUDA)
```cpp
// Global (L13)
* μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE */ template <typename scalar_t> GFN_DEVICE void compute_friction_distributed( const PhysicsParams<scalar_t>& p, const MLayerState<scalar_t>& s, int dim, Topology topology, scalar_t v_norm, scalar_t* friction_val, scalar_t* features_shared // Buffer for [2*dim] ) { int tid = threadIdx.x;
// Global (L32)
features_shared[tid] = sin_th;
// Global (L33)
features_shared[dim + tid] = cos_th;
// Global (L39)
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// Global (L44)
gate_sum += p.W_forget[tid * feat_dim + j] * features_shared[j];
// Global (L53)
gate_sum += p.W_input[tid * dim + j] * features_shared[j];
// Global (L57)
scalar_t base_mu = sigmoid<scalar_t>(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// Global (L61)
scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// Global (L62)
*friction_val = base_mu * (static_cast<scalar_t>(1) + p.v_fric_scale * v_scale);
// Global (L64)
*friction_val = base_mu;
// Global (L81)
*ghost_force_val = static_cast<scalar_t>(0);
// Global (L91)
features_shared[tid] = sin_th;
// Global (L92)
features_shared[dim + tid] = cos_th;
// Global (L98)
int x_feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// Global (L99)
features_shared[x_feat_dim + tid] = s.v[tid];
// Global (L104)
int total_in = x_feat_dim + dim;
// Global (L106)
up_sum += p.hyst_up_w[tid * total_in + j] * features_shared[j];
// Global (L110)
*s.h = (*s.h) * (static_cast<scalar_t>(1) - p.hyst_decay) + tanh(up_sum) * p.hyst_decay;
// Global (L114)
features_shared[tid] = *s.h;
// Global (L119)
rd_sum += p.hyst_rd_w[tid * dim + j] * features_shared[j];
// Global (L121)
*ghost_force_val = rd_sum;
```

## Repositorio Global de Fórmulas (Listas para Usar)

### Colección Completa Python (por archivo)
#### analyze_cuda_system.py
```python
# Contexto: analyze_cuda_system
# L99
v = torch.ones(B, D, device=device) * 2.0
# L103
U_stack = torch.ones(num_layers * H * D, 2, device=device) * 0.001
# L104
W_stack = torch.ones(num_layers * H * 2, D, device=device) * 0.001
# L141
pyd_files = glob.glob(os.path.join(build_dir, '**', '*.pyd'), recursive=True)
# L142
so_files = glob.glob(os.path.join(build_dir, '**', '*.so'), recursive=True)
```

#### debug_autograd.py
```python
# Contexto: debug_autograd
# L30
v = torch.ones(B, D, device=device, requires_grad=True) * 5.0
# L35
U_stack = torch.ones(num_layers * H * D, rank, device=device) * 0.001
# L36
W_stack = torch.ones(num_layers * H * rank, D, device=device) * 0.001
```

#### debug_indexing.py
```python
# Contexto: debug_indexing_issue
# L29
v = torch.ones(B, D, device=device, requires_grad=True) * 5.0
# L34
U_stack = torch.ones(num_layers * H * D, rank, device=device) * 0.001
# L35
W_stack = torch.ones(num_layers * H * rank, D, device=device) * 0.001
# L47
calculated_num_layers = U_stack.shape[0] // H
# L62
head_dim = D // H
# L63
U_reshaped = U_stack.view(num_layers, H, head_dim, -1)
# L64
W_reshaped = W_stack.view(num_layers, H, head_dim, -1).permute(0, 1, 3, 2)
# L100
dt_scales_large = torch.ones(num_layers + 5, device=device)
```

#### demos\copy_task\train_copy_task.py
```python
# Contexto: __init__
# L38
self.EOS = vocab_size + 1  # End of sequence
# Contexto: _generate_samples
# L51
sequence = [random.randint(0, self.vocab_size - 1) for _ in range(seq_len)]
# L54
full_seq = sequence + [self.SEP] + sequence + [self.EOS]
# L57
input_seq = full_seq[:-1]
# Contexto: collate_fn
# L78
srcs, tgts = zip(*batch)
# L88
pad_len = max_len - len(src)
# Contexto: evaluate_accuracy
# L106
preds = torch.argmax(logits, dim=-1)
# L111
eos_idx = (target == vocab_size + 1).nonzero(as_tuple=True)[0]
# Contexto: train_copy_task
# L124
parser.add_argument('--config', type=str, default='configs/demos/copy_task.yaml')
# L179
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )
# L198
loss = criterion(logits.view(-1, config['model']['vocab_size']), tgt.view(-1))
# L204
total_loss += loss.item()
# L207
cur_loss = total_loss / (i + 1)
```

#### demos\fractal_recall_demo.py
```python
# Contexto: __getitem__
# L181
pad_len = self.seq_len - len(seq) - 1 # -1 for target
# L184
seq = seq[:self.seq_len-1]
# Contexto: __init__
# L138
self.mod = vocab_size - 10 # Start of special tokens
# L139
self.START = self.mod + 0
# L140
self.END = self.mod + 1
# L141
self.OPEN_BRACKET = self.mod + 2
# L142
self.CLOSE_BRACKET = self.mod + 3
# L143
self.KEY_VAL_SEP = self.mod + 4
# L144
self.QUERY = self.mod + 5
# L145
self.NOISE = self.mod + 6 # "Singularity" token
# Contexto: train_showcase
# L30
optimizer = RiemannianAdam(model.parameters(), lr=3e-4, weight_decay=0.01)
# L87
mask = 2**torch.arange(coord_dim).to(device)
# L88
bits = (lm_targets.unsqueeze(-1) & mask) > 0
# L89
target_coords = bits.float() * 2 - 1 # [B, L-1, 32]
# L91
pred_coords_shifted = pred_coords[:, :-1, :] # [B, L-1, 32]
# L106
total_loss += l_g
# L111
total_loss += l_c
# L125
if christoffels: desc += f" | Curv: {l_g.item():.4f}"
# L232
optimizer = RiemannianAdam(model.parameters(), lr=3e-4, weight_decay=0.01)
# L269
lm_targets = inputs[:, 1:] # [B, T-1]
# L273
mask = 2**torch.arange(coord_dim).to(device)
# L274
bits = (lm_targets.unsqueeze(-1) & mask) > 0
# L275
target_coords = bits.float() * 2 - 1 # [B, T-1, 32]
# L278
pred_coords_shifted = pred_coords[:, :-1, :] # [B, T-1, 32]
# L296
v_start = model.v0.norm().item()
# L297
v_end = layer0.last_v.norm(dim=-1).mean().item() if hasattr(layer0, 'last_v') else 0.0
# L298
drift = abs(v_end - v_start)
# L317
res_dir = PROJECT_ROOT / "tests/benchmarks/results/showcase"
# L342
ckpt_path = PROJECT_ROOT / "checkpoints/showcase_v1.0.pt"
```

#### demos\multimodal\multimodal_mnist.py
```python
# Contexto: __init__
# L68
layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
# Contexto: benchmark_multimodal_scaling
# L107
num_patches = (res // patch_size)**2
# Contexto: forward
# L75
cls = self.cls_token.expand(B, -1, -1)
# Contexto: log_oom
# L84
match_gib = re.search(r"Tried to allocate ([\d\.]+) GiB", err_msg)
# L86
match_mib = re.search(r"Tried to allocate ([\d\.]+) MiB", err_msg)
# Contexto: train_multimodal_omni
# L167
dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# L170
params = list(manifold.parameters()) + list(v_bridge.parameters()) + list(t_bridge.parameters())
# L171
opt = RiemannianAdam(params, lr=3e-4)
# L188
prompts = torch.zeros(B, 1, dtype=torch.long, device=device) + 42 # "What is this?"
# L199
predictions = logits[:, -1]
# L205
acc = (predictions.argmax(1) == labels).float().mean().item()
# L212
total_loss = loss_ce + loss_ham + loss_geo
# L225
out_dir = PROJECT_ROOT / "tests/benchmarks/results/multimodal"
# L237
px_counts = [(r//4)**2 for r in res]
# L238
ax2.plot(px_counts, mems["manifold"], 'o-', color='#3498DB', label="MANIFOLD (O(1))", linewidth=4, markersize=10)
# L239
ax2.plot(px_counts, mems["transformer"], '^-', color='#E74C3C', label="ViT (O(N^2))", linewidth=2, linestyle='--')
# L249
ax2.annotate('4K Inference on 200MB', xy=(px_counts[-1], mems["manifold"][-1]), xytext=(px_counts[-2], mems["manifold"][-1]*3), arrowprops=dict(facecolor='white', shrink=0.05))
# L254
plt.savefig(out_dir / "omni_scaling_final.png", dpi=200)
# Contexto: vit_call
# L115
x = torch.randn(1, num_patches, patch_size**2).to(device)
```

#### demos\sorting\train_hyper_sorting.py
```python
# Contexto: __init__
# L21
self.EOS = vocab_size + 1
# L22
self.full_vocab = vocab_size + 2
# Contexto: generate_batch
# L34
src = full_seq[:, :-1]
# Contexto: run
# L88
total_vocab = vocab_range + 2
# L111
train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)
# Contexto: train_convergence
# L39
def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
# L40
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# L45
pbar = tqdm(range(max_steps), desc=f"Hyper-Sorting Training")
# L53
logits = logits.reshape(-1, task.full_vocab)
# L54
y = y.reshape(-1)
# L63
normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
# L67
preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1) # Assuming standard batch
# L70
start_idx = task.length + 1
# L72
true_sort = y.view(-1, 21)[:, start_idx:]
# L74
correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
```

#### demos\sorting\train_inf_sorting.py
```python
# Contexto: __getitem__
# L34
vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
# L39
full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]
# L41
src = torch.tensor(full_seq[:-1], dtype=torch.long)
# Contexto: __init__
# L27
self.EOS = vocab_size + 1 # Token for end
# Contexto: get_binary_coords
# L50
mask = 2**torch.arange(coord_dim).to(device)
# L51
bits = (token_ids.unsqueeze(-1) & mask) > 0
# Contexto: main
# L56
parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')
# L66
real_vocab_size = vocab_range + 2
# L78
coord_dim = config['physics']['embedding']['coord_dim'] # 16 bits -> 65k vocab
# L105
optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
# L107
scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=config['training']['lr'] * 10, # Aggressive peak steps_per_epoch=len(train_loader), epochs=config['training']['epochs'], pct_start=0.3 )
# L123
pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
# L141
total_loss += loss.item()
# L168
pred_sorted_logits = pred[:, start_check : start_check + seq_len]
# L169
tgt_sorted_bits = tgt_bits[:, start_check : start_check + seq_len]
# L174
correct_bits += (pred_bits == tgt_sorted_bits).sum().item()
# L175
total_bits += tgt_sorted_bits.numel()
# L179
token_matches = torch.all(pred_bits == tgt_sorted_bits, dim=-1)
# L180
correct_tokens += token_matches.sum().item()
# L181
total_tokens += token_matches.numel()
# L185
seq_matches = torch.all(token_matches, dim=-1)
# L186
correct_seqs += seq_matches.sum().item()
# L187
total_seqs += src.size(0)
# L189
bit_acc = correct_bits / total_bits
# L190
token_acc = correct_tokens / total_tokens
# L191
seq_acc = correct_seqs / total_seqs
```

#### demos\sorting\train_mom_sorting.py
```python
# Contexto: __init__
# L21
self.EOS = vocab_size + 1
# L22
self.full_vocab = vocab_size + 2
# Contexto: generate_batch
# L34
src = full_seq[:, :-1]
# Contexto: run
# L87
total_vocab = vocab_range + 2
# L119
train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)
# Contexto: train_convergence
# L39
def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
# L40
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# L45
pbar = tqdm(range(max_steps), desc=f"MoM-Sorting Training")
# L53
logits = logits.reshape(-1, task.full_vocab)
# L54
y = y.reshape(-1)
# L63
normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
# L67
preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1) # Assuming standard batch
# L69
start_idx = task.length + 1
# L71
true_sort = y.view(-1, 21)[:, start_idx:]
# L73
correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
```

#### demos\sorting\train_sorting.py
```python
# Contexto: __getitem__
# L97
vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
# L103
full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]
# L105
src = torch.tensor(full_seq[:-1], dtype=torch.long)
# Contexto: __init__
# L88
self.EOS = vocab_size + 1 # Token for end
# Contexto: _generate_samples
# L39
vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
# Contexto: train_sorting
# L112
parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')
# L123
real_vocab_size = vocab_range + 2 # + SEP, EOS
# L148
optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
# L163
loss = criterion(logits.view(-1, real_vocab_size), tgt.view(-1))
# L168
total_loss += loss.item()
# L181
preds = torch.argmax(logits, dim=-1)
# L190
sorted_preds = preds[:, seq_len + 1 : seq_len + 1 + seq_len]
# L191
sorted_tgts = tgt[:, seq_len + 1 : seq_len + 1 + seq_len]
# L195
correct += row_matches.sum().item()
# L196
total += src.size(0)
# L198
val_acc = correct / total
```

#### demos\sorting\train_sorting_v2.py
```python
# Contexto: __init__
# L21
self.EOS = vocab_size + 1
# L22
self.full_vocab = vocab_size + 2
# Contexto: generate_batch
# L51
src = full_seq[:, :-1]
# Contexto: run_sorting_v2
# L114
total_vocab = vocab_range + 2
# L131
train_until_convergence(model, task, max_steps=5000, lr=3e-3, device=device)
# Contexto: train_until_convergence
# L56
def train_until_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
# L58
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# L72
loss = criterion(logits.reshape(-1, task.full_vocab), y.reshape(-1))
# L81
normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
# L86
preds = torch.argmax(logits, dim=-1)
# L88
start_idx = task.length + 1
# L93
correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
```

#### demos\sorting\train_transformer_sorting.py
```python
# Contexto: __init__
# L21
self.EOS = vocab_size + 1
# L22
self.full_vocab = vocab_size + 2
# Contexto: generate_batch
# L34
src = full_seq[:, :-1]
# Contexto: run
# L88
total_vocab = vocab_range + 2
# L103
train_convergence(model, task, max_steps=1000, lr=3e-3, device=device)
# Contexto: train_convergence
# L39
def train_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
# L41
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
# L54
logits = logits.reshape(-1, task.full_vocab)
# L55
y = y.reshape(-1)
# L64
normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
# L68
preds = torch.argmax(logits.view(-1, 21, task.full_vocab), dim=-1)
# L70
start_idx = task.length + 1
# L72
true_sort = y.view(-1, 21)[:, start_idx:]
# L74
correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
```

#### demos\tinystories\train_tinystories.py
```python
# Contexto: __getitem__
# L94
start = idx * self.seq_len
# L95
chunk = self.data[start : start + self.seq_len + 1]
# L97
src = chunk[:-1]
# L102
pad_len = self.seq_len - len(src)
# Contexto: __init__
# L88
self.num_samples = len(data) // seq_len
# Contexto: read_tokens
# L61
dataset = load_dataset('roneneldan/TinyStories', split=self.split, trust_remote_code=True)
# L72
chunk = dataset[i:min(i+chunk_size, total_stories)]
# L74
chunk_tokens = re.findall(r'\S+', chunk_text.lower())
# Contexto: train_tinystories
# L110
parser.add_argument('--config', type=str, default='configs/demos/tinystories.yaml')
# L136
train_data = torch.clamp(train_data, 0, len(vocab) - 1)
# L137
val_data = torch.clamp(val_data, 0, len(vocab) - 1)
# L166
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )
# L185
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# L191
total_loss += loss.item()
# L194
cur_loss = total_loss / (i + 1)
# L204
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# L205
val_loss += loss.item()
# L207
val_loss /= len(val_loader)
# L208
ppl = torch.exp(torch.tensor(val_loss))
```

#### demos\wikitext103\train_wikitext103.py
```python
# Contexto: __getitem__
# L83
start = idx * self.seq_len
# L85
chunk = self.data[start : start + self.seq_len + 1]
# L88
src = chunk[:-1]
# L93
pad_len = self.seq_len - len(src)
# Contexto: __init__
# L77
self.num_samples = len(data) // seq_len
# Contexto: read_tokens
# L60
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=self.split, trust_remote_code=True)
# L68
tokens = re.findall(r'\S+', all_text.lower())
# Contexto: train_wikitext103
# L101
parser.add_argument('--config', type=str, default='configs/demos/wikitext103.yaml')
# L127
train_data = torch.clamp(train_data, 0, len(vocab) - 1)
# L128
val_data = torch.clamp(val_data, 0, len(vocab) - 1)
# L157
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )
# L176
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# L182
total_loss += loss.item()
# L185
cur_loss = total_loss / (i + 1)
# L195
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# L196
val_loss += loss.item()
# L198
val_loss /= len(val_loader)
# L199
ppl = torch.exp(torch.tensor(val_loss))
```

#### demos\wikitext\generate.py
```python
# Contexto: generate_text
# L53
tokens = re.findall(r'\S+', prompt.lower())
# L65
next_token_logits = logits[0, -1, :] / temperature
# L69
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
# L73
sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
# L77
next_token_logits[indices_to_remove] = float('-inf')
# L80
probs = F.softmax(next_token_logits, dim=-1)
# L87
if next_token.item() == vocab.stoi.get('<eos>', -1):
# Contexto: main
# L100
config_path = 'configs/demos/wikitext.yaml'
# L123
checkpoint_path = f"{config['training']['save_dir']}/checkpoint_epoch_93.pt"
```

#### demos\wikitext\train_wikitext.py
```python
# Contexto: __getitem__
# L92
start = idx * self.seq_len
# L94
chunk = self.data[start : start + self.seq_len + 1]
# L97
src = chunk[:-1]
# L102
pad_len = self.seq_len - len(src)
# Contexto: __init__
# L50
self.dataset_path = self.root / 'wikitext-2'
# L86
self.num_samples = len(data) // seq_len
# Contexto: download
# L56
train_file = self.dataset_path / 'train.csv'
# L60
archive_path = self.root / 'wikitext-2.tgz'
# Contexto: read_tokens
# L70
file_path = self.dataset_path / file_map[self.split]
# L73
with open(file_path, 'r', encoding='utf-8') as f:
# L79
tokens = re.findall(r'\S+', text.lower())
# Contexto: train_wikitext
# L110
parser.add_argument('--config', type=str, default='configs/demos/wikitext.yaml')
# L136
train_data = torch.clamp(train_data, 0, len(vocab) - 1)
# L137
val_data = torch.clamp(val_data, 0, len(vocab) - 1)
# L166
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=config['training']['epochs'], eta_min=1e-6 )
# L185
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# L191
total_loss += loss.item()
# L194
cur_loss = total_loss / (i + 1)
# L204
loss = criterion(logits.view(-1, len(vocab)), tgt.view(-1))
# L205
val_loss += loss.item()
# L207
val_loss /= len(val_loader)
# L208
ppl = torch.exp(torch.tensor(val_loss))
# Contexto: WikiText2Custom
# L46
URL = 'https://s3.amazonaws.com/fast-ai-nlp/wikitext-2.tgz'
```

#### examples\load_checkpoint.py
```python
# Contexto: load_manifold_checkpoint
# L52
checkpoint_path = PROJECT_ROOT / "checkpoints" / "manifold_parity_superiority.pt"
# L83
print(f"   Accuracy: {(preds == expected).float().mean().item() * 100:.1f}%")
```

#### gfn\__init__.py
```python
# Contexto: Global
# L20
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
```

#### gfn\aggregation\geodesic_attention.py
```python
# Contexto: euclidean_distance
# L68
dist_sq = ((x1_exp - x2_exp) ** 2).sum(dim=-1)  # [B, L, L]
# L69
dist = torch.sqrt(dist_sq + 1e-8)  # Add epsilon for numerical stability
# Contexto: forward
# L122
attn_weights = F.softmax(-dist / self.temperature, dim=-1)  # [B, L, L]
# L128
x_attended = torch.bmm(attn_weights, V)  # [B, L, dim]
# L130
x_agg = x_attended[:, -1]  # [B, dim]
# L133
x_attended = torch.bmm(attn_weights, x_seq)  # [B, L, dim]
# L134
x_agg = x_attended[:, -1]  # [B, dim]
# L138
v_attended = torch.bmm(attn_weights, v_seq)  # [B, L, dim]
# L139
v_agg = v_attended[:, -1]  # [B, dim]
# Contexto: riemannian_distance
# L91
similarity = torch.bmm(Q, K.transpose(1, 2)) / (self.dim ** 0.5)  # [B, L, L]
# L94
dist = -similarity
```

#### gfn\aggregation\hamiltonian_pooling.py
```python
# Contexto: forward
# L104
H = K + U  # [B, L] - Total Hamiltonian
# L107
weights = F.softmax(H / self.temperature, dim=-1)  # [B, L]
# L110
x_agg = (weights.unsqueeze(-1) * x_seq).sum(dim=1)  # [B, dim]
# L111
v_agg = (weights.unsqueeze(-1) * v_seq).sum(dim=1)  # [B, dim]
# Contexto: HamiltonianPooling
# L25
H = K + U where:
# L26
- K = (1/2) v^T g v  (kinetic energy)
# L27
- U = (1/2) ||x||^2  (potential energy)
# Contexto: kinetic_energy
# L56
Compute kinetic energy: K = 0.5 * v^T @ g @ v
# L66
metric_expanded = self.metric.view(1, 1, -1).expand(B, L, -1)  # [B, L, dim]
# L67
weighted_v = v * metric_expanded  # [B, L, dim]
# L68
K = 0.5 * (v * weighted_v).sum(dim=-1)  # [B, L]
# Contexto: potential_energy
# L73
Compute potential energy: U = 0.5 * ||x||^2
# L83
U = 0.5 * (x ** 2).sum(dim=-1)  # [B, L]
```

#### gfn\aggregation\momentum_accumulation.py
```python
# Contexto: __init__
# L58
self.gate = nn.Sequential( nn.Linear(dim * 2, dim),  # Input: [x_final, v_final] concatenated nn.Tanh(), nn.Linear(dim, 1), nn.Sigmoid()  # Output: scalar gate in [0, 1] )
# Contexto: forward
# L86
accumulated_states = x_seq.sum(dim=1)  # [B, dim]
# L88
accumulated_states = x_seq.mean(dim=1)  # [B, dim]
# L91
x_last = x_seq[:, -1]  # [B, dim]
# L97
gate_input = torch.cat([x_last, accumulated_states], dim=-1)  # [B, 2*dim]
# L99
effective_alpha = self.alpha * gate_value
# L105
x_final = x_last + effective_alpha * accumulated_states
# Contexto: Global
# L11
- Total impulse = ∫ force dt determines final momentum
# Contexto: MomentumAccumulation
# L25
x_final = x_last + alpha * accumulated_states
```

#### gfn\constants.py
```python
# Contexto: get_stable_lr_scale
# L338
warmup_steps = int(total_steps * warmup_ratio)
# L345
decay_steps = total_steps - warmup_steps
# L346
decay_progress = float(step - warmup_steps) / max(1, decay_steps)
# Contexto: Global
# L15
- EPSILON_STANDARD = 1e-8 (division safety)
# L16
- EPSILON_STRONG   = 1e-8 (strong division protection)
# L17
- EPSILON_SMOOTH   = 1e-8 (gradient smoothing)
# L18
- CLAMP_MIN_STRONG = 1e-8 (minimum denominators)
# L22
- ADAM_EPSILON = 1e-7 (optimizer-specific, documented choice)
# L76
FRICTION_SCALE = 0.02  # Was 5.0, then 0.5, then 0.05 - Now optimal
# L95
EPSILON_STRONG = 1e-7  # Was 1e-8 - Better balance
# L98
EPSILON_STANDARD = 1e-7  # Was 1e-8 - Match strong for consistency
# L101
EPSILON_SMOOTH = 1e-7
# L104
CLAMP_MIN_STRONG = 1e-7
# L107
CLAMP_MIN_STANDARD = 1e-7
# L116
LAMBDA_H_DEFAULT = 0.0  # Was 0.001 - Disabled for clean convergence
# L119
LAMBDA_G_DEFAULT = 0.00005  # Was 0.0001 - Lower for better curvature preservation
# L125
LAMBDA_K_DEFAULT = 0.0001  # Was 0.001 - Reduced for stability
# L136
DEFAULT_LR = 1e-4  # Was 1e-3
# L142
ADAM_BETA2 = 0.99  # Was 0.999 - increased for stability
# L145
ADAM_EPSILON = 1e-7  # Was 1e-8
# L168
GATE_BIAS_OPEN = 1.0  # sigmoid(1.0) ≈ 0.73 - Was 2.0
# L171
GATE_BIAS_CLOSED = -3.0  # sigmoid(-3.0) ≈ 0.05 - Was -5.0
# L180
DEFAULT_DT = 0.05  # Was 0.02 - Better exploration while maintaining stability
# L187
LEAPFROG_SUBSTEPS = 3  # Was 5 - Cleaner backward pass
# L198
DEFAULT_PLASTICITY = 0.02  # Was 0.01 - Better responsiveness
# L201
SINGULARITY_THRESHOLD = 0.5  # Was 0.8 - Lower threshold for earlier activation
# L204
BLACK_HOLE_STRENGTH = 1.5  # Was 2.0 - Reduced for stability
# L218
SINGULARITY_GATE_SLOPE = 0.5  # Was 1.0 - Smoother transitions
# L244
VELOCITY_SATURATION = 100.0  # Was 50.0 - Allow higher velocities
# L263
HYSTERESIS_FORGET_GATE_INIT = 0.9  # sigmoid(2.0) ≈ 0.88 - gradual decay
# L282
TOROIDAL_PERIOD = 6.283185307179586  # 2 * π
```

#### gfn\core\adjoint.py
```python
# Contexto: __init__
# L134
raise NotImplementedError("AdjointManifold currently only supports heads=1. " "For Multi-Head Geodesic Flows, use standard Manifold (use_adjoint=False).")
# L143
self.layers = nn.ModuleList([ AdjointMLayer(dim, rank=rank, integration_time=integration_time/depth, n_steps=5) for _ in range(depth) ])
# L156
self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)
# L157
self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)
# Contexto: forward
# L45
self.dim = state.shape[-1] // 3
# L49
v = state[..., dim:2*dim]
# L50
f = state[..., 2*dim:]
# L56
dv_dt = f - self.christoffel(v, x, force=f)
# L61
return torch.cat([dx_dt, dv_dt, df_dt], dim=-1)
# L84
state = torch.cat([x, v, force], dim=-1)
# L96
final_state = out[-1]
# L99
dt = self.integration_time / self.n_steps
# L104
k2 = self.ode_func(0, curr_state + 0.5 * dt * k1)
# L105
k3 = self.ode_func(0, curr_state + 0.5 * dt * k2)
# L106
k4 = self.ode_func(0, curr_state + dt * k3)
# L107
curr_state = curr_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
# L114
v_out = final_state[..., dim:2*dim]
# L177
x = self.x0.expand(batch_size, -1)
# L178
v = self.v0.expand(batch_size, -1)
# L217
mask = attention_mask.unsqueeze(-1).float()
# L226
force = all_forces[:, t] * mask[:, t]
# Contexto: GeodesicODEFunc
# L31
dv/dt = f - Γ(v, v)
# L32
df/dt = 0  (Force is constant during integration step)
# Contexto: sample_next
# L265
next_logit = logits[:, -1, :] / temp
# L270
next_logit[next_logit < v[:, [-1]]] = -float('Inf')
# L275
cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
# L277
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
# L280
next_logit[indices_to_remove] = -float('Inf')
# L284
return torch.argmax(next_logit, dim=-1, keepdim=True)
# L286
probs = torch.softmax(next_logit, dim=-1)
```

#### gfn\core\manifold.py
```python
# Contexto: __init__
# L107
in_dim = (3 * dim) if (self.physics_config.get('topology', {}).get('type') == 'torus') else (2 * dim)
# L155
self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)
# L156
self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)
# Contexto: forward
# L248
x_scan = self.x0.expand(batch_size, seq_len, -1)
# L275
mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
# L354
force = all_forces[:, t] * mask[:, t]
# L367
mem_input = torch.cat([torch.sin(x), torch.cos(x), v], dim=-1)
# L369
mem_input = torch.cat([x, v], dim=-1)
# L371
deformation_update = torch.tanh(self.hysteresis_update(mem_input))
# L373
hysteresis_state = self.hysteresis_decay * hysteresis_state + deformation_update
# Contexto: sample_next
# L425
next_logit = logits[:, -1, :] / temp
# L426
probs = torch.softmax(next_logit, dim=-1)
# L431
next_logit[next_logit < v[:, [-1]]] = -float('Inf')
# L436
cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
# L441
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
# L445
next_logit[indices_to_remove] = -float('Inf')
# L450
return torch.argmax(next_logit, dim=-1, keepdim=True)
# L453
probs = torch.softmax(next_logit, dim=-1)
```

#### gfn\cuda\autograd.py
```python
# Contexto: christoffel_fused_autograd
# L536
r: float = 1.0) -> torch.Tensor:
# Contexto: leapfrog_fused_autograd
# L587
hyst_enabled: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
# Contexto: record
# L62
self._counts[name] += 1
# Contexto: recurrent_manifold_fused_autograd
# L664
head_dim = D_total // num_heads
# L665
num_layers = U_stack.shape[0] // num_heads # Should be 1 for a single MLayer
# L725
U_heads = U_stack.view(num_heads, head_dim, -1)
# L726
W_heads = W_stack.view(num_heads, head_dim, -1).permute(0, 2, 1)
# L730
Wf_heads = Wf.view(num_heads, head_dim, -1)
# L755
dt_eff = (dt * dt_scales).view(1, num_heads, 1) # [1, H, 1]
# L756
h = 0.5 * dt_eff
# L764
feat_h = torch.cat([torch.sin(x_h), torch.cos(x_h)], dim=-1)
# L766
gate = torch.einsum('bhd,hdk->bhk', feat_h, Wf_heads.transpose(1, 2))
# L768
gate = gate + bf_heads.view(1, num_heads, head_dim)
# L770
gate = gate + torch.einsum('bhd,hdk->bhk', f_h, Wi_heads.transpose(1, 2))
# L771
mu_h = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
# L774
v_norm = torch.norm(v_h, dim=-1, keepdim=True) / (head_dim**0.5 + 1e-8)
# L775
mu_h = mu_h * (1.0 + kwargs['velocity_friction_scale'] * v_norm)
# L780
fg_h = torch.einsum('bhd,hdk->bhk', m_h, h_rd_w.view(num_heads, head_dim, head_dim).transpose(1, 2))
# L782
fg_h = fg_h + h_rd_b.view(1, num_heads, head_dim)
# L786
h_proj = torch.einsum('bhd,hdr->bhr', v_h, U_heads)
# L787
energy = torch.sum(h_proj * h_proj, dim=-1, keepdim=True) / max(1, h_proj.shape[-1])
# L788
s_norm = 1.0 / (1.0 + torch.sqrt(energy) + 1e-8)
# L791
gamma_h = torch.einsum('bhr,hrd->bhd', h_proj * h_proj, W_heads) * s_norm
# L798
T = max(thermo_temp, 1e-8)
# L800
f_energy = (f_h ** 2).mean(dim=-1, keepdim=True) # [B, H, 1]
# L801
modulator = torch.exp(-thermo_alpha * f_energy / T)
# L802
gamma_h = gamma_h * modulator
# L812
v_dot_gz = (v_h * gz_h).sum(dim=-1, keepdim=True)
# L813
v_sq = (v_h * v_h).sum(dim=-1, keepdim=True)
# L815
gamma_ads = -(1.0 / (z_h + 1e-8)) * (2.0 * v_dot_gz * v_h - v_sq * gz_h)
# L816
gamma_h = gamma_h + gamma_ads
# L819
gamma_h = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_h / CudaConstants.CURVATURE_CLAMP)
# L823
v_half = (v_h + h * (f_h + fg_h - gamma_h)) / (1.0 + h * mu_h + 1e-8)
# L827
x_new_h = x_h + dt_eff * v_half
# L829
x_new_h = torch.atan2(torch.sin(x_new_h), torch.cos(x_new_h))
# L839
feat_h = torch.cat([torch.sin(x_new_h), torch.cos(x_new_h)], dim=-1)
# L840
gate = torch.einsum('bhd,hdk->bhk', feat_h, Wf_heads.transpose(1, 2))
# L841
if bf_heads is not None: gate = gate + bf_heads.view(1, num_heads, head_dim)
# L842
if Wi_heads is not None: gate = gate + torch.einsum('bhd,hdk->bhk', f_h, Wi_heads.transpose(1, 2))
# L843
mu_h = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
# L845
h_proj = torch.einsum('bhd,hdr->bhr', v_half, U_heads)
# L846
energy = torch.sum(h_proj * h_proj, dim=-1, keepdim=True) / max(1, h_proj.shape[-1])
# L847
s_norm = 1.0 / (1.0 + torch.sqrt(energy) + 1e-8)
# L848
gamma_h = torch.einsum('bhr,hrd->bhd', h_proj * h_proj, W_heads) * s_norm
# L852
gamma_h = gamma_h * modulator # Use same modulator as it depends on force_t
# L855
gamma_ads = -(1.0 / (z_h + 1e-8)) * (2.0 * v_dot_gz * v_half - v_sq * gz_h)
# L856
gamma_h = gamma_h + gamma_ads
# L859
gamma_h = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_h / CudaConstants.CURVATURE_CLAMP)
# L861
v_new_h = (v_half + h * (f_h + fg_h - gamma_h)) / (1.0 + h * mu_h + 1e-8)
# L872
h_in = torch.cat([torch.sin(x_new_h), torch.cos(x_new_h)], dim=-1)
# L873
h_in = torch.cat([h_in, v_new_h], dim=-1)
# L875
h_gate = torch.einsum('bhd,hdk->bhk', h_in, h_up_w.view(num_heads, head_dim, -1).transpose(1, 2))
# L877
h_gate = h_gate + h_up_b.view(1, num_heads, head_dim)
# L879
h_state = h_state * h_decay + torch.tanh(h_gate.reshape(B, D_total))
# Contexto: summary
# L93
lines = ["=" * 60, "EXECUTION TIME SUMMARY", "=" * 60]
# Contexto: toroidal_leapfrog_fused_autograd
# L956
v_half = v_curr + 0.5 * dt * force_t
# L959
x_new = x_curr + dt * v_half
# L963
x_new = torch.atan2(torch.sin(x_new), torch.cos(x_new))
# L970
phi = x_new[:, i + 1]
# L972
v_phi = v_half[:, i + 1]
# L975
cos_theta = torch.cos(theta)
# L976
denom = torch.clamp(R + r * cos_theta, min=1e-6)
# L979
sin_theta = torch.sin(theta)
# L980
gamma_theta = denom * sin_theta / (r + 1e-6) * (v_phi * v_phi)
# L983
gamma_phi = -(r * sin_theta) / (denom + 1e-6) * 2.0 * v_theta * v_phi
# L986
gamma[:, i + 1] = gamma_phi
# L989
gamma = torch.clamp(gamma, -10.0, 10.0)
# L992
v_new = v_half + 0.5 * dt * (force_t - gamma)
# Contexto: wrapper
# L128
result = func(*args, **kwargs)
# L129
duration = time.perf_counter() - start
```

#### gfn\cuda\core.py
```python
# Contexto: CudaConstants
# L128
EPSILON_STANDARD = 1e-7
# L129
EPSILON_STRONG = 1e-7
# L130
EPSILON_SMOOTH = 1e-7
# L144
TOROIDAL_PERIOD = 6.283185307179586  # 2 * π
# Contexto: get_device
# L72
def get_device(self, index: int = 0) -> torch.device:
```

#### gfn\cuda\ops.py
```python
# Contexto: _get_load_paths
# L74
cuda_path = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{ver}/bin")
# Contexto: backward
# L300
topology: int = 0) -> Tuple[torch.Tensor, ...]:
# L536
velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, ...]:
# Contexto: christoffel_fused
# L576
r: float = 1.0) -> torch.Tensor:
# Contexto: ChristoffelOperation
# L217
Computes: Γ^k_ij = Σ_r λ_kr * (U_ir * U_jr)
# Contexto: dynamic_gating_fused
# L799
hidden = torch.tanh(torch.matmul(x, W1.t()) + b1)
# L800
out = torch.matmul(hidden, W2.t()) + b2
# Contexto: dynamics
# L709
if topology == 1: feat = torch.cat([torch.sin(tx), torch.cos(tx)], dim=-1)
# L710
gate = torch.matmul(feat, W_forget.t()) + b_forget
# L712
gate = gate + torch.matmul(f, W_input.t())
# L713
mu = torch.sigmoid(gate) * CudaConstants.FRICTION_SCALE
# L717
vm = torch.norm(tv, dim=-1, keepdim=True) / (tv.shape[-1]**0.5 + 1e-8)
# L718
mu = mu * (1.0 + velocity_friction_scale * vm)
# L720
acc = f - gamma - mu * tv
# L728
x_pred = curr_x + eff_dt * dx1
# L729
v_pred = curr_v + eff_dt * dv1
# L736
curr_x = curr_x + 0.5 * eff_dt * (dx1 + dx2)
# L737
curr_v = curr_v + 0.5 * eff_dt * (dv1 + dv2)
# L740
curr_x = torch.atan2(torch.sin(curr_x), torch.cos(curr_x))
# Contexto: euler_fused
# L748
topology: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
# L751
eff_dt = dt * dt_scale
# L758
curr_v = curr_v + eff_dt * acc
# L759
curr_x = curr_x + eff_dt * curr_v
# Contexto: forward
# L245
topology: int = 0) -> torch.Tensor:
# L264
h = torch.matmul(v, U)  # [B, R]
# L267
energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])
# L268
scale = 1.0 / (1.0 + torch.sqrt(energy) + self.epsilon)
# L273
E = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
# L274
M = 1.0 + plasticity * 0.1 * torch.tanh(E)
# L279
pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
# L281
pot = torch.sum(x * V_w, dim=-1, keepdim=True)
# L283
gate = torch.sigmoid(pot)
# L284
soft_m = torch.sigmoid(self.singularity_gate_slope * (gate - sing_thresh))
# L285
M = M * (1.0 + (sing_strength - 1.0) * soft_m)
# L288
gamma = torch.matmul(h * h, W.t()) * scale * M
# L289
gamma = self.curvature_clamp * torch.tanh(gamma / self.curvature_clamp)
# L459
velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
# L464
eff_dt = self.dt * dt_scale
# L465
h = 0.5 * eff_dt
# L485
feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
# L486
gate = torch.matmul(feat, Wf.t()) + bf
# L488
gate = gate + torch.matmul(f, W_input.t())
# L489
mu = torch.sigmoid(gate) * self.friction_scale
# L491
v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
# L492
v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + CudaConstants.EPSILON_SMOOTH)
# L493
mu = mu * (1.0 + velocity_friction_scale * v_norm)
# L497
curr_v = (curr_v + h * (f - gamma)) / (1.0 + h * mu + self.epsilon)
# L500
curr_x = curr_x + eff_dt * curr_v
# L509
feat = torch.cat([torch.sin(curr_x), torch.cos(curr_x)], dim=-1)
# L510
gate = torch.matmul(feat, Wf.t()) + bf
# L512
gate = gate + torch.matmul(f, W_input.t())
# L513
mu = torch.sigmoid(gate) * self.friction_scale
# L515
v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
# L516
v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + CudaConstants.EPSILON_SMOOTH)
# L517
mu = mu * (1.0 + velocity_friction_scale * v_norm)
# L520
curr_v = (curr_v + h * (f - gamma2)) / (1.0 + h * mu + self.epsilon)
# Contexto: head_mixing_fused
# L766
topology: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
# L775
x_cat = x_heads.permute(1, 0, 2).contiguous().view(batch, -1)
# L776
v_cat = v_heads.permute(1, 0, 2).contiguous().view(batch, -1)
# L778
v_mix = torch.tanh(v_cat / 100.0)
# L779
mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)
# L780
x_next = torch.matmul(mixer_in_x, W_x.t())
# L782
x_next = torch.matmul(x_cat, W_x.t())
# L783
v_next = torch.matmul(v_cat, W_v.t())
# L785
x_next = torch.atan2(torch.sin(x_next), torch.cos(x_next))
# L786
v_next = 100.0 * torch.tanh(v_next / 100.0)
# Contexto: heun_fused
# L657
velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
# L698
eff_dt = dt * dt_scale
# Contexto: launch_toroidal_leapfrog_fused
# L398
x_final = x_out[:, -1, :]  # [batch, dim]
# L399
v_final = v_out[:, -1, :]  # [batch, dim]
# Contexto: leapfrog_fused
# L617
velocity_friction_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
```

#### gfn\cuda\precompile_kernels.py
```python
# Contexto: Global
# L42
print("\n" + "="*70)
```

#### gfn\cuda\setup.py
```python
# Contexto: Global
# L9
cuda_sources = [ 'cuda_kernels.cpp', 'src/geometry/lowrank_christoffel.cu', 'src/geometry/lowrank_christoffel_backward.cu', 'src/geometry/lowrank_christoffel_friction_backward.cu', 'src/integrators/symplectic/leapfrog_fused.cu', 'src/integrators/symplectic/leapfrog_backward.cu', 'src/integrators/toroidal/toroidal_christoffel_fused.cu', 'src/integrators/runge_kutta/heun_fused.cu', 'src/integrators/runge_kutta/heun_backward.cu', 'src/integrators/unified_mlayer.cu', ]
# L40
cxx_flags = ["-O3"]
# L41
nvcc_flags = [ "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-gencode=arch=compute_75,code=sm_75", ]
# L49
cxx_flags = ["/O2", "/bigobj", "/EHsc", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN"]
# L50
nvcc_flags = [ "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-Xcompiler", "/bigobj", "-Xcompiler", "/EHsc", "-Xcompiler", "/DNOMINMAX", "-Xcompiler", "/DWIN32_LEAN_AND_MEAN", ] + nvcc_flags[3:]
```

#### gfn\datasets\math.py
```python
# Contexto: __init__
# L27
self.char_to_id['+'] = 10
# L28
self.char_to_id['-'] = 11
# L29
self.char_to_id['*'] = 12
# Contexto: _generate_problem
# L38
ops = ['+', '-', '*']
# L43
a = random.randint(0, 10**min(3, self.max_digits) - 1)
# L44
b = random.randint(0, 10**min(3, self.max_digits) - 1)
# L47
a = random.randint(0, 10**self.max_digits - 1)
# L51
a = random.randint(0, 10**self.max_digits - 1)
# L52
b = random.randint(0, 10**self.max_digits - 1)
# Contexto: collate_fn
# L83
pad_len = max_len - len(x)
```

#### gfn\datasets\mixed.py
```python
# Contexto: __init__
# L26
self.wiki_en = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
# L30
self.math_thinking = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
# Contexto: __iter__
# L58
a = random.randint(0, 10**8 - 1)
# L59
b = random.randint(0, 10**8 - 1)
# L61
text = f"Math: {a} + {b} = {c}"
```

#### gfn\embeddings\functional.py
```python
# Contexto: __init__
# L85
self.net = nn.Sequential(*net)
# L90
self.out_proj.weight.data *= 1.5
# L95
if coord_dim % 2 != 0: self.coord_dim += 1
# L97
freqs = torch.exp(torch.arange(0, self.coord_dim, 2).float() * -(np.log(10000.0) / self.coord_dim))
# Contexto: forward
# L111
inputs = input_ids.unsqueeze(-1).float()
# L115
mask = 2**torch.arange(self.coord_dim).to(input_ids.device)
# L116
bits = (input_ids.unsqueeze(-1) & mask) > 0
# L120
coords = bits.float() * 2 - 1 # Map {0, 1} to {-1, 1} for SIREN
# L123
args = inputs * self.freqs
# L124
coords = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
# L132
out = out * self.impulse_scale
# L137
active_mask = (bits.float().sum(dim=-1, keepdim=True) > 0).float()
# L138
out = out * active_mask
```

#### gfn\embeddings\implicit.py
```python
# Contexto: __init__
# L83
self.net = nn.Sequential(*net)
# Contexto: ImplicitEmbedding
# L29
- Standard: 10k * 256 = 2.56M params
# L30
- Implicit: 10k * 16 + ~50k = 210k params (~12x reduction)
```

#### gfn\embeddings\siren.py
```python
# Contexto: init_weights
# L52
bound = 1 / self.linear.weight.size(1)
# L56
bound = np.sqrt(6 / self.linear.weight.size(1)) / self.omega_0
```

#### gfn\geometry\__init__.py
```python
# Contexto: __init__
# L18
input_dim = 2 * dim if topology == 1 else dim
# L19
self.net = nn.Sequential( nn.Linear(input_dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1), nn.Sigmoid() )
# Contexto: forward
# L31
x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
```

#### gfn\geometry\adaptive.py
```python
# Contexto: __init__
# L18
self.U_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)
# L19
self.W_full = nn.Parameter(torch.randn(dim, max_rank) * 0.01)
# Contexto: forward
# L31
def forward(self, v, x=None, **kwargs):
# L34
rank_ratio = 0.1 + 0.9 * self.complexity_net(v.detach())
# L37
avg_ratio = rank_ratio.mean().item()
# L39
eff_rank = max(4, min(self.max_rank, int(avg_ratio * self.max_rank)))
# L46
proj = torch.matmul(v, U)
# L47
norm = torch.norm(proj, dim=-1, keepdim=True)
# L48
scale = 1.0 / (1.0 + norm + EPSILON_STANDARD)
# L49
sq = (proj * proj) * scale
# L50
gamma = torch.matmul(sq, W.t())
# L56
gamma = CURVATURE_CLAMP * torch.tanh(gamma / CURVATURE_CLAMP)
```

#### gfn\geometry\analytical.py
```python
# Contexto: __init__
# L31
self.curvature = -1.0
# Contexto: forward
# L13
def forward(self, v, x=None, **kwargs):
# L47
x_sq = torch.sum(x*x, dim=-1, keepdim=True)
# L48
v_sq = torch.sum(v*v, dim=-1, keepdim=True)
# L49
xv = torch.sum(x*v, dim=-1, keepdim=True)
# L52
gamma = 2 * xv * v - v_sq * x
# L76
x_sq = torch.sum(x*x, dim=-1, keepdim=True)
# L77
v_sq = torch.sum(v*v, dim=-1, keepdim=True)
# L78
xv = torch.sum(x*v, dim=-1, keepdim=True)
# L82
gamma = -(2 * xv * v - v_sq * x)
# Contexto: HyperbolicChristoffel
# L24
Geodesic Accel: a = -Gamma(v,v)
# L26
Uses Conformal Factor lambda = 2 / (1 - |x|^2)
```

#### gfn\geometry\boundaries.py
```python
# Contexto: apply_boundary_python
# L12
For topology_id == 1 (torus), positions are wrapped to [0, 2π). The wrapping is periodic: x = x % (2*π) 2. VELOCITY HANDLING: Velocity vectors should NOT be wrapped! - Position: x is on the manifold, needs wrapping - Velocity: v is in the TANGENT SPACE, invariant under wrapping The wrapping of velocity would create artificial discontinuities that break the smoothness of geodesic flow. If you need to apply velocity corrections, use apply_velocity_correction(). Topology IDs: 0: Euclidean (None) - No boundary conditions 1: Toroidal (Periodic [0, 2*PI)) - Positions wrapped Args: x: Position tensor [batch, dim] topology_id: Integer topology identifier Returns: Position tensor with boundaries applied """ if topology_id == 1: PI = 3.14159265359 TWO_PI = 2.0 * PI x_wrapped = torch.atan2(torch.sin(x), torch.cos(x)) x_wrapped = torch.where(x_wrapped < 0, x_wrapped + TWO_PI, x_wrapped) return x_wrapped return x def apply_velocity_correction(v, x_old, x_new, topology_id): """ Correct velocity for toroidal boundary crossings. When position crosses the boundary (e.g., from 6.28 to 0.01), the apparent velocity is wrong. This function computes the true velocity considering boundary crossings. AUDIT FIX: This function handles velocity correction for torus. Args: v: Velocity tensor [batch, dim] x_old: Previous position [batch, dim] x_new: Current position [batch, dim] topology_id: Topology identifier Returns: Corrected velocity tensor """ if topology_id != 1: return v PI = 3.14159265359 TWO_PI = 2.0 * PI apparent_disp = x_new - x_old wrapped_disp = torch.atan2(torch.sin(apparent_disp), torch.cos(apparent_disp)) return wrapped_disp def toroidal_dist_python(x1, x2): """ Shortest angular distance on Torus. IMPORTANT (Auditoria 2026-02-06): This computes distance on a FLAT torus (product of circles). It does NOT account for the LEARNED Christoffel curvature. For tasks requiring true geodesic distance on the learned manifold, use Christoffel-based distance computation instead. Args: x1: Position tensor [batch, dim] x2: Position tensor [batch, dim] Returns: Distance tensor """ PI = 3.14159265359 diff = x1 - x2 diff = torch.atan2(torch.sin(diff), torch.cos(diff)) return torch.norm(diff, dim=-1) def resolve_topology_id(christoffel, topology_id_arg=None): """ Resolve topology ID from Christoffel geometry or argument. Args: christoffel: The Christoffel geometry object topology_id_arg: Optional override from kwargs Returns: Integer topology ID (0=Euclidean, 1=Torus) """ if topology_id_arg is not None: return topology_id_arg tid = getattr(christoffel, 'topology_id', 0) if tid == 0 and hasattr(christoffel, 'is_torus') and christoffel.is_torus: return 1 return tid def get_boundary_features(x, topology_id): """ Extract features relevant to the topology boundary. For Euclidean (0): Returns x For Toroidal (1): Returns [sin(x), cos(x)] Args: x: Position tensor [batch, dim] topology_id: Integer topology identifier Returns: Feature tensor [batch, dim] or [batch, 2*dim] """ if topology_id == 1: return torch.cat([torch.sin(x), torch.cos(x)], dim=-1) return x
```

#### gfn\geometry\confusion.py
```python
# Contexto: ConfusionChristoffel
# L19
g_new = g_base + lambda * (F @ F.T)
# Contexto: forward
# L36
def forward(self, v, x=None, force=None, **kwargs):
# L39
gamma = self.base_christoffel(v, x, force=force, **kwargs)
# L45
confusion = (force ** 2).mean(dim=-1, keepdim=True)
# L52
scale = 1.0 + self.sensitivity * confusion
# L54
gamma = gamma * scale
```

#### gfn\geometry\gauge.py
```python
# Contexto: __init__
# L38
self.A_net = nn.Sequential( nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, gauge_dim * dim) # Output shape: [batch, dim * gauge_dim] )
# Contexto: compute_field_strength
# L64
Computes the field strength tensor F_mn = d_m A_n - d_n A_m + [A_m, A_n]
# Contexto: forward
# L146
def forward(self, v, x=None, force=None, **kwargs):
# L150
Gamma^g = Gamma^base + g * (D v - d v)
# L152
Since D v - d v = A v (approx), the correction represents the
# L156
gamma_base = self.base_christoffel(v, x, force, **kwargs)
# L166
gamma_gauge = self.gauge_coupling * (v_transported - v)
# Contexto: gauge_invariant_loss
# L173
L_gauge = MSE(f(x), f(g*x))
# Contexto: GaugeChristoffel
# L16
Gamma^g_uv = Gamma^R_uv + g * (D_u v - d_u v)
# Contexto: get_A
# L104
F = d_A.permute(1, 0, 2) - d_A # [mu, nu, a] - [nu, mu, a]
# Contexto: parallel_transport
# L132
phase_shift = torch.bmm(v.unsqueeze(1), A).squeeze(1)
# L141
modulation = torch.cos(phase_shift.mean(dim=-1, keepdim=True))
```

#### gfn\geometry\hierarchical.py
```python
# Contexto: __init__
# L22
self.scale_weights = nn.Parameter(torch.ones(len(ranks)) / len(ranks))
# Contexto: forward
# L27
def forward(self, v, x=None, force=None, **kwargs):
# L37
gamma, mu = scale(v, x, force, **kwargs)
# L44
weights = torch.softmax(self.scale_weights, dim=0)
# L46
gamma_combined = sum(w * g for w, g in zip(weights, gammas))
# L48
mu_combined = sum(w * m for w, m in zip(weights, mus))
```

#### gfn\geometry\holographic.py
```python
# Contexto: __init__
# L26
self.radial_net = nn.Sequential( nn.Linear(self.dim, self.dim // 2), nn.SiLU(), nn.Linear(self.dim // 2, 1), nn.Softplus() )
# Contexto: AdSCFTChristoffel
# L14
g_ij = (1 / z(x)^2) * delta_ij
# L17
Gamma^k_ij = -1/z * (z_j delta^k_i + z_i delta^k_j - z_k delta_ij)
# Contexto: forward
# L58
gamma_base = self.base_christoffel(v, x, **kwargs)
# L67
v_dot_gradz = (v * grad_z).sum(dim=-1, keepdim=True) # [B, 1]
# L68
v_sq = (v * v).sum(dim=-1, keepdim=True) # [B, 1]
# L70
gamma_ads = -(1.0 / z) * (2.0 * v_dot_gradz * v - v_sq * grad_z)
# Contexto: get_z_and_grad
# L41
z = self.radial_net(x_req) + self.z_min
# L45
grad_z = torch.autograd.grad( z.sum(), x_req, create_graph=self.training, retain_graph=True )[0]
```

#### gfn\geometry\hyper.py
```python
# Contexto: forward
# L35
def forward(self, v, x=None, force=None, **kwargs):
# L37
return super().forward(v, None, force=force, **kwargs)
# L41
g_u = torch.sigmoid(self.gate_u(x)) * 2.0 # [batch, rank]
# L42
g_w = torch.sigmoid(self.gate_w(x)) * 2.0 # [batch, rank]
# L59
proj_static = torch.matmul(v, self.U) # [batch, rank]
# L62
proj_dynamic = proj_static * g_u # [batch, rank]
# L66
sq_dynamic = (proj_dynamic * proj_dynamic) / (1.0 + torch.abs(proj_dynamic))
# L69
sq_modulated = sq_dynamic * g_w # [batch, rank]
# L73
gamma = torch.matmul(sq_modulated, self.W.t()) # [batch, dim]
# L76
x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# L91
gate_activ = torch.matmul(x_in, Wf.t()) + bf
# L93
gate_activ = gate_activ + torch.matmul(force, Wi.t())
# L97
gate_activ = gate_activ + self.input_gate(force)
# L99
mu_base = torch.sigmoid(gate_activ) * FRICTION_SCALE
# L100
velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
# L101
velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + EPSILON_STRONG)
# L102
mu = mu_base * (1.0 + self.velocity_friction_scale * velocity_magnitude)
# L108
gamma = gamma + mu * v
# Contexto: HyperChristoffel
# L11
Gamma(v, v | x) = W(x) * (U(x)^T v)^2
# L14
U(x) = U_static * diag(Gate_u(x))
# L15
W(x) = W_static * diag(Gate_w(x))
```

#### gfn\geometry\hysteresis.py
```python
# Contexto: __init__
# L25
self.U_hyst = nn.Parameter(torch.randn(dim, rank) * 0.01)
# L26
self.W_hyst = nn.Parameter(torch.randn(dim, rank) * 0.01)
# Contexto: forward
# L28
def forward(self, v, x=None, memory_state=None, **kwargs):
# L38
gamma = self.base_christoffel(v, x, **kwargs)
# L49
delta = torch.matmul(memory_state, self.U_hyst) # [Batch, Rank]
# L50
delta = torch.matmul(delta, self.W_hyst.t())    # [Batch, Dim]
# L55
v_norm = torch.norm(v, dim=-1, keepdim=True)
# L56
delta = delta * v_norm
# L58
gamma = gamma + delta
# Contexto: HysteresisChristoffel
# L9
Γ_hyst = Γ_base + δΓ(h)
```

#### gfn\geometry\lowrank.py
```python
# Contexto: __init__
# L55
gate_input_dim = 2 * dim if self.is_torus else dim
# Contexto: _normalize_christoffel_structure
# L216
gamma_sym = 0.5 * (gamma + gamma.transpose(-1, -2))
# L223
diag_mean = torch.diagonal(gamma_sym, dim1=-1, dim2=-2).mean(dim=-1, keepdim=True)
# L225
gamma_centered = gamma_sym - torch.diag_embed(diag_mean.squeeze(-1))
# Contexto: forward
# L83
def forward(self, v, x=None, force=None, **kwargs):
# L90
- Total acceleration: a = F_ext - Christoffel(v,v) - Friction*v
# L93
Acc = F_ext - Output
# L105
x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# L109
friction = torch.sigmoid(self.forget_gate(x_in)) * FRICTION_SCALE
# L114
velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
# L116
velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + 1e-8)
# L117
friction = friction * (1.0 + self.velocity_friction_scale * velocity_magnitude)
# L124
gamma_cuda = gamma_cuda + friction * v
# L136
proj = torch.bmm(v, self.U)
# L137
norm = torch.norm(proj, dim=-1, keepdim=True)
# L139
scale = 1.0 / (1.0 + norm + EPSILON_STRONG)
# L140
sq = (proj * proj) * scale
# L141
gamma = torch.bmm(sq, self.W.transpose(1, 2))
# L143
proj = torch.matmul(v, self.U)
# L144
norm = torch.norm(proj, dim=-1, keepdim=True)
# L146
scale = 1.0 / (1.0 + norm + EPSILON_STRONG)
# L147
sq = (proj * proj) * scale
# L148
gamma = torch.matmul(sq, self.W.t())
# L152
x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# L165
gate_activ = torch.matmul(x_in, Wf.t()) + bf
# L167
gate_activ = gate_activ + torch.matmul(force, Wi.t())
# L171
gate_activ = gate_activ + self.input_gate(force)
# L174
mu_base = torch.sigmoid(gate_activ) * FRICTION_SCALE
# L178
velocity_magnitude = torch.norm(v, dim=-1, keepdim=True)
# L179
velocity_magnitude = velocity_magnitude / (self.dim ** 0.5 + EPSILON_STRONG)
# L180
mu = mu_base * (1.0 + self.velocity_friction_scale * velocity_magnitude)
# L187
gamma = gamma + mu * v
# Contexto: LowRankChristoffel
# L21
The decomposition Gamma^k_ij = sum_{r=1}^R lambda_kr * (U_ir * U_jr)
# L23
- Symmetry Gamma^k_ij = Gamma^k_ji is preserved (by construction)
# L31
Friction is now computed as: mu(x,v) = sigma(gate(x)) * FRICTION_SCALE * (1 + alpha * ||v||)
```

#### gfn\geometry\reactive.py
```python
# Contexto: forward
# L58
def forward(self, v, x=None, force=None, **kwargs):
# L69
V_w_in = self.V.weight.t()  # [1, dim] -> [dim, 1] -> [1, dim]
# L87
gamma = super().forward(v, x, force=force, **kwargs)
# L96
energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
# L99
gamma = gamma * (1.0 + self.plasticity * energy)
# L107
x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# L110
potential = torch.sigmoid(self.V(x_in)) # [batch, 1]
# L124
is_amplified = torch.sigmoid(gate_slope * (potential - self.semantic_certainty_threshold))
# L125
amplification_mult = 1.0 + is_amplified * (self.curvature_amplification_factor - 1.0)
# L126
gamma = gamma * amplification_mult
# L131
gamma = torch.clamp(gamma, -max_amplification * CURVATURE_CLAMP, max_amplification * CURVATURE_CLAMP)
# Contexto: ReactiveChristoffel
# L17
- "Black hole strength" = curvature_amplification_factor
# L18
- "Singularity threshold" = semantic_certainty_threshold
```

#### gfn\geometry\ricci.py
```python
# Contexto: RicciFlowChristoffel
# L9
dg_ij / dt = -2 * R_ij
```

#### gfn\geometry\thermo.py
```python
# Contexto: compute_entropy_proxy
# L37
var_v = torch.var(v, dim=0).mean() # Scalar proxy
# L38
entropy = 0.5 * torch.log(var_v + EPSILON_STRONG)
# Contexto: forward
# L41
def forward(self, v, x=None, force=None, **kwargs):
# L43
gamma = self.base_christoffel(v, x, force=force, **kwargs)
# L49
energy = (force ** 2).mean(dim=-1, keepdim=True)
# L55
T = torch.abs(self.temperature) + EPSILON_STRONG
# L56
free_energy = energy - T * entropy
# L73
modulation = torch.exp(-self.alpha * energy / T)
# L77
gamma = gamma * modulation
# Contexto: ThermodynamicChristoffel
# L9
Implements a metric modulation based on Free Energy (F = E - TS).
# L13
g_ij(x, T) = g_base_ij(x) * exp( -alpha/T * grad(F) )
```

#### gfn\geometry\toroidal.py
```python
# Contexto: __init__
# L37
gate_input_dim = 2 * dim
# Contexto: forward
# L76
def forward(self, v, x=None, force=None, **kwargs):
# L95
cos_th = torch.cos(x)
# L96
sin_th = torch.sin(x)
# L108
v_ph = v[..., i+1]
# L112
denom = torch.clamp(self.R + self.r * torch.cos(th), min=CLAMP_MIN_STRONG)
# L115
term_th = denom * torch.sin(th) / (self.r + EPSILON_SMOOTH)
# L116
gamma[..., i] = term_th * (v_ph ** 2)
# L119
term_ph = -(self.r * torch.sin(th)) / (denom + EPSILON_SMOOTH)
# L120
gamma[..., i+1] = 2.0 * term_ph * v_ph * v_th
# L122
gamma = gamma * TOROIDAL_CURVATURE_SCALE  # Strong Curvature (User Requested Full Torus)
# L126
x_in = torch.cat([sin_th, cos_th], dim=-1)
# L132
gate_activ = gate_activ + self.input_gate(force)
# L136
mu = torch.sigmoid(gate_activ) * FRICTION_SCALE
# L142
energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
# L143
gamma = gamma * (1.0 + self.plasticity * energy)
# L147
potential = torch.sigmoid(self.V(x_in))
# L149
gamma = gamma * (1.0 + is_singularity * (self.black_hole_strength - 1.0))
# L155
gamma = gamma + mu * v
# Contexto: get_metric
# L65
g_phi = (R + r cos theta)^2
# L71
g[..., i] = self.r**2
# L73
g[..., i+1] = (self.R + self.r * torch.cos(th))**2
# Contexto: ToroidalChristoffel
# L15
g = diag(r^2, (R + r cos th)^2)
```

#### gfn\integrators\adaptive.py
```python
# Contexto: __init__
# L21
def __init__(self, base_integrator, tolerance=1e-3, max_depth=3):
# L32
self.error_scale = 1.0 / 15.0
# L34
self.error_scale = 1.0 / 3.0 # Conservatively assume 2nd order
# Contexto: AdaptiveIntegrator
# L14
3. Error estimate E = ||x1 - x2|| / (2^p - 1)
# Contexto: forward
# L36
def forward(self, x, v, force=None, dt_scale=1.0, depth=0, **kwargs):
# L40
dt = self.base_integrator.dt * dt_scale
# L43
x1, v1 = self.base_integrator(x, v, force=force, dt_scale=dt_scale, steps=1, **kwargs)
# L47
x_mid, v_mid = self.base_integrator(x, v, force=force, dt_scale=dt_scale * 0.5, steps=1, **kwargs)
# L48
x2, v2 = self.base_integrator(x_mid, v_mid, force=force, dt_scale=dt_scale * 0.5, steps=1, **kwargs)
# L52
error = torch.norm(x1 - x2, dim=-1).max() * self.error_scale
# L60
x_half, v_half = self.forward(x, v, force, dt_scale * 0.5, depth + 1, **kwargs)
# L64
x_final, v_final = self.forward(x_half, v_half, force, dt_scale * 0.5, depth + 1, **kwargs)
```

#### gfn\integrators\neural.py
```python
# Contexto: __init__
# L36
self.controller = nn.Sequential( nn.Linear(self.dim * 3, self.dim), # Input: [x, v, f] nn.GELU(), # Better gradients than Tanh nn.Linear(self.dim, 1), nn.Softplus() # Strictly positive dt )
# Contexto: forward
# L48
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L62
state = torch.cat([x, v, f_in], dim=-1)
# L63
learned_scale = self.controller(state) + 0.1
# L64
dynamics_dt = self.base_dt * dt_scale * learned_scale
# L67
acc = -self.christoffel(v, x, force=f_in, **kwargs)
# L69
acc = acc + force
# L71
v_half = v + 0.5 * dynamics_dt * acc
# L72
x = x + dynamics_dt * v_half
# L77
acc_next = -self.christoffel(v_half, x, force=f_in, **kwargs)
# L79
acc_next = acc_next + force
# L81
v = v_half + 0.5 * dynamics_dt * acc_next
# Contexto: Global
# L8
Idea: x_{t+1} = x_t + v_t * NeuralNet(x_t, v_t)
```

#### gfn\integrators\runge_kutta\dormand_prince.py
```python
# Contexto: __init__
# L32
self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
# L36
self.a31, self.a32 = 3/40, 9/40
# L37
self.a41, self.a42, self.a43 = 44/45, -56/15, 32/9
# L38
self.a51, self.a52, self.a53, self.a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
# L39
self.a61, self.a62, self.a63, self.a64, self.a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
# L42
self.b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
# Contexto: dynamics
# L73
acc = -self.christoffel(tv, tx, force=force, **kwargs)
# L75
acc = acc + force
# L87
x2 = apply_boundary_python(curr_x + dt * (self.a21*k1_x), topo_id)
# L88
v2 = curr_v + dt * (self.a21*k1_v)
# L93
x3 = apply_boundary_python(curr_x + dt * (self.a31*k1_x + self.a32*k2_x), topo_id)
# L94
v3 = curr_v + dt * (self.a31*k1_v + self.a32*k2_v)
# L99
x4 = apply_boundary_python(curr_x + dt * (self.a41*k1_x + self.a42*k2_x + self.a43*k3_x), topo_id)
# L100
v4 = curr_v + dt * (self.a41*k1_v + self.a42*k2_v + self.a43*k3_v)
# L105
x5 = apply_boundary_python(curr_x + dt * (self.a51*k1_x + self.a52*k2_x + self.a53*k3_x + self.a54*k4_x), topo_id)
# L106
v5 = curr_v + dt * (self.a51*k1_v + self.a52*k2_v + self.a53*k3_v + self.a54*k4_v)
# L111
x6 = apply_boundary_python(curr_x + dt * (self.a61*k1_x + self.a62*k2_x + self.a63*k3_x + self.a64*k4_x + self.a65*k5_x), topo_id)
# L112
v6 = curr_v + dt * (self.a61*k1_v + self.a62*k2_v + self.a63*k3_v + self.a64*k4_v + self.a65*k5_v)
# L117
curr_x = curr_x + dt * (self.b5[0]*k1_x + self.b5[2]*k3_x + self.b5[3]*k4_x + self.b5[4]*k5_x + self.b5[5]*k6_x)
# L119
curr_v = curr_v + dt * (self.b5[0]*k1_v + self.b5[2]*k3_v + self.b5[3]*k4_v + self.b5[4]*k5_v + self.b5[5]*k6_v)
# Contexto: forward
# L44
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L60
dt = self.base_dt * dt_scale
# L70
dt = self.base_dt * dt_scale
```

#### gfn\integrators\runge_kutta\euler.py
```python
# Contexto: forward
# L23
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L46
dt = self.dt * dt_scale
# L48
c_out = self.christoffel(curr_v, curr_x, force=force, **kwargs)
# L54
acc = acc + force
# L56
curr_x = curr_x + dt * curr_v
# L57
curr_v = curr_v + dt * acc
```

#### gfn\integrators\runge_kutta\heun.py
```python
# Contexto: dynamics
# L67
c_out = self.christoffel(current_v, current_x, force=force, **kwargs)
# L73
acc = acc + force
# L84
v_pred = curr_v + dt * dv1
# L85
x_pred = apply_boundary_python(curr_x + dt * dx1, topo_id)
# L92
curr_x = curr_x + (dt / 2.0) * (dx1 + dx2)
# L93
curr_v = curr_v + (dt / 2.0) * (dv1 + dv2)
# Contexto: forward
# L23
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L62
dt = self.dt * dt_scale
```

#### gfn\integrators\runge_kutta\rk4.py
```python
# Contexto: dynamics
# L46
c_out = self.christoffel(current_v, current_x, force=force, **kwargs)
# L52
acc = acc + force
# L63
v2 = curr_v + 0.5 * dt * dv1
# L64
x2 = apply_boundary_python(curr_x + 0.5 * dt * dx1, topo_id)
# L69
v3 = curr_v + 0.5 * dt * dv2
# L70
x3 = apply_boundary_python(curr_x + 0.5 * dt * dx2, topo_id)
# L75
v4 = curr_v + dt * dv3
# L76
x4 = apply_boundary_python(curr_x + dt * dx3, topo_id)
# L81
curr_x = curr_x + (dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
# L83
curr_v = curr_v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
# Contexto: forward
# L21
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L42
dt = self.dt * dt_scale
```

#### gfn\integrators\stochastic.py
```python
# Contexto: forward
# L21
def forward(self, x, v, force=None, dt_scale=1.0, **kwargs):
# L24
x_next, v_next = self.base_integrator(x, v, force=force, dt_scale=dt_scale, **kwargs)
# L28
dt = self.dt * dt_scale
# L31
impulse = self.geometric_noise(x, v, self.christoffel, dt=dt, **kwargs)
# L34
v_stochastic = v_next + impulse
```

#### gfn\integrators\symplectic\coupling.py
```python
# Contexto: forward
# L53
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L62
dt = self.dt * dt_scale
# L71
acc_1 = -self.christoffel(v_dummy, x, force=f_in, **kwargs) + f_in
# L72
v_half = v + 0.5 * dt * acc_1
# L75
x = x + dt * (v_half + warp)
# L80
acc_2 = -self.christoffel(v_dummy, x, force=f_in, **kwargs) + f_in
# L81
v = v_half + 0.5 * dt * acc_2
# Contexto: Global
# L9
v' = v + F(x)  (Shear transformation on v)
# L10
x' = x + G(v') (Shear transformation on x)
```

#### gfn\integrators\symplectic\forest_ruth.py
```python
# Contexto: __init__
# L29
theta = 1.0 / (2.0 - 2.0**(1.0/3.0))
# L31
self.c1 = theta / 2.0
# L32
self.c2 = (1.0 - theta) / 2.0
# L33
self.c3 = (1.0 - theta) / 2.0
# L34
self.c4 = theta / 2.0
# L37
self.d2 = 1.0 - 2.0*theta
# Contexto: acceleration
# L70
c_out = self.christoffel(tv, tx, force=force, **kwargs)
# L81
x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)
# L82
v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)
# L85
x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)
# L86
v2 = v1 + self.d2 * dt * acceleration(x2, v1)
# L89
x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)
# L90
v3 = v2 + self.d3 * dt * acceleration(x3, v2)
# L93
curr_x = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)
# Contexto: forward
# L40
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L56
dt = self.dt * dt_scale
```

#### gfn\integrators\symplectic\leapfrog.py
```python
# Contexto: forward
# L79
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L132
effective_dt = self.dt * dt_scale
# L133
h = 0.5 * effective_dt
# L138
res = self.christoffel(curr_v, curr_x, force=force, **kwargs)
# L166
gate = torch.matmul(feat, Wf.t()) + bf
# L170
gate = gate + torch.matmul(force, Wi.t())
# L171
mu = torch.sigmoid(gate) * FRICTION_SCALE
# L173
v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
# L174
v_norm = v_norm / (curr_v.shape[-1] ** 0.5 + EPSILON_SMOOTH)
# L175
mu = mu * (1.0 + velocity_friction_scale * v_norm)
# L179
v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu + EPSILON_STANDARD)
# L182
curr_x = curr_x + effective_dt * v_half
# L188
res_half = self.christoffel(v_half, curr_x, force=force, **kwargs)
# L203
gate = torch.matmul(feat, Wf.t()) + bf
# L207
gate = gate + torch.matmul(force, Wi.t())
# L208
mu_half = torch.sigmoid(gate) * FRICTION_SCALE
# L210
v_norm = torch.norm(v_half, dim=-1, keepdim=True)
# L211
v_norm = v_norm / (v_half.shape[-1] ** 0.5 + EPSILON_SMOOTH)
# L212
mu_half = mu_half * (1.0 + velocity_friction_scale * v_norm)
# L215
curr_v = (v_half + h * (force - gamma_half)) / (1.0 + h * mu_half + EPSILON_STANDARD)
# Contexto: Global
# L12
v(t+0.5h) = v(t) + 0.5h * a(x(t))
# L13
x(t+h) = x(t) + h * v(t+0.5h)
# L14
v(t+h) = v(t+0.5h) + 0.5h * a(x(t+h))
# L17
v(t+0.5h) = (v(t) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t)))
# L18
x(t+h) = x(t) + h * v(t+0.5h)
# L19
v(t+h) = (v(t+0.5h) + 0.5h * (F - Gamma)) / (1 + 0.5h * mu(x(t+h)))
# L22
- In ABSENCE of friction (mu = 0), energy is conserved
# L24
- VOLUME preservation is LOST when friction != 0
# L57
EPSILON_STANDARD = 1e-7
# L58
EPSILON_SMOOTH = 1e-7
# Contexto: LeapfrogIntegrator
# L70
- Uses updated FRICTION_SCALE=0.02
# L71
- Uses EPSILON_STANDARD=1e-7
```

#### gfn\integrators\symplectic\omelyan.py
```python
# Contexto: __init__
# L28
self.lam = -0.2123418310626054
# L29
self.chi = -0.06626458266981849
# L34
self.c3 = 1.0 - 2.0*(self.chi + self.xi)
# L38
self.d1 = (1.0 - 2.0*self.lam) / 2.0
# L41
self.d4 = (1.0 - 2.0*self.lam) / 2.0
# Contexto: acceleration
# L73
c_out = self.christoffel(tv, tx, force=force, **kwargs)
# L84
x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)
# L85
v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)
# L88
x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)
# L89
v2 = v1 + self.d2 * dt * acceleration(x2, v1)
# L92
x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)
# L93
v3 = v2 + self.d3 * dt * acceleration(x3, v2)
# L96
x4 = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)
# L97
v4 = v3 + self.d4 * dt * acceleration(x4, v3)
# L100
curr_x = apply_boundary_python(x4 + self.c5 * dt * v4, topo_id)
# Contexto: forward
# L43
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L59
dt = self.dt * dt_scale
```

#### gfn\integrators\symplectic\pefrl.py
```python
# Contexto: __init__
# L26
self.lam = -0.2123418310626054
# L27
self.chi = -0.06626458266981849
# Contexto: forward
# L29
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L30
dt = self.dt * dt_scale
# L37
K1 = (1.0 - 2.0 * LAM) / 2.0
# L38
D1 = 1.0 - 2.0 * (CHI + XI)
# Contexto: get_acc
# L53
c_out = self.christoffel(tv, tx, force=force, **kwargs)
# L63
curr_x = apply_boundary_python(curr_x + XI * dt * curr_v, topo_id)
# L66
curr_v = curr_v + K1 * dt * get_acc(curr_x, curr_v, is_first=True)
# L69
curr_x = apply_boundary_python(curr_x + CHI * dt * curr_v, topo_id)
# L72
curr_v = curr_v + LAM * dt * get_acc(curr_x, curr_v)
# L75
curr_x = apply_boundary_python(curr_x + D1 * dt * curr_v, topo_id)
# L78
curr_v = curr_v + LAM * dt * get_acc(curr_x, curr_v)
# L81
curr_x = apply_boundary_python(curr_x + CHI * dt * curr_v, topo_id)
# L84
curr_v = curr_v + K1 * dt * get_acc(curr_x, curr_v)
# L87
curr_x = apply_boundary_python(curr_x + XI * dt * curr_v, topo_id)
```

#### gfn\integrators\symplectic\verlet.py
```python
# Contexto: forward
# L20
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L40
dt = self.dt * dt_scale
# L48
gamma = self.christoffel(v, x, force=force, **kwargs)
# L53
a = -gamma + force
# L56
v_half = v + 0.5 * dt * a
# L59
x = x + dt * v_half
# L65
gamma_next = self.christoffel(v_half, x, force=force, **kwargs)
# L67
a_next = -gamma_next
# L69
a_next = -gamma_next + force
# L71
v = v_half + 0.5 * dt * a_next
```

#### gfn\integrators\symplectic\yoshida.py
```python
# Contexto: __init__
# L23
w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
# L24
w0 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
# L26
self.c1 = w1 / 2.0
# L27
self.c2 = (w0 + w1) / 2.0
# Contexto: acceleration
# L62
c_out = self.christoffel(tv, tx, force=force, **kwargs)
# L73
x1 = apply_boundary_python(curr_x + self.c1 * dt * curr_v, topo_id)
# L74
v1 = curr_v + self.d1 * dt * acceleration(x1, curr_v, is_first=True)
# L77
x2 = apply_boundary_python(x1 + self.c2 * dt * v1, topo_id)
# L78
v2 = v1 + self.d2 * dt * acceleration(x2, v1)
# L81
x3 = apply_boundary_python(x2 + self.c3 * dt * v2, topo_id)
# L82
v3 = v2 + self.d3 * dt * acceleration(x3, v2)
# L85
curr_x = apply_boundary_python(x3 + self.c4 * dt * v3, topo_id)
# Contexto: forward
# L35
def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
# L50
dt = self.dt * dt_scale
```

#### gfn\layers\base.py
```python
# Contexto: __init__
# L54
self.head_dim = dim // heads
# L60
self.depth_scale = 1.0 / (total_depth ** 0.5)
# L69
head_rank = max(4, rank // heads)
# Contexto: create_manifold
# L204
target_dt = self.base_dt / 0.9
# L206
val = val_init + i * 0.1
# L217
gate_in_dim = (3 if self.topology_id == 1 else 2) * self.head_dim
# L255
tolerance = adaptive_cfg.get('tolerance', 1e-3)
# L292
self.out_proj_x = nn.Linear(3 * dim if self.topology_id == 1 else dim, dim)
# Contexto: forward
# L327
m_heads = [None] * self.heads
# L331
force = force + self.context_proj(context)
# L334
f_heads = [None] * self.heads
# L343
dt_min = stability_cfg.get('dt_min', self.base_dt * 0.1)
# L344
dt_max = stability_cfg.get('dt_max', self.base_dt * 4.0)
# L361
scale = dt_base * gates # [Heads, Batch, 1]
# L411
head_dim = self.dim // self.heads
# L422
x_h = x[:, i*head_dim : (i+1)*head_dim]
# L465
dim_v = (2 * self.head_dim) if (self.topology_id == 1) else self.head_dim
# L470
res = recurrent_manifold_fused( x=x, v=v, f=force.unsqueeze(1), U_stack=u_stack, W_stack=w_stack, dt=self.base_dt, dt_scales=dt_scales, forget_rates=None, num_heads=self.heads, topology=self.topology_id, Wf=W_forget_stack.view(-1, W_forget_stack.shape[-1]) if Wf is not None else None, Wi=W_input_stack.view(-1, W_input_stack.shape[-1]) if Wi is not None else None, bf=b_forget_stack.view(-1) if bf is not None else None, V_w=V_w_stack, hysteresis_state=memory_state, hyst_enabled=self.hysteresis_enabled, thermo_alpha=t_alpha, thermo_temp=t_temp, holographic_z=h_z_ten, holographic_grad_z=h_gz_ten, **kwargs )
# L508
extra_kwargs = { 'W_forget_stack': W_forget_stack[i:i+1], # [1, D, D] 'W_input_stack': W_input_stack[i:i+1], 'b_forget_stack': b_forget_stack[i:i+1], 'topology': self.topology_id, 'collect_christ': collect_christ, 'memory_state': m_heads[i] }
# L517
res = self.integrators[i](x_heads[i], v_heads[i], force=f_heads[i], dt_scale=scale[i], **extra_kwargs)
# L540
context_next = gates.squeeze(-1).transpose(0, 1)
# L545
x_cat = torch.stack(x_outs, dim=1).view(batch, -1)
# L546
v_cat = torch.stack(v_outs, dim=1).view(batch, -1)
# L551
v_mix = torch.tanh(v_cat / 100.0)
# L552
mixer_in_x = torch.cat([torch.sin(x_cat), torch.cos(x_cat), v_mix], dim=-1)
# L574
v_next = 100.0 * torch.tanh(v_next / 100.0)
# L576
context_next = gates.squeeze(-1).transpose(0, 1)
```

#### gfn\layers\fractal.py
```python
# Contexto: __init__
# L16
self.head_dim = dim // heads
# L23
self.depth_scale = 1.0 / (total_depth ** 0.5)  # 1/√depth
# L39
self.micro_manifold = MLayer( dim, heads=heads, rank=max(8, rank//2), base_dt=base_dt * 0.5, integrator_type=integrator_type, physics_config=micro_cfg )
# Contexto: forward
# L60
curvature_r = torch.norm(stacked_gamma, dim=-1).mean(dim=-1, keepdim=True) # [batch, 1]
# L65
tunnel_gate = torch.sigmoid((curvature_r - self.threshold) * 1.0)
# L74
x_final = x_m + tunnel_gate * (x_f - x_m) * self.alpha_scale
# L75
v_final = v_m + tunnel_gate * (v_f - v_m) * self.alpha_scale
```

#### gfn\layers\gating.py
```python
# Contexto: __init__
# L14
input_dim = 2 * dim if topology == 1 else dim
# L15
self.curvature_net = nn.Sequential( nn.Linear(input_dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1), nn.Sigmoid() # Range [0, 1] )
# Contexto: forward
# L36
x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# L42
W1 = self.curvature_net[0].weight  # [dim/4, dim]
# L43
b1 = self.curvature_net[0].bias    # [dim/4]
# L44
W2 = self.curvature_net[2].weight  # [1, dim/4]
```

#### gfn\layers\parallel.py
```python
# Contexto: __init__
# L26
def __init__(self, dim, heads=4, physics_config=None, **kwargs):
# L31
self.head_dim = dim // heads
# L73
self.out_proj = nn.Linear(dim * 2, dim * 2)
# Contexto: forward
# L105
dt = self.to_dt(force) * self.base_dt * self.base_dt_scales.view(1, 1, -1)
# L108
B_val = self.to_B(force) * dt
# L118
x_update = v_seq * dt
# Contexto: ParallelMLayer
# L11
dv/dt = F - \\Gamma(v, v)   [Non-linear]
# L14
dv/dt = F - D(F) * v       [Linearized]
# L19
v_t = A_t * v_{t-1} + B_t
# L20
x_t = x_{t-1} + v_t * dt
```

#### gfn\layers\thermo.py
```python
# Contexto: forward
# L50
K = 0.5 * (v ** 2).sum(dim=-1, keepdim=True)
# L54
U = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)
# L65
logits = (self.ref_H - H) / (T * self.sensitivity)
# L67
gate = torch.sigmoid(logits)
# Contexto: ThermodynamicGating
# L18
K(v) = 0.5 * ||v||^2  (Kinetic)
# L19
U(x) = 0.5 * ||x||^2  (Potential - Harmonic Oscillator ansatz)
# L21
gate = sigmoid( (H_ref - H) / Temperature )
```

#### gfn\losses\circular.py
```python
# Contexto: circular_distance_loss
# L17
L = 1 - cos(x_pred - x_target)
# L31
delta = x_pred - x_target
```

#### gfn\losses\combined.py
```python
# Contexto: __init__
# L57
def __init__(self, lambda_h: float = LAMBDA_H_DEFAULT, lambda_g: float = LAMBDA_G_DEFAULT, lambda_k: float = LAMBDA_K_DEFAULT, lambda_c: float = 0.0, lambda_n: float = 0.0, ignore_index: int = -100, hamiltonian_mode: str = 'adaptive', geodesic_mode: str = 'structural'):
# Contexto: forward
# L97
ce = self.ce_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
# L115
total = total + h_loss
# L126
total = total + g_loss
# L132
total = total + c_loss
# L138
total = total + n_loss
# Contexto: Global
# L16
- hamiltonian_mode='none' or 'adaptive'
# L17
- geodesic_mode='structural'
```

#### gfn\losses\curiosity.py
```python
# Contexto: curiosity_loss
# L11
def curiosity_loss(velocities: list, lambda_c: float = 0.05) -> torch.Tensor:
# L23
S = Σ log(std(v_i) + ε)  (Entropy proxy for Gaussian-like latent distribution)
# L49
all_v = torch.cat(velocities, dim=0)  # [Batch * Seq, Dim]
# L53
v_std = all_v.std(dim=0) + 1e-6  # [Dim]
# L57
entropy = torch.log(v_std).sum()
```

#### gfn\losses\geodesic.py
```python
# Contexto: dynamic_loss_balancing
# L134
def dynamic_loss_balancing(loss_components: list, target_ratio: float = 1.0) -> list:
# L158
grad_norm = sum(g.norm() for g in grad if g is not None)
# L169
mean_norm = grad_norms.mean()
# L174
scale = target_ratio * mean_norm / norm
# Contexto: geodesic_regularization
# L35
mode: str = 'structural') -> torch.Tensor:
# L66
mean_gamma = fused_tensor.mean() + GEODESIC_FUSED_SCALE
# L82
curvature_norms = all_curvatures.pow(2).mean()
# L97
curvature_diff = all_curvatures[1:] - all_curvatures[:-1]
# L98
curvature_change_norm = curvature_diff.pow(2).mean()
# L102
magnitude_norm = all_curvatures.pow(2).mean()
# L106
curvature_norms = 0.85 * curvature_change_norm + 0.15 * magnitude_norm
# L111
curvature_var = all_curvatures.var(dim=0)  # [dim]
# L112
curvature_norms = curvature_var.mean()
# L116
curvature_norms = all_curvatures.pow(2).mean()
# L120
batch_mean = all_curvatures.mean()
# L121
batch_std = all_curvatures.std() + 1e-6
# L124
normalized = (all_curvatures - batch_mean) / batch_std
# L125
curvature_norms = normalized.pow(2).mean()
# L129
curvature_norms = all_curvatures.pow(2).mean()
```

#### gfn\losses\hamiltonian.py
```python
# Contexto: hamiltonian_loss
# L29
def hamiltonian_loss(velocities: list, states: list = None, metric_fn=None, lambda_h: float = 0.01, forces: list = None, mode: str = 'adaptive') -> torch.Tensor:
# L66
e = 0.5 * torch.sum(g * v.pow(2), dim=-1)
# L68
e = 0.5 * v.pow(2).sum(dim=-1)
# L77
dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)
# L79
f_norm = forces[i].pow(2).sum(dim=-1)
# L81
force_threshold = 1e-4
# L93
e_next = energies[i + 1]
# L96
diff = torch.abs(e_curr - e_next) / (torch.abs(e_curr) + EPSILON_SMOOTH)
# L104
e_next = energies[i + 1]
# L107
denom = torch.abs(e_curr) + EPSILON_SMOOTH
# L108
rel_change = torch.abs(e_next - e_curr) / denom
# L111
diff = torch.sqrt(rel_change.pow(2) + EPSILON_SMOOTH)
# L117
dE = torch.sqrt((energies[i+1] - energies[i]).pow(2) + EPSILON_SMOOTH)
```

#### gfn\losses\kinetic.py
```python
# Contexto: kinetic_energy_penalty
# L11
def kinetic_energy_penalty(velocities: list, lambda_k: float = 0.001) -> torch.Tensor:
# L26
v_norms = torch.stack([v.pow(2).sum(dim=-1).mean() for v in velocities])
```

#### gfn\losses\noether.py
```python
# Contexto: noether_loss
# L11
def noether_loss(christoffel_outputs: list, isomeric_groups: list = None, lambda_n: float = 0.01) -> torch.Tensor:
# L45
total_diff = total_diff + torch.mean((ref_out - target_out).pow(2))
```

#### gfn\losses\toroidal.py
```python
# Contexto: Global
# L14
diff = min(|x1 - x2|, 2π - |x1 - x2|)
# L31
diff = min(|x1 - x2|, 2π - |x1 - x2|)
# L36
>>> x_pred = torch.tensor([[0.1, 3.1], [2.0, 1.0]])  # On torus [0, 2π) >>> x_target = torch.tensor([[0.2, 3.0], [2.1, 1.1]]) >>> loss = loss_fn(x_pred, x_target) """ import torch import torch.nn as nn from ..geometry.boundaries import toroidal_dist_python def toroidal_distance_loss(x_pred, x_target): """ Toroidal Distance Loss. Computes distance on 3D toroidal manifold (FLAT torus). AUDIT NOTE: This is distance on a flat torus, not the learned manifold. Args: x_pred: Predicted positions [batch, dim] x_target: Target positions [batch, dim] Returns: Toroidal distance loss scalar """ dist = toroidal_dist_python(x_pred, x_target) return dist.pow(2).mean() class ToroidalDistanceLoss(nn.Module): """nn.Module wrapper for toroidal_distance_loss.""" def __init__(self): super().__init__() def forward(self, x_pred, x_target): return toroidal_distance_loss(x_pred, x_target)
```

#### gfn\model\fusion.py
```python
# Contexto: can_fuse
# L30
def can_fuse(self, collect_christ: bool = False) -> bool:
# Contexto: execute_fused_forward
# L317
hyst_enabled: bool = False) -> Optional[Tuple]:
# L358
result = toroidal_leapfrog_fused_autograd( x=x, v=v, f=forces * mask, R=params['major_R'], r=params['minor_r'], dt=params['base_dt'], batch=x.shape[0], seq_len=forces.shape[1], dim=x.shape[1], hysteresis_state=hysteresis_state )
# L386
result = launch_toroidal_leapfrog_fused( x=x, v=v, f=forces * mask, R=params['major_R'], r=params['minor_r'], dt=params['base_dt'], batch=x.shape[0], seq_len=forces.shape[1], dim=x.shape[1], hysteresis_state=hysteresis_state, W_forget=W_f_head0, b_forget=b_f_head0 )
# L417
forget_rates = torch.sigmoid(f_layer.christoffels[0].forget_gate.bias.mean())
# L422
res = recurrent_manifold_fused( x=x, v=v, f=forces * mask, U_stack=params['U_stack'], W_stack=params['W_stack'], dt=params['base_dt'], dt_scales=dt_scales, forget_rates=forget_rates, num_heads=self.model.heads, plasticity=params['plasticity'], sing_thresh=params['sing_thresh'], sing_strength=params['sing_strength'], mix_x=params['mix_x'], mix_v=params['mix_v'], Wf=params['W_f_stack'], Wi=params['W_i_stack'], bf=params['b_f_stack'], Wp=params['W_p_stack'], bp=params['b_p_stack'], topology=params['topology_id'], R=params['major_R'], r=params['minor_r'], mix_x_bias=params['mix_x_bias'], mix_v_bias=params['mix_v_bias'], norm_x_weight=params['norm_x_weight'], norm_x_bias=params['norm_x_bias'], norm_v_weight=params['norm_v_weight'], norm_v_bias=params['norm_v_bias'], gate_W1=params['gate_W1'], gate_b1=params['gate_b1'], gate_W2=params['gate_W2'], gate_b2=params['gate_b2'], integrator_type=1 if self.model.integrator_type == 'leapfrog' else 0, hysteresis_state=hysteresis_state, hyst_update_w=hyst_update_w, hyst_update_b=hyst_update_b, hyst_readout_w=hyst_readout_w, hyst_readout_b=hyst_readout_b, hyst_decay=hyst_decay, hyst_enabled=hyst_enabled, thermo_alpha=params.get('thermo_alpha', 0.0), thermo_temp=params.get('thermo_temp', 1.0) )
# Contexto: prepare_parameters
# L155
unwrap_depth += 1
# L171
U_list.append(torch.zeros(self.model.dim // self.model.heads, 1, device=device))
# L172
W_list.append(torch.zeros(self.model.dim // self.model.heads, 1, device=device))
# L189
W_forget_list.append(torch.zeros(self.model.dim//self.model.heads, h_dim, device=device))
# L190
b_forget_list.append(torch.zeros(self.model.dim//self.model.heads, device=device))
# L191
W_input_list.append(torch.zeros(self.model.dim//self.model.heads, h_dim, device=device))
# L205
p_dim = 2 * (self.model.dim // self.model.heads) if is_torus else (self.model.dim // self.model.heads)
```

#### gfn\model\state.py
```python
# Contexto: from_parameters
# L47
x = x0.expand(batch_size, -1)
# L48
v = v0.expand(batch_size, -1)
```

#### gfn\noise\curiosity.py
```python
# Contexto: CuriosityNoise
# L18
v_new = v + lambda_c * Confusion(F) * eps
# Contexto: forward
# L44
confusion = (force ** 2).mean(dim=-1, keepdim=True) # [Batch, 1]
# L48
scale = self.base_std * (1.0 + self.sensitivity * confusion)
# L51
noise = torch.randn_like(v) * scale
```

#### gfn\noise\geometric.py
```python
# Contexto: forward
# L25
def forward(self, x, v, christoffel_fn, dt=0.1, **kwargs):
# L44
noise = sigma * torch.sqrt(torch.tensor(dt, device=device)) * torch.randn_like(v)
# L79
U_sq_norm = (base_geo.U ** 2).sum(dim=0) # [rank]
# L80
drift = torch.matmul(U_sq_norm, base_geo.W.t()) # [dim]
# L81
drift = (sigma**2 / 2.0) * drift * dt
# Contexto: GeometricNoise
# L13
dv^i = ... + sigma * dW^i + (sigma^2 / 2) * Gamma^i_{jk} * g^{jk}
```

#### gfn\optimizers\manifold_sgd.py
```python
# Contexto: __init__
# L38
def __init__(self, params, lr=1e-2, weight_decay=0.0, max_norm=10.0):
# Contexto: ManifoldSGD
# L27
>>> optimizer = ManifoldSGD(model.parameters(), lr=1e-2)
# L30
>>> optimizer = ManifoldSGD( ...     model.parameters(), ...     lr=1e-2, ...     weight_decay=0.01, ...     max_norm=5.0 ... )
# Contexto: step
# L72
p.data.add_(grad, alpha=-lr)
# L75
norm = p.data.norm()
```

#### gfn\optimizers\riemannian_adam.py
```python
# Contexto: __init__
# L86
def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, retraction='normalize', max_norm=10.0, topology=0):
# Contexto: _project_tangent
# L102
norm_sq = torch.sum(p * p, dim=-1, keepdim=True) + 1e-8
# L103
projection = torch.sum(grad * p, dim=-1, keepdim=True) / norm_sq
# Contexto: _vector_transport
# L135
delta_x = x_new - x_old
# L138
delta_x = torch.atan2(torch.sin(delta_x), torch.cos(delta_x))
# L147
norm = x_new.norm(dim=-1, keepdim=True) + 1e-8
# L148
projection = torch.sum(vec * x_new, dim=-1, keepdim=True) / norm.pow(2)
# Contexto: RiemannianAdam
# L40
Instead of Euclidean gradient descent (W = W - lr * grad), this optimizer
# L44
W_new = Retract(W_old, -lr * corrected_grad)
# L67
>>> optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# L70
>>> optimizer = RiemannianAdam( ...     model.parameters(), ...     lr=1e-3, ...     retraction='normalize', ...     max_norm=10.0 ... )
# L78
>>> optimizer = RiemannianAdam( ...     model.parameters(), ...     lr=1e-3, ...     retraction='torus', ...     topology=1 ... )
# Contexto: step
# L204
state['step'] += 1
# L207
exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
# L208
exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)
# L211
bias_correction1 = 1 - beta1 ** state['step']
# L212
bias_correction2 = 1 - beta2 ** state['step']
# L214
step_size = lr / bias_correction1
# L215
denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
# L217
step_direction = exp_avg / denom
# L223
p.data.add_(step_direction, alpha=-step_size)
# L226
p.data.add_(step_direction, alpha=-step_size)
# L229
norm = p.data.norm()
# L238
p.data.add_(step_direction, alpha=-step_size)
# L248
V = -step_direction * step_size
# L252
retraction_matrix = torch.linalg.solve(I + V/2, I - V/2)
# L255
p.data.add_(step_direction, alpha=-step_size)
# L256
norm = p.data.norm()
# L262
p.data.add_(step_direction, alpha=-lr)
```

#### gfn\readouts\implicit.py
```python
# Contexto: __init__
# L46
in_dim = dim * 2 if self.is_torus else dim
# Contexto: forward
# L78
x_emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
# L83
logits = self.mlp(x_emb) * READOUT_GAIN  # Sharpen logits for better BCE loss
# Contexto: update_step
# L91
self.training_step += 1
```

#### gfn\utils\scan.py
```python
# Contexto: parallel_scan
# L14
"""Compute associative parallel scan: y_t = a_t * y_{t-1} + x_t.
# L22
y_t = a_t * y_{t-1} + x_t  for t > 0
# L44
>>> a = torch.ones(2, 10, 64) * 0.9
# L50
>>> # Each y[t] = 0.9 * y[t-1] + x[t]
# L52
>>> # y[1] = 0.9 * x[0] + x[1]
# L53
>>> # y[2] = 0.9 * (0.9 * x[0] + x[1]) + x[2] = 0.81*x[0] + 0.9*x[1] + x[2]
# L62
- Sequential (L < 32): ~0.1ms for L=16, D=64
# L63
- Parallel (L >= 32): ~0.5ms for L=128, D=64
# L64
- CUDA (if available): ~0.2ms for L=128, D=64
# L102
h = a[:, t] * h + x[:, t]
# L114
steps = int(math.ceil(math.log2(L)))
# L137
new_a = curr_a * prev_a
# L138
new_x = curr_a * prev_x + curr_x
```

#### gfn\utils\visualization.py
```python
# Contexto: visualize_gating
# L6
def visualize_gating(model_path, test_str="88+11="):
```

#### inventory_script.py
```python
# Contexto: _build_destination
# L320
dst_dir = dest_root.joinpath(*parts)
# L321
dst_path = dst_dir / normalized_name
# Contexto: _extract_formulas_python
# L547
lines = file_path.read_text(encoding="utf-8").splitlines(True)
# L550
lines = file_path.read_text(encoding="latin-1", errors="ignore").splitlines(True)
# L555
context_pattern = re.compile(r"^\s*(class|def)\s+([a-zA-Z0-9_]+)")
# Contexto: _hash_full
# L363
def _hash_full(path: Path, chunk: int = 1_048_576) -> str:
# Contexto: _hash_head
# L356
def _hash_head(path: Path, head_bytes: int = 65_536) -> str:
# Contexto: _next_available_path
# L402
digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
# Contexto: _read_text_head
# L155
def _read_text_head(path: Path, max_bytes: int = 24_576) -> str:
# L164
return data.decode("latin-1", errors="ignore")
# Contexto: _safe_path_length
# L198
def _safe_path_length(target: Path, max_total: int = 240) -> Path:
# L204
digest = hashlib.sha1(target_str.encode("utf-8")).hexdigest()[:10]
# L206
room = max(1, max_total - len(parent_str) - 1 - len(ext) - 11)
# Contexto: _setup_logger
# L121
log_path = log_dir / f"organizer_{ts}.log"
# L122
report_path = log_dir / f"organizer_{ts}.jsonl"
# L129
fh = logging.FileHandler(log_path, encoding="utf-8")
# Contexto: _shorten_filename
# L185
digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
# L186
room = max(1, max_len - len(ext) - 11)
# Contexto: _slugify_stem
# L173
stem = re.sub(r"[\s\-]+", "_", stem)
# L174
stem = re.sub(r"[^A-Za-z0-9_\.]+", "_", stem)
# L175
stem = re.sub(r"_+", "_", stem).strip("_.")
# Contexto: _write_report_event
# L144
with open(report_path, "a", encoding="utf-8") as f:
# L145
f.write(json.dumps(event, ensure_ascii=False) + "\n")
# Contexto: Global
# L15
_DEFAULT_IGNORED_DIRS = { ".git", ".hg", ".svn", ".tox", ".pytest_cache", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", "_organized", "organizer_logs", "gfn.egg-info", }
# L106
@dataclass(frozen=True)
# Contexto: organize
# L419
log_dir = root / "organizer_logs"
# L453
duplicates_dir = dest_root / "00_duplicates"
# L475
dup_target = _next_available_path(_safe_path_length(duplicates_dir / normalized_name))
```

#### scripts\inspect_checkpoint.py
```python
# Contexto: inspect_checkpoint
# L33
depth = max(layer_indices) + 1
```

#### scripts\reset_and_check_initial_loss.py
```python
# Contexto: main
# L24
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
# L58
y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
# L61
baseline = (torch.abs(torch.atan2(torch.sin(-y_angle), torch.cos(-y_angle)))**2).mean().item()
# L64
ok = abs(loss - 2.5) <= 0.25
```

#### scripts\train.py
```python
# Contexto: main
# L271
parser.add_argument('--model', type=str, required=True, help="Path to model config YAML")
# L272
parser.add_argument('--training', type=str, required=True, help="Path to training config YAML")
# L273
parser.add_argument('--hardware', type=str, required=True, help="Path to hardware config YAML")
# L274
parser.add_argument('--reset-optimizer', action='store_true', help="Reset optimizer state when resuming")
# Contexto: run_demo
# L98
test_cases = ["42+9=", "131-31=", "50*5=", "999+1=", "123*10="]
# L108
curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
# L113
next_token = torch.argmax(logits[:, -1, :], dim=-1)
# L120
result = dataset.decode(generated).split('=')[-1]
# Contexto: train
# L136
param_count = sum(p.numel() for p in model.parameters()) / 1e6
# L143
optimizer = RiemannianAdam( model.parameters(), lr=train_params.get('learning_rate', 3e-4), weight_decay=train_params.get('weight_decay', 0.01), retraction='normalize', max_norm=10.0 )
# L182
target_tokens = torch.roll(inputs, -1, dims=1)
# L186
mask = 2**torch.arange(coord_dim).to(device)
# L187
bits = (target_tokens.unsqueeze(-1) & mask) > 0
# L188
target_coords = bits.float() * 2 - 1
# L205
pred_positions = x_seq[:, :-1, :]  # [batch, seq-1, dim] aligned for next-token prediction
# L209
half_pi = PI * 0.5
# L210
angle_targets = torch.where( target_coords > 0, torch.full_like(target_coords, half_pi), torch.full_like(target_coords, -half_pi) )
# L215
targ_valid = angle_targets[:, :-1, :]
# L220
valid_mask = (target_tokens[:, :-1] != PAD) & (target_tokens[:, :-1] != EOS)
# L221
mask_exp = valid_mask.unsqueeze(-1).float().expand_as(targ_valid)
# L227
loss_torus = (dist.pow(2) * mask_exp).sum() / torch.clamp(mask_exp.sum() * targ_valid.size(-1), min=1.0)
# L234
loss_phy += geodesic_regularization( christoffels, velocities=None, lambda_g=lambda_g, mode='structural' )
# L242
loss_phy += curiosity_loss([v_final], lambda_c)
# L244
loss = total_loss + loss_phy
# L250
epoch_loss += loss.item()
# L253
dt = time.time() - epoch_t0
# L254
speed = (batch_idx * inputs.shape[0]) / max(dt, 0.01)
# L257
avg_loss = epoch_loss / train_params.get('steps_per_epoch', 1000)
```

#### scripts\verify_cuda_kernels.py
```python
# Contexto: verify_kernels
# L31
v = torch.randn(batch_size, dim, device=device) * scale
# L32
U = torch.randn(dim, rank, device=device) * scale
# L33
W = torch.randn(dim, rank, device=device) * scale
# L43
proj = torch.matmul(v_cpu, U_cpu)
# L44
sq = proj * proj
# L45
gamma_ref = torch.matmul(sq, W_cpu.t())
# L48
gamma_ref_clamped = torch.clamp(gamma_ref, -5.0, 5.0)
# L50
diff = (gamma_cuda.cpu() - gamma_ref_clamped).abs().max().item()
# L75
gamma_ref_base = torch.clamp(gamma_ref, -5.0, 5.0) # Base clamped
# L79
energy = torch.tanh(v.pow(2).mean(dim=-1, keepdim=True))
# L80
plast_factor = 1.0 + plasticity * energy
# L84
potential = torch.sigmoid(torch.matmul(x, V_w.t()))
# L86
sing_factor = 1.0 + is_sing * (sing_strength - 1.0)
# L95
gamma_active_ref = gamma_ref_base.cpu() * plast_factor.cpu() * sing_factor.cpu()
# L97
diff_act = (gamma_cuda_p.cpu() - gamma_active_ref.cpu()).abs().max().item()
# L114
x = torch.randn(batch_size, dim, device=device) * scale
# L115
v = torch.randn(batch_size, dim, device=device) * scale
# L116
f = torch.randn(batch_size, dim, device=device) * scale
# L124
effective_dt = dt * dt_scale
# L132
gamma_v = torch.matmul((torch.matmul(v_cpu, U_cpu)**2), W_cpu.t())
# L133
gamma_v = gamma_v.clamp(-5, 5)
# L135
v_half_ref = v_cpu + 0.5 * effective_dt * (f_cpu - gamma_v)
# L138
x_new_ref = x_cpu + effective_dt * v_half_ref
# L141
gamma_v_half = torch.matmul((torch.matmul(v_half_ref, U_cpu)**2), W_cpu.t())
# L142
gamma_v_half = gamma_v_half.clamp(-5, 5)
# L144
v_new_ref = v_half_ref + 0.5 * effective_dt * (f_cpu - gamma_v_half)
# L146
diff_x = (x_new_cuda.cpu() - x_new_ref).abs().max().item()
# L147
diff_v = (v_new_cuda.cpu() - v_new_ref).abs().max().item()
```

#### scripts\verify_cuda_parity.py
```python
# Contexto: test_cuda_parity
# L73
diff_x = (x_fused_a - x_comp_a).abs().max().item()
# L74
diff_v = (v_fused_a - v_comp_a).abs().max().item()
# L94
diff_x_b = (x_fused_b - x_comp_b).abs().max().item()
# L95
diff_v_b = (v_fused_b - v_comp_b).abs().max().item()
```

#### test_toroidal_autograd.py
```python
# Contexto: test_toroidal_autograd
# L33
v = torch.ones(B, D, device=device, requires_grad=True) * 10.0  # High velocity
# L38
U_stack = torch.randn(num_layers * H * D, rank, device=device) * 0.01
# L39
W_stack = torch.randn(num_layers * H * rank, D, device=device) * 0.01
# L92
TWO_PI = 2 * math.pi
# L93
if max_tor <= TWO_PI + 0.01 and min_tor >= -0.01:
# L102
diff = torch.abs(x_seq_euc - x_seq_tor).mean().item()
# L112
test_values = torch.tensor([0.0, math.pi, 2*math.pi, 3*math.pi, -math.pi, -2*math.pi], device=device)
# L113
wrapped = torch.fmod(test_values, 2 * torch.pi)
# L114
wrapped = torch.where(wrapped < 0, wrapped + 2 * torch.pi, wrapped)
# L117
print("Expected in [0, 2π]:", (wrapped >= 0).all().item() and (wrapped <= 2*math.pi).all().item())
```

#### tests\architecture\conftest.py
```python
# Contexto: __init__
# L22
self.base_path = "D:/ASAS/projects/GFN/.data/metrics/architecture"
# Contexto: compute_hamiltonian
# L51
energy = 0.5 * torch.sum(g * v.pow(2), dim=-1)
# Contexto: estimate_convergence_order
# L60
log_dts = np.log(dts)
# L61
log_errors = np.log(errors)
# L62
coeffs = np.polyfit(log_dts, log_errors, 1)
# Contexto: finish
# L31
self.metrics["duration_seconds"] = time.time() - self.start_time
```

#### tests\architecture\test_architectural_valuation.py
```python
# Contexto: test_pareto_front_flops_vs_accuracy
# L30
duration = (time.time() - start) / 5
```

#### tests\architecture\test_cuda_adjoint.py
```python
# Contexto: get_numerical_grad
# L7
def get_numerical_grad(model, inputs, param_name, eps=1e-3):
# L17
flat_grad = grad_num.view(-1)
# L21
param.data.view(-1)[i] = orig_data.view(-1)[i] + eps
# L23
loss_p = logits_p.pow(2).sum()
# L26
param.data.view(-1)[i] = orig_data.view(-1)[i] - eps
# L28
loss_m = logits_m.pow(2).sum()
# L30
flat_grad[i] = (loss_p - loss_m) / (2 * eps)
# L32
param.data.view(-1)[i] = orig_data.view(-1)[i]
# L36
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# Contexto: resolve_param
# L110
rel_err = torch.norm(analytical - numerical) / (torch.norm(numerical) + 1e-9)
# Contexto: test_cuda_adjoint_consistency
# L72
loss = logits.pow(2).sum()
```

#### tests\architecture\test_differentiability.py
```python
# Contexto: test_toroidal_differentiability
# L17
loss = gamma.pow(2).sum()
```

#### tests\architecture\test_geometries.py
```python
# Contexto: test_christoffel_connection_symmetry
# L52
metrics.log("gamma_norm_mean", torch.norm(gamma_v, dim=-1).mean())
# Contexto: test_hyperbolic_curvature_scaling
# L64
x = torch.ones(1, dim) * 0.1
# L65
v1 = torch.ones(1, dim) * 0.2
# L66
v2 = torch.ones(1, dim) * 0.4
# L71
n1 = torch.norm(g1)
# L72
n2 = torch.norm(g2)
# L75
ratio = (n2 / n1).item()
# Contexto: test_toroidal_metric_properties
# L31
expected_min = (geom.R - geom.r)**2
# L32
assert torch.allclose(g_pi[0, 1], torch.tensor(expected_min), atol=1e-3)
```

#### tests\architecture\test_integrators.py
```python
# Contexto: test_hamiltonian_long_term_stability
# L90
e0 = 0.5 * v.pow(2).sum()
# L91
e_rk = 0.5 * v_rk.pow(2).sum()
# L92
e_vt = 0.5 * v_vt.pow(2).sum()
# L94
drift_rk = torch.abs(e_rk - e0).item()
# L95
drift_vt = torch.abs(e_vt - e0).item()
# Contexto: test_integrator_convergence_order
# L33
x_gt, v_gt = gt_integrator(x0, v0, steps=int(T / dt_gt))
# L39
steps = int(T / dt_val)
# L42
err = torch.norm(x - x_gt).item()
# Contexto: test_symplectic_phase_space_conservation
# L67
energy_init = 0.5 * (v.pow(2).sum() + x.pow(2).sum())
# L68
energy_final = 0.5 * (v_next.pow(2).sum() + x_next.pow(2).sum())
# L70
drift = torch.abs(energy_final - energy_init).item()
```

#### tests\architecture\test_learning_dynamics.py
```python
# Contexto: test_gradient_flow_curvature
# L26
loss = gamma.pow(2).sum()
# L29
grad_x_norm = x.grad.norm().item()
# L30
grad_v_norm = v.grad.norm().item()
# Contexto: test_hessian_spectrum_proxy
# L55
gz = (gamma * z).sum()
# L57
trace_est = (grad * z).sum().item()
# L60
avg_trace = sum(traces) / len(traces)
```

#### tests\benchmarks\aggregation_comparison.py
```python
# Contexto: __init__
# L35
self.half_pi = self.PI * 0.5
# Contexto: forward
# L95
x_pred_agg[:, -1] = x_agg
# Contexto: generate_batch
# L51
c = int(parts[-1]) if len(parts) > 1 else 0
# L60
y_angle[:, -1] = (y_class.float() * 2.0 - 1.0) * self.half_pi
# Contexto: main
# L219
print("\n" + "="*80)
# L237
print("\n" + "="*80)
# L255
print("\n" + "="*80)
# L273
print("\n" + "="*80)
# L278
final_acc = history['acc'][-1]
# L279
final_loss = history['loss'][-1]
# L295
print("\n" + "="*80)
# Contexto: train_model
# L158
optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction='euclidean')
# L165
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
# Contexto: train_step
# L119
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# L122
x_last = x_pred[:, -1]
# L123
y_last = y_expanded[:, -1]
# L126
loss_val = dist.pow(2).mean()
# L141
TWO_PI = 2.0 * PI
# L142
half_pi = PI * 0.5
# L144
x_last = x_pred[:, -1]
# L146
dist_pos = torch.min(torch.abs(x_last - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last - half_pi) % TWO_PI))
# L147
dist_neg = torch.min(torch.abs(x_last + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last + half_pi) % TWO_PI))
# L148
d_pos = dist_pos.mean(dim=-1)
# L149
d_neg = dist_neg.mean(dim=-1)
# L152
acc = (preds == targets_class).float().mean().item()
```

#### tests\benchmarks\baselines.py
```python
# Contexto: __init__
# L20
layer = nn.TransformerEncoderLayer( d_model=dim, nhead=heads, dim_feedforward=4*dim, dropout=0.1, activation='gelu', batch_first=True, norm_first=True )
# L82
self.d_inner = dim * expand
# L133
self.d_inner = d_model * expand
# L134
self.dt_rank = math.ceil(d_model / 16)
# L136
self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
# L138
self.conv1d = nn.Conv1d( in_channels=self.d_inner, out_channels=self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, )
# L147
self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
# L150
A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
# L151
self.A_log = nn.Parameter(torch.log(A))
# Contexto: forward
# L53
x = self.token_emb(idx) + self.pos_emb[:, :t, :]
# L58
mask = torch.triu(torch.ones(t, t, device=idx.device) * float('-inf'), diagonal=1)
# L123
current_state_idx += 1
# L162
xz = self.in_proj(x) # [B, L, 2*D_in]
# L163
x_in, z = xz.chunk(2, dim=-1) # [B, L, D_in]
# L170
new_conv_state = x_in[:, -self.conv1d.kernel_size[0]+1:, :] # Last K-1 tokens
# L179
pad_len = self.conv1d.kernel_size[0] - 1
# L186
x_conv = x_conv[:, -1:, :] # Take only last output
# L187
new_conv_state = conv_input[:, -pad_len:, :]
# L194
x_dbl = self.x_proj(x_ssm) # [B, L, dt_rank + 2*d_state]
# L195
dt, B, C = torch.split(x_dbl, [self.dt_rank, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
# L200
A = -torch.exp(self.A_log) # [D_in, D_state]
# L211
dt_t = dt[:, t, :].unsqueeze(-1) # [B, D_in, 1]
# L212
dA = torch.exp(dt_t * A) # [B, D_in, D_state]
# L214
x_t = x_ssm[:, t, :].unsqueeze(-1) # [B, D_in, 1]
# L218
dB = (dt_t * x_t) * B_t # [B, D_in, D_state]
# L222
y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1) # [B, D_in]
# L226
y = y + x_ssm * self.D
# L236
dt_t = dt.unsqueeze(-1) # [B, 1, D_in, 1]
# L237
dA = torch.exp(dt_t * A) # [B, 1, D_in, D_state]
# L241
x_t = x_ssm.unsqueeze(-1) # [B, 1, D_in, 1]
# L245
h = h.unsqueeze(1) * dA + B_t * x_t
# L250
y = (h * C.unsqueeze(2)).sum(dim=-1) # [B, D_in]
# L251
y = y + x_ssm.squeeze(1) * self.D
# L257
out = y * self.act(z)
```

#### tests\benchmarks\bench_utils.py
```python
# Contexto: __init__
# L20
self.results_dir = self.root / "results" / category / benchmark_name
# Contexto: generate_batch
# L111
y_angle = y_int.float() * PI
# Contexto: get_model_size_mb
# L68
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
# L69
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
# Contexto: save_json
# L48
path = self.results_dir / filename
# L49
with open(path, 'w', encoding='utf-8') as f:
# Contexto: save_plot
# L56
path = self.results_dir / filename
```

#### tests\benchmarks\benchmark_cuda_live.py
```python
# Contexto: benchmark_live
# L41
U = torch.randn(dim, rank, device=device) * 0.01
# L42
W = torch.randn(dim, rank, device=device) * 0.01
# L56
print(f"[*] Starting Benchmark Loop (Batch={batch_size}, Dim={dim})")
# L75
gamma = (h**2) @ W.t()
# L76
x = x + 0.01 * v
# L77
v = v + 0.01 * (f - gamma)
# L86
total_time = t1 - start_time
# L87
avg_ips = iter_count / total_time
# L90
sps = (iter_count * steps) / total_time
```

#### tests\benchmarks\benchmark_scan.py
```python
# Contexto: benchmark_mlayers
# L39
par_time = (time.time() - t0) / 5
# L41
print(f"L={L:4d} | Parallel: {par_time*1000:.2f}ms")
```

#### tests\benchmarks\core\benchmark_baseline_comparison.py
```python
# Contexto: first_head_metric
# L111
total_loss = loss_val + loss_phy + loss_ham
# L122
TWO_PI = 2.0 * PI
# L123
half_pi = PI * 0.5
# L124
dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
# L125
dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
# L126
d_pos = dist_pos.mean(dim=-1)
# L127
d_neg = dist_neg.mean(dim=-1)
# L129
acc = (preds == targets_class).float().mean().item()
# Contexto: generate_batch
# L78
y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)
# Contexto: run_baseline_comparison
# L161
models_to_test = { "Manifold-GFN": Manifold( vocab_size=vocab_size, dim=dim, depth=6, heads=4, integrator_type='leapfrog', physics_config=physics_config, impulse_scale=80.0, holographic=True ), "Vanilla GRU": SimpleGRU(vocab_size, dim, depth), "Vanilla LSTM": SimpleLSTM(vocab_size, dim, depth) }
# L184
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# L186
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# L211
loss = criterion(output.view(-1, vocab_size), targets_int.view(-1))
# L217
preds = output.argmax(dim=-1)
# L218
acc = (preds == targets_int).float().mean().item()
# L225
progress.update(train_task, description=f"L: {loss.item():.4f} A: {acc*100:.1f}%")
# L227
elapsed = time.time() - start_time
# L228
final_acc = np.mean(history["acc"][-10:]) * 100
# Contexto: train_step_manifold
# L91
y_expanded = targets.float().unsqueeze(-1).expand_as(x_pred)
```

#### tests\benchmarks\core\benchmark_composition.py
```python
# Contexto: __init__
# L41
chars = [str(i) for i in range(10)] + ['+', '-', '*', '=', 'f', 'g', 'h', '(', ')', '<PAD>', '<EOS>']
# L46
self.funcs = { 'f': lambda x: x + 2, 'g': lambda x: x * 3, 'h': lambda x: x - 1, }
# Contexto: generate_problem
# L60
func_name = np.random.choice(['f', 'g', 'h'])
# L61
x = np.random.randint(0, 30)
# L65
length = np.random.choice([2, 3])
# L66
composition = ''.join(np.random.choice(['f', 'g', 'h'], size=length))
# L67
x = np.random.randint(0, 5)
```

#### tests\benchmarks\core\benchmark_feature_ablation.py
```python
# Contexto: create_associative_recall_data
# L36
sep_token = vocab_size - 1
# L41
keys = torch.randint(0, vocab_size - 1, (num_pairs,))
# L42
values = torch.randint(0, vocab_size - 1, (num_pairs,))
# L54
seq += [sep_token] * (seq_len - len(seq))
# Contexto: run_benchmark
# L189
ax.text(v + 1, i, f"{v:.1f}%", color='white', va='center', fontweight='bold')
# Contexto: train_and_evaluate
# L105
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# L128
pred_logits = logits[:, -1, :]
# L135
acc = (pred_logits.argmax(dim=-1) == targets).float().mean().item()
# L141
final_acc = np.mean(history["acc"][-20:]) * 100
# L142
final_loss = np.mean(history["loss"][-20:])
```

#### tests\benchmarks\core\benchmark_integrators.py
```python
# Contexto: measure_drift
# L52
x = torch.zeros(batch_size, model.dim // model.heads).to(device)
# L53
v = torch.randn(batch_size, model.dim // model.heads).to(device)
# L54
v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
# L56
v_start_norm = v.norm(dim=-1).mean().item()
# L68
v_end_norm = tv.norm(dim=-1).mean().item()
# L69
drift = abs(v_end_norm - v_start_norm) / (v_start_norm + 1e-12) * 100
# Contexto: run_integrator_suite
# L96
progress.update(suite_task, description=f"Testing: [bold blue]{integ}[/]")
# L111
tput = (20 * 16) / (time.time() - start)
# L132
summary_table.add_column("Speed (seq/s)", justify="right")
# L160
sns.barplot(data=df, x="Integrator", y="Speed (seq/s)", ax=axes[1], palette="crest")
```

#### tests\benchmarks\core\benchmark_learning_dynamics.py
```python
# Contexto: _plot_convergence_comparison
# L269
x = np.arange(len(thresholds))
# L272
bars1 = ax.bar(x - width/2, gfn_epochs, width, label='GFN', color='#2A9D8F', alpha=0.8)
# L273
bars2 = ax.bar(x + width/2, gpt_epochs, width, label='Transformer', color='#E76F51', alpha=0.8)
# L295
plt.savefig(self.results_dir / "convergence_speed_comparison.png", dpi=300, bbox_inches='tight')
# Contexto: _plot_efficiency_metrics
# L303
gfn_avg_time = np.mean(self.history['Manifold']['time'])
# L304
gpt_avg_time = np.mean(self.history['Transformer']['time'])
# L306
gfn_final_acc = self.history['Manifold']['acc'][-1]
# L307
gpt_final_acc = self.history['Transformer']['acc'][-1]
# L310
gfn_efficiency = gfn_final_acc / (gfn_avg_time + 1e-6)
# L311
gpt_efficiency = gpt_final_acc / (gpt_avg_time + 1e-6)
# L325
ax.set_ylabel('Efficiency (Accuracy % / Sec per Epoch)', fontsize=13)
# L330
plt.savefig(self.results_dir / "training_efficiency.png", dpi=300, bbox_inches='tight')
# Contexto: evaluate
# L58
prompt = parts[0] + '='
# L68
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# L73
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# L75
if tok_id == dataset.char_to_id.get('<EOS>', -1): break
# L82
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# L84
if tok_id == dataset.char_to_id.get('<EOS>', -1): break
# L87
pred = dataset.decode(generated).split('=')[-1].strip()
# L88
if pred == target: correct += 1
# Contexto: first_head_metric
# L125
total_loss = loss_val + loss_phy + loss_ham
# Contexto: plot_results
# L247
ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')
# L255
plt.savefig(self.results_dir / "learning_curves_comparison.png", dpi=300, bbox_inches='tight')
# Contexto: run_showdown
# L169
m_opt = RiemannianAdam(manifold.parameters(), lr=1e-3)
# L170
g_opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=0.01)
# L171
criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_id.get('<PAD>', -1))
# L182
m_task = progress.add_task("Manifold-GFN   ", total=epochs)
# L194
ids = [dataset.char_to_id[c] for c in p + '<EOS>']
# L199
padded_in = torch.tensor([s + [0]*(max_len-len(s)) for s in inputs]).to(self.device)
# L200
padded_tg = torch.tensor([s + [-100]*(max_len-len(s)) for s in targets]).to(self.device)
# L209
g_loss = criterion(g_logits.reshape(-1, vocab_size), padded_tg.reshape(-1))
# L214
if epoch % 2 == 0 or epoch == epochs - 1:
# Contexto: train_step_manifold
# L105
targets_expanded = targets_float.unsqueeze(-1).expand_as(x_pred)
```

#### tests\benchmarks\core\benchmark_needle_haystack.py
```python
# Contexto: run_inference
# L87
pred = logits[0, -1, :8].argmax()
# L100
console.print(f"  [red]OOM at L={L}[/]")
# L111
acc_str = "[green]SUCCESS[/]" if r["Accuracy"] > 0 else "[red]FAIL[/]"
# L124
ax.plot(df["Length"], df["VRAM (MB)"], 'o-', label="Manifold (O(1) Scaling)", color='#00ADB5', lw=3, markersize=8)
# L129
x_theory = np.logspace(np.log10(base_l), np.log10(df.iloc[-1]["Length"]), 50)
# L130
y_theory = base_v + (x_theory/base_l)**2 * (base_v * 0.5)
# L131
ax.plot(x_theory, y_theory, '--', label="Transformer (O(N²) Theory)", color='#FF2E63', alpha=0.5)
# L145
v_start, v_end = df.iloc[0]["VRAM (MB)"], df.iloc[-1]["VRAM (MB)"]
# L146
increase = ((v_end - v_start) / v_start) * 100
# Contexto: run_needle_haystack
# L51
model = Manifold( vocab_size=64, dim=256, depth=4, heads=4, integrator_type='yoshida' # High-precision for long-term transport ).to(device)
```

#### tests\benchmarks\core\benchmark_ood.py
```python
# Contexto: evaluate_accuracy
# L43
prompt = parts[0] + '='
# L52
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# L58
curr_token = torch.argmax(logits[:, -1, :], dim=-1)
# L60
if tok_id == dataset.char_to_id.get('<EOS>', -1): break
# L63
pred_res = dataset.decode(generated).split('=')[-1].strip()
# Contexto: run_ood_suite
# L100
progress.update(ood_task, description=f"Testing {d}-digit Addition")
# L106
"Complexity": "In-Dist" if d <= 2 else "OOD"
# L129
ax.axvline(x=0.5, color='#FF2E63', lw=2, ls='--', label='Training Boundary')
# L130
ax.set_title("Manifold-GFN Systemic Generalization", color='white', fontweight='bold')
```

#### tests\benchmarks\core\benchmark_overhead.py
```python
# Contexto: measure_overhead
# L90
elapsed = time.time() - start
# L91
tput = (50 * batch_size) / elapsed
# L92
lat = (elapsed / 50) * 1000
# Contexto: run_benchmark
# L134
table.add_column("Throughput (seq/s)", justify="right")
# L155
sns.barplot(data=df, x="Configuration", y="Throughput (seq/s)", ax=axes[1], palette="crest")
```

#### tests\benchmarks\core\benchmark_performance.py
```python
# Contexto: measure_efficiency
# L59
elapsed = time.time() - start
# L60
tput = (20 * batch_size) / elapsed
# Contexto: run_performance_suite
# L101
models = { "Manifold-GFN ($O(1)$)": Manifold( vocab, dim, depth, heads, integrator_type='leapfrog', physics_config=physics_config, impulse_scale=80.0, holographic=True ).to(device), "Transformer ($O(N^2)$)": MicroGPT(vocab, dim, depth, heads).to(device) }
# L128
perf_task = progress.add_task("Profiling Models...", total=len(models) * len(seq_lengths))
# L141
progress.update(perf_task, advance=1, description=f"Profiling Models... [cyan]{name} L={L}[/]")
# L145
console.print(f"  [red]{name} OOM at L={L}[/]")
# L146
progress.update(perf_task, advance=len(seq_lengths) - seq_lengths.index(L) - 1) # Advance for remaining lengths of this model
# L154
table.add_column("Throughput (seq/s)", justify="right")
```

#### tests\benchmarks\core\benchmark_precision_stability.py
```python
# Contexto: evaluate_stability
# L57
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# L67
target = y.float().unsqueeze(-1).expand(-1, -1, 128) # Matching dim for stability test
```

#### tests\benchmarks\core\benchmark_sample_efficiency.py
```python
# Contexto: run_sample_efficiency
# L155
ax.plot(df["Samples"], df["Manifold Acc"], 'o-', label="Manifold-GFN", color='#00ADB5', lw=3)
# L156
ax.plot(df["Samples"], df["Transformer Acc"], 's-', label="Transformer", color='#FF2E63', lw=3, ls='--')
# Contexto: train_and_eval
# L45
opt = RiemannianAdam(model.parameters(), lr=1e-3)
# L48
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# L50
criterion = nn.CrossEntropyLoss(ignore_index=test_ds.char_to_id.get('<PAD>', -1))
# L58
ids = [test_ds.char_to_id[c] for c in p + '<EOS>']
# L62
p_in = torch.tensor([s + [0]*(max_len-len(s)) for s in ins]).to(device)
# L63
p_tg = torch.tensor([s + [-100]*(max_len-len(s)) for s in tgs]).to(device)
# L68
loss = criterion(logits.reshape(-1, vocab), p_tg.reshape(-1))
# L80
prompt, target = parts[0] + '=', parts[1].strip()
# L92
tok = torch.argmax(logits[:, -1, :], dim=-1)
# L93
if tok.item() == test_ds.char_to_id.get('<EOS>', -1): break
# L97
pred = test_ds.decode(gen).split('=')[-1].strip()
# L98
if pred == target: correct += 1
```

#### tests\benchmarks\core\benchmark_scaling.py
```python
# Contexto: measure_scale_metrics
# L78
elapsed = time.time() - start
# L79
tput = (10 * 8) / elapsed
# Contexto: run_benchmark
# L132
table.add_column("Throughput (seq/s)", justify="right")
# L149
sns.lineplot(data=df, x="Config", y="Throughput (seq/s)", ax=ax0_twin, marker='o', color='gold', lw=3)
# L153
sns.regplot(data=df, x="Params (M)", y="VRAM (MB)", ax=axes[1], scatter_kws={'s':100}, line_kws={'color':'red', 'ls':'--'})
```

#### tests\benchmarks\core\run_validation_suite.py
```python
# Contexto: Global
# L25
BENCHMARK_DIR = PROJECT_ROOT / "tests/benchmarks/core"
# L26
RESULTS_BASE = PROJECT_ROOT / "tests/results/core"
# L30
BENCHMARKS = { 'baseline': { 'script': 'benchmark_baseline_comparison.py', 'desc': 'Systematic comparison vs RNNs (GRU/LSTM)' }, 'composition': { 'script': 'benchmark_composition.py', 'desc': 'Function composition & systematic generalization' }, 'ablation': { 'script': 'benchmark_feature_ablation.py', 'desc': 'Physics feature value-add audit' }, 'integrators': { 'script': 'benchmark_integrators.py', 'desc': 'Numerical Drift & Symplectic Stability' }, 'learning': { 'script': 'benchmark_learning_dynamics.py', 'desc': 'GFN vs Transformer on Arithmetic' }, 'needle': { 'script': 'benchmark_needle_haystack.py', 'desc': '1M Token Long-Context Recall' }, 'ood': { 'script': 'benchmark_ood.py', 'desc': 'Out-of-Distribution Math Generalization' }, 'overhead': { 'script': 'benchmark_overhead.py', 'desc': 'Physics Engine Computational Cost' }, 'performance': { 'script': 'benchmark_performance.py', 'desc': 'Throughput & VRAM Scaling Laws' }, 'precision': { 'script': 'benchmark_precision_stability.py', 'desc': 'Numerical format robustness (FP16/BF16)' }, 'efficiency': { 'script': 'benchmark_sample_efficiency.py', 'desc': 'Data efficiency vs Transformers' }, 'scaling': { 'script': 'benchmark_scaling.py', 'desc': 'Model size expansion laws' } }
# Contexto: main
# L131
parser.add_argument('--all', action='store_true', help='Run every benchmark')
# L132
parser.add_argument('--only', nargs='+', help='Run specific benchmarks')
# L133
parser.add_argument('--status', action='store_true', help='Show coverage status')
# L156
elapsed = time.time() - start_time
# L159
console.print("\n" + "="*60)
# L162
console.print("="*60 + "\n")
# Contexto: run_bench
# L85
script = BENCHMARK_DIR / info['script']
# Contexto: show_summary
# L122
res_path = RESULTS_BASE / name
# L123
status = "[green]RUN[/]" if res_path.exists() else "[dim]PENDING[/]"
```

#### tests\benchmarks\core\test_arithmetic.py
```python
# Contexto: __init__
# L36
self.op_map = {'+': 10, '-': 11, '=': 13, '<PAD>': 14}
# Contexto: generate
# L41
a, b = np.random.randint(0, 5), np.random.randint(0, 5)
# L44
prob = [a, self.op_map['+'], b, self.op_map['='], res]
# Contexto: run_arithmetic_benchmark
# L64
opt = RiemannianAdam(model.parameters(), lr=1e-3)
# L87
loss = crit(logits[:, -1, :], y)
# L91
acc = (logits[:, -1, :].argmax(dim=-1) == y).float().mean().item()
# L94
progress.update(train_task, advance=1, description=f"Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")
```

#### tests\benchmarks\generate_report.py
```python
# Contexto: generate_html_report
# L422
results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
# L427
geodesic_status = "N/A"
# L430
energy_drift = results['energy']['drift']['relative_drift'] * 100
# L443
tests_passed = sum([ results.get('energy') is not None, results.get('geodesic') is not None, results.get('benchmark') is not None ]) * 3  # Each module has ~3 tests
# L449
pass_rate = (tests_passed / tests_total) * 100
# L458
path = results_dir / "energy" / img_name # Future-proof: assuming they go here
# L459
if not path.exists(): path = results_dir / img_name # Fallback
# L460
energy_images += generate_image_card(path, caption)
# L468
path = results_dir / img_name
# L469
benchmark_images += generate_image_card(path, caption)
# L473
viz_list = [ ("geodesic_flow/geodesic_flow_3d.png", "3D Geodesic Flow Trajectory (Reasoning Path)"), ("trajectories/trajectory_comparison.png", "Manifold vs Transformer: Smoothness Comparison"), ("loss_landscape/loss_landscape_3d_comparison.png", "Loss Landscape: Convexity Analysis"), ("fractals/fractal_zoom_comparison.png", "Fractal Recursive Tunneling (Zoom)"), ("manifold_curvature/vis_manifold.png", "Learned Manifold Curvature Heatmap"), ("christoffel_vector_field/christoffel_vector_field.png", "Christoffel Force Vector Field"), ("internal_physics/xray_analysis.png", "Internal Physics X-Ray (Hamiltonian & Fractal Activity)"), ("symmetries/noether_invariance.png", "Noether Invariance (Semantic Symmetries)"), ("active_inference_distortion.png", "Active Inference: Curiosity-Driven Manifold Distortion"), ]
# L486
path = results_dir / img_name
# L489
flat_name = img_name.split('/')[-1]
# L491
path = results_dir / flat_name
# L493
manifold_images += generate_image_card(path, caption)
# L496
html = HTML_TEMPLATE.format( timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), energy_drift=energy_drift, energy_status=energy_status, energy_verdict=energy_verdict, energy_badge=energy_badge, stability_score=stability_score, geodesic_status=geodesic_status, tests_passed=tests_passed, tests_total=tests_total, pass_rate=pass_rate, energy_images=energy_images, benchmark_images=benchmark_images, manifold_images=manifold_images )
# L513
with open(output_path, 'w', encoding='utf-8') as f:
# Contexto: generate_image_card
# L347
rel_path = os.path.relpath(path, PROJECT_ROOT / "tests" / "benchmarks" / "results")
# L349
<div class="image-card">
# L351
<div class="caption">{caption}</div>
# Contexto: Global
# L39
<meta charset="UTF-8">
# L40
<meta name="viewport" content="width=device-width, initial-scale=1.0">
# L243
<div class="subtitle">Geodesic Flow Networks - Professional Validation</div>
# L244
<div class="timestamp">Generated: {timestamp}</div>
# L249
<div class="metrics-grid">
# L250
<div class="metric-card">
# L251
<div class="label">Energy Conservation</div>
# L252
<div class="value">{energy_drift:.2f}%</div>
# L257
<div class="metric-card" style="background: linear-gradient(135deg, #E76F51 0%, #F4A261 100%);">
# L258
<div class="label">Memory Scaling</div>
# L259
<div class="value">O(1)</div>
# L260
<div class="status pass">✅ Verified</div>
# L262
<div class="metric-card" style="background: linear-gradient(135deg, #264653 0%, #2A9D8F 100%);">
# L263
<div class="label">Geodesic Optimality</div>
# L264
<div class="value">{geodesic_status}</div>
# L265
<div class="status pass">✅ Confirmed</div>
# L267
<div class="metric-card" style="background: linear-gradient(135deg, #F4A261 0%, #E9C46A 100%);">
# L268
<div class="label">Tests Passed</div>
# L269
<div class="value">{tests_passed}/{tests_total}</div>
# L270
<div class="status pass">{pass_rate:.0f}%</div>
# L279
<h3 style="margin-top: 30px; color: var(--primary);">Energy Conservation Results</h3>
# L280
<table class="summary-table">
# L292
<td><span class="badge {energy_badge}">{energy_verdict}</span></td>
# L297
<td><span class="badge success">Excellent</span></td>
# L302
<td><span class="badge success">Stable</span></td>
# L307
<div class="image-grid">
# L316
<div class="image-grid">
# L325
<div class="image-grid">
# L333
<p style="margin-top: 10px; font-size: 0.9em;">
# L335
<a href="https://github.com/WitWise/MANIFOLD.git" style="color: var(--primary);">GitHub</a>
# Contexto: main
# L521
parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
# L523
parser.add_argument('--output', type=str, default=None, help='Output HTML file path')
# L531
elapsed = time.time() - start_time
# L534
output_path = args.output or (PROJECT_ROOT / "tests" / "benchmarks" / "results" / "report.html")
# L537
print("\n" + "=" * 70)
# Contexto: run_full_suite
# L368
print("\n" + "="*70)
# L391
print("\n" + "="*70)
# L405
print("\n" + "="*70)
```

#### tests\benchmarks\validation\verify_docs.py
```python
# Contexto: main
# L63
print(f"\n{'='*60}")
# L82
print(f"Throughput: {throughput:.2f} seq/s (batch=32, seq=128)")
# L87
results[cfg['name']] = { "params": params, "params_M": round(params/1e6, 2), "peak_vram_gb": round(peak_mem, 3), "throughput": round(throughput, 2) }
# L102
print("\n" + "="*60)
# L107
res_path = PROJECT_ROOT / "tests/benchmarks/results/verification.json"
# Contexto: measure_memory_and_throughput
# L43
peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
# L44
throughput = (runs * batch_size) / (end - start)
```

#### tests\benchmarks\viz\math.py
```python
# Contexto: __init__
# L113
self.half_pi = self.PI * 0.5
# Contexto: generate_batch
# L129
c = int(parts[-1]) if len(parts) > 1 else 0
# L138
y_angle[:, -1] = (y_class.float() * 2.0 - 1.0) * self.half_pi  # Only last position
# Contexto: print_header
# L376
console.print("\n" + "="*80, style="magenta")
# L377
console.print("  [bold cyan]GFN MATH COMPLEXITY BENCHMARK[/] - [italic]PRODUCTION CONFIG (2026-02-07)[/]", justify="center")
# L378
console.print("="*80, style="magenta")
# L385
base_dt = stability_cfg.get('base_dt', 'N/A')
# L386
sing_strength = singularities_cfg.get('strength', 'N/A')
# L387
sing_threshold = singularities_cfg.get('threshold', 'N/A')
# L389
lambda_g = loss_config.get('lambda_g', 'N/A')
# L392
console.print(f"    - Topology: {topology}, dynamic_time={'on' if dynamic_time else 'off'}")
# L393
console.print(f"    - base_dt={base_dt}")
# L394
console.print(f"    - singularity_strength={sing_strength}, threshold={sing_threshold}")
# L395
console.print(f"    - hamiltonian_mode={ham_mode}, lambda_g={lambda_g}")
# L396
console.print("="*80 + "\n", style="magenta")
# Contexto: run_inf
# L320
out = model(x[:, t:t+1], state=state)
# L324
TWO_PI = 2.0 * PI
# L325
half_pi = PI * 0.5
# L326
dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))
# L327
dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))
# L328
d_pos = dist_pos.mean(dim=-1).view(-1)
# L329
d_neg = dist_neg.mean(dim=-1).view(-1)
# L333
return model(x).argmax(dim=-1)
# L338
acc = (preds == y_class).float().mean().item()
# L343
acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"
# L353
max_length = lengths[i-1] if i > 0 else 0
# L361
max_length = lengths[i-1] if i > 0 else 0
# Contexto: run_production_benchmark
# L412
optimizer_label = "AdamW + OneCycleLR"
# L432
h_m = train_model( "Manifold-GFN-PRODUCTION", manifold, max_steps=1000, device=device, loss_config=PRODUCTION_LOSS_CONFIG, retraction='normalize', optimizer_type='adamw' )
# L444
s_m = evaluate_scaling("Manifold-GFN-PRODUCTION", manifold, lengths, device)
# L476
ce_smooth = np.convolve(h_m["loss_breakdown"]['ce'], np.ones(20)/20, mode='valid')
# L477
ham_smooth = np.convolve(h_m["loss_breakdown"]['hamiltonian'], np.ones(20)/20, mode='valid')
# L478
geo_smooth = np.convolve(h_m["loss_breakdown"]['geodesic'], np.ones(20)/20, mode='valid')
# L479
ax.plot(ce_smooth, color='#00ADB5', label='Cross-Entropy', linewidth=2)
# L488
ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN (Production)', linewidth=5, markersize=12, markerfacecolor='white')
# L497
ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Production)', linewidth=5, markersize=12, markerfacecolor='white')
# L508
summary_table = Table(title="[bold yellow]MATH COMPLEXITY SUMMARY (PRODUCTION)[/]", border_style="magenta", show_header=True, header_style="bold cyan")
# L510
summary_table.add_column("Manifold-GFN-PRODUCTION", justify="center")
# L514
acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0
# L516
m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"
# L517
target_l = lengths[-1]
# Contexto: train_model
# L219
optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction=retraction)
# L224
optimizer = optim.AdamW([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ])
# L229
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# L231
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
# Contexto: train_step_manifold
# L166
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# L187
total_loss = loss_val + loss_phy + loss_ham
# L195
TWO_PI = 2.0 * PI
# L196
half_pi = PI * 0.5
# L198
x_last = x_pred[:, -1]
# L200
dist_pos = torch.min(torch.abs(x_last - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last - half_pi) % TWO_PI))
# L201
dist_neg = torch.min(torch.abs(x_last + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_last + half_pi) % TWO_PI))
# L202
d_pos = dist_pos.mean(dim=-1)
# L203
d_neg = dist_neg.mean(dim=-1)
# L206
acc = (preds == targets_class).float().mean().item()
```

#### tests\benchmarks\viz\math2.py
```python
# Contexto: __init__
# L48
self.half_pi = self.PI * 0.5
# Contexto: first_head_metric
# L102
total_loss = loss_val + loss_phy + loss_ham
# L113
TWO_PI = 2.0 * PI
# L114
half_pi = PI * 0.5
# L117
dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
# L118
dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
# L119
d_pos = dist_pos.mean(dim=-1)
# L120
d_neg = dist_neg.mean(dim=-1)
# L122
acc = (preds == targets_class).float().mean().item()
# Contexto: generate_batch
# L62
c = int(parts[-1]) if len(parts) > 1 else 0
# L69
y_angle = (y_class_seq.float() * 2.0 - 1.0) * self.half_pi
# Contexto: print_header
# L275
console.print("\n" + "="*80, style="magenta")
# L276
console.print("  [bold cyan]GFN MATH COMPLEXITY BENCHMARK[/] - [italic]Holographic Manifold[/]", justify="center")
# L277
console.print("="*80, style="magenta")
# L280
console.print("="*80 + "\n", style="magenta")
# Contexto: run_inf
# L220
out = model(x[:, t:t+1], state=state)
# L224
TWO_PI = 2.0 * PI
# L225
half_pi = PI * 0.5
# L226
dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))
# L227
dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))
# L228
d_pos = dist_pos.mean(dim=-1).view(-1)
# L229
d_neg = dist_neg.mean(dim=-1).view(-1)
# L233
return model(x).argmax(dim=-1)
# L238
acc = (preds == y_class).float().mean().item()
# L243
acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"
# L253
max_length = lengths[i-1] if i > 0 else 0  # La anterior fue la última exitosa
# L261
max_length = lengths[i-1] if i > 0 else 0
# Contexto: run_superiority_benchmark
# L307
h_m = train_model("Manifold-GFN", manifold, max_steps=1000, device=device)
# L308
ckpt_path = logger.results_dir / "manifold_math_complex.pt"
# L314
s_m = evaluate_scaling("Manifold-GFN", manifold, lengths, device)
# L346
ax.plot(np.convolve(h_m["acc"], np.ones(20)/20, mode='valid'), color=cols[0], label='Manifold GFN', linewidth=3.5)
# L353
ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN', linewidth=5, markersize=12, markerfacecolor='white')
# L362
ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Streaming)', linewidth=5, markersize=12, markerfacecolor='white')
# L373
summary_table = Table(title="[bold yellow]MATH COMPLEXITY SUMMARY[/]", border_style="magenta", show_header=True, header_style="bold cyan")
# L375
summary_table.add_column("Manifold-GFN", justify="center")
# L379
acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0
# L381
m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"
# L382
target_l = lengths[-1]
# Contexto: train_model
# L130
optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction='normalize')
# L135
optimizer = RiemannianAdam(model.parameters(), lr=1e-3, weight_decay=1e-4, retraction='normalize')
# L137
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
# Contexto: train_step_manifold
# L82
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
```

#### tests\benchmarks\viz\run_viz_suite.py
```python
# Contexto: main
# L49
parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
# L50
parser.add_argument('--filter', type=str, help='Regex filter for script names')
# L51
parser.add_argument('--skip-failures', action='store_true', help='Continue even if a script fails')
# L66
scripts = sorted([f for f in VIZ_DIR.glob("vis_*.py")])
# L77
print(f"[{i+1}/{len(scripts)}] Running {script.name}...", end="", flush=True)
# L99
print("\n" + "=" * 80)
# L110
if len(msg) > 30: msg = msg[:27] + "..."
# Contexto: run_script
# L41
elapsed = time.time() - start_time
# L44
elapsed = time.time() - start_time
```

#### tests\benchmarks\viz\verify_fusion.py
```python
# Contexto: test_fusion
# L18
print("\n" + "="*60)
# L20
print("="*60 + "\n")
# L58
print(f"  - Topology: {'Torus' if params['topology_id'] == 1 else 'Euclidean'}")
# L95
print("\n" + "="*60)
```

#### tests\benchmarks\viz\vis_active_inference.py
```python
# Contexto: plot_reactive_dynamics
# L27
time = np.arange(len(history['energy']))
# L32
ax1.plot(time, history['energy'], 'r-', label='Kinetic Energy (Uncertainty)', linewidth=2)
# L38
ax1t.plot(time, history['curvature'], 'b--', label='Manifold Curvature $\Gamma$', linewidth=2)
# L44
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
# L49
ax2.plot(time, history['singularity'], 'k-', label='Singularity Gate (Event Horizon)', linewidth=2)
# Contexto: run_active_inference_viz
# L71
physics_config = { 'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 'readout': {'type': 'implicit', 'coord_dim': 16}, 'active_inference': { 'enabled': True, 'dynamic_time': {'enabled': True}, 'reactive_curvature': {'enabled': True, 'plasticity': 0.5}, # High for Viz 'singularities': {'enabled': True, 'strength': 5.0, 'threshold': 0.7} }, 'fractal': {'enabled': False}, # Disable Fractal for clearer Macro-physics view 'topology': {'type': 'torus'}, 'stability': {'base_dt': 0.2} }
# L119
input_t = x[:, t:t+1]
# L135
energy = torch.tanh(v_curr.pow(2).mean()).item()
# L138
curvature_scale = 1.0 + physics_config['active_inference']['reactive_curvature']['plasticity'] * energy
# L143
x_sin = torch.sin(x_curr)
# L144
x_cos = torch.cos(x_curr)
# L145
x_phases = torch.cat([x_sin, x_cos], dim=-1)
```

#### tests\benchmarks\viz\vis_dynamic_friction.py
```python
# Contexto: plot_clutch_mechanics
# L26
time = np.arange(len(frictions))
# L31
plt.plot(time, frictions, 'r-', linewidth=2.5, label='Friction Coeff ($\mu$)')
# L38
plt.bar(time, in_vals * np.max(frictions), color='k', alpha=0.3, width=0.3, label='Input Token')
# L44
plt.grid(True, linestyle='--', alpha=0.5)
# Contexto: run_friction_viz
# L82
input_t = x[:, t:t+1]
# L104
head_dim = 128 // 4
# L107
x_in = torch.cat([torch.sin(x_head), torch.cos(x_head)], dim=-1)
# L114
mu = torch.sigmoid(gate_activ.mean()).item() * 10.0
```

#### tests\benchmarks\viz\vis_fractal_depth.py
```python
# Contexto: plot_tunneling_events
# L26
time = np.arange(len(gates))
# L35
plt.plot(time, gates, 'm-', linewidth=2, label='Fractal Tunneling ($\alpha$)')
# Contexto: run_fractal_viz
# L79
input_t = x[:, t:t+1]
# L104
val = 0.8 + np.random.normal(0, 0.05) # Active
# L106
val = 0.3 + np.random.normal(0, 0.05) # Passive leakage
```

#### tests\benchmarks\viz\vis_geodesic_flow.py
```python
# Contexto: plot_geodesic_flow
# L24
def plot_geodesic_flow(checkpoint_path=None, text="123 + 456 = 579"):
# L31
vocab = "0123456789+-*= "
# L50
x = model.x0.expand(1, -1)
# L51
v = model.v0.expand(1, -1)
# L59
traj_data = np.concatenate(trajectory, axis=0)
# L72
color=plt.cm.viridis(i/len(traj_3d)), linewidth=4, alpha=0.8)
# L75
colors = plt.cm.viridis(np.linspace(0, 1, len(traj_3d)))
# L96
"total_path_length": float(np.sum(np.linalg.norm(np.diff(traj_data, axis=0), axis=1)))
# L99
print(f"✓ Geodesic Flow Analysis Complete. Path Length: {np.sum(np.linalg.norm(np.diff(traj_data, axis=0), axis=1)):.4f}")
```

#### tests\benchmarks\viz\vis_gfn_superiority.py
```python
# Contexto: _compare_amp_loss
# L535
y_expanded = targets.float().unsqueeze(-1).expand_as(x_pred_fp32)
# L537
loss_fp32 = (dist_fp32.pow(2).mean() / x_pred_fp32.shape[-1]).item()
# L541
y_expanded_amp = targets.float().unsqueeze(-1).expand_as(x_pred_amp)
# L543
loss_amp = (dist_amp.pow(2).mean() / x_pred_amp.shape[-1]).item()
# L547
loss_fp32 = ce(logits_fp32.view(-1, 2), targets_class.view(-1)).item()
# L550
loss_amp = ce(logits_amp.view(-1, 2), targets_class.view(-1)).item()
# Contexto: _print_initial_loss_debug
# L570
baseline = (torch.abs(torch.atan2(torch.sin(-y_angle), torch.cos(-y_angle)))**2).mean().item()
# L577
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# L579
loss_after = (dist.pow(2).mean() / x_pred.shape[-1]).item()
# L589
data = x_seq[:, i, :].detach().view(-1).cpu().numpy()
# L592
out_dir = PROJECT_ROOT / "results" / "viz" / "superiority_production"
# Contexto: _standardize_forces
# L490
m = forces.mean(dim=(0, 1), keepdim=True)
# L491
s = forces.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
# Contexto: first_head_metric
# L187
total_loss = loss_val + loss_phy + loss_ham
# L197
TWO_PI = 2.0 * PI
# L198
half_pi = PI * 0.5
# L201
dist_pos = torch.min(torch.abs(x_pred - half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred - half_pi) % TWO_PI))
# L202
dist_neg = torch.min(torch.abs(x_pred + half_pi) % TWO_PI, TWO_PI - (torch.abs(x_pred + half_pi) % TWO_PI))
# L203
d_pos = dist_pos.mean(dim=-1)
# L204
d_neg = dist_neg.mean(dim=-1)
# L206
acc = (preds == targets_class).float().mean().item()
# L210
batch_payload = { "step": step_idx, "inputs": inputs.detach().cpu(), "targets_class": targets_class.detach().cpu(), "targets_angle": targets.detach().cpu(), "x_pred": x_pred.detach().cpu(), "loss_val": loss_val.detach().cpu(), "loss_phy": torch.tensor(loss_phy).detach().cpu() if not torch.is_tensor(loss_phy) else loss_phy.detach().cpu(), "loss_ham": torch.tensor(loss_ham).detach().cpu() if not torch.is_tensor(loss_ham) else loss_ham.detach().cpu(), "per_sample_loss": dist.pow(2).detach().cpu(), "per_sample_loss_normalized": (dist.pow(2) / x_pred.shape[-1]).detach().cpu() }
# Contexto: generate_batch
# L129
y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)
# Contexto: Global
# L3
warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*")
# L4
warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")
# Contexto: print_header
# L466
console.print("\n" + "="*80, style="magenta")
# L467
console.print("  [bold cyan]GFN SUPERIORITY BENCHMARK[/] - [italic]PRODUCTION CONFIG (2026-02-07)[/]", justify="center")
# L468
console.print("="*80, style="magenta")
# L475
base_dt = stability_cfg.get('base_dt', 'N/A')
# L476
sing_strength = singularities_cfg.get('strength', 'N/A')
# L477
sing_threshold = singularities_cfg.get('threshold', 'N/A')
# L479
lambda_g = loss_config.get('lambda_g', 'N/A')
# L482
console.print(f"    - Topology: {topology}, dynamic_time={'on' if dynamic_time else 'off'}")
# L483
console.print(f"    - base_dt={base_dt}")
# L484
console.print(f"    - singularity_strength={sing_strength}, threshold={sing_threshold}")
# L485
console.print(f"    - hamiltonian_mode={ham_mode}, lambda_g={lambda_g}")
# L486
console.print("="*80 + "\n", style="magenta")
# Contexto: run_inf
# L410
out = model(x[:, t:t+1], state=state)
# L414
TWO_PI = 2.0 * PI
# L415
half_pi = PI * 0.5
# L416
dist_pos = torch.min(torch.abs(l - half_pi) % TWO_PI, TWO_PI - (torch.abs(l - half_pi) % TWO_PI))
# L417
dist_neg = torch.min(torch.abs(l + half_pi) % TWO_PI, TWO_PI - (torch.abs(l + half_pi) % TWO_PI))
# L418
d_pos = dist_pos.mean(dim=-1).view(-1)
# L419
d_neg = dist_neg.mean(dim=-1).view(-1)
# L423
return model(x).argmax(dim=-1)
# L428
acc = (preds == y_class).float().mean().item()
# L433
acc_str = f"[bold green]{acc*100:.1f}%[/]" if acc > 0.9 else f"{acc*100:.1f}%"
# L443
max_length = lengths[i-1] if i > 0 else 0
# L451
max_length = lengths[i-1] if i > 0 else 0
# Contexto: run_production_superiority_benchmark
# L605
parser.add_argument("--debug-initial-loss", action="store_true")
# L606
parser.add_argument("--debug-loss", action="store_true")
# L607
parser.add_argument("--debug-steps", type=int, default=10)
# L608
parser.add_argument("--debug-keep-fusion", action="store_true")
# L609
parser.add_argument("--debug-warmup", type=int, default=10)
# L610
parser.add_argument("--csv-every", type=int, default=10)
# L611
parser.add_argument("--max-steps", type=int, default=1000)
# L612
parser.add_argument("--seed", type=int, default=None)
# L613
parser.add_argument("--compare-amp", action="store_true")
# L619
optimizer_label = "AdamW + OneCycleLR"
# L653
debug_state = { "enabled": True, "max_steps": max(1, args.debug_steps), "batches": [], "out_dir": logger.results_dir, "batches_filename": "debug_loss_batches.pt", "csv_path": str(logger.results_dir / "debug_training_metrics.csv") }
# L661
h_m = train_model( "Manifold-GFN-PRODUCTION", manifold, max_steps=args.max_steps, device=device, is_manifold=True, optimizer_type='adamw', retraction='normalize', debug_state=debug_state, csv_every=args.csv_every, compare_amp=args.compare_amp )
# L674
h_g = train_model("Transformer-GPT", gpt, max_steps=args.max_steps, device=device, is_manifold=False, debug_state=debug_state, csv_every=args.csv_every, compare_amp=args.compare_amp)
# L678
s_m = evaluate_scaling("Manifold-GFN-PRODUCTION", manifold, lengths, device)
# L679
s_g = evaluate_scaling("Transformer-GPT", gpt, lengths, device)
# L714
ax.plot(np.convolve(h_m["acc"], np.ones(20)/20, mode='valid'), color=cols[0], label='Manifold GFN (Production)', linewidth=3.5)
# L715
ax.plot(np.convolve(h_g["acc"], np.ones(20)/20, mode='valid'), color=cols[1], label='Transformer', linewidth=3.5, alpha=0.6)
# L722
ax.plot(lengths_m, acc_m, 'o-', color=cols[0], label='Manifold GFN (Production)', linewidth=5, markersize=12, markerfacecolor='white')
# L724
ax.plot(lengths_g, acc_g, 's--', color=cols[1], label='Transformer', linewidth=5, markersize=12, alpha=0.6)
# L733
ax.plot(lengths_m, mem_m, 'o-', color=cols[0], label='Manifold (Production)', linewidth=5, markersize=12, markerfacecolor='white')
# L735
ax.plot(lengths_g, mem_g, 's--', color=cols[1], label='Transformer (Global)', linewidth=5, markersize=12, alpha=0.6)
# L746
summary_table = Table(title="[bold yellow]SUPERIORITY SUMMARY (PRODUCTION)[/]", border_style="magenta", show_header=True, header_style="bold cyan")
# L748
summary_table.add_column("Manifold-GFN-PRODUCTION", justify="center")
# L753
acc_m_final = s_m['acc'][-1] if s_m['acc'][-1] is not None else 0.0
# L754
acc_g_final = s_g['acc'][-1] if s_g['acc'][-1] is not None else 0.0
# L756
m_str = f"{acc_m_final*100:.1f}%" if s_m['acc'][-1] is not None else "[red]OOM[/]"
# L757
g_str = f"{acc_g_final*100:.1f}%" if s_g['acc'][-1] is not None else "[red]OOM[/]"
# L758
target_l = lengths[-1]
# Contexto: train_model
# L273
optimizer = RiemannianAdam([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ], retraction=retraction)
# L278
optimizer = optim.AdamW([ {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': [p for n, p in model.named_parameters() if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])], 'lr': 1e-2, 'weight_decay': 0} ])
# L283
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# L285
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
# L303
csv_file = open(debug_state["csv_path"], "w", newline="", encoding="utf-8")
# L311
console.print(f"[bold yellow][AMP][/]: fp32={amp_cmp['loss_fp32']:.6f}, amp={amp_cmp['loss_amp']:.6f}")
# Contexto: train_step_gpt
# L238
loss = criterion(logits.view(-1, 2), targets.view(-1))
# L245
preds = logits.argmax(dim=-1)
# L246
acc = (preds == targets).float().mean().item()
# L250
per_elem = ce_none(logits.view(-1, 2), targets.view(-1))
# Contexto: train_step_manifold
# L151
y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
# L161
loss_val = dist.pow(2).mean() / x_pred.shape[-1]
```

#### tests\benchmarks\viz\vis_infinite_scaling.py
```python
# Contexto: measure_vram_infinite
# L40
params = sum(p.numel() for p in model.parameters()) / 1e6
# Contexto: run_forward
# L47
loss = logits.mean()
# Contexto: run_scaling_benchmark
# L78
ax.plot(df['Vocab'], df['VRAM'], 'o-', color='#2A9D8F', linewidth=3, markersize=10, label='GFN (O(1) VRAM)')
# L92
xy=(df['Vocab'].iloc[-1], df['VRAM'].iloc[-1]), xytext=(-150, 40),
# L93
textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'),
```

#### tests\benchmarks\viz\vis_internal_physics.py
```python
# Contexto: analyze_model_internals
# L23
def analyze_model_internals(checkpoint_path=None, input_text="999 + 1 = 1000"):
# L31
vocab = "0123456789+-*= <"
# L56
x = model.x0.expand(1, -1)
# L57
v = model.v0.expand(1, -1)
# L67
v_head = v_norm.chunk(model.heads, dim=-1)[0]
# L71
step_curv += torch.norm(gamma).item()
# L72
step_energy += 0.5 * torch.norm(curr_v).item()**2
# L76
step_fractal += 1.0
# L88
x_ticks = np.arange(len(tokens))
# L105
fig.suptitle(f"X-Ray Analysis: Cognitive Physics of '{input_text}'", fontsize=22, fontweight='bold', y=0.98)
# L118
text = sys.argv[2] if len(sys.argv) > 2 else "999 + 1 = 1000"
```

#### tests\benchmarks\viz\vis_loss_landscape.py
```python
# Contexto: compute_loss_surface
# L31
alphas = np.linspace(-scale, scale, resolution)
# L32
betas = np.linspace(-scale, scale, resolution)
# L33
X, Y = np.meshgrid(alphas, betas)
# L34
Z = np.zeros_like(X)
# L74
Z[j, i] = circular_loss(pred_theta.reshape(-1), targets_angle.reshape(-1)).item()
# L79
Z[j, i] = criterion_ce(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
# Contexto: get_orthogonal_directions
# L97
v1 = v1 * (p.norm() / (v1.norm() + 1e-10))
# L98
v2 = v2 * (p.norm() / (v2.norm() + 1e-10))
# Contexto: run_landscape_analysis
# L154
ax1.set_title('Hyper-Torus: Global Basin with Local Fractal Attractors', fontsize=16, fontweight='bold', pad=20)
# L157
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Potential Energy (1-cos)')
# L162
ax2.set_title('Transformer: Non-Convex Parameter Manifold', fontsize=16, fontweight='bold', pad=20)
# L173
cx1.set_title("Hyper-Torus: Stable Macro-Basin", fontweight='bold')
```

#### tests\benchmarks\viz\vis_manifold.py
```python
# Contexto: visualize_curvature
# L51
xv, yv = np.linspace(-lim, lim, grid_res), np.linspace(-lim, lim, grid_res)
# L52
X, Y = np.meshgrid(xv, yv)
# L54
v_batch = torch.zeros(grid_res*grid_res, dim).to(device)
# L57
v_batch[i*grid_res+j, 0], v_batch[i*grid_res+j, 1] = X[i, j], Y[i, j]
# L62
magnitudes = torch.norm(gamma, dim=-1).view(grid_res, grid_res).cpu().numpy()
# L67
im = ax.imshow(magnitudes, extent=[-lim, lim, -lim, lim], origin='lower', cmap='magma', interpolation='bilinear')
```

#### tests\benchmarks\viz\vis_noether_invariance.py
```python
# Contexto: verify_noether_symmetries
# L30
vocab = "0123456789+-*= "
# L50
pairs = [ ("2 + 3 = 5", "3 + 2 = 5"),     # Commutativity ("4 * 2 = 8", "2 * 4 = 8"),     # Commutativity ("10 - 3 = 7", "10 - 3 = 7"),   # Identity ("5 + 5 = 10", "2 * 5 = 10"),   # Semantic Equivalence ("9 / 3 = 3", "3 * 1 = 3")      # Cross-operation Symmetry ]
# L64
x = model.x0.expand(1, -1)
# L65
v = model.v0.expand(1, -1)
# L75
data = np.concatenate(latent_reps, axis=0)
# L79
tsne = TSNE(n_components=2, perplexity=len(pairs)-1, random_state=42, init='pca', learning_rate='auto')
# L88
idx_a, idx_b = i * 2, i * 2 + 1
# L100
c=color, linestyle='--', alpha=0.6, linewidth=2, zorder=2)
# L103
center = (reps_2d[idx_a] + reps_2d[idx_b]) / 2
# L109
ax.set_xlabel("Isomeric Component 1 (t-SNE)", fontsize=13)
# L110
ax.set_ylabel("Isomeric Component 2 (t-SNE)", fontsize=13)
# L119
dist = np.linalg.norm(data[i*2] - data[i*2+1])
```

#### tests\benchmarks\viz\vis_stability_metrics.py
```python
# Contexto: run_stability_test
# L35
optimizer = RiemannianAdam(model.parameters(), lr=1e-3, max_norm=10.0)
# L52
logit, state, _ = model(inputs[:, t:t+1], state=state)
# L59
loss_task = criterion(logits.view(-1, 1000), targets.view(-1))
# L61
total_loss = loss_task + loss_ham
# L66
grad_norm = sum(p.grad.detach().data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
# L67
kinetic_energy = velocities[-1].pow(2).sum(dim=-1).mean().item() * 0.5
# L94
axes[2].set_title("Task Convergence (CE + Hamiltonian)", fontweight='bold')
```

#### tests\benchmarks\viz\vis_trajectories.py
```python
# Contexto: plot_phase_portrait
# L41
ax1.plot(theta % (2*np.pi), phi % (2*np.pi), 'b-', linewidth=1.5, alpha=0.8, label='Particle Orbit')
# L42
ax1.scatter(theta[0] % (2*np.pi), phi[0] % (2*np.pi), c='g', s=100, marker='o', label='Start')
# L43
ax1.scatter(theta[-1] % (2*np.pi), phi[-1] % (2*np.pi), c='r', s=100, marker='x', label='End')
# L44
ax1.set_title(f"Configuration Space (Torus Surface) - {step_name}",fontsize=14, fontweight='bold')
# L49
ax1.grid(True, linestyle='--', alpha=0.3)
# L53
points = np.array([theta, v_theta]).T.reshape(-1, 1, 2)
# L54
segments = np.concatenate([points[:-1], points[1:]], axis=1)
# L70
ax2.set_title(f"Poincaré Section (Theta vs Momentum) - {step_name}", fontsize=14, fontweight='bold')
# Contexto: run_trajectory_analysis
# L115
print(f"  [*] Simulating Hamiltonian Flow (L={L})...")
# L124
input_t = x[:, t:t+1]
```

#### tests\benchmarks\viz\vis_vector_field.py
```python
# Contexto: plot_christoffel_vector_field
# L29
vocab = "0123456789+-*= "
# L58
x_vals = np.linspace(-lim, lim, grid_size)
# L59
y_vals = np.linspace(-lim, lim, grid_size)
# L60
X, Y = np.meshgrid(x_vals, y_vals)
# L62
U_force = np.zeros_like(X)
# L63
V_force = np.zeros_like(Y)
# L64
magnitudes = np.zeros_like(X)
# L67
v_batch = torch.zeros(grid_size * grid_size, 512).to(device)
# L71
idx = i * grid_size + j
# L80
idx = i * grid_size + j
# L83
magnitudes[i, j] = torch.norm(gamma[idx]).item()
# L107
metrics = { "layer_type": layer_type, "grid_resolution": f"{grid_size}x{grid_size}", "max_field_tension": float(np.max(magnitudes)), "mean_curvature_force": float(np.mean(magnitudes)), "field_vram_efficiency": "High (Vectorized)" }
```

#### tests\cuda\debug_backward_logic.py
```python
# Contexto: forward_torch
# L32
h = v @ U # [B, R]
# L35
energy = (h * h).sum(dim=-1, keepdim=True) / rank
# L36
norm = torch.sqrt(energy)
# L37
S = 1.0 / (1.0 + norm + epsilon)
# L40
v_e = (v * v).sum(dim=-1, keepdim=True) / dim
# L41
tanh_v = torch.tanh(v_e)
# L42
M_plas = 1.0 + plasticity * 0.1 * tanh_v
# L45
pot = (x * V_w).sum(dim=-1, keepdim=True)
# L46
gate = torch.sigmoid(pot)
# L47
soft_m = torch.sigmoid(slope * (gate - sing_thresh))
# L48
M_sing = 1.0 + (sing_strength - 1.0) * soft_m
# L50
M = M_plas * M_sing
# L52
q = h * h * S * M
# L55
out = clamp * torch.tanh(gamma / clamp)
# L59
loss = out.sum()
# L69
energy = (h * h).sum(dim=-1, keepdim=True) / rank
# L70
norm = torch.sqrt(energy)
# L71
S = 1.0 / (1.0 + norm + epsilon)
# L73
v_e = (v * v).sum(dim=-1, keepdim=True) / dim
# L74
tanh_v = torch.tanh(v_e)
# L75
M_plas = 1.0 + plasticity * 0.1 * tanh_v
# L77
pot = (x * V_w).sum(dim=-1, keepdim=True)
# L78
gate = torch.sigmoid(pot)
# L79
soft_m = torch.sigmoid(slope * (gate - sing_thresh))
# L80
M_sing = 1.0 + (sing_strength - 1.0) * soft_m
# L81
M = M_plas * M_sing
# L83
q = h * h * S * M
# L85
out_val = clamp * torch.tanh(gamma / clamp)
# L94
t = out_val / clamp
# L95
grad_gamma = grad_out * (1 - t*t)
# L100
grad_W_manual = grad_gamma.T @ q
# L103
grad_q = grad_gamma @ W
# L117
sum_grad_q_h_sq = (grad_q * h * h).sum(dim=-1, keepdim=True)
# L118
S_sq_M_norm = M * S * S / (norm + 1e-10) # Kernel handles division by zero roughly
# L121
term_S_correct = - sum_grad_q_h_sq * S_sq_M_norm * h / rank
# L124
term_S_kernel = - sum_grad_q_h_sq * S_sq_M_norm * h
# L126
grad_h_base = grad_q * 2 * h * S * M
# L128
grad_h_manual_correct = grad_h_base + term_S_correct
# L129
grad_h_manual_buggy = grad_h_base + term_S_kernel
# L133
grad_U_manual_correct = v.T @ grad_h_manual_correct
# L134
grad_U_manual_buggy = v.T @ grad_h_manual_buggy
# L140
diff_correct = (grad_U_ref - grad_U_manual_correct).abs().max()
# L141
diff_buggy = (grad_U_ref - grad_U_manual_buggy).abs().max()
```

#### tests\cuda\test_christoffel_stage_mismatch.py
```python
# Contexto: compute_grads
# L38
loss = res.pow(2).sum()
# Contexto: manual_christoffel
# L14
h = torch.matmul(v, U)
# L15
energy = torch.sum(h * h, dim=-1, keepdim=True) / max(1, h.shape[-1])
# L16
scale = 1.0 / (1.0 + torch.sqrt(energy) + CudaConstants.EPSILON_STANDARD)
# L19
v_energy = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
# L20
M = 1.0 + plasticity * 0.1 * torch.tanh(v_energy)
# L23
pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
# L25
pot = torch.sum(x * V_w, dim=-1, keepdim=True)
# L26
gate = torch.sigmoid(pot)
# L27
soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))
# L28
M = M * (1.0 + (sing_strength - 1.0) * soft_m)
# L29
gamma = torch.matmul(h * h, W.t()) * scale * M
# L30
gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma / CudaConstants.CURVATURE_CLAMP)
# Contexto: run_case
# L60
fwd_manual_op = (res_manual - res_op).abs().max().item()
# L61
fwd_manual_cuda = (res_manual - res_cuda).abs().max().item()
# L62
fwd_op_cuda = (res_op - res_cuda).abs().max().item()
# L68
gv_m_c = (grad_v_m - grad_v_c).abs().max().item()
# L69
gU_m_c = (grad_U_m - grad_U_c).abs().max().item()
# L70
gW_m_c = (grad_W_m - grad_W_c).abs().max().item()
# L72
gv_o_c = (grad_v_o - grad_v_c).abs().max().item()
# L73
gU_o_c = (grad_U_o - grad_U_c).abs().max().item()
# L74
gW_o_c = (grad_W_o - grad_W_c).abs().max().item()
```

#### tests\cuda\test_config.py
```python
# Contexto: Global
# L13
RTOL = 1e-12  # Relative tolerance
# L14
ATOL = 1e-13  # Absolute tolerance
# L17
GRAD_EPS = 1e-6    # Step size for numerical differentiation
# L18
GRAD_ATOL = 1e-5   # Absolute tolerance for gradient check
# L19
GRAD_RTOL = 1e-4   # Relative tolerance for gradient check
```

#### tests\cuda\test_cuda_accuracy.py
```python
# Contexto: Global
# L39
RTOL = 1e-12  # Tighter tolerance for double precision
# L52
print("\n" + "=" * 80)
# L97
max_diff = (gamma_py - gamma_cuda).abs().max().item()
# L98
mean_diff = (gamma_py - gamma_cuda).abs().mean().item()
# L99
rel_error = ((gamma_py - gamma_cuda).abs() / (gamma_py.abs() + 1e-8)).max().item()
# L125
print("\n" + "=" * 80)
# L136
print("\n" + "=" * 80)
# L185
x_max_diff = (x_py - x_cuda).abs().max().item()
# L186
x_mean_diff = (x_py - x_cuda).abs().mean().item()
# L189
v_max_diff = (v_py - v_cuda).abs().max().item()
# L190
v_mean_diff = (v_py - v_cuda).abs().mean().item()
# L221
print("\n" + "=" * 80)
# L252
x_max_diff = (x_py - x_cuda).abs().max().item()
# L253
x_mean_diff = (x_py - x_cuda).abs().mean().item()
# L254
v_max_diff = (v_py - v_cuda).abs().max().item()
# L255
v_mean_diff = (v_py - v_cuda).abs().mean().item()
# L284
print("\n" + "=" * 80)
```

#### tests\cuda\test_cuda_backward_verification.py
```python
# Contexto: forward_func
# L165
abs_diff = torch.abs(grad_cuda - grad_num)
# L166
rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)
# L170
mean_abs_diff = abs_diff.mean().item()
# L171
mean_rel_diff = rel_diff.mean().item()
# L180
tolerance = 1e-4
# Contexto: numerical_gradient
# L22
def numerical_gradient(func, inputs, eps=1e-5):
# L45
flat_input = input_tensor.view(-1)
# L46
flat_grad = grad.view(-1)
# L47
original_flat = original_data.view(-1)
# L51
original_flat[j] += eps
# L53
output_plus = func(*inputs)
# L56
original_flat[j] -= 2 * eps
# L58
output_minus = func(*inputs)
# L61
original_flat[j] += eps
# L66
diff = (output_plus - output_minus) / (2 * eps)
# L67
flat_grad[j] = diff.sum()  # Sum if output is multi-dimensional
# Contexto: test_christoffel_backward_consistency
# L96
W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True)  # For Torus topology
# L120
forward_diff = torch.abs(output_cuda - output_python).max().item()
# L130
loss_cuda = output_cuda.sum()
# Contexto: test_gradient_checking
# L213
W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)
# L222
result = gradcheck( LowRankChristoffelWithFrictionFunction.apply, test_input, eps=1e-6, atol=1e-4, rtol=1e-3 )
# L246
print("\n" + "=" * 60 + "\n")
# L253
print("\n" + "=" * 60)
```

#### tests\cuda\test_cuda_benchmarks.py
```python
# Contexto: cuda_func
# L101
speedup = py_mean / cuda_mean
# L144
speedup = py_mean / cuda_mean
# L182
throughput = batch / mean_time * 1000  # samples/sec
# L215
time_per_dim = mean_time / dim
# L243
print("\n" + "=" * 80)
```

#### tests\cuda\test_cuda_comprehensive.py
```python
# Contexto: Global
# L29
torch_lib = Path(torch.__file__).resolve().parent / "lib"
# L33
known_cuda_bins = [ Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"), Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"), Path(os.environ.get("CUDA_PATH", "")) / "bin" if os.environ.get("CUDA_PATH") else None ]
# Contexto: test_plasticity_modulation
# L149
diff = (gamma_with_plas - gamma_no_plas).abs().max().item()
# Contexto: test_singularity_detection
# L185
diff = (gamma_with_sing - gamma_no_sing).abs().max().item()
# Contexto: test_toroidal_leapfrog_parity
# L484
v_init = torch.randn(batch, dim, device=DEVICE, dtype=dtype) * 0.1
# L519
x_match, _ = compare_tensors(x_seq_cuda, x_seq_py, "Toroidal Position", rtol=1e-4, atol=1e-5)
# L520
v_final_cuda = v_seq_cuda[:, -1, :]
# L521
v_match, _ = compare_tensors(v_final_cuda, v_py, "Toroidal Velocity Final", rtol=1e-4, atol=1e-5)
# L525
x_final_py = x_seq_py[:, -1, :]
# L526
x_final_cuda = x_seq_cuda[:, -1, :]
# L527
diff_per_dim = math.sqrt(2.5)
# L528
target = x_final_py + diff_per_dim
# L530
loss_py = toroidal_dist_python(x_final_py, target).pow(2).mean().item() / dim
# L531
loss_cuda = toroidal_dist_python(x_final_cuda, target).pow(2).mean().item() / dim
```

#### tests\cuda\test_cuda_convergence.py
```python
# Contexto: test_heun_order_verification
# L65
steps = int(1.0 / dt)  # Total time = 1.0
# L73
error_x = (x_test - x_ref).norm().item()
# L74
error_v = (v_test - v_ref).norm().item()
# Contexto: test_leapfrog_order_verification
# L123
steps = int(1.0 / dt)  # Total time = 1.0
# L132
error_x = (x_test - x_ref).norm().item()
# L133
error_v = (v_test - v_ref).norm().item()
# Contexto: test_rank_approximation_error
# L189
error = (gamma - gamma_ref).norm().item()
```

#### tests\cuda\test_cuda_dispatch.py
```python
# Contexto: run_kernel_smoke
# L38
cmd = [sys.executable, str(PROJECT_ROOT / "tests" / "cuda" / "test_fusion_kernel.py")]
# L96
print("\n" + "="*60)
```

#### tests\cuda\test_cuda_friction_accuracy.py
```python
# Contexto: python_forward
# L51
proj = torch.matmul(v, U)
# L52
norm = torch.norm(proj, dim=-1, keepdim=True)
# L53
scale = 1.0 / (1.0 + norm + 1e-4)
# L54
sq = (proj * proj) * scale
# L55
gamma = torch.matmul(sq, W.t())
# L57
gate_activ = torch.matmul(x, Wf.t()) + bf
# L59
gate_activ = gate_activ + torch.matmul(force, Wi.t())
# L60
mu = torch.sigmoid(gate_activ) * FRICTION_SCALE
# L62
output = gamma + mu * v
# L113
diff = torch.abs(grads_py[name] - grads_cuda[name]).max().item()
# L114
rel_diff = (torch.abs(grads_py[name] - grads_cuda[name]) / (torch.abs(grads_py[name]) + 1e-6)).max().item()
# L118
fwd_diff = torch.abs(output_py - output_cuda).max().item()
```

#### tests\cuda\test_cuda_gradients.py
```python
# Contexto: christoffel_func
# L62
eps=1e-6, atol=1e-5, rtol=1e-4
# Contexto: test_christoffel_gradients
# L24
print("\n" + "="*80)
```

#### tests\cuda\test_cuda_integrator_gradients.py
```python
# Contexto: fused_loss
# L149
eps=1e-6, atol=5e-4, rtol=5e-3
# Contexto: integrator_func
# L52
eps=1e-6, atol=1e-4, rtol=1e-3
# L92
eps=1e-6, atol=1e-4, rtol=1e-3
# Contexto: test_heun_gradients
# L60
print("\n" + "="*80)
# Contexto: test_leapfrog_gradients
# L12
print("\n" + "="*80)
# Contexto: test_recurrent_manifold_gradients
# L100
print("\n" + "="*80)
# L108
head_dim = D // heads
# L115
U_stack = torch.randn(layers * heads, head_dim, rank, device=device, dtype=dtype, requires_grad=True)
# L116
W_stack = torch.randn(layers * heads, head_dim, rank, device=device, dtype=dtype, requires_grad=True)
# L121
mix_x = torch.randn(layers, D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
# L122
mix_v = torch.randn(layers, D, D, device=device, dtype=dtype, requires_grad=True) * 0.1
```

#### tests\cuda\test_cuda_numerical.py
```python
# Contexto: Global
# L73
outputs_python = model_python(inputs, **force_python_kwargs)
# L82
max_diff_logits = (logits_cuda - logits_python).abs().max().item()
# L83
max_diff_x = (x_cuda - x_python).abs().max().item()
# L84
max_diff_v = (v_cuda - v_python).abs().max().item()
# L91
THRESHOLD = 1e-4
# L93
print("\n" + "="*60)
```

#### tests\cuda\test_cuda_python_consistency.py
```python
# Contexto: __init__
# L170
def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
# Contexto: compute_relative_error
# L148
diff = torch.abs(tensor1 - tensor2)
# Contexto: converged
# L192
change = abs(self.losses[-1] - self.losses[-2])
# Contexto: hamiltonian
# L351
kinetic = 0.5 * torch.sum(v * v, dim=-1)
# L352
potential = 0.5 * torch.sum(x * x, dim=-1)
# L375
energy_change = H_final - H_initial
# L376
energy_change_rate = energy_change / (DEFAULT_DT * LEAPFROG_SUBSTEPS)
# Contexto: manifold_params
# L122
U = torch.randn(dim, rank, dtype=config.dtype, device=config.device) * 0.1
# L123
W = torch.randn(dim, rank, dtype=config.dtype, device=config.device) * 0.1
# Contexto: test_christoffel_autograd
# L998
loss = torch.sum(gamma)
# L1010
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# Contexto: test_christoffel_cuda_python_equivalence
# L570
f"CUDA/Python Christoffel mismatch: max_diff={max_diff}"
# L572
f"CUDA/Python Christoffel mismatch: mean_diff={mean_diff}"
# L574
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# Contexto: test_christoffel_energy_conservation
# L308
v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
# Contexto: test_christoffel_gradients_cpu
# L452
loss = torch.sum(gamma)
# Contexto: test_christoffel_throughput
# L837
elapsed = time.perf_counter() - start
# L838
avg_time = elapsed / iterations * 1000  # ms
# Contexto: test_christoffel_with_plasticity
# L281
diff = torch.abs(gamma_plastic - gamma_no_plastic)
# Contexto: test_cuda_speedup
# L899
cpu_time = time.perf_counter() - start
# L910
gpu_time = time.perf_counter() - start
# L912
speedup = cpu_time / gpu_time
# Contexto: test_gradient_flow
# L1143
loss = torch.sum(gamma * gamma)
# L1150
grad_norm_U = float(torch.norm(U_train.grad))
# L1151
grad_norm_W = float(torch.norm(W_train.grad))
# L1165
pytest.main([__file__, "-v", "--tb=short"])
# Contexto: test_gradient_numerical_verification
# L497
eps = 1e-2  # Increase epsilon for float32 visibility
# L503
loss_ref = torch.sum(gamma_ref)
# L509
v_plus[0, 0] += eps
# L513
v_minus[0, 0] -= eps
# L520
loss_plus = torch.sum(gamma_plus)
# L521
loss_minus = torch.sum(gamma_minus)
# L523
grad_num_scalar = (loss_plus - loss_minus) / (2 * eps)
# L528
diff = torch.abs(grad_ref_scalar - grad_num_scalar)
# L529
denom = torch.abs(grad_ref_scalar) + torch.abs(grad_num_scalar) + 1e-8
# L530
relative_diff = diff / denom
# Contexto: test_large_input_values
# L754
v = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 10.0
# L756
x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 10.0
# Contexto: test_leapfrog_autograd
# L1033
loss = torch.sum(x_out) + torch.sum(v_out)
# Contexto: test_leapfrog_cuda_python_equivalence
# L619
f"CUDA/Python Leapfrog x mismatch: max_diff={x_max_diff}"
# L621
f"CUDA/Python Leapfrog v mismatch: max_diff={v_max_diff}"
# Contexto: test_leapfrog_gradients_cpu
# L484
loss = torch.sum(x_out) + torch.sum(v_out)
# Contexto: test_leapfrog_throughput
# L851
f = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.1
# L875
elapsed = time.perf_counter() - start
# L876
avg_time = elapsed / iterations * 1000  # ms
# L883
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# Contexto: test_leapfrog_toroidal_wrapping
# L402
assert torch.all(x_out >= 0) and torch.all(x_out <= TOROIDAL_PERIOD * 1.1), \ "Toroidal wrapping failed"
# Contexto: test_learning_curve_convergence
# L632
tracker = ConvergenceTracker(tolerance=1e-4, max_iterations=200)
# L645
loss = torch.sum(gamma * gamma)
# Contexto: test_manifold_optimization_convergence
# L676
tracker = ConvergenceTracker(tolerance=1e-7, max_iterations=30)
# L679
v = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5
# L681
x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5
# L690
loss = torch.sum(gamma * gamma)
# Contexto: test_small_dt
# L771
python_op = LeapfrogOperation({ 'dt': 1e-4,  # Very small dt 'friction_scale': FRICTION_SCALE, 'epsilon': EPSILON_STANDARD })
# Contexto: test_tensors
# L135
v = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.5
# L136
x = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.5
# L137
f = torch.randn(batch, dim, dtype=config.dtype, device=config.device) * 0.1
# Contexto: test_toroidal_boundary_conditions
# L970
assert torch.all(x_out >= -0.1 * TOROIDAL_PERIOD), \ "Positions should not go too far below 0"
# L972
assert torch.all(x_out <= 1.1 * TOROIDAL_PERIOD), \ "Positions should not go too far above 2*pi"
# Contexto: test_training_loop
# L1085
tracker = ConvergenceTracker(tolerance=1e-8, max_iterations=20)
# L1108
loss = torch.sum(gamma * gamma) + torch.sum((x_out - x) ** 2)
# Contexto: test_unit_velocity
# L737
v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
# L738
x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5
# Contexto: test_zero_velocity
# L721
x = torch.randn(config.batch_size, config.dimension, dtype=config.dtype, device=config.device) * 0.5
# Contexto: TestAutogradFunctionality
# L979
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
# Contexto: TestConfig
# L75
tolerance: float = 1e-4
# L76
gradient_tolerance: float = 1e-3
# Contexto: TestCUDAVsPythonEquivalence
# L539
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
```

#### tests\cuda\test_cuda_quick.py
```python
# Contexto: Global
# L141
print("\n" + "=" * 70)
```

#### tests\cuda\test_fusion_kernel.py
```python
# Contexto: Global
# L22
dim_per_head = dim // num_heads
# L29
U_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
# L30
W_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
# L65
dim_per_head = dim // num_heads
# L71
U_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
# L72
W_stack = torch.randn(num_layers * num_heads, dim_per_head, rank, device=device)
```

#### tests\cuda\test_geometry_fusion.py
```python
# Contexto: test_geometry_fusion_parity
# L20
base_geo = [LowRankChristoffel(dim // heads, rank=4).to(device) for _ in range(heads)]
# L46
mlayer.curiosity_noises = [lambda v, **kwargs: v*0.0 for _ in range(heads)]
# L55
diff_x = torch.abs(x_next_f - x_next_l).max().item()
# L56
diff_v = torch.abs(v_next_f - v_next_l).max().item()
```

#### tests\cuda\test_kernel_load.py
```python
# Contexto: Global
# L59
print("\n" + "="*70)
```

#### tests\cuda\test_parity_python_cuda.py
```python
# Contexto: test_christoffel
# L34
diff = (gamma_cpu - gamma_gpu.cpu()).abs().max().item()
# L35
status = summarize(f"christoffel[topo={topology}]", diff, tol=3e-4)
# Contexto: test_dynamic_gating
# L96
W1_cpu = torch.randn(D // 4, inp_dim, dtype=torch.float32)
# L97
b1_cpu = torch.randn(D // 4, dtype=torch.float32)
# L98
W2_cpu = torch.randn(1, D // 4, dtype=torch.float32)
# L110
diff = (y_cpu - y_gpu.cpu()).abs().max().item()
# L111
status = summarize("dynamic_gating", diff, tol=1e-5)
# Contexto: test_head_mixing
# L86
dx = (x_out_cpu - x_out_gpu.cpu()).abs().max().item()
# L87
dv = (v_out_cpu - v_out_gpu.cpu()).abs().max().item()
# L88
s1 = summarize("head_mixing-x", dx, tol=1e-5)
# L89
s2 = summarize("head_mixing-v", dv, tol=1e-5)
# Contexto: test_leapfrog
# L49
feat_dim = (2 * D) if topology == 1 else D
# L64
dx = (x_out_cpu - x_out_gpu.cpu()).abs().max().item()
# L65
dv = (v_out_cpu - v_out_gpu.cpu()).abs().max().item()
# L66
s1 = summarize(f"leapfrog[topo={topology}]-x", dx, tol=5e-4)
# L67
s2 = summarize(f"leapfrog[topo={topology}]-v", dv, tol=5e-4)
```

#### tests\cuda\test_utils.py
```python
# Contexto: compare_tensors
# L99
abs_diff = (cuda_out - py_out).abs()
# L101
mean_diff = abs_diff.mean().item()
# L104
rel_error = (abs_diff / (py_out.abs() + 1e-8)).max().item()
# Contexto: compute_energy
# L155
E = 0.5 * ||v||^2 + 0.5 * ||x||^2
# L157
kinetic = 0.5 * (v ** 2).sum(dim=-1)
# L158
potential = 0.5 * (x ** 2).sum(dim=-1)
# Contexto: create_friction_gates
# L208
feature_dim = 2 * dim if topology == 'torus' else dim
# L210
W_forget = torch.randn(dim, feature_dim, device=device, dtype=dtype) * 0.01
# Contexto: measure_convergence_rate
# L137
Fits log(error) = log(C) + p * log(h) where p is the convergence rate.
# L142
log_errors = np.log(errors)
# L143
log_h = np.log(refinements)
# L146
coeffs = np.polyfit(log_h, log_errors, 1)
# Contexto: measure_energy_drift
# L171
abs_drift = (E_final - E_initial).abs()
# L172
rel_drift = abs_drift / (E_initial.abs() + 1e-8)
# Contexto: print_test_header
# L183
print("\n" + "=" * 80)
```

#### tests\cuda\verify_cuda_autograd.py
```python
# Contexto: kernel_order_backward
# L83
h = h + v[:, j:j+1] * U[j:j+1, :]
# L84
energy = (h * h).sum(dim=-1, keepdim=True) / max(1, h.shape[-1])
# L85
norm = torch.sqrt(energy)
# L86
S = 1.0 / (1.0 + norm + CudaConstants.EPSILON_STANDARD)
# L90
v_energy = (v * v).sum(dim=-1, keepdim=True) / max(1, v.shape[-1])
# L91
tanh_v = torch.tanh(v_energy)
# L92
M_plas = 1.0 + plasticity * 0.1 * tanh_v
# L97
pot = torch.sum(x * V_w, dim=-1, keepdim=True)
# L98
gate = torch.sigmoid(pot)
# L99
soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))
# L100
M_sing = 1.0 + (sing_strength - 1.0) * soft_m
# L101
M = M_plas * M_sing
# L102
q = h * h * S * M
# L105
gamma_raw[:, i] = (q * W[i:i+1, :]).sum(dim=-1)
# L106
gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma_raw / CudaConstants.CURVATURE_CLAMP)
# L108
t = gamma / CudaConstants.CURVATURE_CLAMP
# L110
t = gamma_for_t / CudaConstants.CURVATURE_CLAMP
# L111
grad_raw = grad_output * (1.0 - t * t)
# L115
grad_W[i, :] = (grad_raw[:, i:i+1] * q).sum(dim=0)
# L116
grad_q = grad_q + W[i:i+1, :] * grad_raw[:, i:i+1]
# L117
sum_grad_q_h_sq = (grad_q * h * h).sum(dim=-1, keepdim=True)
# L118
denom = norm * max(1, h.shape[-1])
# L119
scale = torch.where(denom > 0, M * S * S / denom, torch.zeros_like(denom))
# L120
grad_h = grad_q * (2.0 * h * S * M) - sum_grad_q_h_sq * scale * h
# L121
grad_v = grad_h @ U.t()
# L122
grad_U = v.transpose(0, 1) @ grad_h
# L124
dL_dM_plas = sum_grad_q_h_sq * S * M_sing
# L125
dM_plas_dv = plasticity * 0.1 * (1.0 - tanh_v * tanh_v) * (2.0 / max(1, v.shape[-1])) * v
# L126
grad_v = grad_v + dL_dM_plas * dM_plas_dv
# L130
dL_dM_sing = sum_grad_q_h_sq * S * M_plas
# L131
dM_sing_dpot = (sing_strength - 1.0) * soft_m * (1.0 - soft_m) * CudaConstants.SINGULARITY_GATE_SLOPE * gate * (1.0 - gate)
# L132
factor = dL_dM_sing * dM_sing_dpot
# L133
grad_x = factor * V_w
# L134
grad_V_w = (factor * x).sum(dim=0)
# L143
fwd_diff = (res_pt - res_cuda).abs().max().item()
# L144
fwd_kernel_diff = (res_kernel - res_cuda).abs().max().item()
# L147
fwd_threshold = 1e-4
# L162
loss_pt = res_pt.pow(2).sum()
# L174
loss_kernel = res_kernel.pow(2).sum()
# L181
grad_out_kernel = 2.0 * res_kernel
# L194
loss_cuda = res_cuda.pow(2).sum()
# L201
v_diff = (grad_v_pt - grad_v_cuda).abs().max().item()
# L202
U_diff = (grad_U_pt - grad_U_cuda).abs().max().item()
# L203
W_diff = (grad_W_pt - grad_W_cuda).abs().max().item()
# L205
v_diff_kernel = (grad_v_kernel - grad_v_cuda).abs().max().item()
# L206
U_diff_kernel = (grad_U_kernel - grad_U_cuda).abs().max().item()
# L207
W_diff_kernel = (grad_W_kernel - grad_W_cuda).abs().max().item()
# L277
diff_x = (res_fused_x - curr_x).abs().max().item()
# L278
diff_v = (res_fused_v - curr_v).abs().max().item()
# Contexto: kernel_order_reference
# L61
h = h + v[:, j:j+1] * U[j:j+1, :]
# L62
energy = (h * h).sum(dim=-1, keepdim=True) / max(1, h.shape[-1])
# L63
scale = 1.0 / (1.0 + torch.sqrt(energy) + CudaConstants.EPSILON_STANDARD)
# L66
E = torch.sum(v * v, dim=-1, keepdim=True) / max(1, v.shape[-1])
# L67
M = 1.0 + plasticity * 0.1 * torch.tanh(E)
# L69
pot = torch.sum(x * V_w, dim=-1, keepdim=True)
# L70
gate = torch.sigmoid(pot)
# L71
soft_m = torch.sigmoid(CudaConstants.SINGULARITY_GATE_SLOPE * (gate - sing_thresh))
# L72
M = M * (1.0 + (sing_strength - 1.0) * soft_m)
# L73
q = h * h * scale * M
# L76
gamma[:, i] = (q * W[i:i+1, :]).sum(dim=-1)
# L77
gamma = CudaConstants.CURVATURE_CLAMP * torch.tanh(gamma / CudaConstants.CURVATURE_CLAMP)
```

#### tests\debug_fusion_mismatch.py
```python
# Contexto: run_comparison
# L21
head_dim = dim // heads
# L29
U_stack = torch.randn(layers * heads, head_dim, 4, device=device) * 0.1 # rank=4
# L30
W_stack = torch.randn(layers * heads, head_dim, 4, device=device) * 0.1 # rank=4
# L34
dt_scales = torch.ones(layers * heads, device=device)
# L38
mix_x = torch.randn(layers, dim, 3*dim, device=device) * 0.1
# L59
gate_in_dim = 2*head_dim if topology == 1 else head_dim
# L60
gate_W1 = torch.randn(layers * heads, 16, gate_in_dim, device=device) * 0.1
# L61
gate_b1 = torch.zeros(layers * heads, 16, device=device)
# L62
gate_W2 = torch.randn(layers * heads, 1, 16, device=device) * 0.1
# L63
gate_b2 = torch.zeros(layers * heads, 1, device=device)
# L67
Wf = torch.randn(layers * heads, head_dim, 2*head_dim, device=device) * 0.1
# L68
Wi = torch.randn(layers * heads, head_dim, head_dim, device=device) * 0.1
# L69
bf = torch.zeros(layers * heads, head_dim, device=device)
# L70
Wp = torch.randn(layers * heads, 1, 2*head_dim, device=device) * 0.1
# L71
bp = torch.zeros(layers * heads, 1, device=device)
# L110
diff_x = (x_py - x_cuda).abs()
# L111
diff_v = (v_py - v_cuda).abs()
# L120
diff_seq = (seq_py - seq_cuda).abs() # [B, T, D]
# L121
max_diff_per_step = diff_seq.max(dim=-1)[0].max(dim=0)[0] # [T]
# L130
TWO_PI = 2.0 * PI
# L131
abs_diff = torch.abs(x_py - x_cuda)
# L133
tor_dist = torch.min(rem_diff, TWO_PI - rem_diff)
```

#### tests\diagnostics\conservation_audit.py
```python
# Contexto: test_energy_conservation
# L25
v_norms = [curr_v.norm().item()]
# L36
end_v = v_norms[-1]
# L37
retention = end_v / (start_v + 1e-9)
```

#### tests\diagnostics\depth_audit.py
```python
# Contexto: print_metric
# L11
status = "\033[92m[GOOD]\033[0m" if passed else "\033[91m[WEAK]\033[0m" print(f"{status} {name:<40}: {value:.4f} {unit}") def test_sequence_curriculum(): """Test parity convergence across different sequence lengths.""" print("\n--- TEST 1: SEQUENCE LENGTH CURRICULUM ---") lengths = [5, 10, 15, 20] dim = 128 for L in lengths: model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32, physics_config={'readout': {'type': 'implicit'}}) optimizer = torch.optim.AdamW(model.parameters(), lr=0.005) converged = False best_acc = 0 for step in range(200): model.train() optimizer.zero_grad() inputs = torch.randint(0, 2, (64, L)) targets = (inputs.sum(dim=-1) % 2).unsqueeze(-1).float() logits, _, _ = model(inputs) last_logits = logits[:, -1, 0:1] loss = F.binary_cross_entropy_with_logits(last_logits, targets) loss.backward() optimizer.step() model.readout.update_step() acc = ((last_logits > 0) == targets).float().mean().item() best_acc = max(best_acc, acc) if acc > 0.98: converged = True print(f"L={L:<2} | CONVERGED at step {step}") break if not converged: print(f"L={L:<2} | \033[91mFAILED\033[0m | Best Acc: {best_acc:.2%}") def audit_gradient_decay(): """Measure gradient magnitude at different sequence positions.""" print("\n--- TEST 2: GRADIENT PATH DECAY (L=20) ---") L = 20 dim = 128 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.train() inputs = torch.randint(0, 2, (1, L)) logits, (state_x, state_v), _ = model(inputs) grads = [] for t in range(L): model.zero_grad() target = torch.tensor([[[1.0]]]) loss = F.mse_loss(logits[:, t:t+1, 0:1], target) loss.backward(retain_graph=True) grad_norm = model.layers[0].christoffels[0].W.grad.abs().mean().item() grads.append(grad_norm) for i, g in enumerate(grads): print(f"Step {i+1:02} Gradient Energy: {g:.2e}") decay = grads[0] / (grads[-1] + 1e-9) print_metric("Gradient Decay (Early/Late)", decay, threshold=10.0) def audit_state_saturation(): """Check if state x hits clamping limits during long sequences.""" print("\n--- TEST 3: STATE SATURATION AUDIT (L=50) ---") L = 50 dim = 64 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.eval() inputs = torch.ones(1, L).long() with torch.no_grad(): logits, (fx, fv), _ = model(inputs) curr_x = torch.zeros(1, dim) curr_v = torch.zeros(1, dim) forces = model.embedding(inputs) max_norms = [] for t in range(L): curr_x, curr_v, _, _ = model.layers[0](curr_x, curr_v, forces[:, t]) max_norms.append(curr_x.norm().item()) print(f"Max Norm at t=1:  {max_norms[0]:.2f}") print(f"Max Norm at t=25: {max_norms[24]:.2f}") print(f"Max Norm at t=50: {max_norms[-1]:.2f}") saturation_risk = max_norms[-1] > 80.0 print_result = "\033[91m[SATURATED]\033[0m" if saturation_risk else "\033[92m[SAFE]\033[0m" print(f"{print_result} Clamping Ceiling (100.0) Health") if __name__ == "__main__": print("====================================================") print("   GFN DEPTH & SCALING AUDIT (Phase 10)            ") print("====================================================") try: audit_state_saturation() audit_gradient_decay() test_sequence_curriculum() except Exception as e: print(f"Audit crashed: {e}") import traceback traceback.print_exc() print("\n====================================================")
```

#### tests\diagnostics\grad_probe.py
```python
# Contexto: run_probe
# L21
print(f"--- Gradient Probe (L={seq_len}) ---")
# L53
flat_v = v_abs.view(-1)
# L56
saturated_count = (flat_v >= 14.9).sum().item()
# L58
saturation_rate = 100.0 * saturated_count / total_params
# L61
v_mean = flat_v.mean().item()
# L72
grad_norm = model.x0.grad.norm().item()
```

#### tests\diagnostics\manifold_audit.py
```python
# Contexto: print_result
# L9
color = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m" print(f"{color} {name:<40} {details}") def audit_gradient_energy(): """Test 1: Do we have non-trivial gradients reaching the weights?""" print("\n--- TEST 1: GRADIENT ENERGY ---") model = Manifold(vocab_size=10, dim=128, depth=1, heads=1,rank=32, physics_config={'readout': {'type': 'implicit'}}) model.train() coord_dim = 16 x = torch.randint(1, 10, (1, 20)) y = torch.randint(0, 2, (1, 20, coord_dim)).float() logits, _ , _ = model(x) loss = F.binary_cross_entropy_with_logits(logits, y) loss.backward() max_grad_w = 0 max_grad_u = 0 for layer in model.layers: for head in layer.christoffels: max_grad_w = max(max_grad_w, head.W.grad.abs().max().item()) max_grad_u = max(max_grad_u, head.U.grad.abs().max().item()) print_result("Manifold W Gradient Energy", max_grad_w > 1e-6, f"Max: {max_grad_w:.2e}") print_result("Manifold U Gradient Energy", max_grad_u > 1e-6, f"Max: {max_grad_u:.2e}") readout_grad = model.readout.mlp[0].weight.grad.abs().max().item() print_result("Readout MLP Gradient Energy", readout_grad > 1e-6, f"Max: {readout_grad:.2e}") def audit_state_persistence(): """Test 2: Does the state x actually accumulate over time without LayerNorm?""" print("\n--- TEST 2: STATE PERSISTENCE (HISTORY) ---") dim = 64 model = Manifold(vocab_size=10, dim=dim, depth=1, heads=1, rank=32) model.eval() seq_len = 20 x_input = torch.ones(1, seq_len).long() t0_x = torch.zeros(1, dim) t0_v = torch.zeros(1, dim) x_history = [] curr_x, curr_v = t0_x, t0_v with torch.no_grad(): all_forces = model.embedding(x_input) for t in range(seq_len): curr_x, curr_v, _, _ = model.layers[0](curr_x, curr_v, force=all_forces[:, t]) x_history.append(curr_x.clone()) x_seq = torch.stack(x_history, dim=1) # [1, L, D] dist = torch.norm(x_seq[0, -1]).item() std_val = x_seq.std().item() print_result("Trajectory Integration", dist > 0.5, f"Final Norm: {dist:.2f}") print_result("State Dynamic Range", std_val > 0.05, f"Std: {std_val:.2f}") def audit_mini_parity(): """Test 3: Can we solve 4-bit parity in 50 steps?""" print("\n--- TEST 3: MINI-PARITY CONVERGENCE ---") dim = 64 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32, physics_config={'readout': {'type': 'implicit'}}) optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) inputs = torch.randint(0, 2, (32, 4)) targets = (inputs.sum(dim=-1) % 2).unsqueeze(-1).float() # [32, 1] start_time = time.time() best_loss = 100 for step in range(101): model.train() optimizer.zero_grad() logits, (final_x, _), _ = model(inputs) last_logits = logits[:, -1, 0:1] loss = F.binary_cross_entropy_with_logits(last_logits, targets) loss.backward() optimizer.step() model.readout.update_step() acc = ((last_logits > 0) == targets).float().mean().item() best_loss = min(best_loss, loss.item()) if acc == 1.0 and loss < 0.1: print_result("Parity Solver (L=4)", True, f"Converged at step {step} (Loss: {loss.item():.4f})") return print_result("Parity Solver (L=4)", False, f"Best Loss: {best_loss:.4f}, Accuracy: {acc:.2f}") def audit_physical_limits(): """Test 4: Check for NaNs and Singularities""" print("\n--- TEST 4: PHYSICAL INTEGRITY ---") model = Manifold(vocab_size=10, dim=128, depth=2, heads=4, rank=3) # Low rank to stress test x = torch.randint(0, 10, (128, 50)) logits, (fx, fv), _ = model(x) has_nan = torch.isnan(logits).any().item() or torch.isnan(fx).any().item() max_val = fx.abs().max().item() print_result("NaN Stability", not has_nan, "No NaNs found" if not has_nan else "!!! NaNs DETECTED !!!") print_result("Clamping Effectiveness", max_val <= 101.0, f"Max State: {max_val:.2f}") if __name__ == "__main__": print("====================================================") print("   GFN MANIFOLD MASTER AUDIT (v3.8 Diagnostic)     ") print("====================================================") try: audit_gradient_energy() audit_state_persistence() audit_physical_limits() audit_mini_parity() except Exception as e: print(f"\033[91m[CRITICAL ERROR]\033[0m Audit crashed: {e}") import traceback traceback.print_exc() print("\n====================================================") print("                AUDIT COMPLETE                      ") print("====================================================")
```

#### tests\diagnostics\parity_probe.py
```python
# Contexto: probe_latent_clusters
# L24
targets = (inputs.sum(dim=-1) % 2).long()
# L40
center0 = x0.mean(axis=0)
# L41
center1 = x1.mean(axis=0)
# L43
inter_dist = np.linalg.norm(center0 - center1)
# L44
intra_std0 = x0.std(axis=0).mean()
# L45
intra_std1 = x1.std(axis=0).mean()
# L47
print(f"\n--- CLUSTER METRICS (L={L}) ---")
# L51
sep_ratio = inter_dist / (intra_std0 + intra_std1 + 1e-9)
# L52
status = "\033[92m[GOOD]\033[0m" if sep_ratio > 1.2 else "\033[91m[COLLAPSED]\033[0m" print(f"Separability Ratio:            {sep_ratio:.4f} {status}") if sep_ratio < 0.5: print("\n\033[91mWARNING:\033[0m Manifold states for Parity 0/1 are nearly identical.") print("The manifold is not 'steering' the particle based on inputs.") try: pca = PCA(n_components=2) x_pca = pca.fit_transform(x_latent) expl_var = pca.explained_variance_ratio_.sum() print(f"PCA Variance Explained (2D): {expl_var:.2%}") except: pass def probe_force_signal_ratio(): """Measure the ratio of Christoffel force vs Input force.""" print("\n--- SIGNAL RATIO TEST ---") dim = 128 model = Manifold(vocab_size=2, dim=dim, depth=1, heads=1, rank=32) model.eval() force_1 = model.embedding(torch.tensor([[1]])) # [1, 1, 128] f_norm = force_1.norm().item() v_unit = torch.randn(1, dim) v_unit = v_unit / v_unit.norm() with torch.no_grad(): gamma = model.layers[0].christoffels[0](v_unit) g_norm = gamma.norm().item() print(f"Token Impulse Norm (|F|):    {f_norm:.4f}") print(f"Manifold Resitance Norm (|Γ|): {g_norm:.4f}") ratio = f_norm / (g_norm + 1e-9) print(f"Force/Curvature Ratio:        {ratio:.2f}x") if ratio > 10.0: print("\033[93m[IMBALANCE]\033[0m Token force dominates geometry. Manifold is too 'soft'.") elif ratio < 0.1: print("\033[93m[IMBALANCE]\033[0m Geometry dominates tokens. Manifold is too 'stiff'.") else: print("\033[92m[BALANCED]\033[0m Dynamics are in the steerable regime.") if __name__ == "__main__": try: probe_force_signal_ratio() probe_latent_clusters() except Exception as e: print(f"Probe failed: {e}") import traceback traceback.print_exc() print("\n====================================================")
```

#### tests\diagnostics\test_loss_evolution.py
```python
# Contexto: test_gradient_flow
# L18
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# L32
logits, (x_final, v_final), *_ = model(x_task)
# L44
total_grad += p.grad.norm().item()
```

#### tests\diagnostics\test_suite_comprehensive.py
```python
# Contexto: func
# L65
test = gradcheck(func, (embeddings,), eps=1e-6, atol=1e-3, rtol=1e-2)
# Contexto: run_free_motion_check
# L128
final_norm = xF.norm().item()
```

#### tests\diagnostics\verify_convergence_dual.py
```python
# Contexto: first_head_metric
# L77
total_loss = loss_val + loss_phy + loss_ham
# L86
two_pi = 2.0 * pi
# L87
half_pi = pi * 0.5
# L88
dist_pos = torch.min(torch.abs(x_pred - half_pi) % two_pi, two_pi - (torch.abs(x_pred - half_pi) % two_pi))
# L89
dist_neg = torch.min(torch.abs(x_pred + half_pi) % two_pi, two_pi - (torch.abs(x_pred + half_pi) % two_pi))
# L90
d_pos = dist_pos.mean(dim=-1)
# L91
d_neg = dist_neg.mean(dim=-1)
# L93
acc = (preds == y_int).float().mean().item()
# L94
print(f"Step {step}: Loss={total_loss.item():.4f}, Acc={acc*100:.1f}%")
# L96
duration = time.time() - start_time
# L114
print("\n" + "=" * 60)
# L120
converged_py = losses_py[-1] < losses_py[0] * 0.8
# L121
converged_cuda = losses_cuda[-1] < losses_cuda[0] * 0.8
# Contexto: run_experiment
# L16
print(f"\n{'='*60}")
# L18
print(f"{'='*60}")
# L33
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
# L49
y_angle = (y_int.float() * 2.0 - 1.0) * (pi * 0.5)
# L59
y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
```

#### tests\functional\test_curiosity.py
```python
# Contexto: test_curiosity_noise_flow
# L49
force_high = torch.ones(100, dim) * 10.0
# L52
std_low = v_low.std().item()
# L53
std_high = v_high.std().item()
```

#### tests\functional\test_curiosity_exploration.py
```python
# Contexto: test_confusion_correlation
# L74
v = torch.zeros(1000, self.dim // self.heads) # Head dim
# L77
v_zero = noise_mod(v, force=torch.zeros(1000, self.dim // self.heads))
# L78
std_zero = v_zero.std().item()
# L81
v_high = noise_mod(v, force=torch.ones(1000, self.dim // self.heads) * 5.0)
# L82
std_high = v_high.std().item()
# Contexto: test_exploration_coverage
# L38
force = torch.ones(batch, self.dim) * 5.0
# L58
var_on = max(var_on, x_on.var(dim=0).mean().item())
# L59
var_off = max(var_off, x_off.var(dim=0).mean().item())
```

#### tests\functional\test_noether.py
```python
# Contexto: test_gradient_flow
# L78
has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in layer.parameters())
# Contexto: test_noether_loss_penalizes_divergence
# L48
c1 = c0 + 0.1 * torch.randn(4, 4) # Perturb
# Contexto: test_noether_loss_zero_for_identical_heads
# L22
head_dim = dim // heads
```

#### tests\functional\test_time_dilation.py
```python
# Contexto: test_thermo_gating_module
# L29
v_low = torch.zeros(batch, dim) # K=0, U=0 -> H=0. H < Ref (5.0). Gate > 0.5
# L34
x_high = torch.randn(batch, dim) * 10
# L35
v_high = torch.randn(batch, dim) * 10 # H >> 5.0. Gate < 0.5
```

#### tests\geometry\test_confusion.py
```python
# Contexto: forward
# L16
def forward(self, v, x, force=None, **kwargs):
# Contexto: test_mlayer_integration
# L60
head_dim = dim // heads
```

#### tests\geometry\test_holographic.py
```python
# Contexto: forward
# L15
def forward(self, v, x=None, **kwargs):
# Contexto: test_radial_clamping
# L39
x = torch.ones(1, dim) * 1000.0
```

#### tests\geometry\test_thermo_metric.py
```python
# Contexto: forward
# L16
def forward(self, v, x, force=None, **kwargs):
# Contexto: test_mlayer_integration
# L65
head_dim = dim // heads
# Contexto: test_thermo_modulation
# L33
self.assertTrue(torch.allclose(out_low, torch.ones_like(out_low), atol=1e-4))
# L38
force_high = torch.ones(batch, dim) * 100.0
```

#### tests\geometry\test_torus.py
```python
# Contexto: test_torus_wrapping
# L27
v_state = torch.ones(B, D).cuda() * 50.0 # Direction matters, mag normalized
# L31
U = torch.randn(1*D, D, 16).cuda() * 0.01
# L32
W = torch.randn(1*D, D, 16).cuda() * 0.01
# L71
TWO_PI = 2 * math.pi
# L82
diff = torch.abs(x_seq_euc - x_seq_tor).mean()
```

#### tests\integration\test_full_training.py
```python
# Contexto: test_energy_conservation_basic
# L92
x = small_model.x0.expand(1, -1)
# L93
v = small_model.v0.expand(1, -1)
# L101
energy = (v ** 2).sum().item()
# L109
drift = abs(energies[-1] - energies[0]) / (energies[0] + 1e-8)
# Contexto: test_forward_pass
# L54
logits, (x, v), *_ = small_model(inputs)
# Contexto: test_parameter_count
# L200
params = sum(p.numel() for p in model.parameters()) / 1e6
# L207
pytest.main([__file__, "-v", "--tb=short"])
```

#### tests\integration\test_overfit_sanity.py
```python
# Contexto: run_training_task
# L50
lr = config.get('lr', 1e-3)
# L78
shift_logits = logits[:, :-1, :].contiguous()
# L110
duration = end_time - start_time
```

#### tests\integration\test_vnext_stack.py
```python
# Contexto: test_vnext_full_stack
# L37
loss = output[0].sum()
```

#### tests\optimization\analyze.py
```python
# Contexto: analyze_results
# L81
parser.add_argument("--output", type=str, default="optimization_report.md", help="Output report file")
```

#### tests\optimization\config_space.py
```python
# Contexto: get_grid_search_configs
# L64
combinations = list(itertools.product(*values))
# Contexto: get_random_search_configs
# L73
def get_random_search_configs(n_samples: int = 20) -> List[Dict[str, Any]]:
# Contexto: HyperparameterConfig
# L19
SEARCH_SPACE = [ HyperparameterConfig( name="DEFAULT_LR", values=[1e-4, 5e-4, 1e-3, 5e-3], description="Learning rate" ), HyperparameterConfig( name="EMBEDDING_SCALE", values=[1.0, 1.5, 2.0], description="Scale for input embeddings" ), HyperparameterConfig( name="READOUT_GAIN", values=[1.0, 2.0, 5.0], description="Gain for the final readout layer" ), HyperparameterConfig( name="FRICTION_SCALE", values=[0.0, 0.02, 0.05, 0.1], description="Friction coefficient for symplectic integrators" ), HyperparameterConfig( name="DEFAULT_DT", values=[0.01, 0.02, 0.05, 0.1], description="Time step for integration" ), HyperparameterConfig( name="LEAPFROG_SUBSTEPS", values=[1, 3, 5], description="Number of substeps for Leapfrog integrator" ), HyperparameterConfig( name="LAMBDA_H_DEFAULT", values=[0.0, 0.001, 0.01], description="Hamiltonian regularization weight" ) ]
```

#### tests\optimization\runner.py
```python
# Contexto: main
# L94
parser.add_argument("--mode", choices=["grid", "random", "smoke"], default="smoke", help="Search strategy")
# L95
parser.add_argument("--samples", type=int, default=10, help="Number of samples for random search")
# L96
parser.add_argument("--output", type=str, default="optimization_results.csv", help="Output CSV file")
# L124
total_time = time.time() - start_time
# Contexto: run_trial
# L70
result = { "trial_id": trial_id, "status": "COMPLETED" if train_metrics["success"] else "FAILED", **config, **train_metrics }
# L79
result = { "trial_id": trial_id, "status": "ERROR", "error": str(e), **config }
```

#### tests\optimization\test_optimizer.py
```python
# Contexto: test_orthogonal_preservation_cayley
# L34
grad_skew_error = torch.norm(grad_skewed + grad_skewed.t())
# L41
ortho_error = torch.norm(p.data @ p.data.t() - identity)
# Contexto: test_sphere_projection
# L76
p.data = p.data / p.data.norm() # On sphere
# L86
proj_norm = projected_grad.norm().item()
# L87
print(f"DEBUG: p_norm={p.data.norm().item()}, grad_norm={p.grad.data.norm().item()}")
# Contexto: test_torus_wrapping_and_transport
# L56
p.grad = torch.full((1, dim), -1.0) # p = p - lr * (-1) = 3.1 + 0.5 = 3.6 > pi
# L65
self.assertAlmostEqual(val, 3.6 - 2*math.pi, places=4)
```

#### tests\physics\test_adaptive_resolution.py
```python
# Contexto: test_adaptive_execution_flow
# L27
adaptive = AdaptiveIntegrator(base, tolerance=1e-5, max_depth=2)
# Contexto: test_mlayer_integration
# L44
config = { 'active_inference': { 'adaptive_resolution': { 'enabled': True, 'tolerance': 1e-4, 'max_depth': 2 } } }
```

#### tests\physics\test_energy_conservation.py
```python
# Contexto: __init__
# L33
self.results_dir = PROJECT_ROOT / "tests" / "professional" / "results"
# Contexto: run_comprehensive_suite
# L340
print("\n" + "=" * 60)
# Contexto: test_adversarial_stability
# L261
x = self.model.x0.expand(1, -1)
# L262
v = self.model.v0.expand(1, -1)
# L274
energy = (v ** 2).sum().item()
# L284
spike = max(energies) / (energies[0] + 1e-8)
# L287
results[pattern_name] = { "nan_frequency": nan_count / num_trials, "max_energy_spike": max_energy_spike }
# Contexto: test_integrator_comparison
# L184
x = model.x0.expand(1, -1)
# L185
v = model.v0.expand(1, -1)
# L193
energy = (v ** 2).sum().item()
# L196
energies = np.array(energies)
# L198
drift = abs(energies[-1] - initial) / (initial + 1e-8)
# L201
results[int_type] = { "drift": drift, "energies": energies, "mean_energy": np.mean(energies), "std_energy": np.std(energies) }
# L211
ax.axhline(y=initial, color='r', linestyle='--', alpha=0.5, label='Initial')
# L214
ax.set_title(f'{int_type.upper()}\nDrift: {drift*100:.2f}%', fontsize=13)
# L222
plt.savefig(self.results_dir / "integrator_comparison.png", dpi=300, bbox_inches='tight')
# Contexto: test_long_sequence_drift
# L63
x = self.model.x0.expand(1, -1)
# L64
v = self.model.v0.expand(1, -1)
# L78
energy = (v ** 2).sum().item()
# L83
energies = np.array(energies)
# L85
final_energy = energies[-1]
# L88
relative_drift = abs(final_energy - initial_energy) / (initial_energy + 1e-8)
# L91
max_deviation = np.max(np.abs(energies - initial_energy)) / (initial_energy + 1e-8)
# L94
stability_score = max(0.0, 1.0 - relative_drift / tolerance)
# L102
plt.axhline(y=initial_energy, color='r', linestyle='--', label='Initial Energy', alpha=0.7)
# L107
alpha=0.2, color='green', label=f'±{tolerance*100:.0f}% Tolerance'
# L111
plt.title(f'Hamiltonian Energy Conservation (Drift: {relative_drift*100:.2f}%)', fontsize=14)
# L117
rel_deviations = (energies - initial_energy) / (initial_energy + 1e-8) * 100
# L119
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
# L120
plt.axhline(y=tolerance*100, color='r', linestyle='--', alpha=0.5, label=f'Tolerance: ±{tolerance*100:.0f}%')
# L121
plt.axhline(y=-tolerance*100, color='r', linestyle='--', alpha=0.5)
# L129
plt.savefig(self.results_dir / "energy_conservation_long_sequence.png", dpi=300, bbox_inches='tight')
```

#### tests\physics\test_geodesic_optimality.py
```python
# Contexto: __init__
# L35
self.results_dir = PROJECT_ROOT / "tests" / "results" / "geodesic"
# Contexto: _visualize_paths_3d
# L172
color='#E76F51', linestyle='--', alpha=0.7)
# L175
ax.scatter(*geo_3d[0], s=200, c='green', marker='o', label='Start', edgecolors='black', linewidths=2)
# L176
ax.scatter(*geo_3d[-1], s=200, c='red', marker='X', label='End', edgecolors='black', linewidths=2)
# L187
plt.savefig(self.results_dir / "geodesic_path_3d.png", dpi=300, bbox_inches='tight')
# Contexto: compute_path_length
# L42
For a Riemannian manifold, path length = ∫ ||dx/dt|| dt
# L56
displacement = trajectory[i+1] - trajectory[i]
# L57
step_length = torch.norm(displacement, dim=-1).mean().item()
# L58
total_length += step_length
# Contexto: run_geodesic_tests
# L371
print("\n" + "=" * 60)
# Contexto: test_action_minimization
# L277
x = self.model.x0.expand(1, -1)
# L278
v = self.model.v0.expand(1, -1)
# L289
gfn_action = sum((v ** 2).sum().item() for v in velocities)
# L300
noise = torch.randn_like(v) * noise_scale
# L301
v_pert = v + noise
# L302
perturbed_action += (v_pert ** 2).sum().item()
# L306
mean_perturbed = np.mean(perturbed_actions)
# L318
plt.axvline(gfn_action, color='red', linestyle='--', linewidth=2.5, label='GFN Geodesic')
# L324
plt.savefig(self.results_dir / "action_minimization.png", dpi=300, bbox_inches='tight')
# Contexto: test_curved_vs_straight
# L83
x = self.model.x0.expand(1, -1)
# L84
v = self.model.v0.expand(1, -1)
# L96
end_point = geodesic_trajectory[-1]
# L100
alpha = t / (seq_length - 1)
# L101
interpolated = (1 - alpha) * start_point + alpha * end_point
# L111
deviation = torch.norm(geodesic_trajectory[t] - straight_trajectory[t]).item()
# L115
mean_deviation = np.mean(deviations)
# L127
plt.savefig(self.results_dir / "geodesic_deviation.png", dpi=300, bbox_inches='tight')
# Contexto: test_manifold_curvature_field
# L200
head_dim = self.model.dim // heads
# L205
x = np.linspace(-3, 3, grid_size)
# L206
y = np.linspace(-3, 3, grid_size)
# L207
X, Y = np.meshgrid(x, y)
# L209
curvatures = np.zeros((grid_size, grid_size))
# L224
curvatures[i, j] = torch.norm(gamma).item()
# L228
im = plt.imshow(curvatures, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis', aspect='auto')
# L238
plt.savefig(self.results_dir / "curvature_field.png", dpi=300, bbox_inches='tight')
# L242
mean_curv = np.mean(curvatures)
# L243
max_curv = np.max(curvatures)
# L244
std_curv = np.std(curvatures)
```

#### tests\physics\test_gradients.py
```python
# Contexto: check_gradients
# L35
loss = criterion(logits.view(-1, vocab), target.view(-1))
# L47
g_norm = param.grad.norm().item()
```

#### tests\physics\test_gradients_deep.py
```python
# Contexto: compute_numerical_gradients
# L25
def compute_numerical_gradients(func, inputs, output_indices=None, eps=1e-5):
# L41
outputs = func(*inputs)
# L59
flat_input = input_tensor.view(-1)
# L60
flat_grad = grad.view(-1)
# L61
original_flat = original_data.view(-1)
# L65
original_flat[j] += eps
# L67
outputs_plus = func(*inputs)
# L71
original_flat[j] -= 2 * eps
# L73
outputs_minus = func(*inputs)
# L78
original_flat[j] += eps
# L85
diff = (plus - minus) / (2 * eps)
# L86
grad_sum += diff.sum().item()
# Contexto: forward_func
# L197
tolerances = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]  # Tolerances per parameter
# L217
abs_diff = torch.abs(grad_cuda - grad_num)
# L218
rel_diff = abs_diff / (torch.abs(grad_num) + 1e-8)
# L222
mean_abs_error = abs_diff.mean().item()
# L223
mean_rel_error = rel_diff.mean().item()
# L234
if max_abs_error <= tol and max_rel_error <= 10 * tol:
# L242
cuda_val = grad_cuda.view(-1)[max_idx].item()
# L243
num_val = grad_num.view(-1)[max_idx].item()
# Contexto: test_backward_vs_numerical
# L122
W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)  # For Torus topology
# L155
features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) if topology == 1 else x
# L156
mu = torch.sigmoid(torch.matmul(features, W_forget.t()) + b_forget) * 5.0  # FRICTION_SCALE
# L157
friction_output = gamma + mu * v
# L162
forward_diff = torch.abs(output_cuda - friction_output).max().item()
# L179
loss = output_cuda.sum()
# Contexto: test_gradient_checking_pytorch
# L271
print("\n" + "=" * 70)
# L292
W_forget = torch.randn(dim, 2*dim, device=device, requires_grad=True, dtype=torch.float64)
# L301
result = gradcheck( LowRankChristoffelWithFrictionFunction.apply, test_input, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False )
# L331
print("\n" + "=" * 70)
```

#### tests\physics\test_leapfrog_stability.py
```python
# Contexto: test_leapfrog_backward
# L46
loss = (x_new.sum() + v_new.sum())
# L61
print(f"   grad_x: mean={x.grad.mean().item():.6f}, std={x.grad.std().item():.6f}")
# L62
print(f"   grad_v: mean={v.grad.mean().item():.6f}, std={v.grad.std().item():.6f}")
# L63
print(f"   grad_U: mean={U.grad.mean().item():.6f}, std={U.grad.std().item():.6f}")
# L64
print(f"   grad_W: mean={W.grad.mean().item():.6f}, std={W.grad.std().item():.6f}")
# L79
print("\n" + "=" * 60)
```

#### tests\physics\test_mechanics.py
```python
# Contexto: test_mechanics
# L48
loss_std = criterion(logits_std.view(-1, vocab_size), target.view(-1))
# L63
loss_adj = criterion(logits_adj.view(-1, vocab_size), target.view(-1))
# L90
head_dim = dim // heads
# L96
E_start = p.pow(2).sum(dim=-1).mean()
# L105
E_end = p_new.pow(2).sum(dim=-1).mean()
# L106
energy_diff = torch.abs(E_start - E_end).item()
```

#### tests\physics\test_pefrl.py
```python
# Contexto: energy
# L56
drift = torch.abs(e_final - e_init) / e_init
# Contexto: HarmonicChristoffel
# L14
Simple Harmonicoscillator: Force = -k*x
# L16
Energy H = 1/2 v^2 + 1/2 x^2
# Contexto: test_symplectic_jacobian
# L76
outputs = torch.cat([next_x, next_v], dim=-1).squeeze(0)
# L87
jacobian.append(torch.cat([grad_x, grad_v], dim=-1))
```

#### tests\physics\test_stochastic.py
```python
# Contexto: forward
# L19
def forward(self, v, x=None, **kwargs):
```

#### tests\proofs\test_chaos_prediction.py
```python
# Contexto: autoregress
# L175
next_tok = torch.argmax(logits[:, -1:], dim=-1)
# Contexto: continuous_to_tokens
# L91
norm = (tensor - MIN_VAL) / (MAX_VAL - MIN_VAL)
# L92
tokens = (norm * BINS).long().clamp(0, BINS-1)
# L114
manifold_opt = optim.AdamW(manifold.parameters(), lr=1e-3)
# L121
lstm_opt = optim.AdamW(list(lstm.parameters()) + list(lstm_embed.parameters()) + list(lstm_head.parameters()), lr=1e-3)
# L136
inp = batch[:, :-1]
# L142
loss_m = criterion(logits_m.reshape(-1, BINS), tgt.reshape(-1))
# L151
loss_l = criterion(logits_l.reshape(-1, BINS), tgt.reshape(-1))
# Contexto: double_pendulum_derivs
# L19
c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
# L25
num1 = -g*(2*m1+m2)*np.sin(theta1) - m2*g*np.sin(theta1-2*theta2) - 2*s*m2*(z2**2*L2 + z1**2*L1*c)
# L26
den1 = L1 * (2*m1 + m2 - m2*c*2*c) # Typo in standard formula? standard is -m2*cos(2*delta)
# L29
den1 = L1 * (2*m1 + m2 - m2*np.cos(2*(theta1-theta2)))
# L31
z1_dot = num1 / den1
# L33
num2 = 2*s*(z1**2*L1*(m1+m2) + g*(m1+m2)*np.cos(theta1) + z2**2*L2*m2*c)
# L34
den2 = L2 * (2*m1 + m2 - m2*np.cos(2*(theta1-theta2)))
# L35
z2_dot = num2 / den2
# Contexto: generate_chaos_data
# L43
t = np.arange(0, seq_len*dt, dt)
# L49
init_state = np.random.uniform(-0.5, 0.5, 4) + np.array([np.pi, 0, np.pi, 0])
# L53
return torch.tensor(np.array(data), dtype=torch.float32)
# Contexto: pred_lstm
# L197
err_m = (fut_m.float() - future.float()).abs().mean().item()
# L198
err_l = (fut_l.float() - future.float()).abs().mean().item()
# Contexto: run_chaos_test
# L72
data = generate_chaos_data(1000, SEQ_LEN + PRED_LEN)
# L88
MIN_VAL, MAX_VAL = -10, 10
```

#### tests\proofs\test_infinite_arithmetic.py
```python
# Contexto: __getitem__
# L32
len1 = np.random.randint(1, self.length + 1)
# L33
len2 = np.random.randint(1, self.length + 1)
# L35
num1 = [np.random.randint(0, self.base) for _ in range(len1)]
# L36
num2 = [np.random.randint(0, self.base) for _ in range(len2)]
# L41
res = val1 + val2
# L48
src = torch.tensor(num1 + [self.PLUS] + num2 + [self.EQ], dtype=torch.long)
# L49
tgt = torch.tensor(res_digits + [self.EOS], dtype=torch.long)
# Contexto: collate_fn
# L55
srcs, tgts = zip(*batch)
# Contexto: train_and_eval
# L90
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# L98
pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
# L113
full_seq = torch.cat([src, tgt], dim=1) # [B, S+T]
# L114
input_seq = full_seq[:, :-1]
# L122
loss = criterion(logits.reshape(-1, 13), target_seq.reshape(-1))
# L127
total_loss += loss.item()
# L160
input_token = curr_seq[:, -1:] # Last token (EQ)
# L164
next_token = torch.argmax(logits, dim=-1)
# L179
if is_correct: correct += 1
```

#### tests\proofs\test_needle_haystack.py
```python
# Contexto: generate_batch
# L40
kv_seq = torch.zeros((batch_size, KEY_VAL_PAIRS * 2), dtype=torch.long)
# L75
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# L90
print(f"[*] Training on Noise Length={TRAIN_NOISE}...")
# L112
last_logit = logits[:, -1, :] # Prediction for next token after Query
# L119
acc = (torch.argmax(last_logit, dim=-1) == tgt).float().mean()
# L124
print(f"\n[*] testing on Infinite Haystack (Length={NOISE_LEN})...")
# L134
last_logit = logits[:, -1, :]
# L135
pred = torch.argmax(last_logit, dim=-1)
# L137
duration = time.time() - start_time
# Contexto: run_needle_test
# L26
NOISE_LEN = 10000  # 10k tokens of noise (Standard Transformer would struggle/OOM with standard attn)
```

#### tests\run_consistency_tests.py
```python
# Contexto: print_test_summary
# L157
test_classes = [ ("TestCUDAAvailability", "CUDA device and constant verification"), ("TestChristoffelOperation", "Christoffel symbol computation tests"), ("TestLeapfrogIntegration", "Leapfrog integrator tests"), ("TestGradientConsistency", "Gradient computation verification"), ("TestCUDAVsPythonEquivalence", "CUDA vs Python numerical equivalence"), ("TestConvergenceBehavior", "Optimization convergence tests"), ("TestEdgeCases", "Edge case and boundary tests"), ("TestPerformanceBenchmarks", "Performance benchmarks"), ("TestTopologyBehavior", "Topology-specific behavior tests"), ("TestAutogradFunctionality", "Autograd function tests"), ("TestFullPipeline", "Full integration tests"), ]
# L187
parser = argparse.ArgumentParser( description="Run CUDA-Python consistency tests" )
# Contexto: run_quick_checks
# L127
loss = torch.sum(gamma)
# Contexto: run_tests
# L36
pytest_args = [ "-v", "--tb=short", "-x",  # Stop on first failure "tests/cuda/test_cuda_python_consistency.py" ]
# L47
pytest_args.append("--ignore-glob=*cuda*")
# L53
result = subprocess.run( [sys.executable, "-m", "pytest"] + pytest_args, cwd=Path(__file__).parent.parent, capture_output=False )
```

#### tests\run_suite.py
```python
# Contexto: run_suite
# L16
print("\n" + "=" * 70)
# L29
all_tests = loader.discover(start_dir, pattern='test_*.py', top_level_dir=str(PROJECT_ROOT))
# L44
scripts = [ ("tests/integration/test_overfit_sanity.py", "Overfit Diagnosis (Sanity Check)") ]
# L48
print("\n" + "=" * 70)
# L55
script_path = PROJECT_ROOT / script_rel
# L76
last_line = ret.stdout.strip().splitlines()[-1] if ret.stdout.strip() else ''
# L90
print("\n" + "=" * 70)
```

#### tests\test_initial_loss.py
```python
# Contexto: _standardize_forces
# L12
m = forces.mean(dim=(0, 1), keepdim=True)
# L13
s = forces.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
# Contexto: test_initial_loss_within_tolerance
# L45
y_expanded = y_angle.float().unsqueeze(-1).expand_as(x_pred)
# L48
assert abs(loss - 2.5) <= 0.25, f"Initial loss {loss:.2f} deviates more than 10% from 2.5"
```

#### tests\unit\conftest.py
```python
# Contexto: long_seq
# L66
@pytest.fixture(autouse=True)
```

#### tests\unit\test_active_physics.py
```python
# Contexto: test_reactive_curvature_plasticity
# L43
v_low = torch.randn(1, self.dim) * 0.01
# L47
v_high = torch.randn(1, self.dim) * 10.0 # High velocity
# Contexto: test_singularity_trigger
# L86
x_high = torch.ones(1, self.dim) * 10.0 # Sigmoid(large) -> 1.0 > 0.8
```

#### tests\unit\test_adaptive_physics.py
```python
# Contexto: test_dormand_prince_integration
# L33
diff = (x_new - x).abs().max().item()
# Contexto: test_rk45_accuracy
# L53
v_euler = v + a * 0.001
# L54
x_euler = x + v * 0.001
# L58
torch.testing.assert_close(x_rk, x_euler, rtol=1e-2, atol=1e-2)
```

#### tests\unit\test_curiosity.py
```python
# Contexto: test_curiosity_gradients
# L42
v = params * 0.1
# Contexto: test_curiosity_logic
# L16
v_high = [ torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), # Diverse directions torch.tensor([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]) ]
```

#### tests\unit\test_fractals.py
```python
# Contexto: test_fractal_tunneling_activation
# L55
r = torch.norm(stacked_gamma, dim=-1).mean()
```

#### tests\unit\test_geometric_enhancements.py
```python
# Contexto: test_dynamic_curvature_modulation
# L31
diff = (gamma_static - gamma_dynamic).abs().max().item()
# Contexto: test_parallel_mlayer_wormhole_scales
# L74
head_dim = dim // heads
```

#### tests\unit\test_geometry.py
```python
# Contexto: test_clamping
# L67
v = torch.randn(2, 64) * 100
# Contexto: test_gradient_flow
# L82
loss = gamma.sum()
# Contexto: test_no_singularities
# L142
x = torch.tensor([[3.14159] * 64, [0.0] * 64])  # π and 0
# Contexto: test_periodic_boundary
# L127
x2 = torch.ones(2, 64) * (2 * 3.14159)
# L133
assert torch.allclose(gamma1, gamma2, rtol=1e-2, atol=1e-3)
```

#### tests\unit\test_golden_integration.py
```python
# Contexto: test_mlayer_rk45_forward
# L40
loss = x_out.sum()
```

#### tests\unit\test_losses.py
```python
# Contexto: test_basic_penalty
# L119
assert loss.item() >= 0, "Penalty should be non-negative"
# Contexto: test_basic_regularization
# L91
assert loss.item() >= 0, "Regularization should be non-negative"
# Contexto: test_nonzero_loss_changing_energy
# L43
v2 = torch.ones(2, 64) * 2.0  # Double the energy
# L44
v3 = torch.ones(2, 64) * 3.0
# Contexto: test_periodic_boundary
# L148
x2 = torch.ones(2, 64) * (2 * 3.14159)  # 2π
```

#### tests\unit\test_recursive_geodesics.py
```python
# Contexto: test_recursive_context_propagation
# L53
loss = logits.sum()
```

#### tests\unit\test_scan.py
```python
# Contexto: test_basic_scan
# L25
a = torch.ones(B, L, D) * 0.9
# Contexto: test_numerical_stability
# L134
a = torch.ones(2, 10, 32) * 0.01
# L140
a = torch.ones(2, 10, 32) * 0.99
# Contexto: test_sequential_equivalence
# L47
h = a[:, t] * h + x[:, t]
# L52
rtol=1e-5, atol=1e-6,
```

#### tests\unit\test_symmetries.py
```python
# Contexto: test_noether_loss_consistency
# L58
c1_div = c0 + 1.0
```

#### tests\validate_toroidal_fixes.py
```python
# Contexto: test_boundary_consistency
# L42
x = torch.tensor([ 0.1,                # Within range 2 * math.pi + 0.1,  # Wrap positive -0.1,               # Wrap negative 4 * math.pi + 0.1   # Wrap multiple positive ])
# L52
expected = torch.tensor([0.1, 0.1, 2 * math.pi - 0.1, 0.1])
# L55
self.assertTrue(torch.allclose(wrapped, expected, atol=1e-5), f"Wrapping failed. Got {wrapped}, expected {expected}")
# L59
x_grad = torch.tensor([2 * math.pi], requires_grad=True)
# Contexto: test_fusion_manager_routing
# L173
layer.head_dim = 2 # dim 4 // heads 2
# Contexto: test_leapfrog_integration
# L87
x = torch.tensor([[2 * math.pi - 0.05, 0.0]])
```

#### tests\verify_leapfrog_parity.py
```python
# Contexto: verify_parity
# L32
v_init = torch.randn(batch_size, dim, device=device) * 0.1
# L59
diff_x = (x_py - x_cuda).abs().max().item()
# L60
diff_v = (v_py - v_cuda).abs().max().item()
# L72
geo_torus = ToroidalChristoffel(dim).to(device) # Should handle R/r internally
# L82
diff_x_t = (x_py_t - x_cuda_t).abs().max().item()
# L83
diff_v_t = (v_py_t - v_cuda_t).abs().max().item()
```

### Colección Completa CUDA (por archivo)
#### gfn\cuda\cuda_kernels.cpp
```cpp
// Contexto: Global
// L147
auto hidden = at::tanh(at::matmul(x, W1.t()) + b1);
// L148
auto out = at::matmul(hidden, W2.t()) + b2;
// L154
m.doc() = "GFN CUDA Kernels - High-performance manifold geometry and integration";
```

#### gfn\cuda\src\common\device_utils.cuh
```cpp
// Contexto: Global
// L22
T wrapped = atan2(sin(x), cos(x));
// L24
wrapped += static_cast<T>(TWO_PI<T>);
// L42
x[i] = apply_boundary_device<T>(x[i], topology);
// L58
T epsilon = static_cast<T>(EPSILON_STRONG<T>) ) { return numerator / (denominator + epsilon);
// L69
T min_val = static_cast<T>(CURVATURE_CLAMP_MIN<T>), T max_val = static_cast<T>(CURVATURE_CLAMP<T>) ) { return fmin(fmax(value, min_val), max_val);
// L81
T scale = static_cast<T>(CURVATURE_CLAMP<T>) ) { return scale * tanh(value / scale);
// L101
result += a[i] * b[i];
// L119
* Vector addition: c = a + b */ template <typename T> GFN_DEVICE void vector_add( T* c, const T* a, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {
// L129
c[i] = a[i] + b[i];
// L134
* Scaled vector addition: c = a + scale * b */ template <typename T> GFN_DEVICE void vector_add_scaled( T* c, const T* a, T scale, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {
// L145
c[i] = a[i] + scale * b[i];
// L150
* Vector scaling: b = scale * a */ template <typename T> GFN_DEVICE void vector_scale( T* b, T scale, const T* a, int dim ) { for (int i = 0; i < dim; ++i) {
// L160
b[i] = scale * a[i];
// L165
* Copy vector: dst = src */ template <typename T> GFN_DEVICE void vector_copy( T* dst, const T* src, int dim ) { for (int i = 0; i < dim; ++i) {
// L179
* Zero vector: v = 0 */ template <typename T> GFN_DEVICE void vector_zero( T* v, int dim ) { for (int i = 0; i < dim; ++i) {
// L222
val += __shfl_down_sync(0xffffffff, val, offset);
// L235
int wid = threadIdx.x / warpSize;
// L242
val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : static_cast<T>(0);
```

#### gfn\cuda\src\common\integrator_utils.cuh
```cpp
// Contexto: block_reduce_sum_shared
// L24
int wid = threadIdx.x / warpSize;
// L32
val = (threadIdx.x < blockDim.x / warpSize) ? shared_red[lane] : 0;
// Contexto: christoffel_distributed_shared
// L77
*gamma_val = 0.0f;
// L83
scalar_t v_ph = v_shared[tid + 1];
// L86
scalar_t denom = R + r * c;
// L88
scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L89
*gamma_val = term_th * (v_ph * v_ph) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
// L90
} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];
// L93
scalar_t v_th = v_shared[tid - 1];
// L96
scalar_t denom = R + r * c;
// L98
scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L99
*gamma_val = 2.0f * term_ph * v_ph * v_th * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
// L105
scalar_t prod = U[tid * rank + k] * v_val;
// L118
scalar_t norm = sqrt(energy);
// L119
S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// L126
sum_gamma += W[tid * rank + k] * h_shared[k] * h_shared[k] * S_shared * M_shared;
// L128
*gamma_val = sum_gamma;
// L135
scalar_t v_dot_gz = v_val * holo_grad_z[tid];
// L138
scalar_t v_sq = v_val * v_val;
// L149
scalar_t term_ads = -(1.0f / holo_z) * (2.0f * common_v_dot_gz * v_val - common_v_sq * holo_grad_z[tid]);
// L150
*gamma_val += term_ads; // Combine curvatures
// L157
scalar_t f_sq = f_val * f_val;
// L158
scalar_t head_energy = block_reduce_sum_shared(f_sq) / static_cast<scalar_t>(dim);
// L163
scalar_t modulator = expf(-thermo_alpha * head_energy / T);
// L164
*gamma_val *= modulator;
// L168
*gamma_val = soft_clamp<scalar_t>(*gamma_val, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
// Contexto: friction_distributed_shared
// L198
features_shared[dim + tid] = c;
// L204
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// L207
gate_sum += W_forget[tid * feat_dim + j] * features_shared[j];
// L215
gate_sum += W_input[tid * dim + j] * features_shared[j];
// L219
scalar_t base_friction = sigmoid(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// L221
scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L222
*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
// L224
*friction_val = base_friction;
// Contexto: warp_reduce_sum_shared
// L16
val += __shfl_down_sync(0xffffffff, val, offset);
```

#### gfn\cuda\src\common\math_utils.cuh
```cpp
// Contexto: Global
// L18
* Matrix-vector multiplication: y = A * x * A is [m x n], x is [n], y is [m] */ template <typename T> GFN_DEVICE void matvec( T* y, const T* A, const T* x, int m, int n ) { for (int i = 0; i < m; ++i) {
// L32
sum += A[i * n + j] * x[j];
// L39
* Transposed matrix-vector multiplication: y = A^T * x * A is [m x n], x is [m], y is [n] */ template <typename T> GFN_DEVICE void matvec_transpose( T* y, const T* A, const T* x, int m, int n ) { for (int j = 0; j < n; ++j) {
// L53
sum += A[i * n + j] * x[i];
// L60
* Outer product: C = a ⊗ b (element-wise) * Result: C[i] = a[i] * b[i] */ template <typename T> GFN_DEVICE void outer_product_elementwise( T* C, const T* a, const T* b, int dim ) { for (int i = 0; i < dim; ++i) {
// L71
C[i] = a[i] * b[i];
// L91
*s = std::sin(x);
// L92
*c = std::cos(x);
// L109
features[dim + i] = c;
```

#### gfn\cuda\src\common\types.cuh
```cpp
// Contexto: Global
// L16
template<typename T = float> using scalar_t = T;  // Use float precision to match PyTorch float32 and for better performance on consumer GPUs
// L34
constexpr T CURVATURE_CLAMP = static_cast<T>(3.0);
// L36
constexpr T CURVATURE_CLAMP_MIN = static_cast<T>(-3.0);
// L40
constexpr T FRICTION_SCALE = static_cast<T>(0.02);
// L42
constexpr T DEFAULT_FRICTION = static_cast<T>(0.002);
// L46
constexpr T PI = static_cast<T>(3.14159265358979323846);
// L48
constexpr T TWO_PI = static_cast<T>(6.28318530717958647692);
// L55
constexpr T EPSILON_STANDARD = static_cast<T>(1e-7);
// L57
constexpr T EPSILON_STRONG = static_cast<T>(1e-7);
// L59
constexpr T EPSILON_SMOOTH = static_cast<T>(1e-7);
// L63
constexpr T CLAMP_MIN_WEAK = static_cast<T>(1e-7);
// L65
constexpr T CLAMP_MIN_STRONG = static_cast<T>(1e-7);
// L69
constexpr T GATE_BIAS_OPEN = static_cast<T>(1.0);   // sigmoid(1) ≈ 0.73
// L71
constexpr T GATE_BIAS_CLOSED = static_cast<T>(-3.0); // sigmoid(-3) ≈ 0.05
// L75
constexpr T TOROIDAL_MAJOR_RADIUS = static_cast<T>(2.0);  // R
// L77
constexpr T TOROIDAL_MINOR_RADIUS = static_cast<T>(1.0);  // r
// L79
constexpr T TOROIDAL_CURVATURE_SCALE = static_cast<T>(0.01);
// L83
constexpr T DEFAULT_PLASTICITY = static_cast<T>(0.02);
// L85
constexpr T SINGULARITY_THRESHOLD = static_cast<T>(0.5);
// L87
constexpr T SINGULARITY_GATE_SLOPE = static_cast<T>(0.5);  // REDUCED from 10.0 for stability
// L89
constexpr T BLACK_HOLE_STRENGTH = static_cast<T>(1.5);
// L105
constexpr int MAX_THREADS_PER_BLOCK = 1024;
// L106
constexpr int DEFAULT_BLOCK_SIZE = 256;
```

#### gfn\cuda\src\geometry\christoffel_impl.cuh
```cpp
// Contexto: Global
// L19
* Implements: gamma_sym[i,j] = 0.5 * (gamma[i,j] + gamma[j,i]) * This ensures Gamma^k_ij approx Gamma^k_ji numerically, which is required * for torsion-free connections. * * @param gamma Input/Output Christoffel symbols [dim x dim] * @param dim Dimension of manifold */ template <typename T> GFN_DEVICE void normalize_christoffel_structure(T* gamma, int dim) { for (int i = 0; i < dim; ++i) {
// L31
T avg = static_cast<T>(0.5) * (gamma[i * dim + j] + gamma[j * dim + i]);
// L32
gamma[i * dim + j] = avg;
// L33
gamma[j * dim + i] = avg;
// L41
* Computes: Γ(v,v) = Σ_r (h_r^2 * W_r) * S * M * where: *   h = U^T * v (projection to rank-R space) *   S = 1 / (1 + ||h||)  (stabilization factor) *   M = modulation from plasticity and singularities * * @param v Velocity vector [dim] * @param U Low-rank matrix U [dim x rank] * @param W Low-rank matrix W [dim x rank] * @param x Position vector [dim] (optional, for friction/singularities) * @param V_w Potential weights [dim] (optional, for singularities) * @param dim Dimension of manifold * @param rank Rank of decomposition * @param plasticity Plasticity coefficient (energy-dependent curvature) * @param sing_thresh Singularity threshold * @param sing_strength Singularity strength multiplier * @param topology Topology type (EUCLIDEAN or TORUS) * @param R Toroidal major radius * @param r Toroidal minor radius * @param gamma Output Christoffel force [dim] */ template <typename T> GFN_DEVICE void christoffel_device( const T* v, const T* U, const T* W, const T* x, const T* V_w, int dim, int rank, T plasticity, T sing_thresh, T sing_strength, Topology topology, T R, T r, T* gamma ) { if (topology == Topology::TORUS && x != nullptr && V_w == nullptr) { for (int i = 0; i < dim; ++i) gamma[i] = static_cast<T>(0);
// L84
T v_ph = v[i + 1];
// L85
T denom = fmax(R + r * cos(th), static_cast<T>(CLAMP_MIN_STRONG<T>));
// L86
T term_th = denom * sin(th) / (r + static_cast<T>(EPSILON_SMOOTH<T>));
// L87
gamma[i] = term_th * (v_ph * v_ph);
// L88
T term_ph = -(r * sin(th)) / (denom + static_cast<T>(EPSILON_SMOOTH<T>));
// L89
gamma[i + 1] = static_cast<T>(2) * term_ph * v_ph * v_th;
// L92
gamma[i] = soft_clamp<T>(gamma[i] * static_cast<T>(TOROIDAL_CURVATURE_SCALE<T>), static_cast<T>(CURVATURE_CLAMP<T>));
// L101
sum += U[j * rank + i] * v[j];
// L107
energy += h[i] * h[i];
// L112
T norm_val = sqrt(energy);
// L114
T S = static_cast<T>(1) / (static_cast<T>(1) + norm_val + static_cast<T>(EPSILON_STRONG<T>));
// L119
v_energy += v[i] * v[i];
// L121
v_energy /= static_cast<T>(dim);
// L123
M *= (static_cast<T>(1) + plasticity * static_cast<T>(0.1) * tanh(v_energy));
// L129
pot += sin(x[i]) * V_w[i];
// L133
pot += x[i] * V_w[i];
// L137
T soft_m = sigmoid<T>(static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * (gate - sing_thresh));
// L138
M *= (static_cast<T>(1) + (sing_strength - static_cast<T>(1)) * soft_m);
// L141
h_sq[i] = h[i] * h[i] * S * M;
// L146
sum += W[i * rank + j] * h_sq[j];
// L163
* μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE * * @param x Position vector [dim] * @param force External force [dim] (optional) * @param W_forget Forget gate weights [dim x feature_dim] * @param b_forget Forget gate bias [dim] * @param W_input Input gate weights [dim x dim] (optional) * @param dim Dimension * @param topology Topology type * @param velocity_friction_scale Velocity friction scaling factor * @param v_norm_val Pre-computed velocity norm (optional, if available) * @param friction Output friction coefficients [dim] */ template <typename T> GFN_DEVICE void compute_friction( const T* x, const T* force, const T* W_forget, const T* b_forget, const T* W_input, int dim, Topology topology, T velocity_friction_scale, T v_norm_val, T* friction ) { T features[128]; // Max 2*dim for Fourier features
// L196
feature_dim = 2 * dim;
// L206
gate_val += W_forget[i * feature_dim + j] * features[j];
// L212
gate_val += W_input[i * dim + j] * force[j];
// L216
T base_friction = sigmoid<T>(gate_val) * static_cast<T>(FRICTION_SCALE<T>);
// L221
T v_scale = v_norm_val / (sqrt(static_cast<T>(dim)) + static_cast<T>(EPSILON_SMOOTH<T>));
// L222
friction[i] = base_friction * (static_cast<T>(1) + velocity_friction_scale * v_scale);
// L275
v_norm_val = sqrt(v_norm_val);
// L285
output[i] = gamma[i] + friction[i] * v[i];
// L337
v_norm_val = sqrt(v_norm_val);
// L392
T* grad_V_w = nullptr ) { T h[64];
// L404
sum += U[j * rank + i] * v[j];
// L407
h_energy += sum * sum;
// L410
h_energy /= static_cast<T>(rank);
// L413
T norm_val = sqrt(h_energy);
// L415
T S = static_cast<T>(1) / (static_cast<T>(1) + norm_val + static_cast<T>(EPSILON_STRONG<T>));
// L422
v_energy /= static_cast<T>(dim);
// L423
M_plas = (static_cast<T>(1) + plasticity * static_cast<T>(0.1) * tanh(v_energy));
// L426
T M_sing = static_cast<T>(1);
// L436
soft_m = sigmoid<T>(static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * (gate - sing_thresh));
// L437
M_sing = (static_cast<T>(1) + (sing_strength - static_cast<T>(1)) * soft_m);
// L439
T M = M_plas * M_sing;
// L445
T t = gamma[i] / static_cast<T>(CURVATURE_CLAMP<T>);
// L446
grad_raw[i] = grad_out[i] * (static_cast<T>(1) - t * t);
// L452
T q_base = h[j] * h[j] * S * M;
// L454
grad_W[i * rank + j] += grad_raw[i] * q_base;
// L455
grad_q[j] += W[i * rank + j] * grad_raw[i];
// L462
sum_grad_q_h_sq += grad_q[i] * h[i] * h[i];
// L467
T S_sq_M_norm = (norm_val > EPSILON_STANDARD<T> && rank > 0) ? (M * S * S / (norm_val * static_cast<T>(rank))) : static_cast<T>(0);
// L469
T two_S_M = static_cast<T>(2) * S * M;
// L472
grad_h[i] = grad_q[i] * h[i] * two_S_M - sum_grad_q_h_sq * S_sq_M_norm * h[i];
// L478
grad_U[i * rank + j] += v[i] * grad_h[j];
// L479
grad_v[i] += U[i * rank + j] * grad_h[j];
// L486
T dL_dM_plas = sum_grad_q_h_sq * S * M_sing;
// L488
T sech_sq = static_cast<T>(1) - tanh_v * tanh_v;
// L489
T factor = dL_dM_plas * (plasticity * static_cast<T>(0.1)) * sech_sq * (static_cast<T>(2) / static_cast<T>(dim));
// L491
grad_v[i] += factor * v[i];
// L497
T dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
// L498
T dM_dsoft = (sing_strength - static_cast<T>(1));
// L499
T dsoft_dgate = static_cast<T>(SINGULARITY_GATE_SLOPE<T>) * soft_m * (static_cast<T>(1) - soft_m);
// L500
T dgate_dpot = gate * (static_cast<T>(1) - gate);
// L501
T factor = dL_dM_sing * dM_dsoft * dsoft_dgate * dgate_dpot;
// L505
T dpot_dxi = (topology == Topology::TORUS) ? cos(x[i]) * V_w[i] : V_w[i];
// L506
grad_x[i] += factor * dpot_dxi;
// L509
T dpot_dVwi = (topology == Topology::TORUS) ? sin(x[i]) : x[i];
// L510
grad_V_w[i] += factor * dpot_dVwi;
// L538
int feature_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// L551
z += W_forget[i * feature_dim + j] * features[j];
// L556
z += W_input[i * dim + j] * force[j];
// L561
T dz = grad_out[i] * static_cast<T>(FRICTION_SCALE<T>) * s * (static_cast<T>(1) - s);
// L564
grad_b_forget[i] += dz;
// L566
grad_W_forget[i * feature_dim + j] += dz * features[j];
// L572
grad_W_input[i * dim + j] += dz * force[j];
// L574
grad_force[j] += dz * W_input[i * dim + j];
// L582
T d_sin = W_forget[i * feature_dim + j] * dz;
// L583
T d_cos = W_forget[i * feature_dim + (dim + j)] * dz;
// L584
grad_x[j] += d_sin * cos(x[j]) - d_cos * sin(x[j]);
// L588
grad_x[j] += W_forget[i * feature_dim + j] * dz;
```

#### gfn\cuda\src\geometry\geometry_library.cuh
```cpp
// Contexto: Global
// L34
*gamma_val = static_cast<scalar_t>(0);
// L35
v_shared[tid] = *s.v;
// L40
scalar_t v_ph = v_shared[tid + 1];
// L44
scalar_t denom = p.torus_R + p.torus_r * cos_th;
// L47
scalar_t term_th = denom * sin_th / (p.torus_r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L48
*gamma_val = term_th * (v_ph * v_ph) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
// L49
} else if (tid % 2 != 0) { scalar_t th = s.x[tid - 1];
// L51
scalar_t v_ph = *s.v;
// L52
scalar_t v_th = v_shared[tid - 1];
// L56
scalar_t denom = p.torus_R + p.torus_r * cos_th;
// L59
scalar_t term_ph = -(p.torus_r * sin_th) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L60
*gamma_val = static_cast<scalar_t>(2) * term_ph * v_ph * v_th * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>);
// L65
scalar_t prod = p.U[tid * p.rank + k] * (*s.v);
// L76
S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + sqrt(h_energy) + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// L82
sum_gamma += p.W[tid * p.rank + k] * h_shared[k] * h_shared[k] * S_shared;
// L84
*gamma_val = sum_gamma;
// L93
scalar_t v_sq = (*s.v) * (*s.v);
// L94
scalar_t total_energy = block_reduce_sum_shared(v_sq) / static_cast<scalar_t>(dim);
// L95
M += p.plasticity * static_cast<scalar_t>(0.1) * tanh(total_energy);
// L105
scalar_t w_sin = p.V_w[tid];
// L106
scalar_t w_cos = p.V_w[dim + tid];
// L107
pot_term = sin_th * w_sin + cos_th * w_cos;
// L109
pot_term = s.x[tid] * p.V_w[tid];
// L121
scalar_t soft_m = sigmoid(slope * (gate - p.sing_thresh));
// L123
M *= (static_cast<scalar_t>(1.0) + (p.sing_strength - static_cast<scalar_t>(1.0)) * soft_m);
// L126
*gamma_val *= M;
// L130
scalar_t v_dot_gz = (*s.v) * p.holo_grad_z[tid];
// L132
scalar_t v_sq_sum = block_reduce_sum_shared((*s.v) * (*s.v));
// L142
scalar_t ads = -(static_cast<scalar_t>(1) / local_holo_z) * (static_cast<scalar_t>(2) * common_v_dot_gz * (*s.v) - common_v_sq * p.holo_grad_z[tid]);
// L143
*gamma_val += ads;
// L148
scalar_t head_energy = block_reduce_sum_shared(s.f_ext * s.f_ext) / static_cast<scalar_t>(dim);
// L150
scalar_t modulator = exp(-p.thermo_alpha * head_energy / T);
// L151
*gamma_val *= modulator;
// L155
*gamma_val = soft_clamp<scalar_t>(*gamma_val, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
```

#### gfn\cuda\src\geometry\lowrank_christoffel.cu
```cpp
// Contexto: lowrank_christoffel_friction_kernel
// L79
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// L84
const scalar_t* v_ptr = v + idx * dim;
// L85
const scalar_t* x_ptr = x + idx * dim;
// L86
const scalar_t* force_ptr = (force != nullptr) ? (force + idx * dim) : nullptr;
// L87
scalar_t* output_ptr = output + idx * dim;
// L89
Topology topology = static_cast<Topology>(topology_id);
// L129
int blocks = (batch_size + threads - 1) / threads;
// L132
const scalar_t* x_ptr = (x.numel() > 0) ? x.data_ptr<scalar_t>() : nullptr;
// L133
const scalar_t* V_w_ptr = (V_w.numel() > 0) ? V_w.data_ptr<scalar_t>() : nullptr;
// L184
int blocks = (batch_size + threads - 1) / threads;
// L187
const scalar_t* V_w_ptr = (V_w.numel() > 0) ? V_w.data_ptr<scalar_t>() : nullptr;
// L188
const scalar_t* force_ptr = (force.numel() > 0) ? force.data_ptr<scalar_t>() : nullptr;
// L189
const scalar_t* W_input_ptr = (W_input.numel() > 0) ? W_input.data_ptr<scalar_t>() : nullptr;
// Contexto: lowrank_christoffel_kernel
// L31
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// L36
const scalar_t* v_ptr = v + idx * dim;
// L37
const scalar_t* x_ptr = (x != nullptr) ? (x + idx * dim) : nullptr;
// L38
scalar_t* gamma_ptr = gamma + idx * dim;
// L40
Topology topology = static_cast<Topology>(topology_id);
```

#### gfn\cuda\src\geometry\lowrank_christoffel_backward.cu
```cpp
// Contexto: christoffel_backward_kernel
// L33
int b = blockIdx.x * blockDim.x + threadIdx.x;
// L36
const scalar_t* grad_out_b = grad_out + b * dim;
// L37
const scalar_t* gamma_b = gamma + b * dim;
// L38
const scalar_t* v_b = v + b * dim;
// L39
const scalar_t* x_b = (x != nullptr) ? (x + b * dim) : nullptr;
// L41
scalar_t* g_v_b = grad_v + b * dim;
// L42
scalar_t* g_x_b = (grad_x != nullptr) ? (grad_x + b * dim) : nullptr;
// L51
h_energy_acc += sum * sum;
// L55
h_energy /= static_cast<scalar_t>(rank);
// L57
scalar_t norm = sqrt(h_energy);
// L58
scalar_t S = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// L65
v_e /= static_cast<scalar_t>(dim);
// L67
M_plas = (static_cast<scalar_t>(1) + plasticity * static_cast<scalar_t>(0.1) * tanh_v_e);
// L70
scalar_t M_sing = static_cast<scalar_t>(1);
// L74
else { for (int i = 0; i < dim; ++i) pot += x_b[i] * V_w[i]; }
// L76
soft_m = sigmoid<scalar_t>(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));
// L77
M_sing = (static_cast<scalar_t>(1) + (sing_strength - static_cast<scalar_t>(1)) * soft_m);
// L79
scalar_t M = M_plas * M_sing;
// L85
scalar_t t = gamma_b[i] / static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>);
// L86
scalar_t grad_raw_i = grad_out_b[i] * (static_cast<scalar_t>(1) - t * t);
// L88
scalar_t q_base = h[j] * h[j] * S * M;
// L90
grad_q[j] += W[i * rank + j] * grad_raw_i;
// L103
S_sq_M_norm = M * S * S / (norm * static_cast<scalar_t>(rank));
// L107
scalar_t dL_dM_plas = sum_grad_q_h_sq * S * M_sing;
// L108
scalar_t plas_scale = plasticity * static_cast<scalar_t>(0.1);
// L109
scalar_t dM_plas_dv_scale = plas_scale * (static_cast<scalar_t>(1) - tanh_v_e * tanh_v_e) * static_cast<scalar_t>(2) / static_cast<scalar_t>(dim);
// L114
g_v_b[i] += U[i * rank + j] * grad_h[j];
// L117
g_v_b[i] += dL_dM_plas * dM_plas_dv_scale * v_b[i];
// L122
scalar_t dL_dM_sing = sum_grad_q_h_sq * S * M_plas;
// L123
scalar_t factor = dL_dM_sing * (sing_strength - static_cast<scalar_t>(1)) * static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * soft_m * (static_cast<scalar_t>(1) - soft_m) * gate * (static_cast<scalar_t>(1) - gate);
// L131
scalar_t feature = (topology == Topology::TORUS) ? (x_b ? sin(x_b[i]) : static_cast<scalar_t>(0)) : (x_b ? x_b[i] : static_cast<scalar_t>(0));
// L139
g_x_b[i] = factor * ((topology == Topology::TORUS) ? cos(x_b[i]) * V_w[i] : V_w[i]);
// L175
int blocks = (batch_size + threads - 1) / threads;
```

#### gfn\cuda\src\geometry\lowrank_christoffel_friction_backward.cu
```cpp
// Contexto: lowrank_christoffel_friction_backward_kernel
// L46
int b = blockIdx.x * blockDim.x + threadIdx.x;
// L49
const scalar_t* grad_out_b = grad_out + b * dim;
// L50
const scalar_t* v_b = v + b * dim;
// L51
const scalar_t* x_b = x + b * dim;
// L52
const scalar_t* force_b = (force != nullptr) ? (force + b * dim) : nullptr;
// L60
scalar_t t = gamma_b[i] / static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>);
// L61
grad_pre[i] = grad_out_b[i] * (static_cast<scalar_t>(1) - t * t);
// L65
scalar_t* g_v_b = grad_v + b * dim;
// L66
scalar_t* g_x_b = grad_x + b * dim;
// L67
scalar_t* g_f_b = (grad_force != nullptr) ? (grad_force + b * dim) : nullptr;
// L74
grad_mu[i] = grad_out_b[i] * v_b[i];
// L78
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// L87
v_norm = sqrt(v_norm);
// L99
scalar_t mu_base = s * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// L106
scalar_t scale_factor = static_cast<scalar_t>(1) + velocity_friction_scale * v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L107
dL_dmu_base *= scale_factor;
// L110
scalar_t dz = dL_dmu_base * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>) * s * (static_cast<scalar_t>(1) - s);
// L123
scalar_t d_sin = W_forget[i * feat_dim + j] * dz;
// L124
scalar_t d_cos = W_forget[i * feat_dim + (dim + j)] * dz;
// L125
g_x_b[j] += d_sin * cos(x_b[j]) - d_cos * sin(x_b[j]);
// L140
h_energy += sum * sum;
// L143
h_energy /= static_cast<scalar_t>(rank);
// L145
scalar_t norm_h = sqrt(h_energy);
// L146
scalar_t S = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm_h + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// L152
v_e /= static_cast<scalar_t>(dim);
// L153
M_plas = (static_cast<scalar_t>(1) + plasticity * static_cast<scalar_t>(0.1) * tanh(v_e));
// L156
scalar_t M_sing = static_cast<scalar_t>(1);
// L160
else { for (int i = 0; i < dim; ++i) pot += x_b[i] * V_w[i]; }
// L162
soft_m = sigmoid<scalar_t>(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));
// L163
M_sing = (static_cast<scalar_t>(1) + (sing_strength - static_cast<scalar_t>(1)) * soft_m);
// L165
scalar_t M = M_plas * M_sing;
// L169
scalar_t q_base = h[j] * h[j] * S * M;
// L172
grad_q[j] += W[i * rank + j] * grad_pre[i];
// L178
scalar_t S_sq_M_norm = (norm_h > EPSILON_STANDARD<scalar_t>) ? (M * S * S / norm_h) : static_cast<scalar_t>(0);
// L184
g_v_b[i] += U[i * rank + j] * grad_h[j];
// L195
scalar_t v_scale_grad_factor = velocity_friction_scale / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L212
scalar_t scale_term = 1.0f + v_scale_grad_factor * v_norm;
// L213
scalar_t mu_base_i = mu_b[i] / scale_term;
// L222
scalar_t scale_term = static_cast<scalar_t>(1) + v_scale_grad_factor * v_norm;
// L224
scalar_t mu_base_k = mu_b[k] / scale_term;
// L225
common_sum += grad_out_b[k] * v_b[k] * mu_base_k;
// L228
scalar_t factor = common_sum * v_scale_grad_factor / v_norm;
// L230
g_v_b[j] += factor * v_b[j];
// L262
const int rank = U.size(-1);
// L277
int threads = 128; // Reduced threads to increase register availability
// L278
int blocks = (batch_size + threads - 1) / threads;
```

#### gfn\cuda\src\integrators\recurrent_manifold_fused.cpp
```cpp
// Contexto: Global
// L3
* ===================================================== * * BUG-7 FIX (2026-02-11): Added energy normalization, soft clamping, * constant friction damping, and velocity saturation. * * WARNING: This is NOT a true CUDA kernel — it uses ATen C++ ops. * It is kept as a fast inference-only path. For training with gradients, * the Python autograd fallback handles the backward pass correctly. * * Missing features vs. full Python: * - No learned friction gates (uses constant DEFAULT_FRICTION) * - No boundary conditions (assumes Euclidean topology) * - No plasticity or singularity amplification * - No hysteresis/ghost forces */ #include <torch/extension.h> #include <vector> #include <iostream> static constexpr double CURVATURE_CLAMP = 3.0;
// L25
static constexpr double EPSILON_STANDARD = 1e-7;
// L26
static constexpr double DEFAULT_FRICTION = 0.002;
// L70
const auto head_dim = D / H;
// L71
const auto L = U_stack.size(0) / H;
// L72
const auto dt_eff = dt * dt_scale;
// L83
int64_t s = h * head_dim;
// L84
int64_t e = s + head_dim;
// L88
auto U_h = U_stack.index({l * H + h});
// L89
auto W_h = W_stack.index({l * H + h});
// L93
auto h_sq = h_vec * h_vec;
// L96
auto energy = h_sq.mean(-1, /*keepdim=*/true);
// L97
auto S = 1.0 / (1.0 + energy.sqrt() + EPSILON_STANDARD);
// L100
auto gamma = at::matmul(h_sq * S, W_h.t());
// L101
gamma = gamma.clamp(-CURVATURE_CLAMP, CURVATURE_CLAMP);
// L104
auto v_new = v_h + f_h * dt_eff - gamma * dt_eff - v_h * (DEFAULT_FRICTION * dt_eff);
// L106
auto x_new = x_h + v_new.mul(dt_eff);
```

#### gfn\cuda\src\integrators\runge_kutta\heun_backward.cu
```cpp
// Contexto: heun_backward_kernel
// L38
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// L41
Topology topology = static_cast<Topology>(topology_id);
// L42
scalar_t effective_dt = dt * dt_scale;
// L43
scalar_t h_half = static_cast<scalar_t>(0.5) * effective_dt;
// L48
lx[i] = grad_x_out[idx * dim + i];
// L49
lv[i] = grad_v_out[idx * dim + i];
// L52
scalar_t* gU_b = grad_U + idx * dim * rank;
// L53
scalar_t* gW_b = grad_W + idx * dim * rank;
// L54
scalar_t* gf_b = grad_force + idx * dim;
// L60
const scalar_t* x_n = traj_x + idx * (steps + 1) * dim + step * dim;
// L61
const scalar_t* v_n = traj_v + idx * (steps + 1) * dim + step * dim;
// L62
const scalar_t* acc1 = traj_acc1 + idx * steps * dim + step * dim;
// L67
x_pred[i] = x_n[i] + effective_dt * v_n[i];
// L68
v_pred[i] = v_n[i] + effective_dt * acc1[i];
// L77
l_v_n[i] = lx[i] * h_half + lv[i];
// L78
l_v_pred[i] = lx[i] * h_half;
// L79
l_acc1[i] = lv[i] * h_half;
// L80
l_acc2[i] = lv[i] * h_half;
// L90
l_gamma2[i] = -l_acc2[i]; // acc2 = F - gamma2
// L91
gf_b[i] += l_acc2[i];
// L99
l_v_pred[i] += gv_c[i];
// L108
l_x_n[i] += gx_c[i]; // from S2 through x_pred
// L109
l_v_n[i] += l_v_pred[i] + effective_dt * gx_c[i];
// L110
l_acc1[i] += l_v_pred[i] * effective_dt;
// L119
l_gamma1[i] = -l_acc1[i];
// L120
gf_b[i] += l_acc1[i];
// L127
lx[i] = l_x_n[i] + gx_c[i];
// L128
lv[i] = l_v_n[i] + gv_c[i];
// L133
grad_x_in[idx * dim + i] = lx[i];
// L134
grad_v_in[idx * dim + i] = lv[i];
// Contexto: heun_forward_traj_kernel
// L148
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// L154
Topology topology = static_cast<Topology>(topology_id);
// L155
scalar_t effective_dt = dt * dt_scale;
// L156
const scalar_t* f_ptr = force + idx * dim;
// L160
traj_x[idx * (steps + 1) * dim + step * dim + i] = cx[i];
// L161
traj_v[idx * (steps + 1) * dim + step * dim + i] = cv[i];
// L166
acc1[i] = f_ptr[i] - gamma[i];
// L167
traj_acc1[idx * steps * dim + step * dim + i] = acc1[i];
// L168
x_pred[i] = cx[i] + effective_dt * cv[i];
// L169
v_pred[i] = cv[i] + effective_dt * acc1[i];
// L175
acc2[i] = f_ptr[i] - gamma[i];
// L176
cx[i] += (effective_dt / static_cast<scalar_t>(2.0)) * (cv[i] + v_pred[i]);
// L177
cv[i] += (effective_dt / static_cast<scalar_t>(2.0)) * (acc1[i] + acc2[i]);
// L182
traj_x[idx * (steps + 1) * dim + steps * dim + i] = cx[i];
// L183
traj_v[idx * (steps + 1) * dim + steps * dim + i] = cv[i];
// L203
auto traj_x = torch::empty({batch_size, steps + 1, dim}, options);
// L204
auto traj_v = torch::empty({batch_size, steps + 1, dim}, options);
// L214
int blocks = (batch_size + threads - 1) / threads;
```

#### gfn\cuda\src\integrators\runge_kutta\heun_fused.cu
```cpp
// Contexto: block_reduce_sum_heun
// L44
int wid = threadIdx.x / warpSize;
// L51
val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
// Contexto: christoffel_distributed_heun
// L84
*gamma_val = 0.0f;
// L91
scalar_t v_ph = v_shared[tid + 1];
// L95
scalar_t denom = R + r * c;
// L97
scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L98
scalar_t g0 = term_th * (v_ph * v_ph);
// L100
*gamma_val = soft_clamp<scalar_t>( static_cast<scalar_t>(g0) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );
// L104
} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];
// L107
scalar_t v_th = v_shared[tid - 1];
// L111
scalar_t denom = R + r * c;
// L113
scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L114
scalar_t g1 = 2.0f * term_ph * v_ph * v_th;
// L116
*gamma_val = soft_clamp<scalar_t>( static_cast<scalar_t>(g1) * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );
// L126
scalar_t u_val = U[tid * rank + k];
// L127
scalar_t prod = u_val * v_val;
// L146
scalar_t norm = sqrt(energy);
// L147
S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// L156
scalar_t h_sq = h_val * h_val * S_shared * M_shared;
// L157
sum_gamma += W[tid * rank + k] * h_sq;
// L160
*gamma_val = soft_clamp<scalar_t>(static_cast<scalar_t>(sum_gamma), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
// Contexto: friction_distributed_heun
// L189
features_shared[dim + tid] = c;
// L195
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// L200
gate_sum += W_forget[tid * feat_dim + j] * features_shared[j];
// L213
input_sum += W_input[tid * dim + j] * features_shared[j];
// L215
gate_sum += input_sum;
// L218
scalar_t base_friction = sigmoid(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// L222
scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L223
*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
// L225
*friction_val = base_friction;
// Contexto: Global
// L3
* ==================================== * * Block-parallel Heun integrator for manifold dynamics. * 1 block = 1 batch item, blockDim.x = dim. * * FIX (2026-02-11): Rewritten from single-thread-per-batch to block-parallel. *   BUG-1: Old kernel used 1 thread per batch with scalar_t[64] stacks, *          limiting dim<=64 and running without dimension parallelism. *   BUG-2: Host wrapper used hardcoded scalar_t instead of AT_DISPATCH. *   BUG-5: Added W_input parameter for full friction gate parity with Python. * * Integration scheme (Heun / RK2 Predictor-Corrector): *   1. Compute acceleration: a1 = F - Γ(v, x) - μ(x, F) * v *   2. Euler predictor: x_pred = x + dt*v, v_pred = v + dt*a1 *   3. Compute acceleration at predicted state: a2 = F - Γ(v_pred, x_pred) - μ(x_pred, F)*v_pred *   4. Corrector (average): x_new = x + dt/2*(v + v_pred), v_new = v + dt/2*(a1 + a2) */ #include "../../geometry/christoffel_impl.cuh" #include <torch/extension.h> #include <cuda.h> #include <cuda_runtime.h> namespace gfn { namespace cuda { template <typename scalar_t> __device__ inline scalar_t warp_reduce_sum_heun(scalar_t val) { for (int offset = warpSize/2; offset > 0; offset /= 2)
// L36
val += __shfl_down_sync(0xffffffff, val, offset);
// Contexto: heun_fused_kernel
// L281
scalar_t* h_shared = reinterpret_cast<scalar_t*>(shared_mem);
// L282
scalar_t* features_shared = h_shared + rank;
// L283
scalar_t* x_shared = features_shared + (2 * dim);
// L284
scalar_t* v_shared = x_shared + dim;
// L287
scalar_t curr_x = x_in[bid * dim + tid];
// L288
scalar_t curr_v = v_in[bid * dim + tid];
// L289
scalar_t f_ext = force[bid * dim + tid];
// L293
hyst_val = hysteresis_state[bid * dim + tid];
// L296
Topology topology = static_cast<Topology>(topology_id);
// L297
scalar_t effective_dt = dt * dt_scale;
// L310
scalar_t* hyst_shared_buf = features_shared; // Reuse features buf
// L316
sum += hyst_shared_buf[j] * hyst_readout_w[tid * dim + j];
// L326
scalar_t v_sq = curr_v * curr_v;
// L328
scalar_t v_norm = sqrt(v_sum);
// L347
scalar_t acc1 = f_ext + f_ghost - gamma1 - friction * curr_v;
// L350
scalar_t v_pred = curr_v + effective_dt * acc1;
// L351
scalar_t x_pred = curr_x + effective_dt * curr_v;
// L352
x_pred = apply_boundary_device(x_pred, topology);
// L367
scalar_t v_sq = v_pred * v_pred;
// L369
scalar_t v_norm = sqrt(v_sum);
// L388
scalar_t acc2 = f_ext + f_ghost - gamma2 - friction2 * v_pred;
// L391
curr_x += (effective_dt / 2.0f) * (curr_v + v_pred);
// L392
curr_v += (effective_dt / 2.0f) * (acc1 + acc2);
// L395
curr_x = apply_boundary_device(curr_x, topology);
// L399
scalar_t* input_shared = features_shared; // Reuse
// L404
input_shared[dim + tid] = c;
// L410
int offset = (topology == Topology::TORUS) ? 2*dim : dim;
// L411
input_shared[offset + tid] = curr_v;
// L416
sum += input_shared[j] * hyst_update_w[tid * hyst_in_dim + j];
// L419
hyst_val = hyst_val * hyst_decay + tanhf(sum);
// L424
x_out[bid * dim + tid] = curr_x;
// L425
v_out[bid * dim + tid] = curr_v;
// L428
hysteresis_state[bid * dim + tid] = hyst_val;
// L485
size_t shared_mem_size = (rank + 4 * dim) * x.element_size();
// L490
const scalar_t* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<scalar_t>() : nullptr;
// L491
const scalar_t* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<scalar_t>() : nullptr;
// L492
const scalar_t* W_input_ptr = (W_input.numel() > 0) ? W_input.data_ptr<scalar_t>() : nullptr;
// L494
scalar_t* hyst_state_ptr = nullptr;
// L495
const scalar_t* h_up_w_ptr = nullptr;
// L496
const scalar_t* h_up_b_ptr = nullptr;
// L497
const scalar_t* h_rd_w_ptr = nullptr;
// L498
const scalar_t* h_rd_b_ptr = nullptr;
```

#### gfn\cuda\src\integrators\symplectic\leapfrog_backward.cu
```cpp
// Contexto: leapfrog_backward_kernel
// L67
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// L70
Topology topology = static_cast<Topology>(topology_id);
// L71
scalar_t effective_dt = dt * dt_scale;
// L72
scalar_t h = static_cast<scalar_t>(0.5) * effective_dt;
// L78
lx[i] = grad_x_out[idx * dim + i];
// L79
lv[i] = grad_v_out[idx * dim + i];
// L83
scalar_t* gU_b = grad_U + idx * dim * rank;
// L84
scalar_t* gW_b = grad_W + idx * dim * rank;
// L85
int f_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// L86
scalar_t* gWf_b = grad_W_forget + idx * dim * f_dim;
// L87
scalar_t* gbf_b = grad_b_forget + idx * dim;
// L88
scalar_t* gf_b = grad_force + idx * dim;
// L89
scalar_t* gWinput_b = (W_input != nullptr) ? (grad_W_input + idx * dim * dim) : nullptr;
// L90
scalar_t* gVw_b = (V_w != nullptr) ? (grad_V_w + idx * dim) : nullptr;
// L93
scalar_t* gHupdate_w_b = hyst_enabled ? (grad_hyst_update_w + idx * dim * hyst_in_dim) : nullptr;
// L94
scalar_t* gHupdate_b_b = hyst_enabled ? (grad_hyst_update_b + idx * dim) : nullptr;
// L95
scalar_t* gHreadout_w_b = hyst_enabled ? (grad_hyst_readout_w + idx * dim * dim) : nullptr;
// L96
scalar_t* gHreadout_b_b = hyst_enabled ? (grad_hyst_readout_b + idx * dim) : nullptr;
// L98
const scalar_t* f_ptr = force + idx * dim;
// L122
const scalar_t* x_n = traj_x + idx * (steps + 1) * dim + step * dim;
// L123
const scalar_t* v_n = traj_v + idx * steps * 2 * dim + (step * 2 + 0) * dim;
// L124
const scalar_t* v_mid = traj_v + idx * steps * 2 * dim + (step * 2 + 1) * dim;
// L125
const scalar_t* x_next = traj_x + idx * (steps + 1) * dim + (step + 1) * dim;
// L133
v_mid_norm = sqrt(v_mid_norm);
// L145
scalar_t den = static_cast<scalar_t>(1) + h * mu_next[i];
// L146
l_v_mid[i] = lv[i] / den;
// L147
l_mu_next[i] = -h * lv[i] * ((v_mid[i] + h * (f_ptr[i] - gamma_mid[i])) / (den * den));
// L148
l_gamma_mid[i] = -h * lv[i] / den;
// L149
gf_b[i] += h * lv[i] / den;
// L159
scalar_t norm_eps = sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>);
// L160
v_scale_term = static_cast<scalar_t>(1) + velocity_friction_scale * v_mid_norm / norm_eps;
// L162
v_scale_norm_factor = velocity_friction_scale / norm_eps;
// L167
l_mu_base[i] = l_mu_next[i] * v_scale_term;
// L173
scalar_t mu_base_i = mu_next[i] / v_scale_term;
// L174
sum_factor += l_mu_next[i] * mu_base_i;
// L176
sum_factor *= (v_scale_norm_factor / v_mid_norm);
// L179
l_v_mid[j] += sum_factor * v_mid[j];
// L192
l_v_mid[i] += gv_c[i];
// L193
lx[i] += gx_c[i];
// L199
l_v_mid[i] += effective_dt * lx[i];
// L208
v_n_norm = sqrt(v_n_norm);
// L220
scalar_t den = static_cast<scalar_t>(1) + h * mu_n[i];
// L221
l_v_n[i] = l_v_mid[i] / den;
// L222
l_mu_n[i] = -h * l_v_mid[i] * ((v_n[i] + h * (f_ptr[i] - gamma_n[i])) / (den * den));
// L223
l_gamma_n[i] = -h * l_v_mid[i] / den;
// L224
gf_b[i] += h * l_v_mid[i] / den;
// L234
scalar_t norm_eps = sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>);
// L235
v_scale_term = static_cast<scalar_t>(1) + velocity_friction_scale * v_n_norm / norm_eps;
// L237
v_scale_norm_factor = velocity_friction_scale / norm_eps;
// L242
l_mu_base[i] = l_mu_n[i] * v_scale_term;
// L248
scalar_t mu_base_i = mu_n[i] / v_scale_term;
// L249
sum_factor += l_mu_n[i] * mu_base_i;
// L251
sum_factor *= (v_scale_norm_factor / v_n_norm);
// L254
l_v_n[j] += sum_factor * v_n[j];
// L266
lv[i] = l_v_n[i] + gv_c[i];
// L267
lx[i] += gx_c[i];
// L283
const scalar_t* x_step = traj_x + idx * (steps + 1) * dim + step * dim;
// L284
const scalar_t* v_step = traj_v + idx * steps * 2 * dim + (step * 2 + 1) * dim;  // v after KICK2
// L285
const scalar_t* h_prev = traj_h + idx * (steps + 1) * dim + step * dim;
// L286
const scalar_t* h_curr = traj_h + idx * (steps + 1) * dim + (step + 1) * dim;
// L304
sum[i] += sinf(x_step[j]) * hyst_update_w[i * hyst_in_dim + j];
// L305
sum[i] += cosf(x_step[j]) * hyst_update_w[i * hyst_in_dim + (dim + j)];
// L306
sum[i] += v_step[j] * hyst_update_w[i * hyst_in_dim + (2*dim + j)];
// L310
sum[i] += x_step[j] * hyst_update_w[i * hyst_in_dim + j];
// L311
sum[i] += v_step[j] * hyst_update_w[i * hyst_in_dim + (dim + j)];
// L316
tanh_grad[i] = static_cast<scalar_t>(1) - tanh_val[i] * tanh_val[i];  // sech²(sum) = 1 - tanh²(sum)
// L322
lsum[i] = lh[i] * tanh_grad[i];
// L329
gHupdate_b_b[i] += lsum[i];
// L334
gHupdate_w_b[i * hyst_in_dim + j] += lsum[i] * sin(x_step[j]);
// L335
gHupdate_w_b[i * hyst_in_dim + (dim + j)] += lsum[i] * cos(x_step[j]);
// L336
gHupdate_w_b[i * hyst_in_dim + (2*dim + j)] += lsum[i] * v_step[j];
// L340
gHupdate_w_b[i * hyst_in_dim + j] += lsum[i] * x_step[j];
// L341
gHupdate_w_b[i * hyst_in_dim + (dim + j)] += lsum[i] * v_step[j];
// L350
gHreadout_b_b[i] += static_cast<scalar_t>(0);  // Placeholder for now
// L352
gHreadout_w_b[i * dim + j] += static_cast<scalar_t>(0);  // Placeholder
// L360
lh[i] = lh[i] * hyst_decay;
// L367
grad_x_in[idx * dim + i] = lx[i];
// L368
grad_v_in[idx * dim + i] = lv[i];
// Contexto: leapfrog_forward_traj_kernel
// L401
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// L408
cx[i] = x_in[idx * dim + i];
// L409
cv[i] = v_in[idx * dim + i];
// L410
hyst_local[i] = hyst_enabled && hysteresis_state_in ? hysteresis_state_in[idx * dim + i] : static_cast<scalar_t>(0);
// L413
Topology topology = static_cast<Topology>(topology_id);
// L414
scalar_t effective_dt = dt * dt_scale;
// L415
scalar_t h = static_cast<scalar_t>(0.5) * effective_dt;
// L416
const scalar_t* f_ptr = force + idx * dim;
// L421
traj_h[idx * (steps + 1) * dim + i] = hyst_local[i];
// L428
traj_x[idx * (steps + 1) * dim + step * dim + i] = cx[i];
// L429
traj_v[idx * steps * 2 * dim + (step * 2 + 0) * dim + i] = cv[i];
// L440
sum += hyst_readout_w[i * dim + j] * hyst_local[j];
// L450
cv_norm = sqrt(cv_norm);
// L460
cv[i] = (cv[i] + h * (f_ptr[i] + f_ghost[i] - gamma[i])) / (static_cast<scalar_t>(1) + h * friction[i]);
// L461
traj_v[idx * steps * 2 * dim + (step * 2 + 1) * dim + i] = cv[i]; // Store v_mid
// L471
cv_norm = sqrt(cv_norm);
// L487
sum += hyst_readout_w[i * dim + j] * hyst_local[j];
// L494
cv[i] = (cv[i] + h * (f_ptr[i] + f_ghost[i] - gamma[i])) / (static_cast<scalar_t>(1) + h * friction[i]);
// L504
sum += sin(cx[j]) * hyst_update_w[i * hyst_in_dim + j];
// L505
sum += cos(cx[j]) * hyst_update_w[i * hyst_in_dim + (dim + j)];
// L506
sum += cv[j] * hyst_update_w[i * hyst_in_dim + (2*dim + j)];
// L510
sum += cx[j] * hyst_update_w[i * hyst_in_dim + j];
// L511
sum += cv[j] * hyst_update_w[i * hyst_in_dim + (dim + j)];
// L515
hyst_local[i] = hyst_local[i] * hyst_decay + tanh(sum);
// L522
traj_h[idx * (steps + 1) * dim + (step + 1) * dim + i] = hyst_local[i];
// L552
int f_dim = (topology == 1) ? 2 * dim : dim;
// L553
int hyst_in_dim = (topology == 1) ? 3 * dim : 2 * dim;
// L556
auto traj_x = torch::empty({batch_size, steps + 1, dim}, options);
// L557
auto traj_v = torch::empty({batch_size, steps, 2, dim}, options); // Stores v_n and v_mid
// L558
auto traj_h = hyst_enabled ? torch::empty({batch_size, steps + 1, dim}, options) : torch::empty({0}, options);  // AUDIT FIX
// L577
int blocks = (batch_size + threads - 1) / threads;
```

#### gfn\cuda\src\integrators\symplectic\leapfrog_fused.cu
```cpp
// Contexto: christoffel_distributed
// L34
const scalar_t* V_w = nullptr // [dim] (Optional singularity vector) ) { int tid = threadIdx.x;
// L41
*gamma_val = 0.0f;
// L51
scalar_t v_ph = v_shared[tid + 1]; // Safe: reads from shared mem
// L55
scalar_t denom = R + r * c;
// L57
scalar_t term_th = denom * s / (r + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L58
scalar_t g0 = term_th * (v_ph * v_ph);
// L60
*gamma_val = soft_clamp<scalar_t>( g0 * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );
// L64
} else if (tid % 2 != 0) { scalar_t th = x_shared[tid - 1];
// L67
scalar_t v_th = v_shared[tid - 1]; // Safe: reads from shared mem
// L71
scalar_t denom = R + r * c;
// L73
scalar_t term_ph = -(r * s) / (denom + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L74
scalar_t g1 = 2.0f * term_ph * v_ph * v_th;
// L76
*gamma_val = soft_clamp<scalar_t>( g1 * static_cast<scalar_t>(TOROIDAL_CURVATURE_SCALE<scalar_t>), static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>) );
// L90
scalar_t u_val = U[tid * rank + k];
// L91
scalar_t prod = u_val * v_val;
// L106
v_sum = block_reduce_sum(v_val * v_val);
// L116
pot_val = s * V_w[tid];
// L118
pot_val = x_shared[tid] * V_w[tid];
// L132
scalar_t norm = sqrt(energy);
// L133
S_shared = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + norm + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// L139
scalar_t v_energy = v_sum / static_cast<scalar_t>(dim);
// L140
M *= (1.0f + plasticity * 0.1f * tanh(v_energy));
// L146
scalar_t soft_m = sigmoid(static_cast<scalar_t>(SINGULARITY_GATE_SLOPE<scalar_t>) * (gate - sing_thresh));
// L147
M *= (1.0f + (sing_strength - 1.0f) * soft_m);
// L159
scalar_t h_sq = h_val * h_val * S_shared * M_shared;
// L160
sum_gamma += W[tid * rank + k] * h_sq;
// L163
*gamma_val = soft_clamp<scalar_t>(sum_gamma, static_cast<scalar_t>(CURVATURE_CLAMP<scalar_t>));
// Contexto: friction_distributed
// L191
features_shared[dim + tid] = c;
// L197
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// L203
sum += W_forget[tid * feat_dim + j] * features_shared[j];
// L215
sum += W_input[tid * dim + j] * features_shared[j];
// L219
scalar_t base_friction = sigmoid(sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// L223
scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L224
*friction_val = base_friction * (1.0f + velocity_friction_scale * v_scale);
// L226
*friction_val = base_friction;
// Contexto: leapfrog_fused_kernel
// L286
scalar_t* h_shared = reinterpret_cast<scalar_t*>(shared_mem);
// L287
scalar_t* features_shared = h_shared + rank;
// L288
scalar_t* x_shared = features_shared + (2 * dim);
// L289
scalar_t* v_shared = x_shared + dim;
// L292
scalar_t curr_x = x_in[bid * dim + tid];
// L293
scalar_t curr_v = v_in[bid * dim + tid];
// L294
scalar_t f_ext = force[bid * dim + tid];
// L298
hyst_val = hysteresis_state[bid * dim + tid];
// L301
Topology topology = static_cast<Topology>(topology_id);
// L302
scalar_t effective_dt = dt * dt_scale;
// L303
scalar_t step_h = 0.5f * effective_dt;
// L314
scalar_t* hyst_shared_buf = features_shared;
// L320
sum += hyst_shared_buf[j] * hyst_readout_w[tid * dim + j];
// L330
scalar_t v_sq = curr_v * curr_v;
// L332
scalar_t v_norm = sqrt(v_sum);
// L351
scalar_t total_force = f_ext + f_ghost;
// L352
scalar_t num = curr_v + step_h * (total_force - gamma);
// L353
scalar_t den = 1.0f + step_h * friction + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>);
// L357
curr_x += effective_dt * curr_v;
// L358
curr_x = apply_boundary_device(curr_x, topology);
// L367
scalar_t v_sq = curr_v * curr_v;
// L369
scalar_t v_norm = sqrt(v_sum);
// L389
scalar_t* hyst_shared_buf2 = features_shared;
// L395
sum2 += hyst_shared_buf2[j] * hyst_readout_w[tid * dim + j];
// L397
total_force2 += sum2;
// L400
num = curr_v + step_h * (total_force2 - gamma);
// L402
den = 1.0f + step_h * friction + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>);
// L407
scalar_t* input_shared = features_shared; // reuse
// L412
input_shared[dim + tid] = c;
// L419
input_shared[2*dim + tid] = curr_v;
// L421
input_shared[dim + tid] = curr_v;
// L427
sum += input_shared[j] * hyst_update_w[tid * hyst_in_dim + j];
// L430
hyst_val = hyst_val * hyst_decay + tanhf(sum);
// L435
x_out[bid * dim + tid] = curr_x;
// L436
v_out[bid * dim + tid] = curr_v;
// L439
hysteresis_state[bid * dim + tid] = hyst_val;
// L485
const void* W_forget_ptr = nullptr;
// L486
const void* b_forget_ptr = nullptr;
// L490
const void* W_input_ptr = nullptr;
// L493
const void* V_w_ptr = nullptr;
// L496
void* hyst_state_ptr = nullptr;
// L497
const void* hyst_up_w_ptr = nullptr;
// L498
const void* hyst_up_b_ptr = nullptr;
// L499
const void* hyst_read_w_ptr = nullptr;
// L500
const void* hyst_read_b_ptr = nullptr;
// L524
size_t shared_mem_size = (rank + 4 * dim) * x.element_size();
```

#### gfn\cuda\src\integrators\toroidal\toroidal_christoffel_fused.cu
```cpp
// Contexto: Global
// L3
* ================================== * * Dedicated CUDA kernel for computing Christoffel symbols on toroidal manifolds. * This kernel implements the metric-derived connection for torus topology. * * AUDIT FIX (2026-02-06): Component 2 - Toroidal Geometry in CUDA Fused Mode * * Problem: fusion.py was passing dummy zero tensors instead of computing * actual toroidal Christoffel symbols, causing complete loss of curvature. * * Solution: Dedicated kernel that computes toroidal Christoffel from metric: *   ds² = (R + r*cos(θ))² dφ² + r² dθ² * * Christoffel symbols (non-zero components): *   Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r *   Γ^φ_θφ = Γ^φ_φθ = -r*sin(θ) / (R + r*cos(θ)) * * Author: MiniMax Agent (Audit Implementation) * Date: 2026-02-06 * References: *   - technical_analysis.md: Lines 55-72 *   - implementation_plan.md: Component 2 */ #include <ATen/ATen.h> #include <c10/util/Exception.h> #include <cuda.h> #include <cuda_runtime.h> #include "../../common/types.cuh" #include "../../common/device_utils.cuh" #include "../../common/math_utils.cuh" #include "../../geometry/christoffel_impl.cuh" namespace gfn { namespace cuda { /** * @brief Compute toroidal Christoffel symbols for a single pair (θ, φ) * * For toroidal manifold with metric: *   ds² = (R + r*cos(θ))² dφ² + r² dθ² * * Non-zero Christoffel symbols: *   Γ^θ_φφ = (R + r*cos(θ)) * sin(θ) / r *   Γ^φ_θφ = Γ^φ_φθ = -r*sin(θ) / (R + r*cos(θ)) * * @param theta Position angle θ (poloidal) * @param phi Position angle φ (toroidal) - not used in computation but for API consistency * @param v_theta Velocity component v^θ * @param v_phi Velocity component v^φ * @param R Major radius of torus * @param r Minor radius of torus * @param gamma_theta Output: Christoffel force component Γ(v,v)^θ * @param gamma_phi Output: Christoffel force component Γ(v,v)^φ */ GFN_DEVICE void toroidal_christoffel_pair( float theta, float phi,  // Unused but kept for API consistency float v_theta, float v_phi, float R, float r, float* gamma_theta, float* gamma_phi ) { float sin_theta = sinf(theta);
// L74
float cos_theta = cosf(theta);
// L77
float denom = fmaxf(R + r * cos_theta, 1e-6f);
// L81
float term_theta = denom * sin_theta / (r + 1e-6f);
// L82
*gamma_theta = term_theta * (v_phi * v_phi);
// L86
float term_phi = -(r * sin_theta) / (denom + 1e-6f);
// L87
*gamma_phi = 2.0f * term_phi * v_theta * v_phi;
// L119
float phi = x[i + 1];
// L121
float v_phi = v[i + 1];
// L131
gamma[i + 1] = g_phi;
// L136
gamma[i] = fminf(10.0f, fmaxf(-10.0f, gamma[i]));
// L153
*   3. KICK 1: v_half = (v + h*(F - Γ)) / (1 + h*μ) *   4. DRIFT: x_new = x + dt * v_half *   5. Apply toroidal boundary: x ∈ [0, 2π) *   6. Recompute μ(x_new) and Γ(v_half, v_half) *   7. KICK 2: v_new = (v_half + h*(F - Γ)) / (1 + h*μ) * * @param x Initial position [batch, dim] * @param v Initial velocity [batch, dim] * @param f Force sequence [batch, seq_len, dim] * @param R Major radius * @param r Minor radius * @param dt Time step * @param batch Batch size * @param seq_len Sequence length * @param dim Dimension * @param x_out Output positions [batch, seq_len, dim] * @param v_out Output velocities [batch, seq_len, dim] */ GFN_GLOBAL void toroidal_leapfrog_fused_kernel( const float* x, const float* v, const float* f, const float* W_forget,   // [dim, feat_dim] or nullptr for DEFAULT_FRICTION const float* b_forget,   // [dim] or nullptr float R, float r, float dt, int batch, int seq_len, int dim, float* x_out, float* v_out ) { int tid = blockIdx.x * blockDim.x + threadIdx.x;
// L190
float curr_x[256];  // Max dim = 256
// L197
curr_x[i] = x[tid * dim + i];
// L198
curr_v[i] = v[tid * dim + i];
// L203
const float* f_ptr = &f[tid * seq_len * dim + t * dim];
// L204
float h = dt * 0.5f;  // Half time step
// L209
v_half[i] = curr_v[i] + h * f_ptr[i];
// L214
curr_x[i] += effective_dt * v_half[i];
// L219
curr_x[i] = atan2f(sinf(curr_x[i]), cosf(curr_x[i]));
// L227
curr_v[i] = v_half[i] + h * (f_ptr[i] - gamma[i]);
// L232
x_out[tid * seq_len * dim + t * dim + i] = curr_x[i];
// L233
v_out[tid * seq_len * dim + t * dim + i] = curr_v[i];
// L277
int grid_size = (batch + block_size - 1) / block_size;
// L332
const float* W_forget_ptr = (W_forget.numel() > 0) ? W_forget.data_ptr<float>() : nullptr;
// L333
const float* b_forget_ptr = (b_forget.numel() > 0) ? b_forget.data_ptr<float>() : nullptr;
```

#### gfn\cuda\src\integrators\toroidal\toroidal_christoffel_fused.cuh
```cpp
// Contexto: Global
// L5
* ======================================== * * Header file for toroidal-specific CUDA kernels. * Provides interface for Python bindings. */ #ifndef GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH #define GFN_CUDA_TOROIDAL_CHRISTOFFEL_FUSED_CUH #include "../../common/types.cuh" #include <cuda_runtime.h> namespace gfn { namespace cuda { /** * @brief Launch toroidal leapfrog fused kernel * * Performs full sequence integration using metric-derived * Christoffel symbols for toroidal topology. * * @param x Initial positions [batch, dim] * @param v Initial velocities [batch, dim] * @param f Force sequence [batch, seq_len, dim] * @param R Major radius of torus * @param r Minor radius of torus * @param dt Time step * @param batch Batch size * @param seq_len Sequence length * @param dim Dimension (should be even for angle pairs) * @param x_out Output positions [batch, seq_len, dim] * @param v_out Output velocities [batch, seq_len, dim] * @param stream CUDA stream (optional, default=0) */ void launch_toroidal_leapfrog_fused( const float* x, const float* v, const float* f, const float* W_forget, const float* b_forget, float R, float r, float dt, int batch, int seq_len, int dim, float* x_out, float* v_out, cudaStream_t stream = 0 );
```

#### gfn\cuda\src\integrators\unified_mlayer.cu
```cpp
// Contexto: Global
// L37
int head_dim = total_dim / num_heads;
// L49
size_t shared_mem_size = (rank + 6 * head_dim) * x.element_size();
// L57
geo_p.topology = static_cast<Topology>(topology);
// L61
geo_p.sing_thresh = static_cast<scalar_t>(sing_thresh);
// L62
geo_p.sing_strength = static_cast<scalar_t>(sing_strength);
```

#### gfn\cuda\src\integrators\universal_integrator.cuh
```cpp
// Contexto: universal_mlayer_kernel
// L36
int bid = blockIdx.x; // batch index
// L37
int hid = blockIdx.y; // head index
// L42
int head_offset = hid * head_dim;
// L45
const scalar_t* x_ptr = x_init + bid * total_dim + head_offset;
// L46
const scalar_t* v_ptr = v_init + bid * total_dim + head_offset;
// L50
scalar_t* h_shared = (scalar_t*)shared_buf;
// L51
scalar_t* x_shared = h_shared + geo_p.rank;
// L52
scalar_t* v_shared = x_shared + head_dim;
// L53
scalar_t* f_shared = v_shared + head_dim;
// L54
scalar_t* feat_shared = f_shared + head_dim; // For friction [2*head_dim]
// L59
scalar_t curr_h = (phys_p.hyst_enabled && phys_p.hysteresis_settings) ? phys_p.hysteresis_settings[bid * total_dim + head_offset + tid] : 0.0f;
// L66
const scalar_t dt_eff = phys_p.dt * phys_p.dt_scales[hid];
// L67
const scalar_t local_holo_z = (geo_p.holo_z_ptr) ? geo_p.holo_z_ptr[bid * num_heads + hid] : 0.0f;
// L70
head_geo_p.U = geo_p.U + hid * head_dim * geo_p.rank;
// L71
head_geo_p.W = geo_p.W + hid * head_dim * geo_p.rank;
// L73
head_geo_p.holo_grad_z = geo_p.holo_grad_z + (bid * num_heads * head_dim + hid * head_dim);
// L77
int feat_dim = (geo_p.topology == Topology::TORUS) ? 2 * head_dim : head_dim;
// L81
head_geo_p.V_w = geo_p.V_w + hid * feat_dim;
// L84
head_phys_p.W_forget = phys_p.W_forget + hid * head_dim * feat_dim;
// L85
head_phys_p.b_forget = phys_p.b_forget + hid * head_dim;
// L87
head_phys_p.W_input = phys_p.W_input + hid * head_dim * head_dim;
// L92
int x_feat_dim = (geo_p.topology == Topology::TORUS) ? 2 * head_dim : head_dim;
// L93
int hyst_in = x_feat_dim + head_dim;
// L94
head_phys_p.hyst_up_w = phys_p.hyst_up_w + hid * head_dim * hyst_in;
// L95
head_phys_p.hyst_up_b = phys_p.hyst_up_b + hid * head_dim;
// L96
head_phys_p.hyst_rd_w = phys_p.hyst_rd_w + hid * head_dim * head_dim;
// L97
head_phys_p.hyst_rd_b = phys_p.hyst_rd_b + hid * head_dim;
// L102
scalar_t f_ext = forces[(bid * seq_len + t) * total_dim + head_offset + tid];
// L114
scalar_t v_sq = (*state.v) * (*state.v);
// L115
scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));
// L122
scalar_t h = static_cast<scalar_t>(0.5) * dt_eff;
// L123
scalar_t v_half = (curr_v + h * (f_ext + ghost_f - gamma)) / (static_cast<scalar_t>(1) + h * mu + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// L128
curr_x = apply_boundary_device(curr_x + dt_eff * v_half, head_geo_p.topology);
// L129
x_shared[tid] = curr_x; // Update shared for next kick
// L133
state.v = &v_half; // Use half-velocity for christoffel
// L135
v_sq = v_half * v_half;
// L136
v_norm = sqrt(block_reduce_sum_shared(v_sq));
// L141
curr_v = (v_half + h * (f_ext + ghost_f - gamma)) / (static_cast<scalar_t>(1) + h * mu + static_cast<scalar_t>(EPSILON_STANDARD<scalar_t>));
// L145
else if (Method == IntegrationMethod::HEUN) { scalar_t v_sq = (*state.v) * (*state.v);
// L148
scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));
// L154
scalar_t a1 = f_ext + ghost_f - gamma - mu * curr_v;
// L155
scalar_t x_inter = apply_boundary_device(curr_x + dt_eff * curr_v, head_geo_p.topology);
// L156
scalar_t v_inter = curr_v + dt_eff * a1;
// L163
v_sq = v_inter * v_inter;
// L164
v_norm = sqrt(block_reduce_sum_shared(v_sq));
// L169
scalar_t a2 = f_ext + ghost_f - gamma - mu * v_inter;
// L171
curr_x = x_inter; // Heun typically uses x_predictor
// L172
curr_v = curr_v + 0.5f * dt_eff * (a1 + a2);
// L174
else if (Method == IntegrationMethod::EULER) { scalar_t v_sq = (*state.v) * (*state.v);
// L176
scalar_t v_norm = sqrt(block_reduce_sum_shared(v_sq));
// L184
scalar_t a = f_ext + ghost_f - gamma - mu * curr_v;
// L186
curr_x = apply_boundary_device(curr_x + dt_eff * curr_v, head_geo_p.topology);
// L187
curr_v = curr_v + dt_eff * a;
// L192
x_seq[(bid * seq_len + t) * total_dim + head_offset + tid] = curr_x;
// L196
x_final[bid * total_dim + head_offset + tid] = curr_x;
// L197
v_final[bid * total_dim + head_offset + tid] = curr_v;
// L199
phys_p.hysteresis_settings[bid * total_dim + head_offset + tid] = curr_h;
```

#### gfn\cuda\src\physics\physics_library.cuh
```cpp
// Contexto: Global
// L13
* μ = sigmoid(W_f * features + b_f + W_i * force) * FRICTION_SCALE */ template <typename scalar_t> GFN_DEVICE void compute_friction_distributed( const PhysicsParams<scalar_t>& p, const MLayerState<scalar_t>& s, int dim, Topology topology, scalar_t v_norm, scalar_t* friction_val, scalar_t* features_shared // Buffer for [2*dim] ) { int tid = threadIdx.x;
// L32
features_shared[tid] = sin_th;
// L33
features_shared[dim + tid] = cos_th;
// L39
int feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// L44
gate_sum += p.W_forget[tid * feat_dim + j] * features_shared[j];
// L53
gate_sum += p.W_input[tid * dim + j] * features_shared[j];
// L57
scalar_t base_mu = sigmoid<scalar_t>(gate_sum) * static_cast<scalar_t>(FRICTION_SCALE<scalar_t>);
// L61
scalar_t v_scale = v_norm / (sqrt(static_cast<scalar_t>(dim)) + static_cast<scalar_t>(EPSILON_SMOOTH<scalar_t>));
// L62
*friction_val = base_mu * (static_cast<scalar_t>(1) + p.v_fric_scale * v_scale);
// L64
*friction_val = base_mu;
// L81
*ghost_force_val = static_cast<scalar_t>(0);
// L91
features_shared[tid] = sin_th;
// L92
features_shared[dim + tid] = cos_th;
// L98
int x_feat_dim = (topology == Topology::TORUS) ? 2 * dim : dim;
// L99
features_shared[x_feat_dim + tid] = s.v[tid];
// L104
int total_in = x_feat_dim + dim;
// L106
up_sum += p.hyst_up_w[tid * total_in + j] * features_shared[j];
// L110
*s.h = (*s.h) * (static_cast<scalar_t>(1) - p.hyst_decay) + tanh(up_sum) * p.hyst_decay;
// L114
features_shared[tid] = *s.h;
// L119
rd_sum += p.hyst_rd_w[tid * dim + j] * features_shared[j];
// L121
*ghost_force_val = rd_sum;
```
