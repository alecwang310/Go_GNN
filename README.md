  This project uses the pytorch GATv2 framework to create a go engine. 
  The training data is from katagotraining.org, trained from the latest katago self play games.
  All the training up to now is run on a 5080 using cuda 13.2, python 14.3, and pytorch 2.10.0 + cu128
  The python folder contains the training pipeline, graph creation and model architecture, the data_manipulation folder contains python scripts that loads training files and convert them into graph data contained in .pt files. the interactive folder enables connection to online-go.com

  The following is generated automatically by xiaomi mimo-v2.5-pro, and checked for errors by the author.

---

## Project Structure

```
GNN/
├── python/                    # Model architectures, training pipeline, and data extraction
│   ├── GNN.py                 # Original shallow GNN model (10 layers, unique weights per layer)
│   ├── GNN_deep.py            # Deep model v1: extraction + recurrent reasoning (8+14 layers, 4+2 heads)
│   ├── GNN_deep_new.py        # Deep model v2 (current): 8+14 layers, 8+8 heads, string-pooled value head
│   ├── Train.py               # Full training loop with checkpointing, validation, gradient accumulation
│   ├── Data_extract.py        # KataGo NPZ → PyG HeteroData graph conversion
│   ├── export.py              # Export trained model to TorchScript (.pt) for C++ LibTorch inference
│   └── inspect_params.py      # Utility to visualize learned gate sigmoid values for debugging
│
├── data_manipulation/         # Raw data download and preprocessing pipeline
│   ├── fetch_data.ps1         # PowerShell script to bulk-download KataGo training NPZ archives
│   ├── Data_saving.py         # Multi-process NPZ → .pt shard conversion with SSD staging
│   └── shuffle.py             # Multi-process shard-level shuffle for training data randomization
│
├── cpp/                       # C++ inference engine with MCTS search
│   ├── goboard.hpp            # Go board representation (21x21 padded array, legality, captures, scoring)
│   ├── graph_creation.hpp     # C++ graph builder: board state → HeteroData tensors for model input
│   ├── search.hpp             # Multi-threaded MCTS with batched NN inference queue
│   ├── main.cpp               # Interactive terminal Go game (MCTS or pure NN mode)
│   ├── bindings.cpp           # pybind11 module exposing C++ MCTS to Python as `gomcts`
│   └── CMakeLists.txt         # Build config for both the executable and the Python .pyd module
│
├── interactive/               # Playable interfaces and visualization tools
│   ├── gtp_interface.py       # GTP protocol engine (pure NN, no search) for online-go.com integration
│   ├── gtp_interface_search.py# GTP protocol engine with C++ MCTS via gomcts.pyd
│   ├── play.py                # Interactive terminal game against the model (Python)
│   ├── two_bot.py             # Bot-vs-bot self-play with matplotlib visualization and ownership heatmap
│   ├── visualize.py           # KataGo NPZ board state visualizer with feature inspection
│   ├── show.py                # Side-by-side model prediction vs KataGo ground-truth comparison
│   ├── bot_start.bat          # Launcher for the pure-NN GTP bot
│   ├── search_bot_start.bat   # Launcher for the MCTS search GTP bot
│   ├── gomcts.pyd             # Pre-compiled C++ MCTS Python extension
│   └── gtp2ogs-main/          # GTP-to-OGS bridge for online-go.com play
│
├── data/                      # Training data shards (.pt files) and validation/test splits
├── models/                    # Saved checkpoints (.pth)
├── traced_models/             # TorchScript-exported models for C++ inference
├── logs/                      # Training log outputs
└── loss_deep_new.png          # Training loss curve visualization
```

---

## Model Architecture

The project implements a **heterogeneous Graph Attention Network (GATv2)** for Go position evaluation. The board is represented as a multi-type graph with three node types and six edge types, processed through two distinct phases.

### Graph Representation

Each Go board position is converted into a `HeteroData` graph with:

**Node Types:**
- **Stone nodes** (up to 361, one per board intersection) — 18-dimensional features per stone:
  - Channel 0: constant 1.0 (stone exists)
  - Channels 1–2: current player stone, opponent stone (one-hot)
  - Channels 3–5: stone has exactly 1, 2, or ≥3 liberties (one-hot)
  - Channel 6: ko-restricted position flag
  - Channels 7–11: last 5 move history planes (oldest → most recent)
  - Channels 12–13: current player / opponent presence on board
  - Channels 14–17: positional encoding (x, y coordinates + distance-to-edge in both axes)

- **String nodes** (variable count, one per connected group of same-color stones) — 2-dimensional features:
  - Channel 0: string size (number of stones in the group)
  - Channel 1: color (1.0 = current player, 0.0 = opponent)

- **Global node** (exactly 1 per position) — 19-dimensional features:
  - Channels 0–4: pass history flags (last 5 turns)
  - Channel 5: komi / 20.0 (normalized)
  - Channels 6–18: rule flags (ko type, suicide legality, scoring method, handicap, etc.)

**Edge Types (6 directed relations):**
| Edge | From → To | Description |
|------|-----------|-------------|
| `adjacent` | stone → stone | Grid neighbors (horizontal/vertical adjacency on the 19×19 board) |
| `belongs_to` | stone → string | Each stone belongs to its connected string group |
| `contains` | string → stone | Reverse of belongs_to (each string contains its member stones) |
| `adjacent_to` | string → string | Two strings are adjacent if any of their member stones are grid neighbors |
| `reports_to` | string → global_node | Every string reports to the single global node |
| `influences` | global_node → string | The global node influences every string |

### Current Model: `GNN_deep_new.py`

The latest architecture uses a **two-phase design** with 8 extraction layers and 14 recurrent reasoning steps:

**Phase 1 — Extraction (8 unique layers, 8 attention heads each):**
Each layer consists of:
1. LayerNorm on all node features
2. `HeteroConv` with 6 GATv2 attention heads (one per edge type), using **sum aggregation**
3. Gated residual connection: `x = x + sigmoid(gate) * new_x` (gates initialized at 0.5)
4. Feed-forward network (FFN) with LayerNorm → Linear(dim, 2×dim) → GELU → Linear(2×dim, dim) → Dropout

All 8 extraction layers have unique, independently learned weights. This phase builds up rich positional representations by passing messages across the full graph topology.

**Phase 2 — Recurrent Reasoning (14 steps, weight-shared A/B alternating blocks):**
To enable deep reasoning without exploding parameter count, the model alternates between two shared convolution blocks (Block A on even steps, Block B on odd steps):
1. Shared LayerNorm (stable across recurrent iterations)
2. `HeteroConv` with 8 attention heads, sum aggregation — notably, the `string → global_node` edge is **removed** during reasoning to prevent over-smoothing of the global signal
3. Gated residual with per-step gates (initialized at 0.0, learned)
4. FFN (also alternated between A and B)
5. The global node FFN is only applied every 4th step (`i % 4 == 0`) to further control information flow

**Output Heads:**
| Head | Input | Output | Activation | Purpose |
|------|-------|--------|------------|---------|
| Policy (board) | stone features | [361] logits | None | Move probability per intersection |
| Policy (pass) | global node | [1] logit | None | Pass move logit |
| Value | global node + mean-pooled string features (concatenated, 1024-dim) | [1] | Tanh | Win probability from current player's perspective (−1 to +1) |
| Ownership | stone features | [361] | Tanh | Predicted final ownership of each intersection (−1 = white, +1 = black) |

The value head concatenates the global node with `global_mean_pool(string_features)` to incorporate both strategic (global) and tactical (string-level) information.

### Previous Model: `GNN_deep.py`

An earlier variant with the same two-phase structure but different hyperparameters:
- Extraction: 4 heads, Reasoning: 2 heads (vs 8/8 in the current model)
- Value head uses only the global node (no string pooling)
- Reasoning phase applies global FFN on every step (no `i % 4` gating)

### Original Model: `GNN.py`

The initial flat architecture:
- 10 heterogeneous convolution layers, each with unique weights (no weight sharing)
- 8 attention heads per layer
- Uses mean aggregation (vs sum in the deep models)
- Simpler 2-layer policy/value/ownership heads
- No two-phase design — all layers are identical extraction layers with learnable gates

---

## Training Pipeline

### 1. Data Acquisition

Raw training data is downloaded from [katagotraining.org](https://katagotraining.org/) using the PowerShell script `data_manipulation/fetch_data.ps1`:
- Iterates over a date range, downloading `.tgz` archives containing `.npz` files
- Each `.npz` file contains hundreds of positions from KataGo self-play games
- Fields include packed spatial features (`binaryInputNCHWPacked`), policy targets (MCTS visit counts), value targets (win/loss/no-result), ownership targets, and global rule inputs

### 2. Data Conversion (NPZ → Graph Shards)

`data_manipulation/Data_saving.py` orchestrates the conversion pipeline:

1. **Multi-process parsing**: Uses `ProcessPoolExecutor` with 4 workers. Each worker holds a `KataGoData` instance and processes `.npz` files independently.
2. **NPZ unpacking** (`Data_extract.py :: KataGoData.process_npz`):
   - Unpacks 22-channel spatial features from bit-packed format: `(N, 22, 46) → (N, 22, 19, 19)`
   - Extracts policy targets (362-dim: 361 board positions + pass)
   - Extracts value targets from win/loss counts: `wins - losses`
   - Extracts ownership targets (19×19 → 361-dim flattened)
3. **Graph construction** (`KataGoData._build_single_graph`):
   - Builds `HeteroData` objects with all node features, edge indices, and targets
   - Uses `scipy.ndimage.label` for connected-component string detection
   - Vectorized string-to-string adjacency via board edge mapping
4. **Shard saving**: Workers buffer graphs (up to 50,000 per shard) and save as `.pt` files. Each shard file is a list of `HeteroData` objects.
5. **SSD staging**: Processing happens on a fast SSD (`D:/Code/GNN/data/staging_active`), then shuffled, then moved to a slower storage drive in the background.

### 3. Data Shuffling

`data_manipulation/shuffle.py` randomizes the order of graphs within each shard:
- Uses `ProcessPoolExecutor` with 8 workers
- Each worker loads a `.pt` shard, shuffles the list of `HeteroData` objects in-place, and saves it back
- This ensures the training loop sees data in a different order each epoch

### 4. Training Loop (`python/Train.py`)

The `GoTrainer` class implements a complete training pipeline:

**Initialization:**
- Model: `GoGNN()` from `GNN_deep_new.py` (latest architecture)
- Optimizer: Adam with initial LR of 0.00001
- Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=2)
- Mixed precision: `torch.amp.GradScaler` + `autocast('cuda')`
- Model compilation: `torch.compile(mode="max-autotune", dynamic=True)`

**Double-buffered I/O Strategy:**
To avoid I/O bottlenecks, the trainer uses two SSD buffer directories (`graphs_1`, `graphs_2`):
1. Load shard files into `buf1` from the source drive
2. Train on `buf1` while prefetching the next batch into `buf2` using a background `ThreadPoolExecutor`
3. Alternate between buffers each sub-batch
4. Validation files are cached once on the fast drive at the start

**Per-epoch flow:**
1. Shuffle the list of all training shard files
2. Split shards into sub-batches of 30 files each
3. For each sub-batch:
   - Wait for any pending prefetch to complete
   - Kick off prefetch of the next sub-batch into the alternate buffer
   - Load `.pt` shards into a PyG `DataLoader` (batch_size=256, shuffle=True)
   - Train with gradient accumulation (2 steps)
   - Save checkpoint after each sub-batch
4. Run validation on held-out shards
5. Save epoch checkpoint; save best checkpoint if validation loss improved

**Loss Function:**
```
total_loss = 1.0 × policy_loss + 1.5 × value_loss + 0.5 × ownership_loss
```
- **Policy loss**: KL divergence between model log-softmax and normalized KataGo visit count distribution (362-dim: board + pass)
- **Value loss**: MSE between predicted value and `wins - losses` target
- **Ownership loss**: MSE between predicted ownership and final board ownership

**Training Metrics:**
- Top-1, Top-3, Top-5 policy accuracy (does the model's top-K moves include the KataGo expert move?)
- Per-shard and per-epoch breakdown of all three loss components

**Checkpointing:**
- Saves `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, epoch, and loss
- Handles `torch.compile` prefix (`_orig_mod.`) stripping during load
- Supports shape-mismatch filtering for safe architecture changes (e.g., changing value head dimensions)
- Can freeze the trunk and train only the value head via `set_freeze_trunk(True)`

**Two Training Modes:**
- `production`: Full training with all shards, SSD double-buffering, validation split, and checkpoint management
- `test`: Loads directly from a small test directory for quick iteration and debugging

### 5. Model Export (`python/export.py`)

After training, the model is exported to TorchScript for C++ inference:
1. Load the trained checkpoint into `GoGNN`
2. Wrap in `GoGNNExportWrapper` — converts the dict-based forward signature into flat positional tensors compatible with TorchScript
3. `torch.jit.trace` with dummy data (361 stones, 50 strings, 1 global node)
4. Save as `gognn_traced.pt` for loading in C++ via `torch::jit::load`

---

## C++ Inference Engine

The `cpp/` directory contains a complete C++ Go engine with MCTS search, designed for high-performance inference.

### Board Representation (`goboard.hpp`)

- 21×21 padded 1D array (borders prevent bounds checking)
- Move history tracking (last 5 moves + pass flags)
- Legal move detection with suicide and simple ko rule
- Area scoring with flood-fill territory detection

### Graph Construction (`graph_creation.hpp`)

`FastGoGraphGenerator` converts a `GoBoard` state into the same heterogeneous graph format the Python model expects:
- Uses a **Disjoint Set Union (DSU)** for fast connected-component string detection
- Pre-computes static stone-to-stone adjacency edges and positional encodings at construction time
- Bitset-based liberty counting for speed
- Produces all 10 tensors the exported model expects: `stone_x`, `string_x`, `global_x`, `string_batch_index`, and 6 edge index tensors

### MCTS Search (`search.hpp`)

A multi-threaded MCTS implementation with batched neural network inference:

**Tree structure:**
- `MCTSNode` stores visits, value sum, prior, children, and virtual loss for thread coordination
- PUCT formula: `score = Q + cpuct × prior × √(parent_visits) / (1 + visits)`
- FPU (First Play Urgency): unvisited children start with `parent_Q - fpu_value`

**Batched Inference (`InferenceBatchQueue`):**
- A dedicated inference thread collects requests from MCTS worker threads
- Waits for a batch to fill (or a short timeout) before running a single batched forward pass
- Automatically concatenates heterogeneous graph data from multiple positions into a single batched graph
- Results are dispatched back to the requesting worker via a condition variable

**Worker threads:**
- Each thread runs its assigned share of simulations
- **Select**: Walk the tree under mutex, choosing best children via PUCT
- **Expand**: Generate graph for the leaf position, submit to inference queue
- **Backup**: Update visit counts and value sums along the path, flipping value at each level
- Virtual loss is applied during selection to prevent duplicate evaluation of the same leaf

### Interactive Terminal (`main.cpp`)

A complete terminal Go game:
- Choose MCTS search or pure NN mode
- Configurable simulation count and thread count
- Real-time display of NN priors vs MCTS visit distributions
- Area scoring at game end

### Python Bindings (`bindings.cpp`)

The C++ MCTS engine is exposed to Python via pybind11 as `gomcts`:
- `GoBoard`: board management (reset, play_move, is_legal, calculate_area_score)
- `MCTS`: search engine (search, set_num_simulations, set_num_threads)
- Used by `interactive/gtp_interface_search.py` for OGS play with search

---

## Interactive Play

### GTP Interface (`interactive/gtp_interface.py`)

Implements the Go Text Protocol (GTP) for connecting to online-go.com:
- Full board state management in Python (captures, ko, history)
- Feature generation matching the KataGo format (22 channels + 19 global features)
- Policy-based move selection with legality masking
- Compatible with gtp2ogs bridge for online play

### GTP Search Interface (`interactive/gtp_interface_search.py`)

A more advanced GTP bot using the C++ MCTS engine:
- Loads the TorchScript model and creates a `gomcts.MCTS` instance
- Dynamically adjusts simulation count based on time controls
- Scales simulations down when running low on time

### Self-Play Visualization (`interactive/two_bot.py`)

Pits two model checkpoints against each other with matplotlib:
- Real-time board rendering with ownership heatmap overlay
- Keyboard controls (1/2) to switch between each bot's ownership view
- Automatic game termination on two consecutive passes with final scoring

### Diagnostic Tools
- `show.py`: Compare model predictions against KataGo ground truth (top-5 move comparison, win probability)
- `visualize.py`: Visualize raw KataGo NPZ board states with feature channel inspection
- `inspect_params.py`: Analyze learned gate values to understand how much each layer contributes
