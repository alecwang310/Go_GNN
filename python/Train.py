import torch
import os
import numpy as np
from torch_geometric.loader import DataLoader
import torch.amp as amp
import torch.optim as optim
import torch.nn.functional as F
from GNN_deep_new import GoGNN
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

class GoTrainer:
    def __init__(self, model, lr=0.0001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.scaler = amp.GradScaler('cuda')
        self.lr = lr

    def loss_function(self, pred, targets, batch_size):
        p_board, p_pass, v_pred, o_pred = pred
        t_policy, t_value, t_own = targets

        # Policy Loss: Combine board + pass logits [Batch, 362]
        p_full = torch.cat([p_board.view(batch_size, 361), p_pass.view(batch_size, 1)], dim=1)

        t_policy = t_policy.view(batch_size, 362)
        t_pol_dist = t_policy / (t_policy.sum(dim=1, keepdim=True) + 1e-8)

        # log_softmax for KLDiv
        log_p_dist = F.log_softmax(p_full, dim=1)
        pol_loss = F.kl_div(log_p_dist, t_pol_dist, reduction='batchmean')

        # Value & Ownership Loss (MSE)
        val_loss = F.mse_loss(v_pred.view(-1), t_value.view(-1))
        own_loss = F.mse_loss(o_pred.view(-1), t_own.view(-1))
        
        return (1 * pol_loss) + (1.5 * val_loss) + (0.5 * own_loss), (pol_loss, val_loss, own_loss)
    
    def train_epoch(self, device, shard_dir, shard_files, accumulation_steps = 2):
        np.random.shuffle(shard_files)
        total_loss_sum = 0
        total_pol_sum = 0
        total_val_sum = 0
        total_own_sum = 0
        total_batches = 0

        total_top1, total_top3, total_top5 = 0, 0, 0
        
        for shard_name in shard_files:
            shard_top1, shard_top3, shard_top5 = 0, 0, 0

            shard_path = os.path.join(shard_dir, shard_name)
            try:
                graphs = torch.load(shard_path, weights_only=False)
            except Exception as e:
                print(f"!!! CRITICAL: Skipping corrupted shard {shard_name}: {e}")
                continue 
            loader = DataLoader(graphs, batch_size=256, shuffle=True)

            self.model.train()
            shard_loss_sum = 0
            shard_pol_loss = 0
            shard_val_loss = 0
            shard_own_loss = 0
            
            self.optimizer.zero_grad()

            for i, batch in enumerate(loader):
                # These are helpers to determine wich node blongs to the same graph
                # Batching is just joining many graphs into many disjoint parts
                batch = batch.to(device)
                s_batch = batch['string'].batch

                with amp.autocast('cuda'):
                    # Pass through the model
                    out = self.model(batch.x_dict, batch.edge_index_dict, s_batch)
                    targets = (batch.y_policy, batch.y_value, batch.y_ownership)
                    loss, loss_detailed = self.loss_function(out, targets, batch.num_graphs)
                    
                    p_board, p_pass, _, _ = out
                    p_full = torch.cat([p_board.view(batch.num_graphs, 361), p_pass.view(batch.num_graphs, 1)], dim=1)
                    
                    # Target indices (the move the teacher/expert made)
                    t_policy = batch.y_policy.view(batch.num_graphs, 362)
                    target_moves = t_policy.argmax(dim=1) 
                    
                    # Get top 5 indices from model
                    _, top5_indices = p_full.topk(5, dim=1)
                    
                    # Check matches
                    correct = top5_indices.eq(target_moves.view(-1, 1))
                    top1 = correct[:, :1].sum().item()
                    top3 = correct[:, :3].sum().item()
                    top5 = correct[:, :5].sum().item()
                    
                    shard_top1 += top1 / batch.num_graphs
                    shard_top3 += top3 / batch.num_graphs
                    shard_top5 += top5 / batch.num_graphs
                    # 2. Normalize the loss to maintain the correct gradient magnitude
                    scaled_loss = loss / accumulation_steps

                # 3. Backward pass (accumulates gradients automatically)
                self.scaler.scale(scaled_loss).backward()

                # 4. Step the optimizer only when we hit the accumulation target
                # OR if it is the very last batch of the shard
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.optimizer.zero_grad()
                
                # Accumulate for shard reporting (use original unscaled loss for accurate metrics)
                shard_loss_sum += loss.item()
                shard_pol_loss += loss_detailed[0].item()
                shard_val_loss += loss_detailed[1].item()
                shard_own_loss += loss_detailed[2].item()
                
                # Accumulate for global epoch reporting
                total_loss_sum += loss.item()
                total_pol_sum += loss_detailed[0].item()
                total_val_sum += loss_detailed[1].item()
                total_own_sum += loss_detailed[2].item()

                total_top1 += (top1 / batch.num_graphs)
                total_top3 += (top3 / batch.num_graphs)
                total_top5 += (top5 / batch.num_graphs)
                total_batches += 1

            num_batches = len(loader)
            print(f"Shard {shard_name} | "
                  f"Avg Loss: {shard_loss_sum / num_batches:.4f} | "
                  f"Pol: {shard_pol_loss / num_batches:.4f} | "
                  f"Val: {shard_val_loss / num_batches:.4f} | "
                  f"Own: {shard_own_loss / num_batches:.4f} | "
                  f"Top1: {shard_top1/num_batches:.1%} | "
                  f"Top3: {shard_top3/num_batches:.1%} | "
                  f"Top5: {shard_top5/num_batches:.1%}"
                  )
            
            del graphs, loader
            import gc; gc.collect()

        # Return averages for this sub-batch collection
        return (
            total_loss_sum / total_batches,
            total_pol_sum / total_batches,
            total_val_sum / total_batches,
            total_own_sum / total_batches
        )

    def validate(self, device, shard_dir, val_files):
        self.model.eval()
        val_loss_sum = 0
        total_batches = 0
        
        print("--- Running Validation ---")
        with torch.no_grad():
            with amp.autocast('cuda'):
                for shard_name in val_files:
                    shard_path = os.path.join(shard_dir, shard_name)
                    graphs = torch.load(shard_path, weights_only=False)
                    loader = DataLoader(graphs, batch_size = 256, shuffle=False)

                    for batch in loader:
                        batch = batch.to(device)
                        s_batch = batch['string'].batch
                        out = self.model(batch.x_dict, batch.edge_index_dict, s_batch)
                        targets = (batch.y_policy, batch.y_value, batch.y_ownership)
                        loss, _ = self.loss_function(out, targets, batch.num_graphs)
                        
                        val_loss_sum += loss.item()
                        total_batches += 1
                    
                    del graphs, loader
                    import gc; gc.collect()

        avg_val_loss = val_loss_sum / total_batches
        print(f"Validation Complete | Avg Val Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    def save_checkpoint(self, epoch, loss, filename="checkpoint.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'scaler_state_dict': self.scaler.state_dict(), 
            'loss': loss,
        }
        torch.save(checkpoint, filename)
        print(f"--- Checkpoint saved: {filename} ---")

    def load_checkpoint(self, filename, force_lr = None, reinit_opt = True):
        reinit_opt = False

        if os.path.isfile(filename):
            print(f"--- Loading Checkpoint: {filename} ---")
            checkpoint = torch.load(filename, map_location=next(self.model.parameters()).device, weights_only=False)
            
            state_dict = checkpoint['model_state_dict']

            # Create a new state_dict without the '_orig_mod.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                # Remove the prefix added by torch.compile
                name = k.replace('_orig_mod.', '') 
                new_state_dict[name] = v
            
            model_dict = self.model.state_dict()
            filtered_dict = {}
            
            for k, v in new_state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        # This captures the value_head mismatch and skips it safely
                        print(f"  [Skipping] {k}: Shape mismatch. Checkpoint: {v.shape}, Model: {model_dict[k].shape}")
                else:
                    print(f"  [Skipping] {k}: Parameter not found in current architecture")

            # Load the cleaned state dict
            self.model.load_state_dict(filtered_dict, strict=False)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"--- Optimizer state incompatible, starting with fresh optimizer: {e}")

            if reinit_opt:
                print("--- Re-initializing Optimizer ---")
                current_lr = force_lr if force_lr is not None else self.lr
                self.optimizer = optim.Adam(self.model.parameters(), lr=current_lr)
                self.lr = current_lr

                # 3. Restore Scaler
                if 'scaler_state_dict' in checkpoint:
                    try:
                        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    except:
                        print("--- Scaler state incompatible, using fresh scaler ---")

                print("--- Load Success: Trunk restored, Value Head & Optimizer are fresh ---")
            
            else:
                if force_lr is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = force_lr
                    self.lr = force_lr
                    print(f"LR forced to: {force_lr}")
                else:
                    self.lr = self.optimizer.param_groups[0]['lr']
                    print(f"LR loaded from checkpoint: {self.lr}")
                
                if checkpoint['scheduler_state_dict'] and hasattr(self, 'scheduler'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("Load success")
            
            return checkpoint.get('epoch', 0) + 1, checkpoint.get('loss', 10.0)
        
        print(f"--- No checkpoint found at: {filename} ---")
        return 0, float('inf') # start from scratch
    
    def set_freeze_trunk(self, freeze=True):
        """
        True: Only Value Head trains.
        False: Everything trains.
        """
        # First, freeze/unfreeze everything
        for param in self.model.parameters():
            param.requires_grad = not freeze
        
        # If freezing, explicitly turn the Value Head back ON
        if freeze:
            print("--- Strategy: Training VALUE HEAD ONLY (Trunk Frozen) ---")
            for param in self.model.value_head.parameters():
                param.requires_grad = True
        else:
            print("--- Strategy: Training FULL MODEL (Trunk Released) ---")

# Helper function to run in the background thread
def clear_and_copy(files_to_copy, src_dir, dst_dir):
    """Clears the destination directory and copies a new batch of files."""
    # 1. Clear the destination directory
    if os.path.exists(dst_dir):
        # ignore_errors=True is helpful on Windows where PyTorch 
        # might briefly hold onto a file handle.
        shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir, exist_ok=True)
    
    # 2. Copy the new files
    for f in files_to_copy:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)
        shutil.copy2(src_path, dst_path)
        
    return dst_dir

def run_training(mode):
    #Logging every output
    log_file = r'D:/Code/GNN/logs/full_training_output_.txt'
    sys.stdout = Tee(log_file)
    
    print(f"\n{'='*30}")
    print(f"New training session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*30}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Paths Setup ---
    src_dir = r'E:/graphs'
    buf1 = r'D:/Code/GNN/data/graphs_1'
    buf2 = r'D:/Code/GNN/data/graphs_2'
    val_dir = r'D:/Code/GNN/data/val_graphs' # Dedicated fast-drive folder for validation
    
    batch_size = 30 # Number of files to hold in the SSD buffer at once
    if mode == "production":
        all_shards = [f for f in os.listdir(src_dir) if f.endswith('.pt')]
        np.random.shuffle(all_shards)
        
        # Split shards: 1 for validation, the rest for training
        val_files_src = all_shards[:3]
        train_files_src = all_shards[3:]
        
        print(f"Total training shards: {len(train_files_src)}. Validation shards: {len(val_files_src)}.")

        # --- Initialize Validation Set ---
        # We copy the validation set to the fast drive ONCE so we don't 
        # waste I/O bandwidth copying it every single epoch.
        print("Caching validation set to fast drive...")
        clear_and_copy(val_files_src, src_dir, val_dir)
        val_files = os.listdir(val_dir)

        model = GoGNN().to(device)
        trainer = GoTrainer(model)
        trainer.set_freeze_trunk(False)

        resume_file = r'none'
        start_epoch, _ = trainer.load_checkpoint(resume_file, force_lr=0.0001)

        compiled_model = torch.compile(
            trainer.model,
            mode="max-autotune",          # or "reduce-overhead" for faster first compile
            dynamic=True,                 # crucial if batch size or #strings varies slightly
            fullgraph=False               # start False; set True to debug graph breaks
        )

        trainer.model = compiled_model
        num_ep = 40
        best_val_loss = 10

        # Thread pool for background file copying
        executor = ThreadPoolExecutor(max_workers=1)

        for epoch in range(start_epoch, num_ep):
            print(f"\n--- Starting epoch {epoch+1} ---")
            start = time.time()
            
            np.random.shuffle(train_files_src)
            batches = [train_files_src[i:i + batch_size] for i in range(0, len(train_files_src), batch_size)]
            
            print("Pre-loading first batch...")
            clear_and_copy(batches[0], src_dir, buf1)
            
            prefetch_future = None
            # Lists to store metrics for every sub-batch
            epoch_metrics = {
                'total': [],
                'pol': [],
                'val': [],
                'own': []
            }

            for i in range(len(batches)):
                active_buf = buf1 if i % 2 == 0 else buf2
                next_buf = buf2 if i % 2 == 0 else buf1
                
                if prefetch_future is not None:
                    prefetch_future.result()

                if i + 1 < len(batches):
                    prefetch_future = executor.submit(clear_and_copy, batches[i+1], src_dir, next_buf)

                files_to_train = os.listdir(active_buf)
                print(f"  Training sub-batch {i+1}/{len(batches)}...")
                
                # Capture the detailed return
                l_total, l_pol, l_val, l_own = trainer.train_epoch(device, active_buf, files_to_train)

                trainer.save_checkpoint(epoch, l_total, filename = r'D:/Code/GNN/models/temp.pth')
                
                epoch_metrics['total'].append(l_total)
                epoch_metrics['pol'].append(l_pol)
                epoch_metrics['val'].append(l_val)
                epoch_metrics['own'].append(l_own)

            # Calculate final epoch averages
            avg_train_loss = np.mean(epoch_metrics['total'])
            avg_pol_loss = np.mean(epoch_metrics['pol'])
            avg_val_loss_train = np.mean(epoch_metrics['val'])
            avg_own_loss_train = np.mean(epoch_metrics['own'])
            
            print("  Running validation...")
            avg_val_loss = trainer.validate(device, val_dir, val_files)

            trainer.scheduler.step(avg_train_loss)

            # Save Checkpoints
            trainer.save_checkpoint(epoch, avg_train_loss, r'D:/Code/GNN/models/go_gnn_combined_deep_new.pth')
            
            if avg_val_loss < best_val_loss:
                trainer.save_checkpoint(epoch, avg_val_loss, r'D:/Code/GNN/models/best_go_gnn_combined_deep_new.pth')
                best_val_loss = avg_val_loss

            # Final Detailed Printout
            print(f"\n>>> Epoch {epoch+1} Summary <<<")
            print(f"Train Total: {avg_train_loss:.4f} | Pol: {avg_pol_loss:.4f} | Val: {avg_val_loss_train:.4f} | Own: {avg_own_loss_train:.4f}")
            print(f"Val Total:   {avg_val_loss:.4f}")
            print(f"Epoch Time:  {(time.time() - start):.2f}s\n")
        
    if mode == "test":
        print(">>> RUNNING IN TEST MODE: Skipping disk I/O, loading directly from graphs_1")
        # In test mode, we only care about what's already in buf1
        train_files_src = [f for f in os.listdir(buf2) if f.endswith('.pt')]
        val_files = [f for f in os.listdir(val_dir) if f.endswith('.pt')]

        model = GoGNN().to(device)
        trainer = GoTrainer(model)
        trainer.set_freeze_trunk(False)

        resume_file = r'none'
        start_epoch, _ = trainer.load_checkpoint(resume_file, force_lr=0.0001)

        compiled_model = torch.compile(
            trainer.model,
            mode="max-autotune",          # or "reduce-overhead" for faster first compile
            dynamic=True,                 # crucial if batch size or #strings varies slightly
            fullgraph=False               # start False; set True to debug graph breaks
        )

        trainer.model = compiled_model

        num_ep = 40
        best_val_loss = 10

        for epoch in range(start_epoch, num_ep):
            print(f"\n--- Starting epoch {epoch+1} ---")
            start = time.time()
            print(f"  Training directly from {buf2}...")

            l_total, l_pol, l_val, l_own = trainer.train_epoch(device, buf2, train_files_src)
            avg_train_loss, avg_pol_loss, avg_val_loss_train, avg_own_loss_train = l_total, l_pol, l_val, l_own

            print(f"\n>>> Epoch {epoch+1} Summary ({mode}) <<<")
            print(f"Train Total: {avg_train_loss:.4f} | Pol: {avg_pol_loss:.4f} | Val: {avg_val_loss_train:.4f} | Own: {avg_own_loss_train:.4f}")
            print(f"Val Total:   {avg_val_loss:.4f}")
            print(f"Epoch Time:  {(time.time() - start):.2f}s\n")

    executor.shutdown()


#Logging every output to the txt file
import sys
class Tee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8") # "a" for append

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes to disk immediately

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    run_training("production")