
#The ENTIRE multi-core python script is written by Gemini AI, as I am not exactly familiar with python multi-core works
import os
import sys
import torch
import numpy as np
import gc
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

root_path = r'D:/Code/GNN'
if root_path not in sys.path:
    sys.path.append(root_path)

from python.Data_extract import KataGoData
from shuffle import multicore_shuffler

# --- Global worker state ---
worker_processor = None
worker_buffer = []
worker_shard_count = 0
worker_output_dir = ""
worker_chunk_size = 50000

def init_worker(output_dir, chunk_size):
    global worker_processor, worker_output_dir, worker_chunk_size
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    worker_processor = KataGoData()
    worker_output_dir = output_dir
    worker_chunk_size = chunk_size

def process_single_file(file_info):
    global worker_buffer, worker_shard_count
    input_path, filename = file_info
    
    try:
        raw = np.load(os.path.join(input_path, filename))
        graphs = worker_processor.process_npz(raw)
        
        if graphs:
            worker_buffer.extend(graphs)
            
        if len(worker_buffer) >= worker_chunk_size:
            save_worker_shard()
            
        return len(graphs)
    except Exception as e:
        print(f"Error in {filename}: {e}")
        return 0

def save_worker_shard():
    global worker_buffer, worker_shard_count
    if not worker_buffer:
        return
        
    pid = os.getpid()
    save_path = os.path.join(worker_output_dir, f"shard_pid{pid}_{worker_shard_count}.pt")
    torch.save(worker_buffer, save_path)
    
    worker_buffer = []
    worker_shard_count += 1
    gc.collect()

def flush_worker():
    save_worker_shard()
    return True

# --- Modified to process a specific list of files (a batch) ---
def preprocess_batch(path, file_list, output_dir, chunk_size=50000, num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    
    with ProcessPoolExecutor(max_workers=num_workers, 
                             initializer=init_worker, 
                             initargs=(output_dir, chunk_size)) as executor:
        try:
            file_iter = iter(file_list)
            futures = {}
            max_active = num_workers * 2 

            for _ in range(min(max_active, len(file_list))):
                f = next(file_iter)
                futures[executor.submit(process_single_file, (path, f))] = f

            with tqdm(total=len(file_list), desc="Processing Batch") as pbar:
                while futures:
                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    
                    for d in done:
                        futures.pop(d)
                        pbar.update(1)
                        try:
                            next_f = next(file_iter)
                            futures[executor.submit(process_single_file, (path, next_f))] = next_f
                        except StopIteration:
                            pass
            
            # Flush remainders for this specific batch
            flush_futures = [executor.submit(flush_worker) for _ in range(num_workers)]
            wait(flush_futures)

        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

# --- Background Mover Task ---
def move_to_slow_drive(src_dir, dest_dir):
    """Moves files from the fast staging folder to the slow external drive."""
    print(f"\n[Background] Moving finished batch to {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        shutil.move(src_file, dest_file)
    shutil.rmtree(src_dir)
    print(f"\n[Background] Move complete!")


def main_pipeline():
    # --- Paths Setup ---
    raw_data_path = r'D:/Code/GNN/data/25-12-expanded'
    final_dest_dir = r'F:/graphs-25-12'
    
    # Fast SSD working directories
    fast_active_dir = r'D:/Code/GNN/data/staging_active'
    fast_ready_dir = r'D:/Code/GNN/data/staging_ready'

    # Settings
    files_per_batch = 50000  # Adjust based on how many raw files make a good batch
    num_workers = 4
    chunk_size = 50000

    # Get all files and chunk them into batches
    all_files = [f for f in os.listdir(raw_data_path) if f.endswith('.npz')]
    batches = [all_files[i:i + files_per_batch] for i in range(0, len(all_files), files_per_batch)]

    print(f"Found {len(all_files)} files. Splitting into {len(batches)} batches.")

    # Thread pool for our background HDD transfer
    io_executor = ThreadPoolExecutor(max_workers=1)
    transfer_future = None

    for i, batch in enumerate(batches):
        print(f"\n==================================================")
        print(f"Starting Batch {i+1}/{len(batches)} ({len(batch)} files)")
        print(f"==================================================")

        # 1. Clear active directory just in case
        if os.path.exists(fast_active_dir):
            shutil.rmtree(fast_active_dir)
        os.makedirs(fast_active_dir, exist_ok=True)

        # 2. Process batch into the fast active directory
        preprocess_batch(raw_data_path, batch, fast_active_dir, chunk_size, num_workers)

        # 3. Shuffle the batch while it's STILL on the fast SSD
        print("\nShuffling batch on fast SSD...")
        multicore_shuffler(fast_active_dir, max_workers=8)

        # 4. Wait for the PREVIOUS batch's background transfer to finish
        if transfer_future is not None:
            print("\nWaiting for previous HDD transfer to finish...")
            transfer_future.result()

        # 5. Rename active to ready (this is instantaneous)
        if os.path.exists(fast_ready_dir):
            shutil.rmtree(fast_ready_dir)
        os.rename(fast_active_dir, fast_ready_dir)

        # 6. Kick off the background transfer of the new 'ready' directory to E:\
        transfer_future = io_executor.submit(move_to_slow_drive, fast_ready_dir, final_dest_dir)

    # Wait for the very last batch to finish moving
    if transfer_future is not None:
        print("\nWaiting for final HDD transfer to finish...")
        transfer_future.result()

    io_executor.shutdown()
    print("\nAll data processed, shuffled, and moved successfully!")

if __name__ == "__main__":
    main_pipeline()
