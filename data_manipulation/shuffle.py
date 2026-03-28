import torch
import os
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import gc

def shuffle_single_shard(file_info):
    shard_dir, filename = file_info
    file_path = os.path.join(shard_dir, filename)
    try:
        # 1. Load (CPU Intensive)
        graphs = torch.load(file_path, weights_only=False)
        
        # 2. Shuffle (Pointer manipulation)
        random.shuffle(graphs)
        
        # 3. Save (CPU Intensive Compression/Serialization)
        torch.save(graphs, file_path)
        
        # 4. Cleanup
        del graphs
        gc.collect()
        return True
    except Exception as e:
        return f"Error in {filename}: {e}"

def multicore_shuffler(shard_dir="graphs", max_workers=8):
    shard_files = [f for f in os.listdir(shard_dir) if f.endswith('.pt')]
    if not shard_files:
        print("No shards found.")
        return

    # Prepare arguments for the workers
    tasks = [(shard_dir, f) for f in shard_files]

    print(f"Launching {max_workers} workers on 9950X3D...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(shuffle_single_shard, tasks), 
                           total=len(tasks), 
                           desc="Global Shuffle"))

    # Report errors
    errors = [r for r in results if r is not True]
    if errors:
        for err in errors:
            print(err)

if __name__ == "__main__":
    # Use 12-16 workers. Don't use all 32 threads to avoid RAM exhaustion.
    multicore_shuffler("graphs_3.16", max_workers=8)