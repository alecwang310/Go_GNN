import torch
import os
from collections import OrderedDict
from GNN_deep_new import GoGNN

# 1. Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Instantiate Model
model = GoGNN().to(device)

PATH = r'D:/Code/GNN/models/temp.pth'

def load_checkpoint_to_model(model, filename):
    if os.path.isfile(filename):
        print(f"--- Loading Checkpoint: {filename} ---")
        # Use weights_only=False if you are loading complex objects, 
        # but for state_dicts True is safer if supported.
        checkpoint = torch.load(filename, map_location=device)
        
        # Handle the case where the key might be 'model_state_dict' or just the dict itself
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 3. Clean 'torch.compile' prefixes (_orig_mod.)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('_orig_mod.', '') 
            new_state_dict[name] = v
        
        # 4. Filter and Load
        model_dict = model.state_dict()
        filtered_dict = {}
        
        for k, v in new_state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"  [Skipping] {k}: Shape mismatch. Checkpoint: {v.shape}, Model: {model_dict[k].shape}")
            else:
                # This is normal if you have changed the architecture slightly
                pass

        model.load_state_dict(filtered_dict, strict=False)
        print("--- Model weights loaded successfully ---")
        return model
    else:
        print(f"--- ERROR: No checkpoint found at: {filename} ---")
        return None

@torch.no_grad()
def inspect_gate_states(model):
    model.eval()
    gate_stats = {'extraction': {}, 'reasoning': {}}

    # Extraction Phase
    for node_type, param_list in model.extraction_gates.items():
        # param_list[i] is the raw tensor; we sigmoid it to see the active gate value
        vals = torch.sigmoid(torch.stack([p.data for p in param_list]))
        gate_stats['extraction'][node_type] = vals.cpu().numpy().flatten()

    # Reasoning Phase
    for node_type, param_list in model.reason_gates.items():
        vals = torch.sigmoid(torch.stack([p.data for p in param_list]))
        gate_stats['reasoning'][node_type] = vals.cpu().numpy().flatten()

    return gate_stats

# --- Execute ---
loaded_model = load_checkpoint_to_model(model, PATH)

if loaded_model:
    stats = inspect_gate_states(loaded_model)

    print("\n" + "="*50)
    print("GATE STATE ANALYSIS (Sigmoid Values)")
    print("="*50)

    for phase in ['extraction', 'reasoning']:
        print(f"\n>>> {phase.upper()} PHASE")
        for node_type, values in stats[phase].items():
            # Join values with a clearer separator
            formatted_vals = " | ".join([f"{v:.3f}" for v in values])
            print(f"{node_type:12}: {formatted_vals}")