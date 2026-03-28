import torch
import torch.nn as nn
from GNN_deep import GoGNN

class GoGNNExportWrapper(nn.Module):
    """
    Wraps the GoGNN to take flat tensor arguments instead of dictionaries with tuple keys.
    This makes it 100% compatible with TorchScript and C++ LibTorch.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, stone_x, string_x, global_x, 
                edge_s_a_s, edge_s_b_str, edge_str_c_s, 
                edge_str_a_str, edge_str_r_g, edge_g_i_str):
        
        # 1. Reconstruct Node Dictionary
        x_dict = {
            'stone': stone_x,
            'string': string_x,
            'global_node': global_x
        }
        
        # 2. Reconstruct Edge Dictionary (with the tuple keys PyG expects)
        edge_index_dict = {
            ('stone', 'adjacent', 'stone'): edge_s_a_s,
            ('stone', 'belongs_to', 'string'): edge_s_b_str,
            ('string', 'contains', 'stone'): edge_str_c_s,
            ('string', 'adjacent_to', 'string'): edge_str_a_str,
            ('string', 'reports_to', 'global_node'): edge_str_r_g,
            ('global_node', 'influences', 'string'): edge_g_i_str
        }
        
        return self.model(x_dict, edge_index_dict)

if __name__ == "__main__":
    base_model = GoGNN()

    print(f"--- Loading Checkpoint: temp.pth ---")
    checkpoint = torch.load(r'D:/Code/GNN/models/temp.pth', map_location=next(base_model.parameters()).device, weights_only=False)

    state_dict = checkpoint['model_state_dict']

    # Create a new state_dict without the '_orig_mod.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # Remove the prefix added by torch.compile
        name = k.replace('_orig_mod.', '') 
        new_state_dict[name] = v

    # Load the cleaned state dict
    base_model.load_state_dict(new_state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model.to(device)

    base_model.eval()
    wrapper_model = GoGNNExportWrapper(base_model)
    wrapper_model.eval()

    # Create dummy data for tracing
    # N = num_stones, S = num_strings, G = num_global
    N, S, G = 361, 50, 1
    stone_x = torch.randn(N, 18).to(device)
    string_x = torch.randn(S, 2).to(device)
    global_x = torch.randn(G, 19).to(device)

    # Dummy edge indices [2, num_edges]
    e_s_a_s = torch.randint(0, N, (2, 500)).to(device)
    e_s_b_str = torch.stack([torch.randint(0, N, (100,)), torch.randint(0, S, (100,))]).to(device)
    e_str_c_s = torch.stack([torch.randint(0, S, (100,)), torch.randint(0, N, (100,))]).to(device)
    e_str_a_str = torch.randint(0, S, (2, 80)).to(device)
    e_str_r_g = torch.stack([torch.randint(0, S, (50,)), torch.zeros(50, dtype=torch.long)]).to(device)
    e_g_i_str = torch.stack([torch.zeros(50, dtype=torch.long), torch.randint(0, S, (50,))]).to(device)

    # Trace the model
    traced_script_module = torch.jit.trace(wrapper_model, (
        stone_x, string_x, global_x, 
        e_s_a_s, e_s_b_str, e_str_c_s, e_str_a_str, e_str_r_g, e_g_i_str
    ))

    # Save for C++
    traced_script_module.save(r'D:/Code/GNN/traced_models/gognn_traced.pt')
    print("Model exported to gognn_traced.pt")