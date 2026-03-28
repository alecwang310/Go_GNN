import torch
import numpy as np
from torch_geometric.data import Batch

import sys
root_path = r'D:/Code/GNN'
if root_path not in sys.path:
    sys.path.append(root_path)

from python.Data_extract import KataGoData
from python.GNN import GoGNN

def evaluate_sample_row(npz_path, model_path, n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = GoGNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Load Data
    raw = np.load(npz_path)
    sample_raw = {key: raw[key][n:n+1] for key in raw.files}
    
    data_builder = KataGoData(board_size=19)
    graphs = data_builder.process_npz(sample_raw)
    graph = graphs[0]

    # 3. Extract and Normalize KataGo Raw Visit Counts
    # Index 0 is the main policy target (visit counts)
    kt_raw_visits = sample_raw['policyTargetsNCMove'][0, 0, :].astype(np.float32)
    total_visits = np.sum(kt_raw_visits)
    
    # Avoid division by zero if visits are missing
    kt_policy_prob = kt_raw_visits / (total_visits if total_visits > 0 else 1.0)

    # Extract Win/Loss/NoResult (Indices 16, 17, 18)
    # These are already effectively probabilities (summing to ~1.0 or 100 depending on version)
    kt_white_win = sample_raw['globalTargetsNC'][0, 16] 
    
    # 4. Model Inference
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            p_board, p_pass, v_pred, o_pred = model(batch.x_dict, batch.edge_index_dict)
            
    p_full = torch.cat([p_board.view(1, 361), p_pass.view(1, 1)], dim=1)
    model_probs = torch.softmax(p_full, dim=1).cpu().numpy()[0]
    
    def get_coord(idx):
        if idx == 361: return "PASS"
        r, c = idx // 19, idx % 19
        col = chr(ord('A') + c + (1 if c >= 8 else 0))
        return f"{col}{19 - r}"

    # 5. Print Comparison
    print(f"--- Global Diagnostics ---")
    print(f"Total MCTS Visits: {total_visits:.0f}")
    print(f"KataGo Win Chance (White): {kt_white_win:.2%}")
    print(f"Model Predicted Value:    {(v_pred.item() / 2 + 0.5):.2%}")

    print("\n--- Top 5 Moves: Model vs KataGo Visits ---")
    m_top = np.argsort(model_probs)[-5:][::-1]
    k_top = np.argsort(kt_policy_prob)[-5:][::-1]

    print(f"{'Rank':<5} | {'Model Prediction':<20} | {'KataGo Target (Visits)'}")
    print("-" * 60)
    for i in range(5):
        m_idx, k_idx = m_top[i], k_top[i]
        
        m_str = f"{get_coord(m_idx)} ({model_probs[m_idx]:.1%})"
        # Show both the percentage of total visits and the raw count
        k_str = f"{get_coord(k_idx)} ({kt_policy_prob[k_idx]:.1%}) [Count: {kt_raw_visits[k_idx]:.0f}]"
        
        print(f"{i+1:<5} | {m_str:<20} | {k_str}")

if __name__ == "__main__":
    evaluate_sample_row("sample_3.npz", "best_go_gnn_combined.pth", 40)