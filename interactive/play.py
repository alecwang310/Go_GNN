import torch
import numpy as np
from torch_geometric.data import Batch

import sys
root_path = r'D:/Code/GNN'
if root_path not in sys.path:
    sys.path.append(root_path)

from python.Data_extract import KataGoData
from python.GNN import GoGNN

class SimpleGoEngine:
    def __init__(self, size=19):
        self.size = size
        # 0 = empty, 1 = Black, 2 = White
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1 
        self.history = [] # To track the last 5 moves for features 9-13
        self.ko_point = None

    def play_move(self, r, c):
        if r < 0 or r >= self.size or c < 0 or c >= self.size or self.board[r, c] != 0:
            return False
        if (r, c) == self.ko_point:
            return False

        # Place stone
        self.board[r, c] = self.current_player
        self.history.append((r, c))
        if len(self.history) > 5:
            self.history.pop(0)

        # Handle captures
        opp = 3 - self.current_player
        captured_stones = self._clear_dead_groups(opp)
        
        # Self-capture check
        if self._get_liberties(r, c) == 0:
            self.board[r, c] = 0 # Undo
            self.history.pop()
            return False

        # Set Ko point - FIXED: changed .size to len()
        if len(captured_stones) == 1 and len(self._get_group(r, c)) == 1 and self._get_liberties(r, c) == 1:
            self.ko_point = captured_stones[0]
        else:
            self.ko_point = None

        self.current_player = opp
        return True

    def pass_turn(self):
        self.history.append((-1, -1)) # -1, -1 means pass
        self.current_player = 3 - self.current_player

    def _get_group(self, r, c):
        color = self.board[r, c]
        if color == 0: return []
        stack = [(r, c)]
        group = set(stack)
        while stack:
            curr_r, curr_c = stack.pop()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == color and (nr, nc) not in group:
                        group.add((nr, nc))
                        stack.append((nr, nc))
        return group

    def _get_liberties(self, r, c):
        group = self._get_group(r, c)
        liberties = set()
        for gr, gc in group:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == 0:
                        liberties.add((nr, nc))
        return len(liberties)

    def _clear_dead_groups(self, color):
        captured = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == color and self._get_liberties(r, c) == 0:
                    group = self._get_group(r, c)
                    for gr, gc in group:
                        self.board[gr, gc] = 0
                        captured.append((gr, gc))
        return captured

    def generate_features(self):
        # We need 22 channels to match the unpacking logic, 
        # but we will only fill the ones your model's 'keep_idx' uses.
        feat = np.zeros((22, self.size, self.size), dtype=np.float32)
        opp = 3 - self.current_player

        # Basic Stones
        feat[0] = 1.0
        feat[1] = (self.board == self.current_player).astype(np.float32)
        feat[2] = (self.board == opp).astype(np.float32)

        # Liberties
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    libs = self._get_liberties(r, c)
                    if libs == 1: feat[3, r, c] = 1.0
                    elif libs == 2: feat[4, r, c] = 1.0
                    elif libs >= 3: feat[5, r, c] = 1.0

        # Ko
        if self.ko_point:
            feat[6, self.ko_point[0], self.ko_point[1]] = 1.0

        # History (9-13)
        for i, move in enumerate(reversed(self.history)):
            if i < 5 and move != (-1, -1):
                feat[9 + i, move[0], move[1]] = 1.0

        # Features 18 & 19 (The ones you mentioned)
        # Since we don't have the area calculator, we leave them as 0.0.
        # If your model RELIES on these to play, it might stay "scared."
        feat[18] = 0.0 
        feat[19] = 0.0

        # Global variables (19 features)
        global_vars = np.zeros(19, dtype=np.float32)
        global_vars[5] = 7.5 / 20.0 # Komi
        # Add other global constants your model expects here...

        return feat, global_vars
    
    def print_board(self):
        # 1. Prepare Header
        columns = [chr(ord('A') + i + (1 if i >= 8 else 0)) for i in range(self.size)]
        header = "    " + " ".join(columns)
        print("\n" + header)

        # 2. Define Star Point coordinates (0-indexed)
        # For 19x19, these are usually 3, 9, and 15
        star_points = {3, 9, 15}

        for r in range(self.size):
            # Format row number to be 2 digits wide
            row_str = f"{self.size - r:2d} "
            
            for c in range(self.size):
                if self.board[r, c] == 1:
                    row_str += " X" # Black
                elif self.board[r, c] == 2:
                    row_str += " O" # White
                elif r in star_points and c in star_points:
                    row_str += " +" # Star Point (Hoshi)
                else:
                    row_str += " ." # Normal empty point
            
            print(row_str + f" {self.size - r}")
            
        print(header + "\n")

def parse_move(move_str, size=19):
    move_str = move_str.strip().upper()
    if move_str == "PASS": return -1, -1
    col_char = move_str[0]
    row_num = int(move_str[1:])
    c = ord(col_char) - ord('A')
    if c > 8: c -= 1 # Skip 'I'
    r = size - row_num
    return r, c

def move_to_str(r, c, size=19):
    if r == -1: return "PASS"
    col_char = chr(ord('A') + c + (1 if c >= 8 else 0))
    row_num = size - r
    return f"{col_char}{row_num}"

def play_game(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    print("Loading model...")
    model = GoGNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Setup Data Builder & Engine
    data_builder = KataGoData()
    engine = SimpleGoEngine()
    
    print("\n--- Game Started ---")
    print("You are Black (X). Bot is White (O).")
    print("Enter moves like 'D4', 'Q16', or 'pass'.")
    
    while True:
        engine.print_board()
        
        if engine.current_player == 1:
            # Human Turn
            move_str = input("Your move: ")
            if move_str.lower() in ['quit', 'exit', 'resign']:
                break
            try:
                r, c = parse_move(move_str)
                if r == -1:
                    engine.pass_turn()
                elif not engine.play_move(r, c):
                    print("Illegal move, try again.")
            except:
                print("Invalid format. Use D4, etc.")
                
        else:
            # Bot Turn
            print("Bot is thinking...")
            feat, global_vars = engine.generate_features()
            
            # Dummy targets so PyG HeteroData doesn't crash
            d_pol = np.zeros(362)
            d_val = np.zeros(1)
            d_own = np.zeros(361)
            
            graph = data_builder._build_single_graph(feat, d_pol, d_val, d_own, global_vars)
            batch = Batch.from_data_list([graph]).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    p_board, p_pass, v_pred, _ = model(batch.x_dict, batch.edge_index_dict)
                    print(f"Raw Pass Logit: {p_pass.item():.4f}")
                    print(f"Max Board Logit: {p_board.max().item():.4f}")
                    print(f"Min Board Logit: {p_board.min().item():.4f}")
                    win_prob = torch.sigmoid(v_pred).item()
                    print(f"Bot's Win Probability: {win_prob:.2%}")

                    # Look at the top 3 board moves (ignoring the pass)
                    top_values, top_indices = torch.topk(p_board.view(-1), 3)
                    for i in range(3):
                        idx = top_indices[i].item()
                        print(f"Top Move {i+1}: {move_to_str(idx // 19, idx % 19)} (Logit: {top_values[i].item():.2f})")

            # Combine pass and board logits
            p_full = torch.cat([p_board.view(1, 361), p_pass.view(1, 1)], dim=1)
            probs = torch.softmax(p_full, dim=1).cpu().numpy()[0]
            
            # Mask illegal moves
            for i in range(361):
                r, c = i // 19, i % 19
                if engine.board[r, c] != 0 or (r, c) == engine.ko_point:
                    probs[i] = -1.0 # Make impossible to pick
                    
            # Pick highest valid probability
            best_move_idx = np.argmax(probs)
            
            if best_move_idx == 361:
                print("Bot plays: PASS")
                engine.pass_turn()
            else:
                r, c = best_move_idx // 19, best_move_idx % 19
                print(f"Bot plays: {move_to_str(r, c)}")
                print(f"Bot win-prob estimation: {v_pred.item():.2f} / 1.0")
                engine.play_move(r, c)

if __name__ == "__main__":
    # Change this to your best checkpoint path
    play_game("best_go_gnn_combined.pth")