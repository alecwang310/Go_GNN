import sys
import torch
import numpy as np
from torch_geometric.data import Batch

root_path = r'D:/Code/GNN'
if root_path not in sys.path:
    sys.path.append(root_path)

from python.Data_extract import KataGoData
from python.GNN import GoGNN as gnn_shallow
from python.GNN_deep import GoGNN as gnn_deep

class GoGNN_GTP_Engine:
    def __init__(self, model_path, size=19):
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize Board State (The "Internal Engine")
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1 # 1=Black, 2=White
        self.history = []
        self.ko_point = None
        self.komi = 7.5

        # 2. Load Model
        print(f"Loading model to {self.device}...", file=sys.stderr)
        self.model = gnn_deep().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handling the _orig_mod prefix if you used torch.compile
        state_dict = checkpoint['model_state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "") 
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        # 3. Setup Data Builder
        self.data_builder = KataGoData()

    # --- Engine Logic (Integrated) ---
    def play_move(self, r, c, color):
        if r == -1: # Pass
            self.history.append((-1, -1))
            self.current_player = 3 - color
            self.ko_point = None
            return True

        if self.board[r, c] != 0 or (r, c) == self.ko_point:
            return False

        # Place stone
        self.board[r, c] = color
        self.history.append((r, c))
        if len(self.history) > 5: self.history.pop(0)

        # Captures
        opp = 3 - color
        captured = self._clear_dead_groups(opp)
        
        # Suicide check
        if self._get_liberties(r, c) == 0:
            self.board[r, c] = 0
            self.history.pop()
            return False

        # Simple Ko detection
        if len(captured) == 1 and len(self._get_group(r, c)) == 1 and self._get_liberties(r, c) == 1:
            self.ko_point = captured[0]
        else:
            self.ko_point = None

        self.current_player = 3 - color
        return True

    def _get_group(self, r, c):
        color = self.board[r, c]
        if color == 0: return []
        stack, group = [(r, c)], {(r, c)}
        while stack:
            curr_r, curr_c = stack.pop()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == color and (nr, nc) not in group:
                        group.add((nr, nc))
                        stack.append((nr, nc))
        return list(group)

    def _get_liberties(self, r, c):
        group = self._get_group(r, c)
        liberties = set()
        for gr, gc in group:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == 0: liberties.add((nr, nc))
        return len(liberties)

    def _clear_dead_groups(self, color):
        captured = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == color and self._get_liberties(r, c) == 0:
                    for gr, gc in self._get_group(r, c):
                        self.board[gr, gc] = 0
                        captured.append((gr, gc))
        return captured

    def generate_features(self):
        feat = np.zeros((22, self.size, self.size), dtype=np.float32)
        opp = 3 - self.current_player
        feat[0] = 1.0
        feat[1] = (self.board == self.current_player).astype(np.float32)
        feat[2] = (self.board == opp).astype(np.float32)
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    libs = self._get_liberties(r, c)
                    if libs == 1: feat[3, r, c] = 1.0
                    elif libs == 2: feat[4, r, c] = 1.0
                    elif libs >= 3: feat[5, r, c] = 1.0
        if self.ko_point: feat[6, self.ko_point[0], self.ko_point[1]] = 1.0
        for i, move in enumerate(reversed(self.history)):
            if i < 5 and move != (-1, -1): feat[9 + i, move[0], move[1]] = 1.0
        
        g_vars = np.zeros(19, dtype=np.float32)
        g_vars[5] = self.komi / 20.0
        return feat, g_vars

    # --- GTP Communication ---
    def parse_gtp_move(self, move_str):
        move_str = move_str.upper()
        if move_str == "PASS": return -1, -1
        col = ord(move_str[0]) - ord('A')
        if col > 8: col -= 1 
        row = self.size - int(move_str[1:])
        return row, col

    def move_to_gtp(self, r, c):
        if r == -1: return "PASS"
        col_char = chr(ord('A') + c + (1 if c >= 8 else 0))
        return f"{col_char}{self.size - r}"

    def respond(self, message="", success=True):
        print(f"{'=' if success else '?'} {message}\n", flush=True)

    def run(self):
        while True:
            line = sys.stdin.readline()
            if not line: break
            parts = line.strip().split()
            if not parts: continue
            
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == "name": self.respond("GoGNN_Bot")
            elif cmd == "protocol_version": self.respond("2")
            elif cmd == "version": self.respond("1.0")
            elif cmd == "list_commands": self.respond("name\nversion\nprotocol_version\nboardsize\nclear_board\nkomi\nplay\ngenmove\nquit")
            elif cmd == "boardsize": self.respond()
            elif cmd == "clear_board":
                self.board.fill(0)
                self.history = []
                self.ko_point = None
                self.respond()
            elif cmd == "komi":
                self.komi = float(args[0])
                self.respond()
            elif cmd == "play":
                color = 1 if args[0].upper().startswith('B') else 2
                r, c = self.parse_gtp_move(args[1])
                if self.play_move(r, c, color): self.respond()
                else: self.respond("illegal move", False)
            elif cmd == "genmove":
                self.current_player = 1 if args[0].upper().startswith('B') else 2
                feat, g_vars = self.generate_features()
                # Use dummy labels for graph building
                graph = self.data_builder._build_single_graph(feat, np.zeros(362), np.zeros(1), np.zeros(361), g_vars)
                batch = Batch.from_data_list([graph]).to(self.device)
                
                with torch.no_grad():
                    with torch.amp.autocast(self.device.type):
                        p_board, p_pass, _, _ = self.model(batch.x_dict, batch.edge_index_dict)
                
                # Decision logic
                p_full = torch.cat([p_board.view(1, 361), p_pass.view(1, 1)], dim=1)
                probs = torch.softmax(p_full, dim=1).cpu().numpy()[0]
                sorted_idx = np.argsort(probs)[::-1]
                
                found = False
                for idx in sorted_idx:
                    r, c = (idx // 19, idx % 19) if idx < 361 else (-1, -1)
                    if self.play_move(r, c, self.current_player):
                        self.respond(self.move_to_gtp(r, c))
                        found = True
                        break
                if not found: self.respond("resign")
            elif cmd == "quit":
                self.respond()
                break
            else: self.respond("unknown command", False)

if __name__ == "__main__":
    bot = GoGNN_GTP_Engine(r"D:/Code/GNN/models/deep_old_final.pth")
    bot.run()