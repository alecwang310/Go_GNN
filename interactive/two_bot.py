import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
from torch_geometric.data import Batch

import sys
root_path = r'D:/Code/GNN'
if root_path not in sys.path:
    sys.path.append(root_path)

from python.Data_extract import KataGoData
from python.GNN import GoGNN as gnn_shallow
from python.GNN_deep import GoGNN as gnn_deep

from collections import OrderedDict

class SimpleGoEngine:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int) 
        self.current_player = 1
        self.history = []
        self.ko_point = None

    def play_move(self, r, c):
        if not (0 <= r < self.size and 0 <= c < self.size) or self.board[r, c] != 0:
            return False
        if (r, c) == self.ko_point:
            return False

        self.board[r, c] = self.current_player
        self.history.append((r, c))
        if len(self.history) > 5: self.history.pop(0)

        opp = 3 - self.current_player
        captured = self._clear_dead_groups(opp)

        if self._get_liberties(r, c) == 0:
            self.board[r, c] = 0
            self.history.pop()
            return False

        if len(captured) == 1 and len(self._get_group(r, c)) == 1:
            self.ko_point = captured[0]
        else:
            self.ko_point = None

        self.current_player = opp
        return True

    def pass_turn(self):
        self.history.append((-1, -1))
        self.current_player = 3 - self.current_player

    def _get_group(self, r, c):
        color = self.board[r, c]
        if color == 0: return set()
        stack, group = [(r, c)], {(r, c)}
        while stack:
            curr_r, curr_c = stack.pop()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == color and (nr, nc) not in group:
                        group.add((nr, nc)); stack.append((nr, nc))
        return group

    def _get_liberties(self, r, c):
        group = self._get_group(r, c)
        libs = set()
        for gr, gc in group:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == 0:
                    libs.add((nr, nc))
        return len(libs)

    def _clear_dead_groups(self, color):
        captured = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == color and self._get_liberties(r, c) == 0:
                    group = self._get_group(r, c)
                    for gr, gc in group:
                        if self.board[gr, gc] != 0:
                            self.board[gr, gc] = 0
                            captured.append((gr, gc))
        return captured

    def calculate_score(self, komi=7.5):
        scores = {1: 0, 2: 0}
        visited = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0: scores[self.board[r, c]] += 1
                elif (r, c) not in visited:
                    group, owners = self._get_territory_group(r, c)
                    visited.update(group)
                    if len(owners) == 1: scores[list(owners)[0]] += len(group)
        return scores[1], scores[2] + komi

    def _get_territory_group(self, r, c):
        stack, group, owners = [(r, c)], {(r, c)}, set()
        while stack:
            cr, cc = stack.pop()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == 0 and (nr, nc) not in group:
                        group.add((nr, nc)); stack.append((nr, nc))
                    elif self.board[nr, nc] != 0: owners.add(self.board[nr, nc])
        return group, owners

    def generate_features(self):
        feat = np.zeros((22, 19, 19), dtype=np.float32)
        opp = 3 - self.current_player
        feat[0] = 1.0
        feat[1] = (self.board == self.current_player).astype(np.float32)
        feat[2] = (self.board == opp).astype(np.float32)
        for r in range(19):
            for c in range(19):
                if self.board[r, c] != 0:
                    libs = self._get_liberties(r, c)
                    if libs == 1: feat[3,r,c] = 1.0
                    elif libs == 2: feat[4,r,c] = 1.0
                    elif libs >= 3: feat[5,r,c] = 1.0
        if self.ko_point: feat[6, self.ko_point[0], self.ko_point[1]] = 1.0
        for i, move in enumerate(reversed(self.history)):
            if i < 5 and move != (-1, -1): feat[9 + i, move[0], move[1]] = 1.0
        global_vars = np.zeros(19, dtype=np.float32)
        global_vars[5] = 7.5 / 20.0
        return feat, global_vars

class GoBot:
    def __init__(self, model_path, name="Bot", device=None, type="shallow"):
        self.name, self.type = name, type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if type == "shallow": self.model = gnn_shallow().to(self.device)
        else: self.model = gnn_deep().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        sd = checkpoint['model_state_dict']
        if type == "deep":
            new_sd = OrderedDict()
            for k, v in sd.items(): new_sd[k.replace('_orig_mod.', '')] = v
            sd = new_sd
        self.model.load_state_dict(sd)
        self.model.eval()
        self.data_builder = KataGoData()

    @torch.no_grad()
    def get_predictions(self, engine):
        feat, gv = engine.generate_features()
        graph = self.data_builder._build_single_graph(feat, np.zeros(362), np.zeros(1), np.zeros(361), gv)
        batch = Batch.from_data_list([graph]).to(self.device)
        p_board, p_pass, v_pred, ownership = self.model(batch.x_dict, batch.edge_index_dict)
        probs = torch.softmax(torch.cat([p_board.view(1, 361), p_pass.view(1, 1)], dim=1), dim=1).cpu().numpy()[0]
        for i in range(361):
            r, c = divmod(i, 19)
            if engine.board[r, c] != 0 or (r, c) == engine.ko_point: probs[i] = -1e9
        return probs, torch.sigmoid(v_pred).item(), torch.tanh(ownership).view(19, 19).cpu().numpy()

def plot_board(engine, bots, move_num, b_win_rate, ownerships, view_idx, final_score=None):
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_facecolor('#E6B87D')
    
    if ownerships[view_idx] is not None:
        ax.imshow(ownerships[view_idx], extent=[-0.5, 18.5, -0.5, 18.5], origin='upper', cmap='RdBu', alpha=0.4, zorder=0)

    for i in range(19):
        ax.plot([i, i], [0, 18], color='black', lw=0.7, zorder=1)
        ax.plot([0, 18], [i, i], color='black', lw=0.7, zorder=1)
    
    for r in range(19):
        for c in range(19):
            if engine.board[r, c] == 1: ax.add_patch(Circle((c, 18-r), 0.43, color='black', zorder=3))
            elif engine.board[r, c] == 2: ax.add_patch(Circle((c, 18-r), 0.43, color='white', ec='black', lw=1, zorder=3))

    title = f"Move {move_num} | Black Win%: {b_win_rate:.1%}\nViewing: {bots[view_idx].name} (Keys 1/2)"
    if final_score: title = f"FINAL: B {final_score[0]:.1f} - W {final_score[1]:.1f}"
    ax.set_title(title, fontsize=10); ax.invert_yaxis(); plt.axis('off')

def two_bots_battle(path_b, path_w, max_m=500, delay=1.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bot1 = GoBot(path_b, name="Shallow_Black", device=device, type="shallow")
    bot2 = GoBot(path_w, name="Deep_White", device=device, type="deep")
    bots = [bot1, bot2]
    engine = SimpleGoEngine()
    
    # Initialize UI state before anything else
    ui_state = {'view_idx': 0, 'b_wp': 0.5, 'owns': [None, None]}

    plt.ion()
    fig = plt.figure(figsize=(8, 9))
    
    def on_press(event):
        if event.key == '1': ui_state['view_idx'] = 0
        elif event.key == '2': ui_state['view_idx'] = 1
    fig.canvas.mpl_connect('key_press_event', on_press)

    m_idx = 0
    while m_idx < max_m:
        p_idx = 0 if engine.current_player == 1 else 1
        probs, v, own = bots[p_idx].get_predictions(engine)
        
        ui_state['owns'][p_idx] = own
        # Fix Win Rate logic: Normalize to Black's perspective
        ui_state['b_wp'] = v if engine.current_player == 1 else (1 - v)
        
        plot_board(engine, bots, m_idx, ui_state['b_wp'], ui_state['owns'], ui_state['view_idx'])
        plt.pause(delay)

        move = np.argmax(probs)
        if move == 361: engine.pass_turn()
        else: engine.play_move(*divmod(move, 19))
        
        m_idx += 1
        if len(engine.history) >= 2 and engine.history[-1] == (-1,-1) and engine.history[-2] == (-1,-1): break

    plt.ioff()
    plot_board(engine, bots, m_idx, 0, ui_state['owns'], ui_state['view_idx'], engine.calculate_score())
    plt.show()

if __name__ == "__main__":
    MODEL_BLACK = r'D:/Code/GNN/models/best_go_gnn_combined.pth'
    MODEL_WHITE = r'D:/Code/GNN/models/temp.pth'
    two_bots_battle(MODEL_BLACK, MODEL_WHITE, delay=1.0)