import sys
import math
import torch
import numpy as np
from torch_geometric.data import Batch
from collections import OrderedDict

root_path = r'D:/Code/GNN'
if root_path not in sys.path:
    sys.path.append(root_path)

from python.Data_extract import KataGoData
from python.GNN_deep import GoGNN


class GoBoard:
    SIZE = 19

    def __init__(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.ko_point = None
        self.history = []  # most recent last
        self.history_passes = []  # most recent last

    def reset(self):
        self.board.fill(0)
        self.ko_point = None
        self.history = []
        self.history_passes = []

    def play_move(self, r, c, color):
        if r == -1:
            self.history.append(-1)
            self.history_passes.append(True)
            if len(self.history) > 5:
                self.history.pop(0)
                self.history_passes.pop(0)
            self.ko_point = None
            return True

        if self.board[r, c] != 0 or (r, c) == self.ko_point:
            return False

        self.board[r, c] = color
        self.history.append(r * self.SIZE + c)
        self.history_passes.append(False)
        if len(self.history) > 5:
            self.history.pop(0)
            self.history_passes.pop(0)

        opp = 3 - color
        captured = self._clear_dead_groups(opp)

        if self._get_liberties(r, c) == 0:
            self.board[r, c] = 0
            self.history.pop()
            self.history_passes.pop()
            return False

        if len(captured) == 1 and len(self._get_group(r, c)) == 1 and self._get_liberties(r, c) == 1:
            self.ko_point = captured[0]
        else:
            self.ko_point = None

        return True

    def is_legal(self, r, c, color):
        if r == -1:
            return True
        if r < 0 or r >= self.SIZE or c < 0 or c >= self.SIZE:
            return False
        if self.board[r, c] != 0 or (r, c) == self.ko_point:
            return False
        # Temporarily place
        self.board[r, c] = color
        has_libs = self._get_liberties(r, c) > 0
        if not has_libs:
            opp = 3 - color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if self.board[nr, nc] == opp and self._get_liberties(nr, nc) == 0:
                        has_libs = True
                        break
        self.board[r, c] = 0
        return has_libs

    def get_legal_moves(self, color):
        moves = []
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.is_legal(r, c, color):
                    moves.append((r, c))
        moves.append((-1, -1))  # pass always legal
        return moves

    def _get_group(self, r, c):
        color = self.board[r, c]
        if color == 0:
            return []
        stack = [(r, c)]
        group = set(stack)
        while stack:
            cr, cc = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if self.board[nr, nc] == color and (nr, nc) not in group:
                        group.add((nr, nc))
                        stack.append((nr, nc))
        return list(group)

    def _get_liberties(self, r, c):
        group = self._get_group(r, c)
        liberties = set()
        for gr, gc in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                    if self.board[nr, nc] == 0:
                        liberties.add((nr, nc))
        return len(liberties)

    def _clear_dead_groups(self, color):
        captured = []
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.board[r, c] == color and self._get_liberties(r, c) == 0:
                    group = self._get_group(r, c)
                    for gr, gc in group:
                        self.board[gr, gc] = 0
                        captured.append((gr, gc))
        return captured

    def generate_features(self, komi, current_player):
        feat = np.zeros((22, self.SIZE, self.SIZE), dtype=np.float32)
        opp = 3 - current_player

        feat[0] = 1.0
        feat[1] = (self.board == current_player).astype(np.float32)
        feat[2] = (self.board == opp).astype(np.float32)

        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.board[r, c] != 0:
                    libs = self._get_liberties(r, c)
                    if libs == 1:
                        feat[3, r, c] = 1.0
                    elif libs == 2:
                        feat[4, r, c] = 1.0
                    elif libs >= 3:
                        feat[5, r, c] = 1.0

        if self.ko_point:
            feat[6, self.ko_point[0], self.ko_point[1]] = 1.0

        # History channels 9-13, oldest to most recent
        # self.history is oldest first, most recent last
        for i, move_flat in enumerate(self.history):
            if move_flat != -1:
                feat[9 + i, move_flat // self.SIZE, move_flat % self.SIZE] = 1.0

        g_vars = np.zeros(19, dtype=np.float32)
        for i, passed in enumerate(self.history_passes):
            if passed:
                g_vars[i] = 1.0
        g_vars[5] = komi / 20.0

        return feat, g_vars


class MCTSNode:
    def __init__(self, parent=None, move=None, color=None, prior=0.0):
        self.parent = parent
        self.move = move  # (r, c) or (-1, -1) for pass
        self.color = color
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        self.children = []

    def mean_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTS:
    def __init__(self, model, device, data_builder, komi=7.5, num_simulations=200, cpuct=2.0):
        self.model = model
        self.device = device
        self.data_builder = data_builder
        self.komi = komi
        self.num_simulations = num_simulations
        self.cpuct = cpuct

    def search(self, board, current_player):
        root = MCTSNode(color=3 - current_player)

        legal_moves = board.get_legal_moves(current_player)
        if len(legal_moves) == 1:  # only pass
            return -1, -1

        # Expand root
        self._expand_node(root, board, current_player)

        for _ in range(self.num_simulations):
            sim_board = GoBoard()
            sim_board.board = board.board.copy()
            sim_board.ko_point = board.ko_point
            sim_board.history = list(board.history)
            sim_board.history_passes = list(board.history_passes)

            node = root
            sim_player = current_player

            # Select
            while node.is_expanded and node.children:
                node = self._select_child(node)
                r, c = node.move
                sim_board.play_move(r, c, sim_player)
                sim_player = 3 - sim_player

            # Expand & evaluate
            if not node.is_expanded:
                value = self._expand_node(node, sim_board, sim_player)
            else:
                value = 0.0  # terminal

            # Backup
            self._backup(node, value, sim_player)

        # Pick most visited child
        best = max(root.children, key=lambda n: n.visits)
        return best.move

    def _select_child(self, node):
        sqrt_parent = math.sqrt(node.visits)
        best, best_score = None, -float('inf')
        for child in node.children:
            q = child.mean_value()
            u = self.cpuct * child.prior * sqrt_parent / (1.0 + child.visits)
            score = -q + u
            if score > best_score:
                best_score = score
                best = child
        return best

    def _expand_node(self, node, board, current_player):
        node.is_expanded = True

        feat, g_vars = board.generate_features(self.komi, current_player)
        graph = self.data_builder._build_single_graph(
            feat, np.zeros(362), np.zeros(1), np.zeros(361), g_vars
        )
        batch = Batch.from_data_list([graph]).to(self.device)

        with torch.no_grad():
            p_board, p_pass, value_t, _ = self.model(batch.x_dict, batch.edge_index_dict)

        value = value_t.item()

        p_full = torch.cat([p_board.view(1, 361), p_pass.view(1, 1)], dim=1)

        legal_moves = board.get_legal_moves(current_player)

        # Mask illegal moves to -inf before softmax so they get zero probability
        mask = torch.full((1, 362), float('-inf'), device=p_full.device)
        for r, c in legal_moves:
            if r == -1:
                mask[0, 361] = 0.0
            else:
                mask[0, r * 19 + c] = 0.0
        probs = torch.softmax(p_full + mask, dim=1).cpu().numpy()[0]

        for r, c in legal_moves:
            if r == -1:
                prior = probs[361]
            else:
                prior = probs[r * 19 + c]
            node.children.append(MCTSNode(parent=node, move=(r, c), color=current_player, prior=prior))

        return value

    def _backup(self, node, value, player_at_leaf):
        curr = node
        v = value
        while curr is not None:
            curr.visits += 1
            curr.value_sum += v
            v = -v
            curr = curr.parent


class GoGNN_GTP_Engine:
    def __init__(self, model_path, time_increment=10):
        self.size = 19
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.komi = 7.5
        self.time_increment = time_increment
        self.num_simulations = time_increment * 20

        self.board = GoBoard()
        self.current_player = 1  # 1=Black, 2=White

        # Load model
        print(f"Loading model to {self.device}...", file=sys.stderr)
        self.model = GoGNN().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("_orig_mod.", "")] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        self.data_builder = KataGoData()
        self.mcts = MCTS(self.model, self.device, self.data_builder, self.komi, self.num_simulations)

        print("Search bot ready.", file=sys.stderr)

    def parse_gtp_move(self, move_str):
        move_str = move_str.upper()
        if move_str == "PASS":
            return -1, -1
        col = ord(move_str[0]) - ord('A')
        if col > 8:
            col -= 1
        row = self.size - int(move_str[1:])
        return row, col

    def move_to_gtp(self, r, c):
        if r == -1:
            return "PASS"
        col_char = chr(ord('A') + c + (1 if c >= 8 else 0))
        return f"{col_char}{self.size - r}"

    def respond(self, message="", success=True):
        print(f"{'=' if success else '?'} {message}\n", flush=True)

    def run(self):
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            parts = line.strip().split()
            if not parts:
                continue

            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == "name":
                self.respond("GoGNN_Search")
            elif cmd == "protocol_version":
                self.respond("2")
            elif cmd == "version":
                self.respond(f"1.0 ({self.num_simulations} sims)")
            elif cmd == "list_commands":
                self.respond("name\nversion\nprotocol_version\nboardsize\nclear_board\nkomi\nplay\ngenmove\ntime_settings\ntime_left\nquit")
            elif cmd == "boardsize":
                self.respond()
            elif cmd == "clear_board":
                self.board.reset()
                self.respond()
            elif cmd == "komi":
                self.komi = float(args[0])
                self.mcts.komi = self.komi
                self.respond()
            elif cmd == "play":
                color = 1 if args[0].upper().startswith('B') else 2
                r, c = self.parse_gtp_move(args[1])
                if self.board.play_move(r, c, color):
                    self.respond()
                else:
                    self.respond("illegal move", False)
            elif cmd == "genmove":
                self.current_player = 1 if args[0].upper().startswith('B') else 2
                best_r, best_c = self.mcts.search(self.board, self.current_player)
                self.board.play_move(best_r, best_c, self.current_player)
                self.respond(self.move_to_gtp(best_r, best_c))
            elif cmd == "time_settings":
                # time_settings <main_time> <byo_yomi_time> <byo_yomi_stones>
                if len(args) >= 2:
                    byo_yomi_time = float(args[1])
                    if byo_yomi_time > 0:
                        self.time_increment = byo_yomi_time
                        self.num_simulations = int(self.time_increment * 20)
                        self.mcts.num_simulations = self.num_simulations
                        print(f"Time increment set to {self.time_increment}s -> {self.num_simulations} sims", file=sys.stderr)
                self.respond()
            elif cmd == "time_left":
                # time_left <color> <time> <stones>
                if len(args) >= 2:
                    remaining = float(args[1])
                    if remaining > 0 and remaining < self.time_increment:
                        # Scale simulations down if running low on time
                        self.num_simulations = max(1, int(remaining * 20))
                        self.mcts.num_simulations = self.num_simulations
                self.respond()
            elif cmd == "quit":
                self.respond()
                break
            else:
                self.respond("unknown command", False)


if __name__ == "__main__":
    bot = GoGNN_GTP_Engine(r"D:/Code/GNN/models/deep_old_final.pth", time_increment=10)
    bot.run()
