import sys
import os
import site

# Add torch lib to DLL search path so gomcts can find its dependencies
for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
    torch_lib = os.path.join(site_dir, "torch", "lib")
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        break

import gomcts


class GoGNN_GTP_Engine:
    def __init__(self, model_path, time_increment=10):
        self.size = 19
        self.komi = 7.5
        self.time_increment = time_increment
        self.num_simulations = int(time_increment * 150)
        self.num_threads = 16

        self.board = gomcts.GoBoard()
        self.current_player = 1  # 1=Black, 2=White

        device = "cuda"
        print(f"Loading model...", file=sys.stderr)
        self.mcts = gomcts.MCTS(
            model_path, device, self.komi, self.num_simulations, 2.0, 0.25, self.num_threads
        )

        print("Search bot ready.", file=sys.stderr)

    def parse_gtp_move(self, move_str):
        """Convert GTP move string to padded board index. Returns -1 for pass."""
        move_str = move_str.upper()
        if move_str == "PASS":
            return -1
        col = ord(move_str[0]) - ord('A')
        if col > 8:
            col -= 1
        row = self.size - int(move_str[1:])
        return self.board.get_index(row, col)

    def move_to_gtp(self, pos):
        """Convert padded board index to GTP move string."""
        if pos == -1:
            return "PASS"
        r = pos // gomcts.GoBoard.PADDED_SIZE - 1
        c = pos % gomcts.GoBoard.PADDED_SIZE - 1
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
                if args and int(args[0]) != 19:
                    self.respond("unacceptable size", False)
                else:
                    self.respond()
            elif cmd == "clear_board":
                self.board.reset()
                self.respond()
            elif cmd == "komi":
                self.komi = float(args[0])
                self.board.komi = self.komi
                self.respond()
            elif cmd == "play":
                color = gomcts.BLACK if args[0].upper().startswith('B') else gomcts.WHITE
                pos = self.parse_gtp_move(args[1])
                if self.board.play_move(pos, color):
                    self.respond()
                else:
                    self.respond("illegal move", False)
            elif cmd == "genmove":
                color = gomcts.BLACK if args[0].upper().startswith('B') else gomcts.WHITE
                best_pos = self.mcts.search(self.board, color)
                self.board.play_move(best_pos, color)
                self.respond(self.move_to_gtp(best_pos))
            elif cmd == "time_settings":
                # time_settings <main_time> <byo_yomi_time> <byo_yomi_stones>
                if len(args) >= 2:
                    byo_yomi_time = float(args[1])
                    if byo_yomi_time > 0:
                        self.time_increment = byo_yomi_time
                        self.num_simulations = int(self.time_increment * 250)
                        self.mcts.set_num_simulations(self.num_simulations)
                        print(f"Time increment set to {self.time_increment}s -> {self.num_simulations} sims", file=sys.stderr)
                self.respond()
            elif cmd == "time_left":
                # time_left <color> <time> <stones>
                if len(args) >= 2:
                    remaining = float(args[1])
                    if remaining > 0 and remaining < self.time_increment:
                        # Scale simulations down if running low on time
                        self.num_simulations = max(1, int(remaining * 20))
                        self.mcts.set_num_simulations(self.num_simulations)
                self.respond()
            elif cmd == "quit":
                self.respond()
                break
            else:
                self.respond("unknown command", False)


if __name__ == "__main__":
    bot = GoGNN_GTP_Engine(r"D:/Code/GNN/traced_models/gognn_traced.pt", time_increment=10)
    bot.run()
