#include <iostream>
#include <string>
#include <chrono>
#include <sstream>

#include <torch/torch.h>
#include <torch/script.h>

#include "goboard.hpp"
#include "graph_creation.hpp"
#include "search.hpp"

// Convert (row, col) string like "D4" to board index
int parse_move(const std::string& input, const GoBoard& board) {
    if (input == "pass" || input == "PASS" || input == "p") return -1;
    if (input == "resign" || input == "RESIGN" || input == "r") return -2;

    if (input.size() < 2 || input.size() > 3) return -3; // invalid

    char col_char = std::toupper(input[0]);
    if (col_char < 'A' || col_char > 'T' || col_char == 'I') return -3; // skip I

    int col;
    if (col_char > 'I') col = col_char - 'A' - 1;
    else col = col_char - 'A';

    int row;
    try {
        row = std::stoi(input.substr(1)) - 1;
    } catch (...) {
        return -3;
    }

    if (row < 0 || row >= 19 || col < 0 || col >= 19) return -3;

    return board.get_index(row, col);
}

std::string move_to_string(int pos, const GoBoard& board) {
    if (pos == -1) return "pass";
    int r = pos / GoBoard::PADDED_SIZE - 1;
    int c = pos % GoBoard::PADDED_SIZE - 1;
    char col_char;
    if (c >= 8) col_char = 'A' + c + 1; // skip I
    else col_char = 'A' + c;
    return std::string(1, col_char) + std::to_string(r + 1);
}

void print_board(const GoBoard& board, int8_t human_color) {
    std::cout << "\n   ";
    for (int c = 0; c < 19; ++c) {
        char col_char;
        if (c >= 8) col_char = 'A' + c + 1;
        else col_char = 'A' + c;
        std::cout << col_char << ' ';
    }
    std::cout << "\n";

    for (int r = 0; r < 19; ++r) {
        int label = r + 1;
        if (label < 10) std::cout << " ";
        std::cout << label << " ";
        for (int c = 0; c < 19; ++c) {
            int pos = board.get_index(r, c);
            int8_t stone = board.board[pos];
            if (stone == BLACK) std::cout << "X ";
            else if (stone == WHITE) std::cout << "O ";
            else std::cout << ". ";
        }
        std::cout << label << "\n";
    }

    std::cout << "   ";
    for (int c = 0; c < 19; ++c) {
        char col_char;
        if (c >= 8) col_char = 'A' + c + 1;
        else col_char = 'A' + c;
        std::cout << col_char << ' ';
    }
    std::cout << "\n";

    if (human_color == BLACK) std::cout << "You: X (Black)  Bot: O (White)\n";
    else std::cout << "You: O (White)  Bot: X (Black)\n";
    std::cout << "\n";
}

// Pure NN move selection (no search)
int nn_best_move(GoBoard& board, int8_t current_player,
                 torch::jit::script::Module& module, torch::Device device, float komi) {
    FastGoGraphGenerator graph_gen;
    GraphData gd = graph_gen.generate(board, komi, current_player, device);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(gd.stone_x);
    inputs.push_back(gd.string_x);
    inputs.push_back(gd.global_x);
    inputs.push_back(gd.e_s_a_s);
    inputs.push_back(gd.e_s_b_str);
    inputs.push_back(gd.e_str_c_s);
    inputs.push_back(gd.e_str_a_str);
    inputs.push_back(gd.e_str_r_g);
    inputs.push_back(gd.e_g_i_str);

    torch::NoGradGuard no_grad;
    auto output = module.forward(inputs).toTuple();

    torch::Tensor policy_t = output->elements()[0].toTensor().squeeze();
    torch::Tensor pass_val_t = output->elements()[1].toTensor().squeeze();
    torch::Tensor value_t = output->elements()[2].toTensor().squeeze();

    float value = value_t.item<float>();
    std::cout << "  NN value: " << value << " (from " << (current_player == BLACK ? "Black" : "White") << "'s perspective)\n";

    // Combine board logits + pass logit, then single softmax (matching Python play.py)
    torch::Tensor full_logits = torch::cat({policy_t.unsqueeze(0), pass_val_t.unsqueeze(0).unsqueeze(0)}, 1); // [1, 362]
    torch::Tensor full_probs = torch::softmax(full_logits, 1).squeeze(0); // [362]

    auto probs_cpu = full_probs.to(torch::kCPU);
    auto probs_acc = probs_cpu.accessor<float, 1>();

    float pass_prior = probs_acc[361];
    std::cout << "  NN pass prior (softmax): " << pass_prior << "\n";

    // Collect legal moves with their priors
    std::vector<std::pair<float, int>> candidates;
    float best_prior = -1.0f;
    int best_move = -1;

    for (int r = 0; r < 19; ++r) {
        for (int c = 0; c < 19; ++c) {
            int pos = board.get_index(r, c);
            if (board.is_legal(pos, current_player)) {
                int flat = r * 19 + c;
                float p = probs_acc[flat];
                candidates.push_back({p, pos});
                if (p > best_prior) {
                    best_prior = p;
                    best_move = pos;
                }
            }
        }
    }

    // Compare pass prior
    candidates.push_back({pass_prior, -1});

    // Sort by prior descending, show top 10
    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    std::cout << "  Top 10 moves by policy:\n";
    for (int i = 0; i < std::min(10, (int)candidates.size()); ++i) {
        auto [p, pos] = candidates[i];
        std::string name = (pos == -1) ? "pass" : [&]() {
            int rr = pos / GoBoard::PADDED_SIZE - 1;
            int cc = pos % GoBoard::PADDED_SIZE - 1;
            char col_char = (cc >= 8) ? ('A' + cc + 1) : ('A' + cc);
            return std::string(1, col_char) + std::to_string(rr + 1);
        }();
        std::cout << "    " << (i + 1) << ". " << name << " = " << p << "\n";
    }

    // Return best legal move by policy (not pass unless pass is highest)
    float top_board_prior = best_prior;
    if (pass_prior > top_board_prior) return -1;
    return best_move;
}

int main() {
    // Setup device
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA/GPU for inference.\n";
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "CUDA not available. Using CPU.\n";
    }

    // Load model
    std::string model_path = "D:/Code/GNN/traced_models/gognn_traced.pt";
    std::string device_str = (device.is_cuda()) ? "cuda" : "cpu";
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.to(device);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.msg() << "\n";
        return -1;
    }
    std::cout << "Model loaded successfully.\n\n";

    // Choose mode
    std::cout << "Mode: (M)CTS search or (N)N only (no search)? [M]: ";
    std::string mode_input;
    std::getline(std::cin, mode_input);
    bool use_nn_only = (!mode_input.empty() && (mode_input[0] == 'n' || mode_input[0] == 'N'));
    if (use_nn_only) {
        std::cout << "Using pure NN mode (no search).\n\n";
    } else {
        std::cout << "Using MCTS search mode.\n\n";
    }

    // Game setup
    GoBoard board;
    float komi = 7.5f;
    int num_simulations = 200;
    int8_t human_color = BLACK;
    int8_t bot_color = WHITE;
    int consecutive_passes = 0;

    MCTS mcts(model_path, device_str, komi, num_simulations);

    std::cout << "=== Go AI Terminal ===\n";
    std::cout << "Commands during play:\n";
    std::cout << "  sims      - show current simulation count\n";
    std::cout << "  sims <N>  - change simulation count (current: " << num_simulations << ")\n";
    std::cout << "  resign    - resign the game\n";
    std::cout << "  pass      - pass your turn\n";
    std::cout << "  <move>    - e.g. D4, Q16, etc. (no I column)\n\n";

    // Choose color
    std::cout << "Play as (B)lack or (W)hite? [B]: ";
    std::string color_input;
    std::getline(std::cin, color_input);
    if (!color_input.empty() && (color_input[0] == 'w' || color_input[0] == 'W')) {
        human_color = WHITE;
        bot_color = BLACK;
        std::cout << "You are White (O). Bot plays Black (X).\n\n";
    } else {
        std::cout << "You are Black (X). Bot plays White (O).\n\n";
    }

    int8_t current_player = BLACK;
    bool game_over = false;

    while (!game_over) {
        print_board(board, human_color);

        if (current_player == human_color) {
            // Human turn
            std::cout << "Your move: ";
            std::string input;
            if (!std::getline(std::cin, input)) break;

            // Trim
            while (!input.empty() && input.back() == ' ') input.pop_back();
            while (!input.empty() && input.front() == ' ') input.erase(input.begin());

            if (input.empty()) continue;

            // Handle commands
            if (input.substr(0, 4) == "sims") {
                if (input.size() > 5) {
                    try {
                        int new_sims = std::stoi(input.substr(5));
                        if (new_sims > 0) {
                            num_simulations = new_sims;
                            mcts.set_num_simulations(num_simulations);
                            std::cout << "Simulation count set to " << num_simulations << "\n";
                        } else {
                            std::cout << "Simulation count must be > 0\n";
                        }
                    } catch (...) {
                        std::cout << "Usage: sims <number>\n";
                    }
                } else {
                    std::cout << "Current simulation count: " << num_simulations << "\n";
                }
                continue;
            }

            if (input == "resign") {
                std::cout << "You resigned. Bot wins!\n";
                game_over = true;
                continue;
            }

            int pos = parse_move(input, board);
            if (pos == -3) {
                std::cout << "Invalid move. Use format like D4, Q16, or 'pass'.\n";
                continue;
            }

            if (pos != -1 && !board.is_legal(pos, current_player)) {
                std::cout << "Illegal move. Try again.\n";
                continue;
            }

            // Play the move
            if (board.play_move(pos, current_player)) {
                std::cout << "You played: " << move_to_string(pos, board) << "\n";

                if (pos == -1) consecutive_passes++;
                else consecutive_passes = 0;

                current_player = 3 - current_player;
            } else {
                std::cout << "Move failed. Try again.\n";
            }
        } else {
            // Bot turn
            auto start = std::chrono::high_resolution_clock::now();

            int bot_move;
            if (use_nn_only) {
                std::cout << "Bot evaluating (NN only)...\n";
                bot_move = nn_best_move(board, current_player, module, device, komi);
            } else {
                std::cout << "Bot is thinking (" << num_simulations << " simulations)...\n";
                bot_move = mcts.search(board, current_player);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();

            if (bot_move == -1) {
                std::cout << "Bot passes. (" << elapsed << "s)\n";
            } else {
                std::cout << "Bot plays: " << move_to_string(bot_move, board) << " (" << elapsed << "s)\n";
            }

            board.play_move(bot_move, current_player);

            if (bot_move == -1) consecutive_passes++;
            else consecutive_passes = 0;

            current_player = 3 - current_player;
        }

        // Check for two consecutive passes
        if (consecutive_passes >= 2) {
            game_over = true;
            print_board(board, human_color);
            std::cout << "Two consecutive passes. Game over!\n";

            float score = board.calculate_area_score();
            std::cout << "Score (Black - White - komi): " << score << "\n";
            if (score > 0) std::cout << "Black wins by " << score << " points.\n";
            else if (score < 0) std::cout << "White wins by " << -score << " points.\n";
            else std::cout << "It's a tie!\n";
        }
    }

    std::cout << "\nThanks for playing!\n";
    return 0;
}
