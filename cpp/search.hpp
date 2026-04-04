#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <torch/torch.h>
#include <torch/script.h>

#include "goboard.hpp"
#include "graph_creation.hpp"

struct MCTSNode {
    MCTSNode* parent = nullptr;
    int move = -1; // board index, -1 = pass
    int8_t color = BLACK; // color that played to get here
    int visits = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    bool is_expanded = false;
    bool is_terminal = false;
    float terminal_value = 0.0f; // from perspective of the player to move

    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> legal_moves;

    float mean_value() const {
        if (visits == 0) return 0.0f;
        return value_sum / visits;
    }
};

class MCTS {
public:
    MCTS(const std::string& model_path, const std::string& device_str = "cpu",
         float komi = 7.5f, int num_simulations = 200, float cpuct = 2.0f,
         float fpu_value = 1.25f)
        : komi_(komi), num_simulations_(num_simulations), cpuct_(cpuct), fpu_value_(fpu_value),
          device_((device_str == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU)
    {
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
    }

    int search(GoBoard& board, int8_t current_player) {

        MCTSNode root;
        root.color = 3 - current_player; // parent's move color
        root.legal_moves = get_legal_moves(board, current_player);

        if (root.legal_moves.empty()) return -1; // no legal moves = pass

        for (int i = 0; i < num_simulations_; ++i) {
            bool debug = (i == 0);
            // 1. Select
            GoBoard sim_board = board;
            MCTSNode* node = &root;
            int8_t sim_player = current_player;

            while (node->is_expanded && !node->is_terminal && !node->children.empty()) {
                MCTSNode* best_child = select_child(node, sim_player);
                sim_board.play_move(best_child->move, sim_player);
                sim_player = 3 - sim_player;
                node = best_child;
            }

            // 2. Expand & Evaluate
            float value;
            if (is_game_over(sim_board, sim_player)) {
                value = get_terminal_value(sim_board, sim_player);
                node->is_terminal = true;
                node->terminal_value = value;
            } else {
                value = expand_and_evaluate(node, sim_board, sim_player, debug);
            }

            // 3. Backup
            backup(node, value, current_player);
        }

        // Pick move with most visits
        MCTSNode* best = nullptr;
        int best_visits = -1;
        for (auto& child : root.children) {
            if (child->visits > best_visits) {
                best_visits = child->visits;
                best = child.get();
            }
        }

        return best ? best->move : -1;
    }

    void set_num_simulations(int n) { num_simulations_ = n; }
    int get_num_simulations() const { return num_simulations_; }

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    float komi_;
    int num_simulations_;
    float cpuct_;
    float fpu_value_;
    FastGoGraphGenerator graph_gen_;

    std::vector<int> get_legal_moves(GoBoard& board, int8_t color) {
        std::vector<int> moves;
        for (int r = 0; r < 19; ++r) {
            for (int c = 0; c < 19; ++c) {
                int pos = board.get_index(r, c);
                if (board.is_legal(pos, color)) {
                    moves.push_back(pos);
                }
            }
        }
        // Always allow pass
        moves.push_back(-1);
        return moves;
    }

    MCTSNode* select_child(MCTSNode* node, int8_t current_player) {
        float sqrt_parent = std::sqrt((float)node->visits);
        MCTSNode* best = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();

        // FPU: if parent has been visited, use parent's mean value minus a penalty
        // for unvisited children. This gives urgency to try new moves when the
        // best-so-far value is strong.
        float parent_q = (node->visits > 0) ? node->value_sum / node->visits : 0.0f;
        float fpu = parent_q - fpu_value_;

        for (auto& child : node->children) {
            // Modern CPUCT: Q + c * P * sqrt(N_parent) / (1 + N_child)
            float q;
            if (child->visits == 0) {
                q = fpu; // FPU for unvisited children
            } else {
                q = child->value_sum / child->visits;
            }
            // Value is stored from perspective of the player who moved.
            // For selection, we want the parent's perspective, so negate.
            float u = cpuct_ * child->prior * sqrt_parent / (1.0f + child->visits);
            float score = -q + u; // negate because value is from child's perspective

            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }

    float expand_and_evaluate(MCTSNode* node, GoBoard& board, int8_t current_player, bool debug = false) {
        node->is_expanded = true;

        // Build graph and run inference
        GraphData gd = graph_gen_.generate(board, komi_, current_player, device_);

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
        auto output = model_.forward(inputs).toTuple();

        torch::Tensor policy_t = output->elements()[0].toTensor().squeeze();   // [361]
        torch::Tensor pass_val_t = output->elements()[1].toTensor().squeeze(); // scalar
        torch::Tensor value_t = output->elements()[2].toTensor().squeeze();    // scalar

        float value = value_t.item<float>();

        // Combine board logits + pass logit, then single softmax (matching Python play.py)
        torch::Tensor full_logits = torch::cat({policy_t.unsqueeze(0), pass_val_t.unsqueeze(0).unsqueeze(0)}, 1); // [1, 362]
        torch::Tensor full_probs = torch::softmax(full_logits, 1).squeeze(0); // [362]

        auto probs_cpu = full_probs.to(torch::kCPU);
        auto probs_acc = probs_cpu.accessor<float, 1>();

        float pass_prior = probs_acc[361];

        if (debug) {
            std::cout << "  [sim0] NN value: " << value
                      << " (" << (current_player == BLACK ? "Black" : "White") << "'s pov)\n";
            std::cout << "  [sim0] pass prior (softmax over 362): " << pass_prior << "\n";
        }

        // Get legal moves and assign priors
        node->legal_moves = get_legal_moves(board, current_player);
        float total_prior = 0.0f;

        for (int move : node->legal_moves) {
            auto child = std::make_unique<MCTSNode>();
            child->parent = node;
            child->move = move;
            child->color = current_player;
            child->visits = 0;
            child->value_sum = 0.0f;

            if (move == -1) {
                child->prior = pass_prior;
            } else {
                // Convert padded index to 19x19 flat index
                int r = move / GoBoard::PADDED_SIZE - 1;
                int c = move % GoBoard::PADDED_SIZE - 1;
                int flat = r * 19 + c;
                child->prior = probs_acc[flat];
            }
            total_prior += child->prior;
            node->children.push_back(std::move(child));
        }

        // Normalize priors
        if (total_prior > 0.0f) {
            for (auto& child : node->children) {
                child->prior /= total_prior;
            }
        }

        if (debug) {
            std::vector<std::pair<float, int>> sorted;
            for (auto& child : node->children) sorted.push_back({child->prior, child->move});
            std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first > b.first; });
            std::cout << "  [sim0] Top priors after normalize:\n";
            for (int i = 0; i < std::min(10, (int)sorted.size()); ++i) {
                auto [p, pos] = sorted[i];
                std::string name = (pos == -1) ? "pass" : [&]() {
                    int rr = pos / GoBoard::PADDED_SIZE - 1;
                    int cc = pos % GoBoard::PADDED_SIZE - 1;
                    char col = (cc >= 8) ? ('A' + cc + 1) : ('A' + cc);
                    return std::string(1, col) + std::to_string(rr + 1);
                }();
                std::cout << "    " << name << " = " << p << "\n";
            }
            std::cout << "  [sim0] total_prior before norm: " << total_prior << "\n";
            std::cout << "  [sim0] children count: " << node->children.size() << "\n";
        }

        return value;
    }

    void backup(MCTSNode* node, float value, int root_player) {
        // value is from the perspective of the player to move at the leaf node.
        // We need to propagate it up, alternating perspective at each level.
        MCTSNode* curr = node;
        float v = value;
        while (curr != nullptr) {
            curr->visits++;
            curr->value_sum += v;
            v = -v; // alternate perspective for parent
            curr = curr->parent;
        }
    }

    bool is_game_over(GoBoard& board, int8_t current_player) {
        // Two consecutive passes end the game
        // For simplicity in search, check if no legal moves exist beyond pass
        return false; // Let MCTS handle game-over via consecutive passes in a real impl
    }

    float get_terminal_value(GoBoard& board, int8_t current_player) {
        float score = board.calculate_area_score();
        if (score > 0) return (current_player == BLACK) ? 1.0f : -1.0f;
        if (score < 0) return (current_player == WHITE) ? 1.0f : -1.0f;
        return 0.0f;
    }

};

#endif
