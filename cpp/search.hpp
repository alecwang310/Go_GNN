#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <map>
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

    // Virtual loss (multithreading)
    int virtual_visits = 0;
    float virtual_value_sum = 0.0f;

    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> legal_moves;

    float mean_value() const {
        if (visits == 0) return 0.0f;
        return value_sum / visits;
    }
};

// ─── Inference batch queue ───────────────────────────────────────────────

struct InferenceRequest {
    GraphData graph_data;
    MCTSNode* node;
    GoBoard board;
    int8_t current_player;
    uint64_t request_id = 0;
};

struct InferenceResult {
    MCTSNode* node;
    torch::Tensor policy_logits; // [361]
    torch::Tensor pass_logit;    // scalar
    float value;                 // scalar
    GoBoard board;
    int8_t current_player;
    uint64_t request_id = 0;
};

class InferenceBatchQueue {
public:
    InferenceBatchQueue(torch::jit::script::Module& model, torch::Device device,
                        int batch_size, int timeout_ms = 5)
        : model_(model), device_(device), batch_size_(batch_size), timeout_ms_(timeout_ms) {}

    void start() {
        stopped_ = false;
        inference_thread_ = std::thread([this]() { inference_loop(); });
    }

    InferenceResult submit(InferenceRequest&& req) {
        uint64_t key;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            key = next_id_++;
            req.request_id = key;
            pending_requests_.push_back(std::move(req));
        }
        queue_cv_.notify_one();

        std::unique_lock<std::mutex> lock(results_mutex_);
        results_cv_.wait(lock, [this, key] {
            return results_.find(key) != results_.end();
        });

        InferenceResult result = std::move(results_[key]);
        results_.erase(key);
        return result;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopped_ = true;
        }
        queue_cv_.notify_one();
        if (inference_thread_.joinable()) inference_thread_.join();
    }

private:
    torch::jit::script::Module& model_;
    torch::Device device_;
    int batch_size_;
    int timeout_ms_;

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<InferenceRequest> pending_requests_;

    std::mutex results_mutex_;
    std::condition_variable results_cv_;
    std::map<uint64_t, InferenceResult> results_;

    bool stopped_ = false;
    std::thread inference_thread_;
    uint64_t next_id_ = 0;

    void inference_loop() {
        try {
        while (true) {
            std::vector<InferenceRequest> batch;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                // Wait until at least one request arrives or we're stopped
                queue_cv_.wait(lock, [this] {
                    return !pending_requests_.empty() || stopped_;
                });

                if (stopped_ && pending_requests_.empty()) break;

                // Give other workers a moment to submit their requests
                // (unlock during sleep so workers can acquire the mutex)
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                lock.lock();

                // Now drain the full queue
                while (!pending_requests_.empty() && (int)batch.size() < batch_size_) {
                    batch.push_back(std::move(pending_requests_.back()));
                    pending_requests_.pop_back();
                }
            }

            if (batch.empty()) continue;

            auto batch_input = build_batch_input(batch);

            torch::NoGradGuard no_grad;
            auto batch_output = model_.forward(batch_input);

            auto results = decompose_batch_output(batch, batch_output);

            {
                std::lock_guard<std::mutex> lock(results_mutex_);
                for (auto& r : results) {
                    results_[r.request_id] = std::move(r);
                }
            }
            results_cv_.notify_all();
        }
        } catch (const std::exception& e) {
            std::cerr << "  [FATAL] inference thread exception: " << e.what() << "\n";
        } catch (...) {
            std::cerr << "  [FATAL] inference thread unknown exception\n";
        }
    }

    std::vector<torch::jit::IValue> build_batch_input(const std::vector<InferenceRequest>& requests) {
        int B = requests.size();

        // Collect per-graph metadata
        std::vector<int> string_counts(B);
        std::vector<int> s_b_str_counts(B);
        std::vector<int> str_str_counts(B);
        int total_strings = 0;
        int total_s_b_str_edges = 0;
        int total_str_str_edges = 0;

        for (int i = 0; i < B; i++) {
            auto& gd = requests[i].graph_data;
            string_counts[i] = gd.string_x.size(0);
            s_b_str_counts[i] = gd.e_s_b_str.size(1);
            str_str_counts[i] = gd.e_str_a_str.size(1);
            total_strings += string_counts[i];
            total_s_b_str_edges += s_b_str_counts[i];
            total_str_str_edges += str_str_counts[i];
        }

        auto long_opts = torch::TensorOptions().dtype(torch::kInt64);

        // 1. stone_x: [361*B, 18]
        std::vector<torch::Tensor> stone_tensors;
        stone_tensors.reserve(B);
        for (int i = 0; i < B; i++) stone_tensors.push_back(requests[i].graph_data.stone_x);
        auto batch_stone_x = torch::cat(stone_tensors, 0);

        // 2. string_x: [total_strings, 2]
        std::vector<torch::Tensor> string_tensors;
        string_tensors.reserve(B);
        for (int i = 0; i < B; i++) string_tensors.push_back(requests[i].graph_data.string_x);
        auto batch_string_x = torch::cat(string_tensors, 0);

        // 3. global_x: [B, 19]
        std::vector<torch::Tensor> global_tensors;
        global_tensors.reserve(B);
        for (int i = 0; i < B; i++) global_tensors.push_back(requests[i].graph_data.global_x);
        auto batch_global_x = torch::cat(global_tensors, 0);

        // 4. string_batch_index: [total_strings]
        auto str_batch_idx = torch::zeros({total_strings}, long_opts);
        auto sb_acc = str_batch_idx.accessor<int64_t, 1>();
        int offset = 0;
        for (int i = 0; i < B; i++) {
            for (int s = 0; s < string_counts[i]; s++) sb_acc[offset + s] = i;
            offset += string_counts[i];
        }

        // 5. e_s_a_s: [2, 1368*B] — offset stone indices by 361*i
        auto batch_e_s_a_s = torch::zeros({2, 1368 * B}, long_opts);
        auto be_acc = batch_e_s_a_s.accessor<int64_t, 2>();
        auto ref_e_s_a_s = requests[0].graph_data.e_s_a_s.cpu();
        auto ref_acc = ref_e_s_a_s.accessor<int64_t, 2>();
        for (int i = 0; i < B; i++) {
            int64_t stone_off = 361 * i;
            for (int e = 0; e < 1368; e++) {
                be_acc[0][i * 1368 + e] = ref_acc[0][e] + stone_off;
                be_acc[1][i * 1368 + e] = ref_acc[1][e] + stone_off;
            }
        }

        // 6. e_s_b_str: [2, total_s_b_str_edges]
        auto batch_e_s_b_str = torch::zeros({2, std::max(1, total_s_b_str_edges)}, long_opts);
        auto bs_acc = batch_e_s_b_str.accessor<int64_t, 2>();
        int edge_off = 0;
        int str_off = 0;
        for (int i = 0; i < B; i++) {
            int ne = s_b_str_counts[i];
            if (ne > 0) {
                auto e = requests[i].graph_data.e_s_b_str.cpu();
                auto e_acc = e.accessor<int64_t, 2>();
                int64_t stone_off = 361 * i;
                for (int j = 0; j < ne; j++) {
                    bs_acc[0][edge_off + j] = e_acc[0][j] + stone_off;
                    bs_acc[1][edge_off + j] = e_acc[1][j] + str_off;
                }
            }
            edge_off += ne;
            str_off += string_counts[i];
        }
        if (total_s_b_str_edges == 0) edge_off = 1; // keep tensor shape valid

        // 7. e_str_c_s: flip of e_s_b_str
        auto batch_e_str_c_s = batch_e_s_b_str.flip(0);

        // 8. e_str_a_str: [2, total_str_str_edges]
        auto batch_e_str_a_str = torch::zeros({2, std::max(1, total_str_str_edges)}, long_opts);
        auto bsa_acc = batch_e_str_a_str.accessor<int64_t, 2>();
        edge_off = 0;
        str_off = 0;
        for (int i = 0; i < B; i++) {
            int ne = str_str_counts[i];
            if (ne > 0) {
                auto e = requests[i].graph_data.e_str_a_str.cpu();
                auto e_acc = e.accessor<int64_t, 2>();
                for (int j = 0; j < ne; j++) {
                    bsa_acc[0][edge_off + j] = e_acc[0][j] + str_off;
                    bsa_acc[1][edge_off + j] = e_acc[1][j] + str_off;
                }
            }
            edge_off += ne;
            str_off += string_counts[i];
        }
        if (total_str_str_edges == 0) edge_off = 1;

        // 9. e_str_r_g: [2, total_strings] — each string -> its global node
        auto batch_e_str_r_g = torch::zeros({2, std::max(1, total_strings)}, long_opts);
        auto brg_acc = batch_e_str_r_g.accessor<int64_t, 2>();
        int idx = 0;
        for (int i = 0; i < B; i++) {
            for (int s = 0; s < string_counts[i]; s++) {
                brg_acc[0][idx] = idx;
                brg_acc[1][idx] = i;
                idx++;
            }
        }
        if (total_strings == 0) idx = 1;

        // 10. e_g_i_str: flip of e_str_r_g
        auto batch_e_g_i_str = batch_e_str_r_g.flip(0);

        // Move all tensors to model device
        batch_stone_x = batch_stone_x.to(device_);
        batch_string_x = batch_string_x.to(device_);
        batch_global_x = batch_global_x.to(device_);
        str_batch_idx = str_batch_idx.to(device_);
        batch_e_s_a_s = batch_e_s_a_s.to(device_);
        batch_e_s_b_str = batch_e_s_b_str.to(device_);
        batch_e_str_c_s = batch_e_str_c_s.to(device_);
        batch_e_str_a_str = batch_e_str_a_str.to(device_);
        batch_e_str_r_g = batch_e_str_r_g.to(device_);
        batch_e_g_i_str = batch_e_g_i_str.to(device_);

        return std::vector<torch::jit::IValue>{
            batch_stone_x, batch_string_x, batch_global_x, str_batch_idx,
            batch_e_s_a_s, batch_e_s_b_str, batch_e_str_c_s,
            batch_e_str_a_str, batch_e_str_r_g, batch_e_g_i_str
        };
    }

    std::vector<InferenceResult> decompose_batch_output(
            const std::vector<InferenceRequest>& requests,
            torch::jit::IValue batch_output) {
        auto output_tuple = batch_output.toTuple();
        // Model outputs flat tensors: policy [361*B], pass [B], value [B]
        auto policy_flat = output_tuple->elements()[0].toTensor().to(torch::kCPU);
        auto pass_t      = output_tuple->elements()[1].toTensor().to(torch::kCPU);
        auto value_t     = output_tuple->elements()[2].toTensor().to(torch::kCPU);

        int B = requests.size();
        // Reshape flat policy [361*B] -> [B, 361]
        auto policy_t = policy_flat.view({B, 361});

        std::vector<InferenceResult> results;
        results.reserve(B);
        for (int i = 0; i < B; i++) {
            InferenceResult r;
            r.node = requests[i].node;
            r.policy_logits = policy_t[i];            // [361]
            r.pass_logit = pass_t[i];                  // scalar
            r.value = value_t[i].item<float>();
            r.board = requests[i].board;
            r.current_player = requests[i].current_player;
            r.request_id = requests[i].request_id;
            results.push_back(std::move(r));
        }
        return results;
    }
};

// ─── MCTS ────────────────────────────────────────────────────────────────

class MCTS {
public:
    MCTS(const std::string& model_path, const std::string& device_str = "cpu",
         float komi = 7.5f, int num_simulations = 200, float cpuct = 2.0f,
         float fpu_value = 0.25f, int num_threads = 16)
        : komi_(komi), num_simulations_(num_simulations), cpuct_(cpuct),
          fpu_value_(fpu_value), num_threads_(num_threads),
          device_((device_str == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU)
    {
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
    }

    // Diagnostics populated after search()
    float last_nn_value = 0.0f;
    std::vector<std::pair<float, int>> last_nn_priors;
    std::vector<std::pair<int, int>> last_mcts_visits;

    int search(GoBoard& board, int8_t current_player) {
        MCTSNode root;
        root.color = 3 - current_player;
        root.legal_moves = get_legal_moves(board, current_player);

        if (root.legal_moves.empty()) return -1;

        nn_value_captured_ = false;

        // Start inference thread
        auto queue = std::make_unique<InferenceBatchQueue>(model_, device_, num_threads_, 5);
        queue->start();

        // Launch workers
        int sims_per_worker = num_simulations_ / num_threads_;
        int remainder = num_simulations_ % num_threads_;

        std::vector<std::thread> workers;
        for (int t = 0; t < num_threads_; t++) {
            int my_sims = sims_per_worker + (t < remainder ? 1 : 0);
            workers.emplace_back([&, my_sims]() {
                worker_thread(board, current_player, &root, my_sims, queue.get());
            });
        }

        for (auto& w : workers) w.join();
        queue->stop();

        // Pick best move
        MCTSNode* best = nullptr;
        int best_visits = -1;
        for (auto& child : root.children) {
            if (child->visits > best_visits) {
                best_visits = child->visits;
                best = child.get();
            }
        }

        // Collect MCTS visit stats
        last_mcts_visits.clear();
        for (auto& child : root.children) {
            last_mcts_visits.push_back({child->visits, child->move});
        }
        std::sort(last_mcts_visits.begin(), last_mcts_visits.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        return best ? best->move : -1;
    }

    void set_num_simulations(int n) { num_simulations_ = n; }
    int get_num_simulations() const { return num_simulations_; }
    void set_num_threads(int n) { num_threads_ = n; }
    int get_num_threads() const { return num_threads_; }

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    float komi_;
    int num_simulations_;
    float cpuct_;
    float fpu_value_;
    int num_threads_;

    std::mutex tree_mutex_;
    std::atomic<bool> nn_value_captured_{false};

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
        moves.push_back(-1);
        return moves;
    }

    MCTSNode* select_child(MCTSNode* node, int8_t current_player) {
        int eff_parent_visits = node->visits + node->virtual_visits;
        float sqrt_parent = std::sqrt((float)eff_parent_visits);
        float best_score = -std::numeric_limits<float>::infinity();
        MCTSNode* best = nullptr;

        float parent_q = (eff_parent_visits > 0)
            ? (node->value_sum + node->virtual_value_sum) / eff_parent_visits
            : 0.0f;
        float fpu = parent_q - fpu_value_;

        for (auto& child : node->children) {
            int eff_visits = child->visits + child->virtual_visits;
            float q;
            if (eff_visits == 0) {
                q = fpu;
            } else {
                q = (child->value_sum + child->virtual_value_sum) / eff_visits;
            }
            float u = cpuct_ * child->prior * sqrt_parent / (1.0f + eff_visits);
            float score = q + u;

            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }

    void expand_node_with_priors(MCTSNode* node, const torch::Tensor& policy_logits,
                                  const torch::Tensor& pass_logit,
                                  GoBoard& board, int8_t current_player) {
        node->is_expanded = true;

        auto full_logits = torch::cat({policy_logits.unsqueeze(0),
                                        pass_logit.unsqueeze(0).unsqueeze(0)}, 1); // [1, 362]
        auto full_probs = torch::softmax(full_logits, 1).squeeze(0); // [362]
        auto probs_cpu = full_probs.to(torch::kCPU);
        auto probs_acc = probs_cpu.accessor<float, 1>();

        float pass_prior = probs_acc[361];

        node->legal_moves = get_legal_moves(board, current_player);
        float total_prior = 0.0f;

        for (int move : node->legal_moves) {
            auto child = std::make_unique<MCTSNode>();
            child->parent = node;
            child->move = move;
            child->color = current_player;

            if (move == -1) {
                child->prior = pass_prior;
            } else {
                int r = move / GoBoard::PADDED_SIZE - 1;
                int c = move % GoBoard::PADDED_SIZE - 1;
                int flat = r * 19 + c;
                child->prior = probs_acc[flat];
            }
            total_prior += child->prior;
            node->children.push_back(std::move(child));
        }

        if (total_prior > 0.0f) {
            for (auto& child : node->children) {
                child->prior /= total_prior;
            }
        }
    }

    void backup(MCTSNode* node, float value, int root_player) {
        MCTSNode* curr = node;
        float v = value;
        while (curr != nullptr) {
            curr->visits++;
            curr->value_sum += v;
            v = -v;
            curr = curr->parent;
        }
    }

    void apply_virtual_loss(std::vector<MCTSNode*>& path) {
        for (auto* n : path) {
            n->virtual_visits++;
            n->virtual_value_sum -= 1.0f;
        }
    }

    void remove_virtual_loss(std::vector<MCTSNode*>& path) {
        for (auto* n : path) {
            n->virtual_visits--;
            n->virtual_value_sum += 1.0f;
        }
    }

    void worker_thread(GoBoard root_board, int8_t root_player, MCTSNode* root,
                       int num_sims, InferenceBatchQueue* queue) {
        try {
        FastGoGraphGenerator graph_gen; // per-thread instance

        for (int sim = 0; sim < num_sims; sim++) {
            std::vector<MCTSNode*> path;
            float value;
            GoBoard sim_board;
            int8_t sim_player;
            MCTSNode* node;

            // 1. SELECT (under tree mutex)
            {
                std::lock_guard<std::mutex> lock(tree_mutex_);
                sim_board = root_board;
                node = root;
                sim_player = root_player;

                while (node->is_expanded && !node->is_terminal && !node->children.empty()) {
                    MCTSNode* best_child = select_child(node, sim_player);
                    bool ok = sim_board.play_move(best_child->move, sim_player);
                    if (!ok) {
                        node->is_terminal = true;
                        node->terminal_value = -1.0f;
                        break;
                    }
                    sim_player = 3 - sim_player;
                    node = best_child;
                    path.push_back(node);
                }

                apply_virtual_loss(path);

                if (node->is_terminal) {
                    value = node->terminal_value;
                    remove_virtual_loss(path);
                    backup(node, value, root_player);
                    continue;
                }

                if (is_game_over(sim_board, sim_player)) {
                    value = get_terminal_value(sim_board, sim_player);
                    node->is_terminal = true;
                    node->terminal_value = value;
                    remove_virtual_loss(path);
                    backup(node, value, root_player);
                    continue;
                }
            }

            // 2. GENERATE GRAPH (outside lock)
            GraphData gd = graph_gen.generate(sim_board, komi_, sim_player, device_);

            // 3. SUBMIT TO INFERENCE QUEUE
            InferenceRequest req;
            req.graph_data = std::move(gd);
            req.node = node;
            req.board = sim_board;
            req.current_player = sim_player;
            auto result = queue->submit(std::move(req));

            // 4. EXPAND + REMOVE VIRTUAL LOSS + BACKUP (under tree mutex)
            {
                std::lock_guard<std::mutex> lock(tree_mutex_);

                if (!result.node->is_expanded) {
                    expand_node_with_priors(result.node, result.policy_logits,
                                            result.pass_logit, result.board,
                                            result.current_player);

                    // Capture NN priors from root expansion (first worker to do so)
                    if (result.node == root && !nn_value_captured_.exchange(true)
                        && !root->children.empty()) {
                        last_nn_value = result.value;
                        last_nn_priors.clear();
                        for (auto& child : root->children) {
                            last_nn_priors.push_back({child->prior, child->move});
                        }
                        std::sort(last_nn_priors.begin(), last_nn_priors.end(),
                                  [](auto& a, auto& b) { return a.first > b.first; });
                    }
                }

                remove_virtual_loss(path);
                backup(result.node, result.value, root_player);
            }
        }
        } catch (const std::exception& e) {
            std::cerr << "  [FATAL] worker exception: " << e.what() << "\n";
        } catch (...) {
            std::cerr << "  [FATAL] worker unknown exception\n";
        }
    }

    bool is_game_over(GoBoard& board, int8_t current_player) {
        return false;
    }

    float get_terminal_value(GoBoard& board, int8_t current_player) {
        float score = board.calculate_area_score();
        if (score > 0) return (current_player == BLACK) ? 1.0f : -1.0f;
        if (score < 0) return (current_player == WHITE) ? 1.0f : -1.0f;
        return 0.0f;
    }
};

#endif
