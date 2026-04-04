#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <torch/torch.h>
#include <vector>
#include <array>
#include <bitset>
#include <algorithm>

#include "goboard.hpp" 

struct GraphData {
    torch::Tensor stone_x, string_x, global_x;
    torch::Tensor e_s_a_s, e_s_b_str, e_str_c_s;
    torch::Tensor e_str_a_str, e_str_r_g, e_g_i_str;
};

class FastGoGraphGenerator {
private:
    static const int N = 361; 
    static const int MAX_EDGES = 1368; 
    static const int MAX_STRINGS = N;

    int64_t static_s_s_edges[2][MAX_EDGES];
    int s_s_edge_count = 0;
    float pos_enc[N][4];

    // DSU structures
    int parent[N];
    int group_size[N];

    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }

    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            if (group_size[root_i] < group_size[root_j]) std::swap(root_i, root_j);
            parent[root_j] = root_i;
            group_size[root_i] += group_size[root_j];
        }
    }

public:
    FastGoGraphGenerator() {
        for (int r = 0; r < 19; ++r) {
            for (int c = 0; c < 19; ++c) {
                int i = r * 19 + c;
                // Precompute Static Grid
                if (r + 1 < 19) {
                    int j = (r + 1) * 19 + c;
                    static_s_s_edges[0][s_s_edge_count] = i; static_s_s_edges[1][s_s_edge_count] = j; s_s_edge_count++;
                    static_s_s_edges[0][s_s_edge_count] = j; static_s_s_edges[1][s_s_edge_count] = i; s_s_edge_count++;
                }
                if (c + 1 < 19) {
                    int j = r * 19 + (c + 1);
                    static_s_s_edges[0][s_s_edge_count] = i; static_s_s_edges[1][s_s_edge_count] = j; s_s_edge_count++;
                    static_s_s_edges[0][s_s_edge_count] = j; static_s_s_edges[1][s_s_edge_count] = i; s_s_edge_count++;
                }
                // Precompute Positional Encodings
                pos_enc[i][0] = c / 18.0f; 
                pos_enc[i][1] = r / 18.0f;
                pos_enc[i][2] = std::min(c, 18 - c) / 9.0f;
                pos_enc[i][3] = std::min(r, 18 - r) / 9.0f;
            }
        }
    }

    GraphData generate(const GoBoard& board,
                       float komi, int8_t current_player, torch::Device device) {
        // 1. DSU Grouping
        for (int i = 0; i < N; ++i) { parent[i] = i; group_size[i] = 1; }
        for (int i = 0; i < s_s_edge_count; i += 2) {
            int u = static_s_s_edges[0][i], v = static_s_s_edges[1][i];
            int8_t c_u = board.board[board.get_index(u/19, u%19)];
            int8_t c_v = board.board[board.get_index(v/19, v%19)];
            if (c_u != EMPTY && c_u != BORDER && c_u == c_v) unite(u, v);
        }

        // 2. Liberty Counting (Bitset-based for speed)
        std::array<std::bitset<N>, MAX_STRINGS> string_libs;
        int root_to_id[N]; std::fill(root_to_id, root_to_id + N, -1);
        int string_count = 0;

        for (int i = 0; i < N; ++i) {
            int8_t color = board.board[board.get_index(i/19, i%19)];
            if (color != EMPTY && color != BORDER) {
                int root = find(i);
                if (root_to_id[root] == -1) root_to_id[root] = string_count++;
                int sid = root_to_id[root];

                // Check 4 neighbors for liberties
                int r = i / 19, c = i % 19;
                int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};
                for(int d=0; d<4; ++d) {
                    int nr = r + dr[d], nc = c + dc[d];
                    if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19) {
                        if (board.board[board.get_index(nr, nc)] == EMPTY) {
                            string_libs[sid].set(nr * 19 + nc);
                        }
                    }
                }
            }
        }

        // 3. Fill Stone Features [N, 18]
        // Model expects (matching Python keep_idx + pos_enc):
        //   0: constant, 1: current player, 2: opponent
        //   3: lib==1, 4: lib==2, 5: lib==3
        //   6: ko
        //   7: raw[9] (history oldest), 8: raw[10], 9: raw[11], 10: raw[12], 11: raw[13] (history most recent)
        //   12: raw[18] (current player stones on board), 13: raw[19] (opponent stones on board)
        //   14-17: positional encoding
        auto stone_x = torch::zeros({N, 18}, torch::kFloat32);
        auto s_acc = stone_x.accessor<float, 2>();
        for (int i = 0; i < N; ++i) {
            int8_t color = board.board[board.get_index(i/19, i%19)];
            int root = (color != EMPTY && color != BORDER) ? find(i) : -1;
            int sid = (root != -1) ? root_to_id[root] : -1;

            s_acc[i][0] = 1.0f; // Constant
            s_acc[i][1] = (color == current_player) ? 1.0f : 0.0f; // Current player
            s_acc[i][2] = (color != EMPTY && color != BORDER && color != current_player) ? 1.0f : 0.0f; // Opponent

            if (sid != -1) {
                size_t libs = string_libs[sid].count();
                if (libs == 1) s_acc[i][3] = 1.0f;
                if (libs == 2) s_acc[i][4] = 1.0f;
                if (libs == 3) s_acc[i][5] = 1.0f;
            }

            if (board.last_ko_pos != -1 && i == board.last_ko_pos) s_acc[i][6] = 1.0f;

            // History (Channels 7-11), oldest to most recent: raw[9]=oldest .. raw[13]=most recent
            // board.history_moves[0] = most recent, [4] = oldest
            for(int h=0; h<5; ++h) {
                int raw_idx = 4 - h; // [4]=oldest→ch7, [3]→ch8, [2]→ch9, [1]→ch10, [0]=most recent→ch11
                if (board.history_moves[raw_idx] != -1 && i == board.history_moves[raw_idx])
                    s_acc[i][7+h] = 1.0f;
            }

            // Channels 12-13: stones on board (current player / opponent)
            if (color == current_player) s_acc[i][12] = 1.0f;
            else if (color != EMPTY && color != BORDER) s_acc[i][13] = 1.0f;

            // Positional encoding (channels 14-17)
            for(int p=0; p<4; ++p) s_acc[i][14+p] = pos_enc[i][p];
        }

        // 4. Fill String Features [S, 2]
        auto string_x = torch::zeros({std::max(1, string_count), 2}, torch::kFloat32);
        auto str_acc = string_x.accessor<float, 2>();
        for (int i = 0; i < N; ++i) {
            int root = find(i);
            if (root_to_id[root] != -1) {
                int sid = root_to_id[root];
                str_acc[sid][0] = (float)group_size[root];
                int8_t color = board.board[board.get_index(i/19, i%19)];
                str_acc[sid][1] = (color == current_player) ? 1.0f : 0.0f;
            }
        }

        // 5. Fill Global Features [1, 19]
        auto global_x = torch::zeros({1, 19}, torch::kFloat32);
        auto g_acc = global_x.accessor<float, 2>();
        for(int i=0; i<5; ++i) g_acc[0][i] = board.history_passes[i] ? 1.0f : 0.0f; // History of passes
        g_acc[0][5] = komi / 20.0f; 
        g_acc[0][9] = 1.0f; // Area scoring
        // Indices 6, 7, 8, 10-18 are 0.0f for simple rules

        // 6. Build Edge Tensors
        std::vector<int64_t> s_b_str_u, s_b_str_v, str_str_u, str_str_v;
        std::bitset<MAX_STRINGS * MAX_STRINGS> str_adj_matrix;

        for (int i = 0; i < N; ++i) {
            int sid = (board.board[board.get_index(i/19, i%19)] != EMPTY) ? root_to_id[find(i)] : -1;
            if (sid != -1) { s_b_str_u.push_back(i); s_b_str_v.push_back(sid); }
        }

        for (int i = 0; i < s_s_edge_count; i += 2) {
            int u = static_s_s_edges[0][i], v = static_s_s_edges[1][i];
            int su = root_to_id[find(u)], sv = root_to_id[find(v)];
            if (su != -1 && sv != -1 && su != sv) {
                if (!str_adj_matrix.test(su * MAX_STRINGS + sv)) {
                    str_adj_matrix.set(su * MAX_STRINGS + sv); str_adj_matrix.set(sv * MAX_STRINGS + su);
                    str_str_u.push_back(su); str_str_v.push_back(sv);
                    str_str_u.push_back(sv); str_str_v.push_back(su);
                }
            }
        }

        auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
        GraphData gd;
        gd.stone_x = stone_x.to(device);
        gd.string_x = string_x.to(device);
        gd.global_x = global_x.to(device);
        gd.e_s_a_s = torch::from_blob((void*)static_s_s_edges, {2, s_s_edge_count}, torch::kInt64).to(device).clone();
        gd.e_s_b_str = torch::stack({torch::tensor(s_b_str_u, long_opts), torch::tensor(s_b_str_v, long_opts)});
        gd.e_str_c_s = gd.e_s_b_str.flip(0);
        gd.e_str_a_str = str_str_u.empty() ? torch::zeros({2, 0}, long_opts) : 
                         torch::stack({torch::tensor(str_str_u, long_opts), torch::tensor(str_str_v, long_opts)});
        
        std::vector<int64_t> g_u, g_v;
        for(int i=0; i<string_count; ++i) { g_u.push_back(i); g_v.push_back(0); }
        gd.e_str_r_g = torch::stack({torch::tensor(g_u, long_opts), torch::tensor(g_v, long_opts)});
        gd.e_g_i_str = gd.e_str_r_g.flip(0);

        return gd;
    }
};

#endif