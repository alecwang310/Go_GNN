#ifndef BOARD_HPP
#define BOARD_HPP

#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>

enum Color : int8_t { EMPTY = 0, BLACK = 1, WHITE = 2, BORDER = 3 };

class GoBoard {
public:
    static const int SIZE = 19;
    static const int PADDED_SIZE = SIZE + 2; // 21x21
    static const int BOARD_ARRAY_SIZE = PADDED_SIZE * PADDED_SIZE;

    int8_t board[BOARD_ARRAY_SIZE];
    int last_ko_pos = -1;
    float komi = 7.5f;

    // Move history: [0] = most recent, [4] = oldest
    int history_moves[5] = {-1, -1, -1, -1, -1};
    bool history_passes[5] = {false, false, false, false, false};

    // Neighbors: Up, Down, Left, Right
    const int adj[4] = { -PADDED_SIZE, PADDED_SIZE, -1, 1 };

    GoBoard& operator=(const GoBoard& other) {
        if (this != &other) {
            std::copy(other.board, other.board + BOARD_ARRAY_SIZE, board);
            last_ko_pos = other.last_ko_pos;
            komi = other.komi;
            std::copy(other.history_moves, other.history_moves + 5, history_moves);
            std::copy(other.history_passes, other.history_passes + 5, history_passes);
        }
        return *this;
    }

    GoBoard& operator=(GoBoard&& other) noexcept {
        if (this != &other) {
            std::copy(other.board, other.board + BOARD_ARRAY_SIZE, board);
            last_ko_pos = other.last_ko_pos;
            komi = other.komi;
            std::copy(other.history_moves, other.history_moves + 5, history_moves);
            std::copy(other.history_passes, other.history_passes + 5, history_passes);
        }
        return *this;
    }

    GoBoard(const GoBoard& other) {
        std::copy(other.board, other.board + BOARD_ARRAY_SIZE, board);
        last_ko_pos = other.last_ko_pos;
        komi = other.komi;
        std::copy(other.history_moves, other.history_moves + 5, history_moves);
        std::copy(other.history_passes, other.history_passes + 5, history_passes);
    }

    GoBoard() {
        reset();
    }

    void reset() {
        for (int i = 0; i < BOARD_ARRAY_SIZE; ++i) {
            int r = i / PADDED_SIZE;
            int c = i % PADDED_SIZE;
            if (r == 0 || r == PADDED_SIZE - 1 || c == 0 || c == PADDED_SIZE - 1)
                board[i] = BORDER;
            else
                board[i] = EMPTY;
        }
        last_ko_pos = -1;
        for (int i = 0; i < 5; ++i) {
            history_moves[i] = -1;
            history_passes[i] = false;
        }
    }

    // Convert (row, col) to 1D index
    inline int get_index(int r, int c) const { return (r + 1) * PADDED_SIZE + (c + 1); }

    bool is_legal(int pos, int8_t color) {
        if (pos == -1) return true; // Pass is always legal
        if (board[pos] != EMPTY || pos == last_ko_pos) return false;

        // Temporarily place stone to check suicide
        board[pos] = color;
        bool has_libs = has_liberties(pos);
        
        // If no liberties, check if it captures anything (which makes it legal)
        if (!has_libs) {
            int8_t opponent = 3 - color;
            for (int i = 0; i < 4; ++i) {
                int nb = pos + adj[i];
                if (board[nb] == opponent && !has_liberties(nb)) {
                    has_libs = true; // Captures opponent, not suicide
                    break;
                }
            }
        }

        board[pos] = EMPTY; // Cleanup
        return has_libs;
    }

    bool play_move(int pos, int8_t color) {
        if (pos == -1) { // Pass
            last_ko_pos = -1;
            update_history(pos);
            return true;
        }
        if (!is_legal(pos, color)) return false;

        board[pos] = color;
        int8_t opponent = 3 - color;
        int captured_count = 0;
        int last_captured_pos = -1;

        // Check for captures in 4 directions
        for (int i = 0; i < 4; ++i) {
            int nb = pos + adj[i];
            if (board[nb] == opponent && !has_liberties(nb)) {
                int count = remove_group(nb);
                captured_count += count;
                if (count == 1) last_captured_pos = nb;
            }
        }

        // Simple Ko Rule: If exactly 1 stone was captured, and the playing
        // stone now has only 1 liberty, that spot is the new Ko.
        if (captured_count == 1 && count_liberties(pos) == 1) {
            last_ko_pos = last_captured_pos;
        } else {
            last_ko_pos = -1;
        }

        update_history(pos);
        return true;
    }

    float calculate_area_score() {
        float score = -komi;
        std::vector<bool> visited(BOARD_ARRAY_SIZE, false);

        for (int i = 0; i < BOARD_ARRAY_SIZE; ++i) {
            if (board[i] == BORDER || visited[i]) continue;
            if (board[i] == BLACK) score += 1.0f;
            else if (board[i] == WHITE) score -= 1.0f;
            else {
                // Empty area: use flood fill to see who owns it
                int owner = 0; // 0: neutral, 1: black, 2: white, 3: both
                int area_size = 0;
                check_area(i, visited, owner, area_size);
                if (owner == BLACK) score += (float)area_size;
                if (owner == WHITE) score -= (float)area_size;
            }
        }
        return score;
    }

private:
    void update_history(int pos) {
        for (int i = 4; i > 0; --i) {
            history_moves[i] = history_moves[i - 1];
            history_passes[i] = history_passes[i - 1];
        }
        history_moves[0] = pos;
        history_passes[0] = (pos == -1);
    }

    bool has_liberties(int pos) {
        std::vector<int> stack = { pos };
        std::vector<bool> seen(BOARD_ARRAY_SIZE, false);
        seen[pos] = true;
        int8_t color = board[pos];

        while (!stack.empty()) {
            int curr = stack.back();
            stack.pop_back();
            for (int i = 0; i < 4; ++i) {
                int nb = curr + adj[i];
                if (board[nb] == EMPTY) return true;
                if (board[nb] == color && !seen[nb]) {
                    seen[nb] = true;
                    stack.push_back(nb);
                }
            }
        }
        return false;
    }

    int count_liberties(int pos) {
        int libs = 0;
        for (int i = 0; i < 4; ++i) if (board[pos + adj[i]] == EMPTY) libs++;
        return libs;
    }

    int remove_group(int pos) {
        int8_t color = board[pos];
        std::vector<int> stack = { pos };
        int count = 0;
        board[pos] = EMPTY;
        count++;

        while (!stack.empty()) {
            int curr = stack.back();
            stack.pop_back();
            for (int i = 0; i < 4; ++i) {
                int nb = curr + adj[i];
                if (board[nb] == color) {
                    board[nb] = EMPTY;
                    count++;
                    stack.push_back(nb);
                }
            }
        }
        return count;
    }

    void check_area(int pos, std::vector<bool>& global_visited, int& owner, int& size) {
        std::vector<int> q = { pos };
        std::vector<bool> local_visited(BOARD_ARRAY_SIZE, false);
        local_visited[pos] = true;
        global_visited[pos] = true;

        while (!q.empty()) {
            int curr = q.back();
            q.pop_back();
            size++;
            for (int i = 0; i < 4; ++i) {
                int nb = curr + adj[i];
                if (board[nb] == BORDER) continue;
                if (board[nb] == BLACK) owner |= BLACK;
                else if (board[nb] == WHITE) owner |= WHITE;
                else if (!local_visited[nb]) {
                    local_visited[nb] = true;
                    global_visited[nb] = true;
                    q.push_back(nb);
                }
            }
        }
    }
};

#endif