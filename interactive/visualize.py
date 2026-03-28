import matplotlib.pyplot as plt
import numpy as np

def inspect_coordinate(feat, row=0, col=0):
    """
    feat: The [22, 19, 19] numpy array for a single state.
    row, col: The 0-indexed integer coordinates (0,0 is A19).
    """
    # Channel descriptions based on KataGo's source / your KataGoData class
    channel_labels = [
        "0: Stone Present (Any)",
        "1: Current Player Stone",
        "2: Opponent Player Stone",
        "3: 1 Liberty",
        "4: 2 Liberties",
        "5: 3 Liberties",
        "6: Ko Illegal Move",
        "7: [Unused/Raw] Move History", 
        "8: [Unused/Raw] Move History",
        "9: History (1 move ago)",
        "10: History (2 moves ago)",
        "11: History (3 moves ago)",
        "12: History (4 moves ago)",
        "13: History (5 moves ago)",
        "14: [Unused/Raw] Ladder",
        "15: [Unused/Raw] Ladder",
        "16: [Unused/Raw] Ladder",
        "17: [Unused/Raw] Ladder",
        "18: Extra Feature 1",
        "19: Extra Feature 2",
        "20: [Unused/Raw]",
        "21: [Unused/Raw]"
    ]

    coord_name = f"{chr(ord('A') + col + (1 if col >= 8 else 0))}{19 - row}"
    print(f"--- Feature Vector for {coord_name} (Row: {row}, Col: {col}) ---")
    
    for i in range(22):
        val = feat[i, row, col]
        label = channel_labels[i] if i < len(channel_labels) else "Unknown"
        # Bolding active features for visibility
        prefix = ">> " if val > 0 else "   "
        print(f"{prefix}[{i:2d}] {label:25s}: {val:.1f}")

def visualize_board(raw, n, target_move=None, model_move=None, title="Go Board State"):

    packed_nth = raw['binaryInputNCHWPacked'][n]

    spatial_bits = np.unpackbits(packed_nth, axis=-1, bitorder='big')
    spatial_test = spatial_bits[:, :361].reshape(-1, 22, 19, 19).astype(np.float32)
    inspect_coordinate(spatial_test[0], row=11, col=9)

    feat = spatial_bits[:, :361].reshape(22, 19, 19)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Draw the Grid
    ax.set_facecolor('#DCB35C')  # Traditional wood color
    for i in range(19):
        ax.plot([i, i], [0, 18], color='black', linewidth=1, zorder=1)
        ax.plot([0, 18], [i, i], color='black', linewidth=1, zorder=1)
    
    # 2. Draw Star Points (Hoshi)
    star_points = [3, 9, 15]
    for r in star_points:
        for c in star_points:
            ax.scatter(c, 18-r, color='black', s=50, zorder=2)

    # 3. Plot Stones
    # feat[1] is Current Player, feat[2] is Opponent
    # We need to know who is who. Let's assume for visualization:
    # Black = 1, White = -1 (This depends on the 'player' global var, 
    # but usually feat[1] are stones for the player whose turn it is)
    
    # Let's just draw them as Player (Blue-ish) and Opponent (Red-ish) 
    # or Black/White if you prefer.
    player_stones = np.argwhere(feat[1] > 0)
    opp_stones = np.argwhere(feat[2] > 0)
    
    for r, c in player_stones:
        ax.scatter(c, 18-r, s=300, edgecolors='black', color='black', zorder=3)
    for r, c in opp_stones:
        ax.scatter(c, 18-r, s=300, edgecolors='black', color='white', zorder=3)

    # 4. Highlight Special Points
    # Ko point (feat[6])
    ko_point = np.argwhere(feat[6] > 0)
    for r, c in ko_point:
        ax.scatter(c, 18-r, marker='x', color='red', s=100, zorder=4)

    # 5. Mark Moves
    def mark_move(idx, color, label, marker):
        if idx is not None and idx < 361:
            r, c = idx // 19, idx % 19
            ax.scatter(c, 18-r, marker=marker, s=150, edgecolors=color, 
                       facecolors='none', linewidths=3, label=label, zorder=5)

    mark_move(target_move, 'green', 'KataGo Target', 's')
    mark_move(model_move, 'cyan', 'Model Prediction', 'o')

    # Formatting
    ax.set_xticks(range(19))
    ax.set_xticklabels([chr(ord('A') + i + (1 if i >= 8 else 0)) for i in range(19)])
    ax.set_yticks(range(19))
    ax.set_yticklabels(range(1, 20))
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = np.load("sample_3.npz")
    visualize_board(data, 40)