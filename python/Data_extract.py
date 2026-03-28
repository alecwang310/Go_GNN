import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from scipy.ndimage import label

class KataGoData:
    def __init__(self, board_size=19):
        self.size = board_size
        self.num_intersections = board_size**2
        
        # Precompute as numpy for faster worker indexing
        edges = []
        for i in range(self.num_intersections):
            x, y = i // self.size, i % self.size
            for dx, dy in [(0, 1), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    j = nx * self.size + ny
                    edges.append([i, j])
                    edges.append([j, i])
        
        self.adj_index_np = np.array(edges).T # [2, E]
        self.adj_index_torch = torch.from_numpy(self.adj_index_np).long()

        # Precompute positional encodings as a single block
        x = np.linspace(0, 1, 19)
        dist_x = np.minimum(np.arange(19), 18 - np.arange(19)) / 9.0
        xv, yv = np.meshgrid(x, x, indexing='ij')
        dv_x, dv_y = np.meshgrid(dist_x, dist_x, indexing='ij')
        self.pos_enc = np.stack([xv, yv, dv_x, dv_y]).reshape(4, -1).astype(np.float32)

    def _build_single_graph(self, feat, policy, value, ownership, global_vars):
        data = HeteroData()
        
        # 1. Stones (Vectorized feature prep)
        # We only keep these dimensions, they are as follows:
        # 0: always 1.0f for a board position
        # {1, 2}: {Current, opponent}  player stones, 1.0f or 0.0f
        # {3, 4, 5}: Stones that have {1, 2 ,3} liberties, 1.0f or 0.0f
        # 6: Positions that cannot have stones placed in the next turn due to ko, 1.0f or 0.0f
        # {9, 10, 11, 12, 13}: History for the past 5 moves (if they exist). Only the newly placed stones are 
        # 1.0f, others are 0.0f
        # 
        keep_idx = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 18, 19]
        raw_stone_feats = feat[keep_idx, :, :].reshape(len(keep_idx), -1)
        # Use np.concatenate once, then one torch conversion
        full_feats = np.concatenate([raw_stone_feats, self.pos_enc], axis=0).T
        data['stone'].x = torch.from_numpy(full_feats).float()
        data['stone', 'adjacent', 'stone'].edge_index = self.adj_index_torch

        # 2. Global Node
        # The global node data are as follows:
        # {0, 1, 2, 3, 4}: from the latest to oldest, if player passed, 1.0f, if player didn't, 0.0f
        # 5: selfKomi/20.0f
        # {6, 7}: {0.0f, 0.0f}, When using simple Ko, the inputs are 0, 0
        # 8: 0.0f suicide is illegal
        # 9: 0.0f area scoring
        # {10, 11}: {0.0f, 0.0f} no group tax
        # {12, 13}: {0.0f, 0.0f} encore phase
        # 14: If pass will end game, 1.0f, else, 0.0f
        # {15, 16, 17, 18}: all zero, about handicap, button rules, and the last one about encore..
        
        data['global_node'].x = torch.from_numpy(global_vars).float().view(1, -1)

        # 3. Find Strings (Vectorized adjacency and features)
        stones_mask = (feat[1] + feat[2]) > 0
        labels, num_strings = label(stones_mask)
        flat_labels = labels.flatten()

        if num_strings > 0:
            # Vectorized String-to-Stone relations
            stone_indices = np.where(flat_labels > 0)[0]
            string_assignments = flat_labels[stone_indices] - 1
            bt_tensor = torch.stack([
                torch.from_numpy(stone_indices), 
                torch.from_numpy(string_assignments)
            ]).long()
            
            data['stone', 'belongs_to', 'string'].edge_index = bt_tensor
            data['string', 'contains', 'stone'].edge_index = bt_tensor.flip(0)

            # Vectorized String Features (Size and Color)
            # We use bincount to get string sizes without a loop
            counts = np.bincount(flat_labels)[1:] 
            # Get color: check the first stone of each string
            # We find the first index of each label
            _, first_indices = np.unique(flat_labels, return_index=True)
            # unique returns 0 (background) first, so skip it
            first_indices = first_indices[1:]
            is_player = feat[1].flatten()[first_indices]
            
            data['string'].x = torch.stack([
                torch.from_numpy(counts).float(),
                torch.from_numpy(is_player).float()
            ], dim=1)

            # VECTORIZED String-to-String Adjacency
            # Map board edges to their respective string labels
            src_labels = flat_labels[self.adj_index_np[0]]
            dst_labels = flat_labels[self.adj_index_np[1]]
            
            # Filter: Both must be strings, and different strings
            mask = (src_labels > 0) & (dst_labels > 0) & (src_labels != dst_labels)
            if np.any(mask):
                s_adj = np.stack([src_labels[mask]-1, dst_labels[mask]-1])
                # Unique edges only to keep the graph sparse
                s_adj = np.unique(s_adj, axis=1)
                data['string', 'adjacent_to', 'string'].edge_index = torch.from_numpy(s_adj).long()

            # Global Connections
            s_idx = torch.arange(num_strings, dtype=torch.long)
            g_idx = torch.zeros(num_strings, dtype=torch.long)
            data['string', 'reports_to', 'global_node'].edge_index = torch.stack([s_idx, g_idx])
            data['global_node', 'influences', 'string'].edge_index = torch.stack([g_idx, s_idx])
        else:
            # Set empty defaults to prevent PyG loader errors
            data['string'].x = torch.zeros((0, 2))

        # 4. Targets
        data.y_policy = torch.from_numpy(policy).float()
        data.y_value = torch.as_tensor(value).float()
        data.y_ownership = torch.from_numpy(ownership).float()

        return data

    def process_npz(self, raw):
        # 1. Unpack Spatial Bits (N, 22, 46) -> (N, 22, 19, 19)
        packed = raw['binaryInputNCHWPacked']
        spatial = np.unpackbits(packed, axis=-1)[:, :, :self.num_intersections]
        spatial = spatial.reshape(-1, 22, self.size, self.size).astype(np.float32)

        # 2. Extract Targets
        # Policy: (N, 2, 362) -> Index 0 is current move, 362 includes 'pass'
        policy_targets = raw['policyTargetsNCMove'][:, 0, :] 
        # Value: (N, 64) -> Index 0 is White Win Prob (perspective of player to move)
        wins = raw['globalTargetsNC'][:, 16]
        losses = raw['globalTargetsNC'][:, 17]
        value_targets = wins - losses
        # Ownership: (N, 5, 19, 19) -> Index 0 is final ownership [-1, 1]
        ownership_targets = raw['valueTargetsNCHW'][:, 0, :, :].reshape(-1, self.num_intersections)
        # Global variables for the global node
        global_values = raw['globalInputNC'][:, :]

        dataset = []
        for i in range(spatial.shape[0]):
            dataset.append(self._build_single_graph(
                spatial[i], policy_targets[i], value_targets[i], ownership_targets[i], global_values[i]
            ))
        return dataset
    

"""Provided in this directory are bulk-downloadable archives of the rating games, training games, and training data for "kata1", the public run of KataGo hosted at  https://katagotraining.org/

See DISCLAIMER.txt for legal disclaimers regarding this data.

Training games are provided in SGF format, and on each move is labeled: the MCTS-estimated white win probability, black win probability, "no result" probability (see "no result" at https://lightvector.github.io/KataGo/rules.html), expected final score in points, visits, and weight. These SGFs are mainly for human viewing, and only contain a tiny fraction of the actual set of targets and information about each turn of a game. For the full details, see the training data.

Rating games are provided in SGF format, and labeled similarly. For rating games, there is no additional detailed data, the game record and game result between the two players are the data.

Training data is provided in NPZ format (i.e. numpy zipped tensors). Each row is a dictionary of a few fields.

Training data input fields:
The code that generated these two tensors is at: https://github.com/lightvector/KataGo/blob/v1.12.3/cpp/neuralnet/nninputs.cpp#L2145
Please refer to the source code for details about the exact way different channels are computed.

"binaryInputNCHWPacked" - spatial inputs to the model in NCHW format (H=19,W=19), except that HW have been flattened and bitwise packed, from 19x19=361 bits, padded to 368 bits, and written as 46 bytes, using numpy.packbits(). For the channels C, these are 22 different feature planes containing the stones, liberties, ladder info, etc. For details, refer to the source code linked above.

"globalInputNC" - Global (i.e. non-spatial) inputs to the model in NC format, indicating some history information, the rules, komi, etc. There are 19 global features.

Training data outcome and training targets and metadata fields:

For these fields, please refer to the source code and the comments within the source code here:
https://github.com/lightvector/KataGo/blob/v1.11.0/cpp/dataio/trainingwrite.h#L134

"policyTargetsNCMove" - Various policy targets (including auxiliary targets besides the main policy prediction) in NC<Move> format, where <Move> is HW flattened from 19x19 into 361 and then extended to 362 where the last index corresponds to passing.

"globalTargetsNC", - Various non-spatial targets including game outcome, score, various exponential-moving-averaged "short-term" versions of these. Also includes the weighting of rows, a hash identifier, and other per-row metadata.

"scoreDistrN" - The score of the game, expressed as a large tensor N<outcome> where outcome has 1 index for every possible score in a large range, and has 0s everywhere except has a 100 on the index corresponding to the final score, or two adjacent entries that sum to 100 if the score was in between those. Indices correspond to half-point outcomes (e.g. ...,-2.5,-1.5,-0.5,0.5,1.5,2.5,...) and draws will generally have two adjacent entries that sum to 100 based on the utility of a draw (i.e. how many fractional wins a draw counts as for the player).

"valueTargetsNCHW" - Various spatial targets that involve predicting something on every square of the board, such as final ownership, and future stone positions.

"""