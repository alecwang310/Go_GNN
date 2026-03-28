import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

class GoGNN(nn.Module):
    def __init__(self, stone_in=18, string_in=2, global_in=19,
                 stone_dim=128, hyper_dim=512, dropout=0.0, 
                 extraction_heads=4, reasoning_heads=2, reasoning_steps=14):
        super().__init__()
        
        self.reasoning_steps = reasoning_steps
        
        # 1. Embeddings
        self.stone_emb = nn.Linear(stone_in, stone_dim)
        self.string_emb = nn.Linear(string_in, hyper_dim)
        self.global_emb = nn.Linear(global_in, hyper_dim)

        def make_hetero_conv(heads):
            return HeteroConv({
                ('stone', 'adjacent', 'stone'): GATv2Conv(stone_dim, stone_dim // heads, heads=heads),
                ('stone', 'belongs_to', 'string'): GATv2Conv((stone_dim, hyper_dim), hyper_dim // heads, heads=heads, add_self_loops=False),
                ('string', 'contains', 'stone'): GATv2Conv((hyper_dim, stone_dim), stone_dim // heads, heads=heads, add_self_loops=False),
                ('string', 'adjacent_to', 'string'): GATv2Conv(hyper_dim, hyper_dim // heads, heads=heads),
                ('string', 'reports_to', 'global_node'): GATv2Conv(hyper_dim, hyper_dim // heads, heads=heads, add_self_loops=False),
                ('global_node', 'influences', 'string'): GATv2Conv(hyper_dim, hyper_dim // heads, heads=heads, add_self_loops=False),
            }, aggr='mean')
        
        def make_ffn(dim):
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout)
            )
        
        def make_gates(num_layers, base_init):
            gates = nn.ModuleDict()
            for node_type in ['stone', 'string', 'global_node']:
                gates[node_type] = nn.ParameterList([
                    nn.Parameter(torch.full((1,), base_init + 0.05 * torch.randn(1).item())) 
                    for _ in range(num_layers)
                ])
            return gates

        # ====================== EXTRACTION PHASE (UNIQUE) ======================
        self.extraction_layers = nn.ModuleList([make_hetero_conv(extraction_heads) for _ in range(8)])
        self.extraction_stone_ffns = nn.ModuleList([make_ffn(stone_dim) for _ in range(8)])
        self.extraction_hyper_ffns = nn.ModuleList([make_ffn(hyper_dim) for _ in range(8)])
        self.extraction_norms = nn.ModuleDict({
            'stone': nn.ModuleList([nn.LayerNorm(stone_dim) for _ in range(8)]),
            'hyper': nn.ModuleList([nn.LayerNorm(hyper_dim) for _ in range(8)])
        })

        self.extraction_gates = make_gates(8, 0.5)

        # ====================== REASONING PHASE (Alternating A/B) ======================
        # Block A
        self.conv_A = make_hetero_conv(reasoning_heads)
        self.ffn_stone_A = make_ffn(stone_dim)
        self.ffn_hyper_A = make_ffn(hyper_dim)
        
        # Block B
        self.conv_B = make_hetero_conv(reasoning_heads)
        self.ffn_stone_B = make_ffn(stone_dim)
        self.ffn_hyper_B = make_ffn(hyper_dim)

        # Reasoning Gates (Unique for each of the 14 steps)
        self.reason_gates = make_gates(reasoning_steps, 0.0)
        
        # Shared Norms for Reasoning (Helps stability in recurrent loops)
        self.shared_norm_stone = nn.LayerNorm(stone_dim)
        self.shared_norm_hyper = nn.LayerNorm(hyper_dim)

        # ====================== HEADS ======================
        self.policy_head = nn.Sequential(
            nn.Linear(stone_dim, stone_dim),
            nn.GELU(),
            nn.LayerNorm(stone_dim),
            nn.Linear(stone_dim, stone_dim // 2),
            nn.GELU(),
            nn.Linear(stone_dim // 2, 1)
        )
        self.own_head = nn.Sequential(
            nn.Linear(stone_dim, stone_dim),
            nn.GELU(),
            nn.Linear(stone_dim, 1),
            nn.Tanh()
        )
        self.value_head = nn.Sequential(
            nn.Linear(hyper_dim, hyper_dim),
            nn.GELU(),
            nn.LayerNorm(hyper_dim),
            nn.Linear(hyper_dim, hyper_dim // 2),
            nn.GELU(),
            nn.Linear(hyper_dim // 2, 1),
            nn.Tanh()
        )
        self.pass_head = nn.Sequential(
            nn.LayerNorm(hyper_dim),
            nn.Linear(hyper_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict['stone'] = F.gelu(self.stone_emb(x_dict['stone']))
        x_dict['string'] = F.gelu(self.string_emb(x_dict['string']))
        x_dict['global_node'] = F.gelu(self.global_emb(x_dict['global_node']))
        
        # 1. Extraction Phase
        for i in range(8):
            norm_dict = {
                'stone': self.extraction_norms['stone'][i](x_dict['stone']),
                'string': self.extraction_norms['hyper'][i](x_dict['string']),
                'global_node': self.extraction_norms['hyper'][i](x_dict['global_node'])
            }
            new_x = self.extraction_layers[i](norm_dict, edge_index_dict)
            
            for key in x_dict:
                if key in new_x:
                    g = torch.sigmoid(self.extraction_gates[key][i])
                    x_dict[key] = x_dict[key] + g * new_x[key]
            
            x_dict['stone'] = x_dict['stone'] + self.extraction_stone_ffns[i](x_dict['stone'])
            x_dict['string'] = x_dict['string'] + self.extraction_hyper_ffns[i](x_dict['string'])
            x_dict['global_node'] = x_dict['global_node'] + self.extraction_hyper_ffns[i](x_dict['global_node'])

        
        # 2. Reasoning Phase (Recurrent Weight Sharing)
        for i in range(self.reasoning_steps):
            # Select Block
            conv = self.conv_A if i % 2 == 0 else self.conv_B
            ffn_stone = self.ffn_stone_A if i % 2 == 0 else self.ffn_stone_B
            ffn_hyper = self.ffn_hyper_A if i % 2 == 0 else self.ffn_hyper_B

            # Pre-norm
            norm_x = {
                'stone': self.shared_norm_stone(x_dict['stone']),
                'string': self.shared_norm_hyper(x_dict['string']),
                'global_node': self.shared_norm_hyper(x_dict['global_node'])
            }

            # Conv + Gate
            new_x = conv(norm_x, edge_index_dict)
            for k in x_dict:
                if k in new_x:
                    g = torch.sigmoid(self.reason_gates[k][i])
                    x_dict[k] = x_dict[k] + g * new_x[k]

            # FFN
            x_dict['stone'] = x_dict['stone'] + ffn_stone(x_dict['stone'])
            x_dict['string'] = x_dict['string'] + ffn_hyper(x_dict['string'])
            x_dict['global_node'] = x_dict['global_node'] + ffn_hyper(x_dict['global_node'])

        return (self.policy_head(x_dict['stone']).squeeze(-1),
                self.pass_head(x_dict['global_node']).squeeze(-1),
                self.value_head(x_dict['global_node']).squeeze(-1),
                self.own_head(x_dict['stone']).squeeze(-1))