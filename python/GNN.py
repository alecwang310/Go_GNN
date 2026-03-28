import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

class GoGNN(nn.Module):
    def __init__(self, stone_in=18, string_in=2, global_in=19, 
                 stone_dim=128, hyper_dim=512, dropout = 0.1, heads=8):
        super().__init__()
        
        # 1. Heterogeneous Embeddings
        self.stone_emb  = nn.Linear(stone_in, stone_dim)
        self.string_emb = nn.Linear(string_in, hyper_dim)
        self.global_emb = nn.Linear(global_in, hyper_dim)

        #The FFN layers
        self.layers = nn.ModuleList()
        self.stone_ffns = nn.ModuleList()
        self.global_ffns = nn.ModuleList()
        self.string_ffns = nn.ModuleList()

        num_layers = 10

        #Added gates the help with oversmoothing: the model can decide to skip some layers
        self.gates = nn.ModuleDict({
            'stone': nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)]),
            'string': nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)]),
            'global_node': nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        })

        for _ in range(num_layers):
            self.layers.append(HeteroConv({
                # Stone-to-Stone (Tactical)
                ('stone', 'adjacent', 'stone'): 
                    GATv2Conv(stone_dim, stone_dim // heads, heads=heads),
                
                # Stone-to-String (Aggregation) - Stone (128) -> String (512)
                ('stone', 'belongs_to', 'string'): 
                    GATv2Conv((stone_dim, hyper_dim), hyper_dim // heads, heads=heads, add_self_loops=False),
                
                # String-to-Stone (Broadcast) - String (512) -> Stone (128)
                ('string', 'contains', 'stone'): 
                    GATv2Conv((hyper_dim, stone_dim), stone_dim // heads, heads=heads, add_self_loops=False),
                
                # Hyper-to-Hyper (Strategic)
                ('string', 'adjacent_to', 'string'): 
                    GATv2Conv(hyper_dim, hyper_dim // heads, heads=heads),
                ('string', 'reports_to', 'global_node'): 
                    GATv2Conv(hyper_dim, hyper_dim // heads, heads=heads, add_self_loops=False),
                ('global_node', 'influences', 'string'): 
                    GATv2Conv(hyper_dim, hyper_dim // heads, heads=heads, add_self_loops=False),
            }, aggr='mean'))
            
            #Add a FFN layer for every pass layer
            self.string_ffns.append(nn.Sequential(
                nn.LayerNorm(hyper_dim),
                nn.Linear(hyper_dim, hyper_dim * 2),
                nn.GELU(), 
                nn.Linear(hyper_dim * 2, hyper_dim),
                nn.Dropout(dropout)
            ))

            self.global_ffns.append(nn.Sequential(
                nn.LayerNorm(hyper_dim),
                nn.Linear(hyper_dim, hyper_dim * 2),
                nn.GELU(), 
                nn.Linear(hyper_dim * 2, hyper_dim),
                nn.Dropout(dropout)
            ))

            self.stone_ffns.append(nn.Sequential(
                nn.LayerNorm(stone_dim),
                nn.Linear(stone_dim, stone_dim * 4),
                nn.GELU(), 
                nn.Linear(stone_dim * 4, stone_dim),
                nn.Dropout(dropout)
            ))
        
        self.policy_head = nn.Sequential(
            nn.Linear(stone_dim, stone_dim),
            nn.GELU(),
            nn.Linear(stone_dim, 1)
        )

        self.own_head = nn.Sequential(
            nn.Linear(stone_dim, stone_dim),
            nn.GELU(),
            nn.Linear(stone_dim, 1),
            nn.Tanh()
        )

        self.value_head = nn.Sequential(
            nn.Linear(hyper_dim, hyper_dim // 2),
            nn.GELU(),
            nn.Linear(hyper_dim // 2, 1),
            nn.Tanh()
        )

        self.pass_head = nn.Sequential(
            nn.Linear(hyper_dim, hyper_dim // 2),
            nn.GELU(),
            nn.Linear(hyper_dim // 2, 1)
        )

        self.stone_norms = nn.ModuleList([nn.LayerNorm(stone_dim) for _ in range(10)])
        self.string_norms = nn.ModuleList([nn.LayerNorm(hyper_dim) for _ in range(10)])
        self.global_norms = nn.ModuleList([nn.LayerNorm(hyper_dim) for _ in range(10)])


    def forward(self, x_dict, edge_index_dict):
        # 1. Embeddings
        x_dict['stone'] = F.gelu(self.stone_emb(x_dict['stone']))
        x_dict['string'] = F.gelu(self.string_emb(x_dict['string']))
        x_dict['global_node'] = F.gelu(self.global_emb(x_dict['global_node']))
        
        if x_dict['global_node'].dim() == 3:
            x_dict['global_node'] = x_dict['global_node'].squeeze(1)

        # 2. GNN Layers
        for i, conv in enumerate(self.layers):
            norm_x = {
                'stone': self.stone_norms[i](x_dict['stone']),
                'string': self.string_norms[i](x_dict['string']),
                'global_node': self.global_norms[i](x_dict['global_node'])
            }

            # The actual convolution layer where the message passing happens
            new_x = conv(norm_x, edge_index_dict)


            for key in x_dict:
                if key in new_x:
                    g = torch.sigmoid(self.gates[key][i])
                    x_dict[key] = x_dict[key] + g * new_x[key]

            x_dict['stone'] = x_dict['stone'] + self.stone_ffns[i](x_dict['stone'])
            x_dict['string'] = x_dict['string'] + self.string_ffns[i](x_dict['string'])
            x_dict['global_node'] = x_dict['global_node'] + self.global_ffns[i](x_dict['global_node'])


        # Policy & Possession
        policy_board = self.policy_head(x_dict['stone']).squeeze(-1)
        possession = self.own_head(x_dict['stone']).squeeze(-1)

        # Value
        policy_pass = self.pass_head(x_dict['global_node']).squeeze(-1)
        value = self.value_head(x_dict['global_node']).squeeze(-1)

        return policy_board, policy_pass, value, possession