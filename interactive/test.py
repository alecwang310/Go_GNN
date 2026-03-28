import sys
root_path = r'D:/Code/GNN'
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
data = np.load("sample_3.npz")
print(data.files)
var = data['globalTargetsNC']
print(var[1][55])