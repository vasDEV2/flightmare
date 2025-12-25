    # import torch
    # import torch.nn as nn
    # import numpy as np
        

    # param = np.load("sb1_params.npy", allow_pickle=True)

    # print(param.item().keys())

# test_pytorch_loader.py
import numpy as np
import torch
import torch.nn as nn
import re
from collections import defaultdict, OrderedDict

np_path = "sb1_params.npy"   # adjust if needed
params = np.load(np_path, allow_pickle=True).item()

# 1) Print param keys & shapes for inspection
print("Loaded parameter keys and shapes:")
for k, v in params.items():
    print(f"  {k:30s} -> {np.shape(v)}")
print()

# 2) Helper: group params by prefix like "model/pi_fc0", "model/pi"
groups = defaultdict(dict)
for k, v in params.items():
    # e.g. 'model/pi_fc0/w:0' -> prefix='model/pi_fc0', suffix='w' or 'b' or 'logstd'
    m = re.match(r"(?P<prefix>.+?)/(w|b|logstd)(:0)?$", k)
    if m:
        prefix = m.group("prefix")
        if "/w" in k:
            groups[prefix]["w"] = v
        elif "/b" in k:
            groups[prefix]["b"] = v
        elif "logstd" in k:
            groups[prefix]["logstd"] = v
    else:
        # fallback: keep raw
        groups[k]["raw"] = v

# 3) Build a dynamic linear stack class for a named group (pi or vf)
class DynamicMLP(nn.Module):
    def __init__(self, ordered_prefixes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.prefixes = ordered_prefixes  # list of prefixes in forward order
        for pref in ordered_prefixes:
            info = groups[pref]
            if "w" not in info or "b" not in info:
                raise RuntimeError(f"Missing w/b for {pref}")
            w = info["w"]
            # TF store shape: (in_dim, out_dim)
            in_dim = int(w.shape[0])
            out_dim = int(w.shape[1])
            linear = nn.Linear(in_dim, out_dim)
            self.layers.append(linear)

    def forward(self, x):
        # apply tanh activation between layers except after the final output layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)   # SB1 default activation
        return x

# 4) Identify actor (pi) prefixes in order and value (vf) prefixes
def sorted_prefixes_for(keyroot):
    # collect prefixes that start with keyroot (e.g., 'model/pi' or 'model/pi_fc')
    found = [p for p in groups.keys() if p.startswith(keyroot)]
    # sort by numeric suffix if present (e.g. pi_fc0, pi_fc1), otherwise put exact root 'model/pi' last
    def sort_key(p):
        m = re.search(r"_(\d+)$", p)   # matches ..._0, ..._1 at the end
        if m:
            return int(m.group(1))
        # place bare root after numbered layers
        if p == keyroot:
            return 9999
        # try to find trailing digits anywhere
        m2 = re.search(r"(\d+)", p)
        return int(m2.group(1)) if m2 else 10000
    return sorted(found, key=sort_key)

pi_prefixes = sorted_prefixes_for("model/pi")       # will include model/pi_fc0, model/pi_fc1, model/pi
vf_prefixes = sorted_prefixes_for("model/vf")       # model/vf_fc0, model/vf_fc1, model/vf
q_prefixes  = sorted_prefixes_for("model/q")        # possibly model/q (single layer) or model/q_fc...
print("Detected prefixes (actor):", pi_prefixes)
print("Detected prefixes (critic):", vf_prefixes)
print("Detected prefixes (q-head):", q_prefixes)
print()

# 5) Instantiate dynamic modules if present
actor = None
critic = None
q_head = None
pi_logstd = None

if pi_prefixes:
    actor = DynamicMLP(pi_prefixes)
if vf_prefixes:
    critic = DynamicMLP(vf_prefixes)
if q_prefixes:
    q_head = DynamicMLP(q_prefixes)  # handles single linear or multi-layer q

# 6) Load weights into the PyTorch modules (transpose TF weights)
def load_weights_into_module(module, prefixes):
    with torch.no_grad():
        for layer, pref in zip(module.layers, prefixes):
            info = groups[pref]
            w = info["w"]              # TF shape (in, out)
            b = info["b"]
            in_dim = int(w.shape[0]); out_dim = int(w.shape[1])
            # Make sure layer dims match expected
            if layer.in_features != in_dim or layer.out_features != out_dim:
                raise RuntimeError(f"Dimension mismatch for {pref}: TF ({in_dim},{out_dim}) vs PyTorch ({layer.in_features},{layer.out_features})")
            # transpose w -> (out, in) for PyTorch
            layer.weight.copy_(torch.tensor(w.T, dtype=torch.float32))
            layer.bias.copy_(torch.tensor(b.reshape(-1), dtype=torch.float32))

# load actor & critic weights
if actor:
    load_weights_into_module(actor, pi_prefixes)
    # load logstd if present
    root_pi = "model/pi"
    if "model/pi/logstd:0" in params:
        logstd = params["model/pi/logstd:0"]
        # make it a learnable parameter with shape (action_dim,)
        pi_logstd = nn.Parameter(torch.tensor(logstd.reshape(-1), dtype=torch.float32))
    else:
        # check grouped logstd
        if "logstd" in groups.get(root_pi, {}):
            ls = groups[root_pi]["logstd"]
            pi_logstd = nn.Parameter(torch.tensor(ls.reshape(-1), dtype=torch.float32))

if critic:
    load_weights_into_module(critic, vf_prefixes)
if q_head:
    load_weights_into_module(q_head, q_prefixes)

# 7) Wrap into a convenient policy class
class SB1ConvertedPolicy(nn.Module):
    def __init__(self, actor, critic, pi_logstd=None, q_head=None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.q_head = q_head
        if pi_logstd is not None:
            self.pi_logstd = pi_logstd
        else:
            self.pi_logstd = None

    def forward(self, x):
        # expects x: torch tensor shape (B, obs_dim)
        mean = None; value = None; qval = None
        if self.actor:
            mean = self.actor(x)
        if self.critic:
            value = self.critic(x)
        if self.q_head:
            qval = self.q_head(x)
        return {"mean": mean, "logstd": self.pi_logstd, "value": value, "q": qval}

policy = SB1ConvertedPolicy(actor, critic, pi_logstd, q_head)
policy.eval()
print("Successfully built PyTorch policy object.")
print(policy)

# 8) Example: run a dummy forward (make sure dims match)
# Find obs_dim from first actor layer if available, else critic
obs_dim = None
if actor:
    first_w = groups[pi_prefixes[0]]["w"]
    obs_dim = int(first_w.shape[0])
elif critic:
    first_w = groups[vf_prefixes[0]]["w"]
    obs_dim = int(first_w.shape[0])

if obs_dim is not None:
    x = torch.zeros(1, obs_dim, dtype=torch.float32)
    out = policy(x)
    print("Forward pass outputs shapes:")
    for k, v in out.items():
        if v is None:
            print(f"  {k}: None")
        elif isinstance(v, nn.Parameter):
            print(f"  {k}: parameter shape {v.shape}")
        else:
            print(f"  {k}: {v.shape}")
else:
    print("Couldn't infer obs_dim automatically; run the script after printing shapes to determine obs_dim.")

    

