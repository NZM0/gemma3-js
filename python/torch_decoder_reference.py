"""Independent PyTorch oracle for Stage 02's token-at-a-time decoder.

This is NOT a Hugging Face checkpoint parity test. It checks the existing
JavaScript decoder against separately implemented PyTorch operations.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "torch-decoder-reference.json"
torch.set_grad_enabled(False)

config = {
    "hiddenSize": 8,
    "intermediateSize": 12,
    "numLayers": 2,
    "numHeads": 2,
    "numKeyValueHeads": 1,
    "headDim": 4,
    "slidingWindow": 2,
    "rmsNormEps": 1e-6,
    "queryPreAttnScalar": 4,
    "localRopeTheta": 10000,
    "globalRopeTheta": 1000000,
    "layerTypes": ["local", "global"],
}


def tensor(rows, cols, seed):
    values = [math.sin((i + 1) * (seed + 1) * 0.31) * 0.14 for i in range(rows * cols)]
    return torch.tensor(values, dtype=torch.float32).reshape(rows, cols)


def layer(seed):
    h = config["hiddenSize"]
    d = config["headDim"]
    q = config["numHeads"] * d
    kv = config["numKeyValueHeads"] * d
    m = config["intermediateSize"]
    return {
        "inputLayernorm": torch.tensor([math.sin(i + seed) * 0.08 for i in range(h)]),
        "postAttentionLayernorm": torch.tensor([math.cos(i + seed) * 0.08 for i in range(h)]),
        "preFeedforwardLayernorm": torch.tensor([math.sin(i + seed + 2) * 0.08 for i in range(h)]),
        "postFeedforwardLayernorm": torch.tensor([math.cos(i + seed + 2) * 0.08 for i in range(h)]),
        "qNorm": torch.tensor([math.sin(i + seed) * 0.08 for i in range(d)]),
        "kNorm": torch.tensor([math.cos(i + seed) * 0.08 for i in range(d)]),
        "qProj": tensor(q, h, seed + 1),
        "kProj": tensor(kv, h, seed + 2),
        "vProj": tensor(kv, h, seed + 3),
        "oProj": tensor(h, q, seed + 4),
        "gateProj": tensor(m, h, seed + 5),
        "upProj": tensor(m, h, seed + 6),
        "downProj": tensor(h, m, seed + 7),
    }


def norm(x, weight):
    return F.rms_norm(x, (x.shape[-1],), eps=config["rmsNormEps"]) * (1 + weight)


def rope(x, position, theta):
    half = x.shape[-1] // 2
    indices = torch.arange(half, dtype=torch.float32)
    angle = position / theta ** (2 * indices / x.shape[-1])
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()), dim=-1)


def step(x, weights, state, layer_index, position):
    h = config["hiddenSize"]
    d = config["headDim"]
    qh = config["numHeads"]
    kvh = config["numKeyValueHeads"]
    local = config["layerTypes"][layer_index] == "local"
    theta = config["localRopeTheta"] if local else config["globalRopeTheta"]

    z = norm(x, weights["inputLayernorm"])
    q = (weights["qProj"] @ z).reshape(qh, d)
    k = (weights["kProj"] @ z).reshape(kvh, d)
    v = (weights["vProj"] @ z).reshape(kvh, d)
    q = rope(norm(q, weights["qNorm"]), position, theta)
    k = rope(norm(k, weights["kNorm"]), position, theta)

    state["positions"].append(position)
    state["keys"].append(k)
    state["values"].append(v)
    if local and len(state["positions"]) > config["slidingWindow"]:
        for name in ("positions", "keys", "values"):
            state[name].pop(0)

    heads = []
    for qi in range(qh):
        ki = qi // (qh // kvh)
        scores = torch.stack([
            torch.dot(q[qi], key[ki]) / math.sqrt(config["queryPreAttnScalar"])
            for key in state["keys"]
        ])
        probabilities = scores.softmax(dim=0)
        head = sum((p * value[ki] for p, value in zip(probabilities, state["values"])), torch.zeros(d))
        heads.append(head)

    attention = weights["oProj"] @ torch.cat(heads)
    x = x + norm(attention, weights["postAttentionLayernorm"])
    ff = norm(x, weights["preFeedforwardLayernorm"])
    gate = weights["gateProj"] @ ff
    up = weights["upProj"] @ ff
    down = weights["downProj"] @ (F.gelu(gate, approximate="tanh") * up)
    return x + norm(down, weights["postFeedforwardLayernorm"])


def flatten_layer(weights):
    return {name: value.reshape(-1).tolist() for name, value in weights.items()}


def main():
    layers = [layer(1), layer(11)]
    inputs = [torch.tensor([math.cos((i + 1) * (p + 1) * 0.4) for i in range(8)], dtype=torch.float32) for p in range(5)]
    states = [{"positions": [], "keys": [], "values": []} for _ in layers]
    expected = []

    for position, x in enumerate(inputs):
        intermediate = []
        for layer_index, weights in enumerate(layers):
            x = step(x, weights, states[layer_index], layer_index, position)
            intermediate.append(x.tolist())
        expected.append({"position": position, "layerOutputs": intermediate, "output": x.tolist()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "kind": "synthetic-independent-pytorch-oracle",
        "config": config,
        "layers": [flatten_layer(weights) for weights in layers],
        "inputs": [value.tolist() for value in inputs],
        "expected": expected,
    }, indent=4), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
