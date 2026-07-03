import numpy as np
import matplotlib.pyplot as plt

# 3つのユニット A, B, C の全状態
states = ["000", "001", "010", "011", "100", "101", "110", "111"]

# Whole system:
# 1の数が偶数の状態だけが起こるとする
# 000, 011, 101, 110 の4状態が確率 1/4
whole_prob = []

for s in states:
    num_ones = s.count("1")
    if num_ones % 2 == 0:
        whole_prob.append(1 / 4)
    else:
        whole_prob.append(0)

whole_prob = np.array(whole_prob)

# Partitioned system:
# A, B, C を独立に見ると、各状態はすべて同じ確率 1/8
partitioned_prob = np.ones(len(states)) / len(states)

# KL divergence を計算する関数
def kl_divergence(p, q):
    kl = 0
    for pi, qi in zip(p, q):
        if pi > 0:
            kl += pi * np.log2(pi / qi)
    return kl

phi = kl_divergence(whole_prob, partitioned_prob)

# グラフ描画
x = np.arange(len(states))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width / 2, whole_prob, width, label="Whole system")
plt.bar(x + width / 2, partitioned_prob, width, label="Partitioned system")

plt.xticks(x, states)
plt.xlabel("Future state (A, B, C)")
plt.ylabel("Probability")
plt.title(f"Integrated Information Example: 3-Unit Even Parity System\nKL divergence = {phi:.2f} bits")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
