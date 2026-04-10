import numpy as np

clusters = []
for i in range(100):
    size = np.random.randint(10, 500)
    pos = int(size * np.random.uniform(0, 0.4)) if np.random.rand() > 0.3 else 0
    clusters.append({"size": size, "pos_count": pos})

clusters.sort(key=lambda x: x["size"], reverse=True)

target_counts = {"train": 80000, "val": 10000, "test": 10000}
n_mols = sum(c["size"] for c in clusters)
target_counts = {"train": int(n_mols * 0.8), "val": int(n_mols * 0.1), "test": n_mols - int(n_mols * 0.8) - int(n_mols * 0.1)}

train_size, train_pos = 0, 0
val_size, val_pos = 0, 0
test_size, test_pos = 0, 0

def get_ratio(size, pos):
    return pos / size if size > 0 else 0.0

for c in clusters:
    size, pos = c["size"], c["pos_count"]
    
    can_train = train_size + size <= target_counts["train"] * 1.05
    can_val = val_size + size <= target_counts["val"] * 1.05
    can_test = test_size + size <= target_counts["test"] * 1.05
    
    if not (can_train or can_val or can_test):
        rem = {"train": target_counts["train"] - train_size, 
               "val": target_counts["val"] - val_size, 
               "test": target_counts["test"] - test_size}
        best = max(rem, key=rem.get)
        if best == "train": can_train = True
        elif best == "val": can_val = True
        else: can_test = True

    cands = []
    if can_train: cands.append(("train", get_ratio(train_size, train_pos), target_counts["train"]))
    if can_val: cands.append(("val", get_ratio(val_size, val_pos), target_counts["val"]))
    if can_test: cands.append(("test", get_ratio(test_size, test_pos), target_counts["test"]))
    
    if pos > 0:
        # tie breaker: largest target_count
        best_split = min(cands, key=lambda x: (x[1], -x[2]))[0]
    else:
        best_split = max(cands, key=lambda x: (x[1], x[2]))[0]

    if best_split == "train":
        train_size += size; train_pos += pos
    elif best_split == "val":
        val_size += size; val_pos += pos
    else:
        test_size += size; test_pos += pos

print(f"Train: {train_size} elems, {get_ratio(train_size, train_pos):.2%} pos")
print(f"Val:   {val_size} elems, {get_ratio(val_size, val_pos):.2%} pos")
print(f"Test:  {test_size} elems, {get_ratio(test_size, test_pos):.2%} pos")
