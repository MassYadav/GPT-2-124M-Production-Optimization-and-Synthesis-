import os
import pickle
import numpy as np
import tiktoken

# read your raw text
with open("input.txt", "r", encoding="utf-8") as f:
    data = f.read()

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode_ordinary(data)
ids = np.array(ids, dtype=np.uint16)

# 90% train, 10% val
n = int(len(ids) * 0.9)
train_ids = ids[:n]
val_ids = ids[n:]

os.makedirs("data", exist_ok=True)
train_ids.tofile("data/train.bin")
val_ids.tofile("data/val.bin")

meta = {
    "vocab_size": enc.n_vocab,
    "encoder_name": "gpt2",
}
with open("data/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Prepared data: data/train.bin, data/val.bin, data/meta.pkl")
