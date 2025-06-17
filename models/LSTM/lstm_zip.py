import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
import zlib
from torchinfo import summary
from io import StringIO
import sys
import numpy as np

# ──────── 설정 ────────
lstm_data_dir = os.path.abspath('../data/LSTM')
os.makedirs(lstm_data_dir, exist_ok=True)
os.environ['TORCHTEXT_DATADIR'] = lstm_data_dir

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
local_data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(local_data_dir, exist_ok=True)

result_file = os.path.join(local_data_dir, f"compression_results_{timestamp}.csv")
model_summary_path = os.path.join(local_data_dir, "model_summary.txt")
param_detail_path = os.path.join(local_data_dir, "param_details.txt")
model_save_path = os.path.join(local_data_dir, "lstm_model_weights.pth")

# ──────── 데이터 로딩 ────────
tokenizer = get_tokenizer("basic_english")

# ✅ IMDB 데이터를 메모리에 올려서 재사용
train_iter_raw = list(IMDB(split='train'))

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter_raw), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])
pad_idx = vocab["<pad>"]

def text_pipeline(x): return vocab(tokenizer(x))
def label_pipeline(x): return 1 if x == "pos" else 0

def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(torch.tensor(label_pipeline(label), dtype=torch.long))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.long)
        text_list.append(processed_text)
    return pad_sequence(text_list, batch_first=True, padding_value=pad_idx), torch.tensor(label_list)

train_dataloader = DataLoader(train_iter_raw, batch_size=16, shuffle=True, collate_fn=collate_batch)

# ──────── 모델 정의 ────────
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n[-1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(len(vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ──────── 모델 요약 저장 ────────
dummy_input = torch.randint(0, len(vocab), (16, 100)).to(device)
buffer = StringIO()
sys_stdout = sys.stdout
sys.stdout = buffer
summary(model, input_data=dummy_input, col_names=["input_size", "output_size", "num_params"])
sys.stdout = sys_stdout
with open(model_summary_path, "w") as f:
    f.write(buffer.getvalue())

with open(param_detail_path, "w") as f:
    f.write("[LSTM Model Parameter Details]\n\n")
    total_params = 0
    for name, param in model.named_parameters():
        size = param.numel()
        total_params += size
        shape = tuple(param.shape)
        f.write(f"{name:<35} {str(shape):<20} - param: {size:9,} ({size * 4 / 1024 / 1024:.2f} MB)\n")
    f.write(f"Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)\n")

# ──────── gradient 추출 함수 ────────
def extract_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().cpu().numpy().tobytes())
    return b"".join(grads)

# ──────── CSV 헤더 설정 ────────
with open(result_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Epoch", "Original Size", "Compressed Size", "Compression Ratio",
        "Delta Time", "Delta Size", "Delta Ratio", "Compress Time", "Accuracy"
    ])

# ──────── 학습 루프 ────────
prev_grad = None
prev_grad_dict = {}

for epoch in range(5):
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        total_correct += (preds == y).sum().item()
        total += len(y)

    accuracy = total_correct / total

    epoch_dir = os.path.join("epoch_data", f"epoch_{epoch+1:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    curr_grad = extract_gradients(model)
    original_size = len(curr_grad)

    start = time.time()
    compressed = zlib.compress(curr_grad)
    compress_time = time.time() - start
    compressed_size = len(compressed)
    ratio = compressed_size / original_size if original_size > 0 else 0

    delta_time = delta_size = delta_ratio = 0.0
    if prev_grad is not None and len(prev_grad) == len(curr_grad):
        start = time.time()
        delta = bytes([a ^ b for a, b in zip(curr_grad, prev_grad)])
        delta_time = time.time() - start
        delta_compressed = zlib.compress(delta)
        delta_size = len(delta_compressed)
        delta_ratio = delta_size / original_size if original_size > 0 else 0
    else:
        delta = None
        delta_time = delta_size = delta_ratio = -1

    prev_grad = curr_grad

    with open(result_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            original_size,
            compressed_size,
            f"{ratio:.4f}",
            f"{delta_time:.4f}" if delta_time >= 0 else "N/A",
            delta_size if delta_size >= 0 else "N/A",
            f"{delta_ratio:.4f}" if delta_ratio >= 0 else "N/A",
            f"{compress_time:.4f}",
            f"{accuracy:.4f}"
        ])

    print(
        f"[Epoch {epoch+1}] Loss: {total_loss / total:.4f}, "
        f"Accuracy: {accuracy * 100:.2f}%, "
        f"Ratio: {ratio:.4f}, "
        f"Delta Ratio: {delta_ratio:.4f}" if delta_ratio >= 0 else
        f"[Epoch {epoch+1}] Loss: {total_loss / total:.4f}, "
        f"Accuracy: {accuracy * 100:.2f}%, "
        f"Ratio: {ratio:.4f}, Delta: N/A"
    )

    # 파라미터 및 그래디언트 저장
    for name, param in model.named_parameters():
        np.save(os.path.join(epoch_dir, f"param_{name}.npy"), param.data.cpu().numpy())

        if param.grad is not None:
            grad_np = param.grad.detach().cpu().numpy()
            np.save(os.path.join(epoch_dir, f"grad_{name}.npy"), grad_np)

            if name in prev_grad_dict:
                delta_np = grad_np - prev_grad_dict[name]
                np.save(os.path.join(epoch_dir, f"delta_{name}.npy"), delta_np)
            prev_grad_dict[name] = grad_np.copy()

# ──────── 모델 저장 및 grad 복사 ────────
torch.save(model.state_dict(), model_save_path)
state_dict = torch.load(model_save_path, map_location=device)
for name, param in model.named_parameters():
    if name in state_dict:
        param.grad = state_dict[name].clone()
print(f"✅ 저장된 모델의 값을 .grad에 복사해 완료")
