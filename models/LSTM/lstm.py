import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys

# -----------------------
# 경로 설정 및 DualLogger 설정
# -----------------------
os.environ['TORCHTEXT_DATADIR'] = os.path.abspath('../../data')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from jutils import DualLogger
# logger = DualLogger("output_log.txt")
# sys.stdout = logger

# -----------------------
# 하이퍼파라미터
# -----------------------
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_CLASSES = 2
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Tokenizer 및 Vocabulary
# -----------------------
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, line in data_iter:
        yield tokenizer(line)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

# -----------------------
# Text/Label 파이프라인 및 Collate 함수
# -----------------------
def text_pipeline(x): return vocab(tokenizer(x))
def label_pipeline(x): return 1 if x == "pos" else 0

def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(torch.tensor(label_pipeline(label), dtype=torch.long))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.long)
        text_list.append(processed_text)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.stack(label_list)
    return text_list.to(DEVICE), label_list.to(DEVICE)

# -----------------------
# 데이터 로딩
# -----------------------
train_iter, test_iter = IMDB(split=('train', 'test'))
train_dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# -----------------------
# LSTM 분류기 정의
# -----------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        out = self.fc(hn[-1])
        return out

model = LSTMClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# 모델 구조 및 파라미터 정보 출력
# -----------------------
def print_model_details(model):
    print(f"{'Layer':<40} {'Shape':<30} {'Param #':<12} {'Size (MB)':<10}")
    print("-" * 100)
    total_params = 0
    total_bytes = 0

    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        numel = param.numel()
        size_mb = numel * 4 / (1024 ** 2)
        total_params += numel
        total_bytes += numel * 4
        print(f"{name:<40} {str(shape):<30} {numel:<12} {size_mb:.2f} MB")

    print("-" * 100)
    print(f"Total parameters: {total_params:,} ({total_bytes / (1024 ** 2):.2f} MB)")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":

    print_model_details(model)

    # -----------------------
    # 학습
    # -----------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for texts, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(train_dataloader):.4f}")

    # -----------------------
    # 평가
    # -----------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels in test_dataloader:
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    # logger.log.close()
