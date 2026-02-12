import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np

# ───────────────────────────────────────────────────────────────
# MODEL CLASSES (copied from your notebook)
# ───────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(2048, hidden_size)
        self.dropout = nn.Dropout(0.3)  # added to avoid overfitting

    def forward(self, img_feat):
        # img_feat: (B, 2048) ← image features from ResNet
        h0 = torch.tanh(self.fc(img_feat))  # (B, hidden_size)
        h0 = self.dropout(h0)  # added to avoid overfitting
        return h0.unsqueeze(0)  # ← now (1, B, hidden_size)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type.lower()

        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)  # pad_id will be defined later
        self.embed_dropout = nn.Dropout(0.3)  # added to avoid overfitting

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_size,
                batch_first=True,
                dropout=0.3  # added to avoid overfitting (active if num_layers > 1)
            )
        else:
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_size,
                batch_first=True,
                dropout=0.3  # added to avoid overfitting (active if num_layers > 1)
            )

        self.output_dropout = nn.Dropout(0.4)  # added to avoid overfitting
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, cap_in, h0):
        # cap_in: (B, seq_len)
        # h0: (1, B, hidden_size)
        x = self.embed(cap_in)  # (B, seq_len, embed_dim)
        x = self.embed_dropout(x)  # added to avoid overfitting

        if self.rnn_type == "gru":
            out, hn = self.rnn(x, h0)
        else:
            c0 = torch.zeros_like(h0)  # (1, B, hidden_size)
            out, (hn, cn) = self.rnn(x, (h0, c0))

        out = self.output_dropout(out)  # added to avoid overfitting
        logits = self.fc_out(out)  # (B, seq_len, vocab_size)
        return logits

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, rnn_type="lstm"):
        super().__init__()
        self.encoder = Encoder(hidden_size=hidden_size)
        self.decoder = Decoder(
            vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            rnn_type=rnn_type
        )

    def forward(self, img_feat, cap_in):
        h0 = self.encoder(img_feat)       # (1, B, H)
        logits = self.decoder(cap_in, h0) # (B, seq_len, vocab_size)
        return logits

# ───────────────────────────────────────────────────────────────
# DECODE FUNCTIONS (with @torch.no_grad() restored)
# ───────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(img_feat_2048, max_len=30):
    img_feat = torch.tensor(img_feat_2048, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2048)
    
    h0 = model.encoder(img_feat)  # (1, 1, hidden)

    cur = torch.tensor([[start_id]], dtype=torch.long).to(device)  # (1, 1)
    out_tokens = []

    for _ in range(max_len):
        logits = model.decoder(cur, h0)
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1).item()
        
        if next_id == end_id:
            break
        
        out_tokens.append(next_id)
        cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)

    words = [idx2word.get(i, "<unk>") for i in out_tokens]
    return " ".join(words)

@torch.no_grad()
def beam_decode(img_feat_2048, beam_width=3, max_len=30):
    img_feat = torch.tensor(img_feat_2048, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2048)
    
    h_init = model.encoder(img_feat)  # (1, 1, H)

    beams = [([start_id], 0.0, h_init)]
    finished = []

    for step in range(max_len):
        new_beams = []
        
        for seq, score, hidden in beams:
            if seq[-1] == end_id:
                finished.append((seq, score))
                continue

            cur = torch.tensor([seq], dtype=torch.long).to(device)

            logits = model.decoder(cur, hidden)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            topk = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                next_id = topk.indices[0, i].item()
                next_logp = topk.values[0, i].item()
                new_seq = seq + [next_id]
                new_score = score + next_logp
                new_beams.append((new_seq, new_score, hidden))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    finished.extend(beams)
    finished.sort(key=lambda x: x[1], reverse=True)

    best_seq = finished[0][0]
    best_seq = [t for t in best_seq if t not in (start_id, end_id, pad_id)]
    
    words = [idx2word.get(i, "<unk>") for i in best_seq]
    return " ".join(words)

# ───────────────────────────────────────────────────────────────
# LOAD ARTIFACTS (moved up – vocab first, then model)
# ───────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabulary first (critical)
with open('vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)

word2idx = vocab_data['word2idx']
idx2word = vocab_data['idx2word']
vocab_size = vocab_data['vocab_size']
start_id = vocab_data['start_id']
end_id = vocab_data['end_id']
pad_id = vocab_data['pad_id']  # Now defined before Decoder uses it

# Improved for app.py: Create and load model AFTER vocab is loaded
model = Seq2Seq(
    vocab_size=vocab_size,
    embed_dim=256,
    hidden_size=512,
    rnn_type="lstm"
)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.to(device)
model.eval()

# Image preprocessing & ResNet feature extractor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()  # Improved: .eval() here

# ───────────────────────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────────────────────

st.title("Neural Storyteller – Image Captioning (Flickr30k Seq2Seq)")

uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Extracting features & generating..."):
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet(img_tensor).view(1, -1)  # (1, 2048)
            
            # Use beam search (better quality) – change to greedy_decode(...) if you want faster
            caption = beam_decode(feat.squeeze(0).cpu().numpy(), beam_width=3)
            st.success("Generated Caption:")
            st.write(caption)