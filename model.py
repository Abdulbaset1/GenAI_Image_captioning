import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_size)

    def forward(self, x):
        return self.fc(x)

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions, hidden):
        emb = self.embedding(captions)
        h0 = hidden.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(emb, (h0, c0))
        return self.fc(out)

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = CaptionDecoder(vocab_size)

    def forward(self, features, captions):
        hidden = self.encoder(features)
        return self.decoder(captions, hidden)
