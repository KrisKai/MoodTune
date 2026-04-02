import torch
import torch.nn as nn

from config import (
    MIDI_VOCAB_SIZE, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_EMBED_DIM, NUM_CLASSES,
)


class MelodyLSTM(nn.Module):
    """LSTM-based melody generator conditioned on emotion.

    Architecture:
        - Note embedding (MIDI note → dense vector)
        - Emotion embedding (emotion index → dense vector)
        - 2-layer LSTM with dropout
        - FC output head → next note prediction

    The emotion embedding is concatenated with the note embedding at each
    timestep, so the LSTM learns emotion-specific melodic patterns.
    """

    def __init__(
        self,
        vocab_size=MIDI_VOCAB_SIZE,
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        num_emotions=NUM_CLASSES,
        dropout=0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embeddings
        self.note_embed = nn.Embedding(vocab_size, embed_dim)
        self.emotion_embed = nn.Embedding(num_emotions, embed_dim)

        # LSTM: input is note_embed + emotion_embed concatenated
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, notes, emotion_idx, hidden=None):
        """
        Args:
            notes: (batch, seq_len) MIDI note indices
            emotion_idx: (batch,) emotion class indices
            hidden: optional (h0, c0) tuple

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: updated hidden state
        """
        batch_size, seq_len = notes.shape

        # Embed notes: (batch, seq_len, embed_dim)
        note_emb = self.note_embed(notes)

        # Embed emotion and expand to match sequence: (batch, seq_len, embed_dim)
        emo_emb = self.emotion_embed(emotion_idx)  # (batch, embed_dim)
        emo_emb = emo_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate: (batch, seq_len, embed_dim * 2)
        x = torch.cat([note_emb, emo_emb], dim=-1)

        # LSTM
        if hidden is None:
            hidden = self.init_hidden(batch_size, notes.device)
        out, hidden = self.lstm(x, hidden)

        # Output
        out = self.dropout(out)
        logits = self.fc(out)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def generate(self, emotion_idx, start_notes, num_notes=32, temperature=1.0,
                 scale_mask=None, device='cpu'):
        """Generate a melody autoregressively.

        Args:
            emotion_idx: int, emotion class index
            start_notes: list of MIDI notes to seed the generation
            num_notes: number of notes to generate
            temperature: sampling temperature (higher = more random)
            scale_mask: optional tensor of shape (vocab_size,), 1 for valid notes, 0 for invalid
            device: torch device

        Returns:
            list of generated MIDI note numbers
        """
        self.eval()
        generated = list(start_notes)

        # Prepare emotion tensor
        emo = torch.tensor([emotion_idx], device=device)

        # Seed with start notes
        input_notes = torch.tensor([start_notes], device=device)
        hidden = self.init_hidden(1, device)

        # Process seed sequence
        with torch.no_grad():
            logits, hidden = self.forward(input_notes, emo, hidden)

        # Generate note by note
        current_note = torch.tensor([[start_notes[-1]]], device=device)

        for _ in range(num_notes):
            with torch.no_grad():
                logits, hidden = self.forward(current_note, emo, hidden)

            # Apply temperature
            logits = logits[:, -1, :] / temperature

            # Apply scale mask if provided (force notes to stay in key)
            if scale_mask is not None:
                logits[:, scale_mask == 0] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_note = torch.multinomial(probs, 1)

            generated.append(next_note.item())
            current_note = next_note

        return generated
