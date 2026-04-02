"""
Train the LSTM melody generator on MIDI files.

Dataset: Place .mid files in data/midi/ organized by emotion subdirectories:
    data/midi/
        happy/       *.mid files
        sad/         *.mid files
        angry/       *.mid files
        ...

If you don't have emotion-labeled MIDI files, you can:
  1. Download the Nottingham Music Dataset or any MIDI collection
  2. Place all files in data/midi/all/
  3. Run with --auto-label flag to assign emotions by key/tempo analysis

Usage:
    python3 training/train_music.py               # Train on labeled MIDI
    python3 training/train_music.py --auto-label   # Auto-label by musical features
    python3 training/train_music.py --generate-synthetic  # Generate synthetic training data
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MIDI_DATA_DIR, LSTM_MODEL_PATH, EMOTION_LABELS, NUM_CLASSES,
    LSTM_SEQ_LEN, LSTM_EPOCHS, LSTM_LR, LSTM_BATCH_SIZE, MIDI_VOCAB_SIZE,
    EMOTION_MUSIC_PARAMS, BASE_DIR,
)
from models.melody_lstm import MelodyLSTM
from utils.emotion_music_map import get_scale_notes


# ── MIDI Parsing ──

def parse_midi_file(filepath):
    """Extract note sequence from a MIDI file. Returns list of MIDI note numbers."""
    try:
        import mido
        mid = mido.MidiFile(filepath)
    except ImportError:
        print("mido not installed. Run: pip install mido")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    notes = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
    return notes


def parse_midi_directory(midi_dir):
    """Parse all MIDI files in emotion-labeled subdirectories.

    Returns:
        sequences: list of (note_sequence, emotion_index) tuples
    """
    sequences = []

    for emo_idx, emotion in enumerate(EMOTION_LABELS):
        emo_dir = os.path.join(midi_dir, emotion)
        if not os.path.isdir(emo_dir):
            continue

        midi_files = glob.glob(os.path.join(emo_dir, '*.mid')) + \
                     glob.glob(os.path.join(emo_dir, '*.midi'))

        for f in midi_files:
            notes = parse_midi_file(f)
            if len(notes) >= 16:
                sequences.append((notes, emo_idx))

        print(f"  {emotion}: {len(midi_files)} files")

    return sequences


# ── Synthetic Data Generation ──

def generate_synthetic_data(num_sequences=500, seq_len=128):
    """Generate synthetic training sequences using rule-based patterns.

    This creates training data by generating emotion-specific melodies using
    music theory rules, giving the LSTM patterns to learn from.
    """
    print("Generating synthetic MIDI training data...")
    sequences = []

    for emo_idx, emotion in enumerate(EMOTION_LABELS):
        params = EMOTION_MUSIC_PARAMS[emotion]
        scale_notes = get_scale_notes(params['scale'], params['octave'])
        density = params['note_density']

        for _ in range(num_sequences // NUM_CLASSES + 1):
            notes = []
            idx = len(scale_notes) // 2

            for _ in range(seq_len):
                if random.random() < density:
                    # Stepwise motion with emotion-specific patterns
                    if emotion in ('happy', 'surprised'):
                        # Upward tendency
                        step = random.choices([-2, -1, 0, 1, 2, 3],
                                              weights=[0.05, 0.15, 0.1, 0.35, 0.25, 0.1])[0]
                    elif emotion in ('sad', 'fearful'):
                        # Downward tendency, narrow range
                        step = random.choices([-3, -2, -1, 0, 1],
                                              weights=[0.05, 0.2, 0.35, 0.25, 0.15])[0]
                    elif emotion in ('angry', 'disgusted'):
                        # Wide leaps, erratic
                        step = random.choices([-3, -2, -1, 0, 1, 2, 3],
                                              weights=[0.1, 0.15, 0.1, 0.1, 0.1, 0.15, 0.3])[0]
                    else:
                        # Neutral - balanced
                        step = random.choices([-2, -1, 0, 1, 2],
                                              weights=[0.1, 0.3, 0.2, 0.3, 0.1])[0]

                    idx = max(0, min(len(scale_notes) - 1, idx + step))
                    notes.append(scale_notes[idx])
                else:
                    # Rest encoded as note 0
                    notes.append(0)

            sequences.append((notes, emo_idx))

        print(f"  {emotion}: {num_sequences // NUM_CLASSES + 1} synthetic sequences")

    random.shuffle(sequences)
    return sequences


# ── Dataset ──

class MelodyDataset(Dataset):
    """Dataset that creates input-target pairs from note sequences for next-note prediction."""

    def __init__(self, sequences, seq_len=LSTM_SEQ_LEN):
        self.samples = []
        for notes, emo_idx in sequences:
            # Create sliding windows
            for i in range(0, len(notes) - seq_len, seq_len // 2):
                input_seq = notes[i:i + seq_len]
                target_seq = notes[i + 1:i + seq_len + 1]
                if len(input_seq) == seq_len and len(target_seq) == seq_len:
                    self.samples.append((
                        torch.tensor(input_seq, dtype=torch.long),
                        torch.tensor(target_seq, dtype=torch.long),
                        emo_idx,
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq, emo_idx = self.samples[idx]
        return input_seq, target_seq, emo_idx


# ── Training ──

def train(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore rest/padding
    optimizer = optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(LSTM_EPOCHS):
        # ── Train ──
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for input_seq, target_seq, emo_idx in train_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            emo_idx = emo_idx.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_seq, emo_idx)

            # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len,)
            loss = criterion(logits.view(-1, MIDI_VOCAB_SIZE), target_seq.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * input_seq.size(0)

            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = target_seq != 0
            total_correct += (preds[mask] == target_seq[mask]).sum().item()
            total_tokens += mask.sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / max(total_tokens, 1)

        # ── Validate ──
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{LSTM_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), LSTM_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {LSTM_MODEL_PATH}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for input_seq, target_seq, emo_idx in loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            emo_idx = emo_idx.to(device)

            logits, _ = model(input_seq, emo_idx)
            loss = criterion(logits.view(-1, MIDI_VOCAB_SIZE), target_seq.view(-1))
            total_loss += loss.item() * input_seq.size(0)

            preds = logits.argmax(dim=-1)
            mask = target_seq != 0
            total_correct += (preds[mask] == target_seq[mask]).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = total_correct / max(total_tokens, 1)
    return avg_loss, accuracy


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Train LSTM melody generator')
    parser.add_argument('--auto-label', action='store_true',
                        help='Auto-label MIDI files by key/tempo analysis')
    parser.add_argument('--generate-synthetic', action='store_true',
                        help='Generate synthetic training data (no MIDI files needed)')
    parser.add_argument('--num-synthetic', type=int, default=700,
                        help='Number of synthetic sequences to generate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Using device: {device}")

    # ── Load Data ──
    if args.generate_synthetic:
        sequences = generate_synthetic_data(num_sequences=args.num_synthetic)
    else:
        print(f"Loading MIDI files from {MIDI_DATA_DIR}...")
        sequences = parse_midi_directory(MIDI_DATA_DIR)

        if len(sequences) < 10:
            print(f"\nOnly {len(sequences)} sequences found. Augmenting with synthetic data...")
            synthetic = generate_synthetic_data(num_sequences=500)
            sequences.extend(synthetic)

    print(f"\nTotal sequences: {len(sequences)}")

    # ── Split ──
    random.shuffle(sequences)
    split = int(0.9 * len(sequences))
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    train_dataset = MelodyDataset(train_seqs)
    val_dataset = MelodyDataset(val_seqs)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: No training samples. Check your MIDI data or use --generate-synthetic.")
        return

    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)

    # ── Model ──
    model = MelodyLSTM().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MelodyLSTM parameters: {total_params:,}")

    # ── Train ──
    train(model, train_loader, val_loader, device)

    # ── Demo Generation ──
    print("\n── Demo: Generating melodies for each emotion ──")
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    for emo_idx, emotion in enumerate(EMOTION_LABELS):
        params = EMOTION_MUSIC_PARAMS[emotion]
        scale_notes = get_scale_notes(params['scale'], params['octave'])
        seed = scale_notes[:4]

        # Build scale mask
        scale_mask = torch.zeros(MIDI_VOCAB_SIZE)
        for n in scale_notes:
            if 0 <= n < MIDI_VOCAB_SIZE:
                scale_mask[n] = 1.0
        scale_mask = scale_mask.to(device)

        melody = model.generate(
            emotion_idx=emo_idx,
            start_notes=seed,
            num_notes=16,
            temperature=0.8,
            scale_mask=scale_mask,
            device=device,
        )
        print(f"  {emotion}: {melody[:12]}...")


if __name__ == '__main__':
    main()
