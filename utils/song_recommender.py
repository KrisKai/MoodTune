import pandas as pd

from config import SONGS_CSV, MOOD_IMPROVEMENT_MAP


def load_songs():
    """Load the song recommendation dataset."""
    return pd.read_csv(SONGS_CSV)


def recommend_songs(emotion, n=5):
    """Recommend songs to improve the user's mood based on detected emotion.

    Returns a DataFrame of recommended songs and the improvement label.
    """
    songs_df = load_songs()
    mapping = MOOD_IMPROVEMENT_MAP.get(emotion, MOOD_IMPROVEMENT_MAP['neutral'])

    val_min, val_max = mapping['target_valence']
    eng_min, eng_max = mapping['target_energy']
    label = mapping['label']

    # Filter songs within target valence and energy ranges
    filtered = songs_df[
        (songs_df['valence'] >= val_min) & (songs_df['valence'] <= val_max) &
        (songs_df['energy'] >= eng_min) & (songs_df['energy'] <= eng_max)
    ]

    # If not enough matches, relax the filter
    if len(filtered) < n:
        filtered = songs_df[
            (songs_df['valence'] >= val_min - 0.1) & (songs_df['valence'] <= val_max + 0.1) &
            (songs_df['energy'] >= eng_min - 0.1) & (songs_df['energy'] <= eng_max + 0.1)
        ]

    # If still not enough, just return random songs
    if len(filtered) < n:
        filtered = songs_df

    recommended = filtered.sample(n=min(n, len(filtered)))
    return recommended[['title', 'artist', 'genre', 'url']], label
