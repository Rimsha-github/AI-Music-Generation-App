#        Mid-Term Project(Music Generation)
# Developed by: Rimsha Naim and Anam Naeem
# Purpose: This project generates creative music using AI models like LSTM.
# Features: Upload a MIDI file, train a model, and generate unique music tracks.

import os
import streamlit as st
import pretty_midi
import numpy as np
import tensorflow as tf
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore
from io import BytesIO
import matplotlib.pyplot as plt

# Page Configuration for theme and layout
st.set_page_config(
    page_title="AI Music Generator 🎶",
    layout="centered",
    page_icon="🎵",
    initial_sidebar_state="expanded",
)

# App Title
st.title("🎶 AI-Powered Music Generation App")

# Sidebar Settings
st.sidebar.header("Settings")
sequence_length = st.sidebar.slider("Sequence Length", 10, 200, 50)
generate_length = st.sidebar.slider("Generated Music Length", 10, 500, 100)
epochs = st.sidebar.slider("Training Epochs", 1, 50, 5)

# Extracting Notes from MIDI File
def extract_notes_from_midi(file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append((note.pitch, note.velocity / 127.0, note.end - note.start))  # Normalize velocity
        return np.array(notes) if notes else np.array([])
    except Exception as e:
        st.error(f"Error processing MIDI file: {e}")
        return np.array([])

# Preparing Sequences for Training
def prepare_sequences(notes, sequence_length):
    X, y = [], []
    for i in range(len(notes) - sequence_length):
        X.append(notes[i:i + sequence_length])
        y.append(notes[i + sequence_length])
    return np.array(X), np.array(y)

# Building LSTM Model
def build_music_model(sequence_length, input_dim):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, input_dim)),
        LSTM(128),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Generating Music
def generate_music(model, seed_sequence, length):
    generated_sequence = []
    current_sequence = seed_sequence.copy()

    for _ in range(length):
        prediction = model.predict(np.expand_dims(current_sequence, axis=0), verbose=0)
        generated_sequence.append(prediction.flatten())
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = prediction

    return np.array(generated_sequence)

# Saving Generated Music as MIDI
def save_generated_midi(sequence, output_file):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    start_time = 0.0

    for pitch, velocity, duration in sequence:
        pitch = int(min(max(pitch, 0), 127))
        velocity = int(min(max(velocity * 127, 1), 127))  # Scale velocity back to 0-127
        duration = max(duration, 0.1)  # Ensure minimum valid duration
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=start_time + duration
        )
        piano.notes.append(note)
        start_time += duration

    midi.instruments.append(piano)
    midi.write(output_file)

# Plotting MIDI Notes Visualization
def plot_notes(notes):
    if notes.size > 0:
        pitches = notes[:, 0]
        durations = notes[:, 2]
        plt.figure(figsize=(10, 4))
        plt.scatter(range(len(pitches)), pitches, c=durations, cmap='viridis')
        plt.colorbar(label='Note Duration')
        plt.xlabel('Note Index')
        plt.ylabel('Pitch')
        plt.title('MIDI Notes Visualization')
        st.pyplot(plt)
    else:
        st.warning("No notes to visualize.")

# MIDI File Upload
uploaded_file = st.file_uploader("Upload a MIDI file", type=["mid", "midi"])

if uploaded_file is not None:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    notes = extract_notes_from_midi(file_path)

    if notes.size > 0:
        st.success("MIDI file successfully processed!")
        st.write("Extracted Notes (First 10):", notes[:10])
        plot_notes(notes)

        if len(notes) > sequence_length:
            X, y = prepare_sequences(notes, sequence_length)
            input_dim = X.shape[2]

            model = build_music_model(sequence_length, input_dim)
            st.write("Training the model, please wait...")

            with st.spinner(f"Training for {epochs} epoch(s)..."):
                model.fit(X, y, epochs=epochs, batch_size=8, verbose=1)

            seed_sequence = X[np.random.randint(0, len(X))]
            generated_music = generate_music(model, seed_sequence, generate_length)

            midi_file = "generated_song.mid"
            save_generated_midi(generated_music, midi_file)

            with open(midi_file, "rb") as f:
                st.download_button("Download Generated Music 🎵", f, file_name="generated_song.mid")
        else:
            st.error("The uploaded MIDI file is too short. Please upload a longer file.")
    else:
        st.error("No notes found in the uploaded MIDI file. Please upload a valid file.")
else:
    st.info("Upload a MIDI file to start.")

# Footer
st.markdown("---")
st.caption("🎼 Developed with ❤️ for creative music generation")
