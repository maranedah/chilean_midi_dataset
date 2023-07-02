import librosa
import librosa.display
import numpy as np

desired_sample_rate = 16000
audio, sr = librosa.load("audio.mp3", sr=desired_sample_rate)  # audio_samples

window_size = int(sr * 0.025)  # 25 milliseconds window size
hop_length = int(sr * 0.01)  # 10 milliseconds stride
n_mels = 80  # Number of Mel frequency channels

mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=window_size,
    hop_length=hop_length,
    n_mels=n_mels,
)
log_mel_spec = librosa.amplitude_to_db(mel_spec)  # Convert to logarithmic scale
normalized_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)
normalized_spec = np.clip(normalized_spec, -1, 1)
breakpoint()
