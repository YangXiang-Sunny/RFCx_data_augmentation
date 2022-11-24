import numpy as np 
import librosa


def noise_injection(data, sample_rate):
    # noize_factor random between 0.001 to 0.005
    noise_factor = np.random.uniform(0.001, 0.005)
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift_time(data, sampling_rate):
    #shift time between 0.1 to 1 second
    shift_max= np.random.uniform(0.1, 1)
    shift = int(np.round(sampling_rate * shift_max)) # np.random.randint(
    direction = np.random.randint(0, 2)
    if direction == 1:
        shift = -shift
    augmented_data = np.roll(data, shift)
    return augmented_data

def change_pitch(data, sample_rate):
    n_step = np.random.uniform(-2,2)
    augmented_data = librosa.effects.pitch_shift(y=data, sr=sample_rate, n_steps=n_step)
    return augmented_data

def change_speed(data, sample_rate):
    speed_factor= np.random.uniform(0.75, 1.25) 
    augmented_data = librosa.effects.time_stretch(data, speed_factor)
    return augmented_data