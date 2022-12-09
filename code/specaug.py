import numpy as np 
import tensorflow as tf

import nlpaug.augmenter.spectrogram as nas


def freq_mask(data):
    """
    frequency masking
    """
    if (np.random.uniform(0,1) > 0.5):
        zone_l, zone_r = 0, 0.15
    else:
        zone_l, zone_r = 0.85, 1
    
    coverage = np.random.uniform(0.8, 1)
    channel_num = np.random.randint(10, 20)
    aug = nas.FrequencyMaskingAug(zone=(zone_l, zone_r), coverage = coverage, factor=(10, 20))
    
    data_squeezed = tf.squeeze(data).numpy()
    aug_data = aug.augment(data_squeezed)
    spec_aug_data = tf.expand_dims(tf.squeeze(tf.convert_to_tensor(aug_data)), -1)
    
    return spec_aug_data


def time_mask(data):
    """
    time mask 
    """
    if (np.random.uniform(0,1) > 0.5):
        zone_l, zone_r = 0, 0.15
    else:
        zone_l, zone_r = 0.85, 1
    
    coverage = np.random.uniform(0.1, 0.2)
    aug = nas.TimeMaskingAug(zone=(zone_l, zone_r), coverage = coverage)
    
    data_squeezed = tf.squeeze(data).numpy()
    aug_data = aug.substitute(data_squeezed) 
    spec_aug_data = tf.expand_dims(tf.squeeze(tf.convert_to_tensor(aug_data)), -1)
    
    return spec_aug_data


def loud(data):
    """
    Adjust loudness 
    """
    zone_l, zone_r = np.random.uniform(0,0.15), np.random.uniform(0.85, 1)
    
    coverage = np.random.uniform(0.8, 1)
    aug = nas.LoudnessAug(zone=(zone_l, zone_r), coverage=coverage, factor=(0.75, 1.25))
    
    data_squeezed = tf.squeeze(data).numpy()
    aug_data = aug.augment(data_squeezed)
    spec_aug_data = tf.expand_dims(tf.squeeze(tf.convert_to_tensor(aug_data)), -1)
    
    return spec_aug_data

    
