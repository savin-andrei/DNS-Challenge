"""
@author: chkarada
"""

# Note: This single process audio synthesizer will attempt to use each clean
# speech sourcefile once, as it does not randomly sample from these files

import os
import sys
import glob
import argparse
import ast
import configparser as CP
from random import shuffle
import random
import csv
import time

import librosa
import numpy as np
from scipy import signal
from audiolib import audioread, audiowrite, segmental_snr_mixer, activitydetector, is_clipped, add_clipping, normalize
import utils
import telephony_augment

import pandas as pd
from pathlib import Path
from scipy.io import wavfile
import torch
import torchaudio

MAXTRIES = 50
MAXFILELEN = 100

np.random.seed(5)
random.seed(5)

_RESAMPLERS = {}


def _resample_audio(audio, fs_in, fs_out):
    if fs_in == fs_out:
        return audio
    key = (fs_in, fs_out)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(
            orig_freq=fs_in, new_freq=fs_out
        )
    resampler = _RESAMPLERS[key]
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    with torch.no_grad():
        out = resampler(audio_tensor).squeeze(0).cpu().numpy()
    return out

def add_pyreverb(clean_speech, rir):
    
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0 : clean_speech.shape[0]]

    return reverb_speech


def _match_length(audio, target_len):
    if len(audio) > target_len:
        return audio[:target_len]
    if len(audio) < target_len:
        pad = np.zeros(target_len - len(audio))
        return np.append(audio, pad)
    return audio


def build_telephony_noise_mask(params, audio_samples_length):
    """Build telephony noise mask from real telephony noise clips."""
    if 'telephony_noise_files' not in params or not params['telephony_noise_files']:
        return None
    source_files = params['telephony_noise_files']
    remaining_length = audio_samples_length
    output_audio = np.zeros(0)

    tries_left = MAXTRIES
    idx = np.random.randint(0, np.size(source_files))
    while remaining_length > 0 and tries_left > 0:
        idx = (idx + 1) % np.size(source_files)
        input_audio, fs_input = audioread(source_files[idx])
        if input_audio is None or len(input_audio) == 0:
            tries_left -= 1
            continue
        if fs_input != params['fs']:
            input_audio = librosa.resample(input_audio, fs_input, params['fs'])
        if len(input_audio) > remaining_length:
            idx_seg = np.random.randint(0, len(input_audio) - remaining_length)
            input_audio = input_audio[idx_seg:idx_seg + remaining_length]
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

    if len(output_audio) < audio_samples_length:
        output_audio = _match_length(output_audio, audio_samples_length)
    return output_audio

def build_audio(is_clean, params, index, audio_samples_length=-1):
    '''Construct an audio signal from source files'''

    fs_output = params['fs']
    silence_length = params['silence_length']
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length']*params['fs'])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    perf = params.get('perf_detail', None)
    perf_trace = params.get('perf_trace', False)
    t_setup_start = time.perf_counter()

    t0 = time.perf_counter()
    if is_clean:
        source_files = params['cleanfilenames']
        idx = index
    else:
        if 'noisefilenames' in params.keys():
            source_files = params['noisefilenames']
            idx = index
        # if noise files are organized into individual subdirectories, pick a directory randomly
        else:
            noisedirs = params['noisedirs']
            # pick a noise category randomly
            idx_n_dir = np.random.randint(0, np.size(noisedirs))
            source_files = glob.glob(os.path.join(noisedirs[idx_n_dir], 
                                                  params['audioformat']))
            shuffle(source_files)
            # pick a noise source file index randomly
            idx = np.random.randint(0, np.size(source_files))
    source_select_s = time.perf_counter() - t0

    # initialize silence
    t0 = time.perf_counter()
    silence = np.zeros(int(fs_output*silence_length))
    silence_init_s = time.perf_counter() - t0
    setup_total_s = time.perf_counter() - t_setup_start
    if perf is not None:
        perf["source_select_s"] += source_select_s
        perf["silence_init_s"] += silence_init_s
        perf["build_setup_s"] += setup_total_s
    if perf_trace:
        print(
            "Perf build setup: clean={} source_select={:.6f} silence_init={:.6f} "
            "setup_total={:.6f}".format(
                int(is_clean), source_select_s, silence_init_s, setup_total_s
            )
        )

    t_build_start = time.perf_counter()
    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    iter_idx = 0

    def _trace_build_iter(event, file_path, select_s, read_s, empty_check_s,
                          resample_s, crop_s, clip_s, concat_s, silence_s,
                          perf_s, iter_total_s, audio_len, remaining_len):
        if not perf_trace:
            return
        file_name = os.path.basename(file_path)
        comp_sum = (
            select_s + read_s + empty_check_s + resample_s + crop_s + clip_s +
            concat_s + silence_s + perf_s
        )
        gap_s = iter_total_s - comp_sum
        print(
            "Perf build iter: clean={} iter={} event={} file={} select={:.6f} "
            "read={:.6f} empty_chk={:.6f} resample={:.6f} crop={:.6f} "
            "clip={:.6f} concat={:.6f} silence={:.6f} perf={:.6f} "
            "gap={:.6f} iter_total={:.6f} audio_len={} remaining={}".format(
                int(is_clean), iter_idx, event, file_name, select_s, read_s,
                empty_check_s, resample_s, crop_s, clip_s, concat_s, silence_s,
                perf_s, gap_s, iter_total_s, audio_len, remaining_len
            )
        )
    while remaining_length > 0 and tries_left > 0:
        iter_idx += 1
        t_iter_start = time.perf_counter()
        t_attempt_start = time.perf_counter()
        select_s = 0.0
        read_s = 0.0
        empty_check_s = 0.0
        resample_s = 0.0
        crop_s = 0.0
        clip_s = 0.0
        concat_s = 0.0
        silence_s = 0.0
        perf_s = 0.0

        # read next audio file and resample if necessary

        t0 = time.perf_counter()
        idx = (idx + 1) % len(source_files)
        file_path = source_files[idx]
        select_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        input_audio, fs_input = audioread(file_path)
        read_s = time.perf_counter() - t0
        if perf is not None:
            perf["read_s"] += read_s
        t0 = time.perf_counter()
        is_empty = input_audio is None or len(input_audio) == 0
        empty_check_s = time.perf_counter() - t0
        if is_empty:
            if perf is not None:
                perf["read_empty"] += 1
            sys.stderr.write("WARNING: Empty or unreadable audio: %s\n" % file_path)
            _trace_build_iter(
                "empty", file_path, select_s, read_s, empty_check_s, resample_s,
                crop_s, clip_s, concat_s, silence_s, perf_s,
                iter_total_s=time.perf_counter() - t_iter_start, audio_len=0,
                remaining_len=remaining_length
            )
            continue
        t0 = time.perf_counter()
        if fs_input != fs_output:
            input_audio = _resample_audio(input_audio, fs_input, fs_output)
            if perf is not None:
                perf["resample_s"] += time.perf_counter() - t0
        resample_s = time.perf_counter() - t0

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        t0 = time.perf_counter()
        if len(input_audio) > remaining_length and (not is_clean or not params['is_test_set']):
            idx_seg = np.random.randint(0, len(input_audio)-remaining_length)
            input_audio = input_audio[idx_seg:idx_seg+remaining_length]
            crop_s = time.perf_counter() - t0
            if perf is not None:
                perf["crop_s"] += crop_s
        else:
            crop_s = time.perf_counter() - t0

        # check for clipping, and if found move onto next file
        t0 = time.perf_counter()
        clipped = is_clipped(input_audio)
        clip_s = time.perf_counter() - t0
        if clipped:
            if perf is not None:
                perf["clipped"] += 1
            clipped_files.append(file_path)
            tries_left -= 1
            _trace_build_iter(
                "clipped", file_path, select_s, read_s, empty_check_s, resample_s,
                crop_s, clip_s, concat_s, silence_s, perf_s,
                time.perf_counter() - t_iter_start, len(input_audio),
                remaining_length
            )
            continue

        # concatenate current input audio to output audio stream
        t0 = time.perf_counter()
        files_used.append(file_path)
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)
        concat_s = time.perf_counter() - t0
        if perf is not None:
            perf["concat_s"] += concat_s

        # add some silence if we have not reached desired audio length
        t0 = time.perf_counter()
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len
            silence_s = time.perf_counter() - t0
            if perf is not None:
                perf["silence_s"] += silence_s
        else:
            silence_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        if perf is not None:
            perf["attempt_s"] += time.perf_counter() - t_attempt_start
            perf["attempts"] += 1
        perf_s = time.perf_counter() - t0
        _trace_build_iter(
            "ok", file_path, select_s, read_s, empty_check_s, resample_s,
            crop_s, clip_s, concat_s, silence_s, perf_s,
            time.perf_counter() - t_iter_start, len(input_audio),
            remaining_length
        )

    if tries_left == 0 and not is_clean and 'noisedirs' in params.keys():
        print("There are not enough non-clipped files in the " + noisedirs[idx_n_dir] + \
              " directory to complete the audio build")
        return [], [], clipped_files, idx

    if perf is not None:
        perf["build_total_s"] += time.perf_counter() - t_build_start
        if is_clean:
            perf["build_files_clean"] += len(files_used)
        else:
            perf["build_files_noise"] += len(files_used)

    return output_audio, files_used, clipped_files, idx


def gen_audio(is_clean, params, index, audio_samples_length=-1):
    '''Calls build_audio() to get an audio signal, and verify that it meets the
       activity threshold'''

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length']*params['fs'])
    if is_clean:
        activity_threshold = params['clean_activity_threshold']
    else:
        activity_threshold = params['noise_activity_threshold']

    t_gen_start = time.perf_counter()
    perf = params.get('perf_detail', None)
    perf_trace = params.get('perf_trace', False)
    attempts = 0
    build_total_s = 0.0
    activity_total_s = 0.0
    attempt_total_s = 0.0
    while True:
        attempts += 1
        t_attempt_start = time.perf_counter()
        t0 = time.perf_counter()
        audio, source_files, new_clipped_files, index = \
            build_audio(is_clean, params, index, audio_samples_length)
        build_s = time.perf_counter() - t0
        activity_s = 0.0
        percactive = None

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            if perf is not None:
                if is_clean:
                    perf["short_clean"] += 1
                else:
                    perf["short_noise"] += 1
            attempt_total = time.perf_counter() - t_attempt_start
            build_total_s += build_s
            attempt_total_s += attempt_total
            if perf_trace:
                other_s = attempt_total - (build_s + activity_s)
                print(
                    "Perf gen attempt: clean={} attempt={} event=short_audio "
                    "build={:.6f} activity={:.6f} other={:.6f} total={:.6f}".format(
                        int(is_clean), attempts, build_s, activity_s, other_s,
                        attempt_total
                    )
                )
            continue

        if activity_threshold == 0.0:
            attempt_total = time.perf_counter() - t_attempt_start
            build_total_s += build_s
            attempt_total_s += attempt_total
            if perf_trace:
                other_s = attempt_total - (build_s + activity_s)
                print(
                    "Perf gen attempt: clean={} attempt={} event=skip_activity "
                    "build={:.6f} activity={:.6f} other={:.6f} total={:.6f}".format(
                        int(is_clean), attempts, build_s, activity_s, other_s,
                        attempt_total
                    )
                )
            break

        t0 = time.perf_counter()
        percactive = activitydetector(audio=audio)
        activity_s = time.perf_counter() - t0
        if perf is not None:
            perf["activity_s"] += activity_s
            perf["activity_calls"] += 1
        if percactive > activity_threshold:
            attempt_total = time.perf_counter() - t_attempt_start
            build_total_s += build_s
            activity_total_s += activity_s
            attempt_total_s += attempt_total
            if perf_trace:
                other_s = attempt_total - (build_s + activity_s)
                print(
                    "Perf gen attempt: clean={} attempt={} event=accepted "
                    "build={:.6f} activity={:.6f} other={:.6f} total={:.6f} "
                    "percactive={:.6f}".format(
                        int(is_clean), attempts, build_s, activity_s, other_s,
                        attempt_total, percactive
                    )
                )
            break
        else:
            low_activity_files += source_files
            if perf is not None:
                if is_clean:
                    perf["low_activity_clean"] += 1
                else:
                    perf["low_activity_noise"] += 1
            attempt_total = time.perf_counter() - t_attempt_start
            build_total_s += build_s
            activity_total_s += activity_s
            attempt_total_s += attempt_total
            if perf_trace:
                other_s = attempt_total - (build_s + activity_s)
                print(
                    "Perf gen attempt: clean={} attempt={} event=low_activity "
                    "build={:.6f} activity={:.6f} other={:.6f} total={:.6f} "
                    "percactive={:.6f}".format(
                        int(is_clean), attempts, build_s, activity_s, other_s,
                        attempt_total, percactive
                    )
                )

    if perf is not None:
        perf["gen_total_s"] += time.perf_counter() - t_gen_start
        perf["gen_attempts"] += attempts
        if is_clean:
            perf["gen_attempts_clean"] += attempts
        else:
            perf["gen_attempts_noise"] += attempts
    if perf_trace:
        other_total_s = attempt_total_s - (build_total_s + activity_total_s)
        print(
            "Perf gen summary: clean={} attempts={} build_total={:.6f} "
            "activity_total={:.6f} other_total={:.6f} total={:.6f}".format(
                int(is_clean), attempts, build_total_s, activity_total_s,
                other_total_s, attempt_total_s
            )
        )
    return audio, source_files, clipped_files, low_activity_files, index


def main_gen(params):
    '''Calls gen_audio() to generate the audio signals, verifies that they meet
       the requirements, and writes the files to storage'''

    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    clean_index = 0
    noise_index = 0
    file_num = params['fileindex_start']
    train_rows = []
    eval_rows = []
    perf = {
        "clean_gen_s": 0.0,
        "noise_gen_s": 0.0,
        "rir_s": 0.0,
        "telephony_clean_s": 0.0,
        "telephony_noise_s": 0.0,
        "telephony_post_s": 0.0,
        "mix_s": 0.0,
        "telephony_mask_s": 0.0,
        "write_s": 0.0,
        "files": 0,
    }
    perf_detail = {
        "read_s": 0.0,
        "resample_s": 0.0,
        "crop_s": 0.0,
        "concat_s": 0.0,
        "silence_s": 0.0,
        "activity_s": 0.0,
        "build_total_s": 0.0,
        "gen_total_s": 0.0,
        "gen_attempts": 0,
        "gen_attempts_clean": 0,
        "gen_attempts_noise": 0,
        "attempt_s": 0.0,
        "attempts": 0,
        "source_select_s": 0.0,
        "silence_init_s": 0.0,
        "build_setup_s": 0.0,
        "read_empty": 0,
        "clipped": 0,
        "short_clean": 0,
        "short_noise": 0,
        "low_activity_clean": 0,
        "low_activity_noise": 0,
        "build_files_clean": 0,
        "build_files_noise": 0,
        "activity_calls": 0,
    }
    params["perf_detail"] = perf_detail
    perf_interval = params.get('perf_interval', 0)
    perf_trace = params.get('perf_trace', False)
    if perf_trace:
        perf_interval = 0

    while file_num <= params['fileindex_end']:
        # generate clean speech
        t0 = time.perf_counter()
        clean, clean_sf, clean_cf, clean_laf, clean_index = \
            gen_audio(True, params, clean_index)
        perf["clean_gen_s"] += time.perf_counter() - t0

        if params.get('use_rir', True) and params.get('myrir'):
            t0 = time.perf_counter()
            # add reverb with selected RIR
            rir_index = random.randint(0, len(params['myrir']) - 1)

            rir_rel = params['myrir'][rir_index]
            if os.path.isabs(rir_rel):
                my_rir = os.path.normpath(rir_rel)
            else:
                prefix = os.path.normpath(os.path.join('datasets', 'impulse_responses'))
                norm_rir = os.path.normpath(rir_rel)
                if norm_rir.startswith(prefix + os.sep) or norm_rir == prefix:
                    norm_rir = norm_rir[len(prefix):].lstrip(os.sep)
                my_rir = os.path.normpath(os.path.join(params['rir_base_dir'], norm_rir))
            (fs_rir, samples_rir) = wavfile.read(my_rir)

            my_channel = int(params['mychannel'][rir_index])

            if samples_rir.ndim == 1:
                samples_rir_ch = np.array(samples_rir)
            elif my_channel > 1:
                samples_rir_ch = samples_rir[:, my_channel - 1]
            else:
                samples_rir_ch = samples_rir[:, my_channel - 1]
                #print(samples_rir.shape)
                #print(my_channel)

            clean = add_pyreverb(clean, samples_rir_ch)
            perf["rir_s"] += time.perf_counter() - t0
        clean_target_len = len(clean)
        if params.get('telephony') and params['telephony'].get('enable') \
           and params['telephony'].get('apply_to_clean', True):
            t0 = time.perf_counter()
            clean = telephony_augment.apply_telephony_augmentation(
                clean, params['fs'], params['telephony'], rng=random
            )
            clean = _match_length(clean, clean_target_len)
            perf["telephony_clean_s"] += time.perf_counter() - t0

        # generate noise
        t0 = time.perf_counter()
        noise, noise_sf, noise_cf, noise_laf, noise_index = \
            gen_audio(False, params, noise_index, clean_target_len)
        perf["noise_gen_s"] += time.perf_counter() - t0
        if params.get('telephony') and params['telephony'].get('enable') \
           and params['telephony'].get('apply_to_noise', True):
            t0 = time.perf_counter()
            noise = telephony_augment.apply_telephony_augmentation(
                noise, params['fs'], params['telephony'], rng=random
            )
            noise = _match_length(noise, clean_target_len)
            perf["telephony_noise_s"] += time.perf_counter() - t0

        clean_clipped_files += clean_cf
        clean_low_activity_files += clean_laf
        noise_clipped_files += noise_cf
        noise_low_activity_files += noise_laf

        # get rir files and config

        # mix clean speech and noise
        # if specified, use specified SNR value
        if not params['randomize_snr']:
            snr = params['snr']
        # use a randomly sampled SNR value between the specified bounds
        else:
            snr = np.random.randint(params['snr_lower'], params['snr_upper'])

        t0 = time.perf_counter()
        clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(params=params,
                                                                  clean=clean,
                                                                  noise=noise,
                                                                  snr=snr)
        perf["mix_s"] += time.perf_counter() - t0
        # Uncomment the below lines if you need segmental SNR and comment the above lines using snr_mixer
        #clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(params=params, 
        #                                                         clean=clean, 
        #                                                          noise=noise, 
        #                                                         snr=snr)

        post_mix_applied = False
        t_post_start = None
        if params.get('telephony') and params['telephony'].get('enable'):
            t_post_start = time.perf_counter()
            if params['telephony'].get('apply_post_mix_clean', False):
                clean_snr = telephony_augment.apply_telephony_augmentation(
                    clean_snr, params['fs'], params['telephony'], rng=random
                )
                post_mix_applied = True
            if params['telephony'].get('apply_post_mix_noise', False):
                noise_snr = telephony_augment.apply_telephony_augmentation(
                    noise_snr, params['fs'], params['telephony'], rng=random
                )
                post_mix_applied = True
            if params['telephony'].get('apply_post_mix_noisy', False):
                noisy_snr = telephony_augment.apply_telephony_augmentation(
                    noisy_snr, params['fs'], params['telephony'], rng=random
                )
                post_mix_applied = True
        if post_mix_applied:
            clean_snr = normalize(clean_snr, target_level=target_level)
            noise_snr = normalize(noise_snr, target_level=target_level)
            noisy_snr = normalize(noisy_snr, target_level=target_level)
            max_amp = max(abs(noisy_snr))
            if max_amp >= 0.99:
                scale = max_amp / 0.99
                clean_snr = clean_snr / scale
                noise_snr = noise_snr / scale
                noisy_snr = noisy_snr / scale
            if t_post_start is not None:
                perf["telephony_post_s"] += time.perf_counter() - t_post_start

        telephony_mask = None
        if params.get('telephony_noise') and params['telephony_noise'].get('enable'):
            t0 = time.perf_counter()
            telephony_mask = build_telephony_noise_mask(
                params, len(clean_snr)
            )
            if telephony_mask is not None:
                telephony_mask = normalize(
                    telephony_mask, target_level=params['telephony_noise']['level_db']
                )
                clean_snr = clean_snr + telephony_mask
                noisy_snr = noisy_snr + telephony_mask
                max_amp = max(abs(noisy_snr))
                if max_amp >= 0.99:
                    scale = max_amp / 0.99
                    clean_snr = clean_snr / scale
                    noisy_snr = noisy_snr / scale
            perf["telephony_mask_s"] += time.perf_counter() - t0
        # unexpected clipping
        if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
            print("Warning: File #" + str(file_num) + " has unexpected clipping, " + \
                  "returning without writing audio to disk")
            continue

        clean_source_files += clean_sf
        noise_source_files += noise_sf

        # write resultant audio streams to files
        hyphen = '-'
        clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_sf]
        clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
        noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_sf]
        noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]

        noisyfilename = clean_files_joined + '_' + noise_files_joined + '_snr' + \
                        str(snr) + '_tl' + str(target_level) + '_fileid_' + str(file_num) + '.wav'
        cleanfilename = 'clean_fileid_'+str(file_num)+'.wav'
        noisefilename = 'noise_fileid_'+str(file_num)+'.wav'

        is_eval = False
        if params.get('eval_stride', 0):
            eval_idx = file_num - params['fileindex_start'] + 1
            if eval_idx % params['eval_stride'] == 0:
                is_eval = True

        if is_eval:
            noisypath = os.path.join(params['eval_noisyspeech_dir'], noisyfilename)
            cleanpath = os.path.join(params['eval_clean_proc_dir'], cleanfilename)
            noisepath = os.path.join(params['eval_noise_proc_dir'], noisefilename)
        else:
            noisypath = os.path.join(params['noisyspeech_dir'], noisyfilename)
            cleanpath = os.path.join(params['clean_proc_dir'], cleanfilename)
            noisepath = os.path.join(params['noise_proc_dir'], noisefilename)

        audio_signals = [noisy_snr, clean_snr, noise_snr]
        file_paths = [noisypath, cleanpath, noisepath]

        file_num += 1
        t0 = time.perf_counter()
        for i in range(len(audio_signals)):
            try:
                audiowrite(file_paths[i], audio_signals[i], params['fs'])
            except Exception as e:
                print(str(e))
        perf["write_s"] += time.perf_counter() - t0

        if is_eval:
            eval_rows.append((cleanpath, noisepath, noisypath))
        else:
            train_rows.append((cleanpath, noisepath, noisypath))
        perf["files"] += 1
        if perf_interval and perf["files"] % perf_interval == 0:
            avg = {k: perf[k] / max(perf["files"], 1) for k in perf if k.endswith("_s")}
            avg_detail = {k: perf_detail[k] / max(perf["files"], 1) for k in perf_detail}
            print(
                "Perf avg (s/file): clean_gen={:.3f} noise_gen={:.3f} rir={:.3f} "
                "tele_clean={:.3f} tele_noise={:.3f} tele_post={:.3f} mix={:.3f} "
                "tele_mask={:.3f} write={:.3f}".format(
                    avg["clean_gen_s"], avg["noise_gen_s"], avg["rir_s"],
                    avg["telephony_clean_s"], avg["telephony_noise_s"],
                    avg["telephony_post_s"], avg["mix_s"], avg["telephony_mask_s"],
                    avg["write_s"]
                )
            )
            attempt_avg = avg_detail["attempt_s"] / max(avg_detail["attempts"], 1)
            print(
                "Perf build_audio avg (s/file): read={:.3f} resample={:.3f} crop={:.3f} "
                "concat={:.3f} silence={:.3f} activity={:.3f} build_total={:.3f} "
                "gen_total={:.3f} gen_attempts={:.2f} read_attempts={:.2f} "
                "attempt_avg={:.3f}".format(
                    avg_detail["read_s"], avg_detail["resample_s"], avg_detail["crop_s"],
                    avg_detail["concat_s"], avg_detail["silence_s"], avg_detail["activity_s"],
                    avg_detail["build_total_s"], avg_detail["gen_total_s"],
                    avg_detail["gen_attempts"], avg_detail["attempts"], attempt_avg
                )
            )
            clean_files_per_build = perf_detail["build_files_clean"] / max(
                perf_detail["gen_attempts_clean"], 1
            )
            noise_files_per_build = perf_detail["build_files_noise"] / max(
                perf_detail["gen_attempts_noise"], 1
            )
            activity_call_avg = perf_detail["activity_s"] / max(
                perf_detail["activity_calls"], 1
            )
            print(
                "Perf gen counts avg (per file): gen_clean={:.2f} gen_noise={:.2f} "
                "empty_reads={:.2f} clipped_reads={:.2f} short_clean={:.2f} "
                "short_noise={:.2f} low_act_clean={:.2f} low_act_noise={:.2f}".format(
                    avg_detail["gen_attempts_clean"], avg_detail["gen_attempts_noise"],
                    avg_detail["read_empty"], avg_detail["clipped"],
                    avg_detail["short_clean"], avg_detail["short_noise"],
                    avg_detail["low_activity_clean"], avg_detail["low_activity_noise"]
                )
            )
            print(
                "Perf gen build avg: clean_files/build={:.2f} noise_files/build={:.2f} "
                "activity_call_avg={:.3f}".format(
                    clean_files_per_build, noise_files_per_build, activity_call_avg
                )
            )

    return clean_source_files, clean_clipped_files, clean_low_activity_files, \
           noise_source_files, noise_clipped_files, noise_low_activity_files, \
           train_rows, eval_rows, perf


def main_body():
    '''Main body of this file'''

    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument('--cfg', default='noisyspeech_synthesizer.cfg',
                        help='Read noisyspeech_synthesizer.cfg for all the details')
    parser.add_argument('--cfg_str', type=str, default='noisy_speech')
    args = parser.parse_args()

    params = dict()
    params['args'] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f'No configuration file as [{cfgpath}]'

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params['cfg'] = cfg._sections[args.cfg_str]
    cfg = params['cfg']

    clean_dir = os.path.join(os.path.dirname(__file__), 'datasets/clean')
    if cfg.get('speech_dir', 'None') != 'None':
        clean_dir = cfg.get('speech_dir')
    elif cfg.get('speech_dir_root', 'None') != 'None':
        clean_dir = cfg.get('speech_dir_root')
    if not os.path.exists(clean_dir):
        assert False, ('Clean speech data is required')

    noise_dir = os.path.join(os.path.dirname(__file__), 'datasets/noise')

    if cfg['noise_dir'] != 'None':
        noise_dir = cfg['noise_dir']
    if not os.path.exists:
        assert False, ('Noise data is required')

    params['fs'] = int(cfg['sampling_rate'])
    params['audioformat'] = cfg['audioformat']
    params['audio_length'] = float(cfg['audio_length'])
    params['silence_length'] = float(cfg['silence_length'])
    params['total_hours'] = float(cfg['total_hours'])
    
    # clean singing speech
    params['use_singing_data'] = int(cfg['use_singing_data'])
    params['clean_singing'] = str(cfg['clean_singing'])
    params['singing_choice'] = int(cfg['singing_choice'])

    # clean emotional speech
    params['use_emotion_data'] = int(cfg['use_emotion_data'])
    params['clean_emotion'] = str(cfg['clean_emotion'])
    
    # clean mandarin speech
    params['use_mandarin_data'] = int(cfg['use_mandarin_data'])
    params['clean_mandarin'] = str(cfg['clean_mandarin'])
    
    # rir
    params['use_rir'] = utils.str2bool(cfg.get('use_rir', 'True'))
    params['rir_choice'] = int(cfg['rir_choice'])
    params['lower_t60'] = float(cfg['lower_t60'])
    params['upper_t60'] = float(cfg['upper_t60'])
    params['rir_table_csv'] = str(cfg['rir_table_csv'])
    params['clean_speech_t60_csv'] = str(cfg['clean_speech_t60_csv'])
    params['rir_base_dir'] = cfg.get('rir_base_dir',
                                     os.path.join('datasets', 'impulse_responses'))

    if cfg['fileindex_start'] != 'None' and cfg['fileindex_end'] != 'None':
        params['num_files'] = int(cfg['fileindex_end'])-int(cfg['fileindex_start'])
        params['fileindex_start'] = int(cfg['fileindex_start'])
        params['fileindex_end'] = int(cfg['fileindex_end'])
    else:
        params['num_files'] = int((params['total_hours']*60*60)/params['audio_length'])
        params['fileindex_start'] = 0
        params['fileindex_end'] = params['num_files']

    print('Number of files to be synthesized:', params['num_files'])
    
    params['is_test_set'] = utils.str2bool(cfg['is_test_set'])
    params['clean_activity_threshold'] = float(cfg['clean_activity_threshold'])
    params['noise_activity_threshold'] = float(cfg['noise_activity_threshold'])
    params['snr_lower'] = int(cfg['snr_lower'])
    params['snr_upper'] = int(cfg['snr_upper'])
    
    params['randomize_snr'] = utils.str2bool(cfg['randomize_snr'])
    params['target_level_lower'] = int(cfg['target_level_lower'])
    params['target_level_upper'] = int(cfg['target_level_upper'])
    
    if 'snr' in cfg.keys():
        params['snr'] = int(cfg['snr'])
    else:
        params['snr'] = int((params['snr_lower'] + params['snr_upper'])/2)

    params['noisyspeech_dir'] = utils.get_dir(cfg, 'noisy_destination', 'noisy')
    params['clean_proc_dir'] = utils.get_dir(cfg, 'clean_destination', 'clean')
    params['noise_proc_dir'] = utils.get_dir(cfg, 'noise_destination', 'noise')
    params['perf_interval'] = int(cfg.get('perf_interval', 0))
    params['perf_trace'] = utils.str2bool(cfg.get('perf_trace', 'False'))
    params['eval_stride'] = int(cfg.get('eval_stride', 0))
    if params['eval_stride'] > 0:
        params['eval_noisyspeech_dir'] = utils.get_dir(cfg, 'eval_noisy_destination', 'eval_noisy')
        params['eval_clean_proc_dir'] = utils.get_dir(cfg, 'eval_clean_destination', 'eval_clean')
        params['eval_noise_proc_dir'] = utils.get_dir(cfg, 'eval_noise_destination', 'eval_noise')

    def _cfg_bool(key, default=False):
        if key not in cfg:
            return default
        return utils.str2bool(cfg[key])

    def _cfg_float(key, default):
        if key not in cfg:
            return default
        return float(cfg[key])

    def _cfg_int(key, default):
        if key not in cfg:
            return default
        return int(cfg[key])

    params['telephony'] = {
        'enable': _cfg_bool('telephony_enable', False),
        'apply_to_clean': _cfg_bool('telephony_apply_to_clean', True),
        'apply_to_noise': _cfg_bool('telephony_apply_to_noise', True),
        'apply_post_mix_clean': _cfg_bool('telephony_apply_post_mix_clean', False),
        'apply_post_mix_noise': _cfg_bool('telephony_apply_post_mix_noise', False),
        'apply_post_mix_noisy': _cfg_bool('telephony_apply_post_mix_noisy', False),
        'codec_backend': cfg.get('telephony_codec_backend', 'internal'),
        'codec_mix': telephony_augment.parse_codec_mix(
            cfg.get('telephony_codec_mix', 'amr_nb:0.75,amr_wb:0.2,alaw:0.05')
        ),
        'band_low_nb': _cfg_float('telephony_band_low_nb', 300.0),
        'band_high_nb': _cfg_float('telephony_band_high_nb', 3400.0),
        'band_low_wb': _cfg_float('telephony_band_low_wb', 50.0),
        'band_high_wb': _cfg_float('telephony_band_high_wb', 7000.0),
        'nb_target_sr': _cfg_int('telephony_nb_target_sr', 8000),
        'wb_target_sr': _cfg_int('telephony_wb_target_sr', params['fs']),
        'quant_bits_nb': _cfg_int('telephony_quant_bits_nb', 8),
        'quant_bits_wb': _cfg_int('telephony_quant_bits_wb', 10),
        'gain_variation_db': _cfg_float('telephony_gain_variation_db', 0.0),
        'gain_variation_segment_s': _cfg_float('telephony_gain_variation_segment_s', 1.0),
    }

    params['telephony_noise'] = {
        'enable': _cfg_bool('telephony_noise_enable', False),
        'dir': cfg.get('telephony_noise_dir', 'None'),
        'level_db': _cfg_float('telephony_noise_level_db', -40.0),
    }
    if params['telephony_noise']['enable']:
        noise_dir = params['telephony_noise']['dir']
        if noise_dir == 'None' or not os.path.exists(noise_dir):
            raise ValueError('telephony_noise_dir is required and must exist')
        telephony_noise_files = []
        for path in Path(noise_dir).rglob('*.wav'):
            telephony_noise_files.append(str(path.resolve()))
        shuffle(telephony_noise_files)
        params['telephony_noise_files'] = telephony_noise_files

    if 'speech_csv' in cfg.keys() and cfg['speech_csv'] != 'None':
        cleanfilenames = pd.read_csv(cfg['speech_csv'])
        cleanfilenames = cleanfilenames['filename']
    elif 'speech_dir_root' in cfg.keys() and cfg['speech_dir_root'] != 'None':
        root_dir = Path(cfg['speech_dir_root'])
        replicas_subdir = cfg.get('replicas_subdir', 'replicas')
        cleanfilenames = []
        for call_dir in sorted(root_dir.iterdir()):
            if not call_dir.is_dir():
                continue
            replicas_dir = call_dir / replicas_subdir
            if not replicas_dir.is_dir():
                continue
            for path in sorted(replicas_dir.rglob('*.wav')):
                cleanfilenames.append(str(path.resolve()))
    else:
        #cleanfilenames = glob.glob(os.path.join(clean_dir, params['audioformat']))
        cleanfilenames= []
        for path in Path(clean_dir).rglob('*.wav'):
            cleanfilenames.append(str(path.resolve()))

    shuffle(cleanfilenames)
#   add singing voice to clean speech
    if params['use_singing_data'] ==1:
        all_singing= []
        for path in Path(params['clean_singing']).rglob('*.wav'):
            all_singing.append(str(path.resolve()))
            
        if params['singing_choice']==1: # male speakers
            mysinging = [s for s in all_singing if ("male" in s and "female" not in s)]
    
        elif params['singing_choice']==2: # female speakers
            mysinging = [s for s in all_singing if "female" in s]
    
        elif params['singing_choice']==3: # both male and female
            mysinging = all_singing
        else: # default both male and female
            mysinging = all_singing
            
        shuffle(mysinging)
        if mysinging is not None:
            all_cleanfiles= cleanfilenames + mysinging
    else: 
        all_cleanfiles= cleanfilenames
        
#   add emotion data to clean speech
    if params['use_emotion_data'] ==1:
        all_emotion= []
        for path in Path(params['clean_emotion']).rglob('*.wav'):
            all_emotion.append(str(path.resolve()))

        shuffle(all_emotion)
        if all_emotion is not None:
            all_cleanfiles = all_cleanfiles + all_emotion
    else: 
        print('NOT using emotion data for training!')    
        
#   add mandarin data to clean speech
    if params['use_mandarin_data'] ==1:
        all_mandarin= []
        for path in Path(params['clean_mandarin']).rglob('*.wav'):
            all_mandarin.append(str(path.resolve()))

        shuffle(all_mandarin)
        if all_mandarin is not None:
            all_cleanfiles = all_cleanfiles + all_mandarin
    else: 
        print('NOT using non-english (Mandarin) data for training!')           
        

    params['cleanfilenames'] = all_cleanfiles
    params['num_cleanfiles'] = len(params['cleanfilenames'])
    # If there are .wav files in noise_dir directory, use those
    # If not, that implies that the noise files are organized into subdirectories by type,
    # so get the names of the non-excluded subdirectories
    if 'noise_csv' in cfg.keys() and cfg['noise_csv'] != 'None':
        noisefilenames = pd.read_csv(cfg['noise_csv'])
        noisefilenames = noisefilenames['filename']
    else:
        noisefilenames = glob.glob(os.path.join(noise_dir, params['audioformat']))

    if len(noisefilenames)!=0:
        shuffle(noisefilenames)
        params['noisefilenames'] = noisefilenames
    else:
        noisedirs = glob.glob(os.path.join(noise_dir, '*'))
        if cfg['noise_types_excluded'] != 'None':
            dirstoexclude = cfg['noise_types_excluded'].split(',')
            for dirs in dirstoexclude:
                noisedirs.remove(dirs)
        shuffle(noisedirs)
        params['noisedirs'] = noisedirs

    if params['use_rir']:
        temp = pd.read_csv(params['rir_table_csv'], skiprows=[1], sep=',', header=None,
                           names=['wavfile','channel','T60_WB','C50_WB','isRealRIR'])
        temp.keys()

        rir_wav = temp['wavfile'][1:] # 115413
        rir_channel = temp['channel'][1:]
        rir_t60 = temp['T60_WB'][1:]
        rir_isreal= temp['isRealRIR'][1:]

        rir_wav2 = [w.replace('\\', '/') for w in rir_wav]
        rir_channel2 = [w for w in rir_channel]
        rir_t60_2 = [w for w in rir_t60]
        rir_isreal2= [w for w in rir_isreal]

        myrir =[]
        mychannel=[]
        myt60=[]

        lower_t60=  params['lower_t60']
        upper_t60=  params['upper_t60']

        if params['rir_choice']==1: # real 3076 IRs
            real_indices= [i for i, x in enumerate(rir_isreal2) if x == "1"]

            chosen_i = []
            for i in real_indices:
                if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                    chosen_i.append(i)

            myrir= [rir_wav2[i] for i in chosen_i]
            mychannel = [rir_channel2[i] for i in chosen_i]
            myt60 = [rir_t60_2[i] for i in chosen_i]


        elif params['rir_choice']==2: # synthetic 112337 IRs
            synthetic_indices= [i for i, x in enumerate(rir_isreal2) if x == "0"]

            chosen_i = []
            for i in synthetic_indices:
                if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                    chosen_i.append(i)

            myrir= [rir_wav2[i] for i in chosen_i]
            mychannel = [rir_channel2[i] for i in chosen_i]
            myt60 = [rir_t60_2[i] for i in chosen_i]

        elif params['rir_choice']==3: # both real and synthetic
            all_indices= [i for i, x in enumerate(rir_isreal2)]

            chosen_i = []
            for i in all_indices:
                if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                    chosen_i.append(i)

            myrir= [rir_wav2[i] for i in chosen_i]
            mychannel = [rir_channel2[i] for i in chosen_i]
            myt60 = [rir_t60_2[i] for i in chosen_i]

        else:  # default both real and synthetic
            all_indices= [i for i, x in enumerate(rir_isreal2)]

            chosen_i = []
            for i in all_indices:
                if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                    chosen_i.append(i)

            myrir= [rir_wav2[i] for i in chosen_i]
            mychannel = [rir_channel2[i] for i in chosen_i]
            myt60 = [rir_t60_2[i] for i in chosen_i]

        params['myrir'] = myrir
        params['mychannel'] = mychannel
        params['myt60'] = myt60
    else:
        params['myrir'] = []
        params['mychannel'] = []
        params['myt60'] = []

    # Call main_gen() to generate audio
    clean_source_files, clean_clipped_files, clean_low_activity_files, \
    noise_source_files, noise_clipped_files, noise_low_activity_files, \
    train_rows, eval_rows, perf = main_gen(params)

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = utils.get_dir(cfg, 'log_dir', 'Logs')

    utils.write_log_file(log_dir, 'source_files.csv', clean_source_files + noise_source_files)
    utils.write_log_file(log_dir, 'clipped_files.csv', clean_clipped_files + noise_clipped_files)
    utils.write_log_file(log_dir, 'low_activity_files.csv', \
                         clean_low_activity_files + noise_low_activity_files)

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = len(clean_source_files) + len(clean_clipped_files) + len(clean_low_activity_files)
    total_noise = len(noise_source_files) + len(noise_clipped_files) + len(noise_low_activity_files)
    pct_clean_clipped = round(len(clean_clipped_files)/total_clean*100, 1)
    pct_noise_clipped = round(len(noise_clipped_files)/total_noise*100, 1)
    pct_clean_low_activity = round(len(clean_low_activity_files)/total_clean*100, 1)
    pct_noise_low_activity = round(len(noise_low_activity_files)/total_noise*100, 1)

    print("Of the " + str(total_clean) + " clean speech files analyzed, " + \
          str(pct_clean_clipped) + "% had clipping, and " + str(pct_clean_low_activity) + \
          "% had low activity " + "(below " + str(params['clean_activity_threshold']*100) + \
          "% active percentage)")
    print("Of the " + str(total_noise) + " noise files analyzed, " + str(pct_noise_clipped) + \
          "% had clipping, and " + str(pct_noise_low_activity) + "% had low activity " + \
          "(below " + str(params['noise_activity_threshold']*100) + "% active percentage)")
    if perf["files"] > 0 and not params.get('perf_trace', False):
        perf_detail = params.get("perf_detail", {})
        avg = {k: perf[k] / perf["files"] for k in perf if k.endswith("_s")}
        avg_detail = {k: perf_detail[k] / perf["files"] for k in perf_detail}
        print(
            "Perf avg (s/file): clean_gen={:.3f} noise_gen={:.3f} rir={:.3f} "
            "tele_clean={:.3f} tele_noise={:.3f} tele_post={:.3f} mix={:.3f} "
            "tele_mask={:.3f} write={:.3f}".format(
                avg["clean_gen_s"], avg["noise_gen_s"], avg["rir_s"],
                avg["telephony_clean_s"], avg["telephony_noise_s"],
                avg["telephony_post_s"], avg["mix_s"], avg["telephony_mask_s"],
                avg["write_s"]
            )
        )
        attempt_avg = avg_detail["attempt_s"] / max(avg_detail["attempts"], 1)
        print(
            "Perf build_audio avg (s/file): read={:.3f} resample={:.3f} crop={:.3f} "
            "concat={:.3f} silence={:.3f} activity={:.3f} build_total={:.3f} "
            "gen_total={:.3f} gen_attempts={:.2f} read_attempts={:.2f} "
            "attempt_avg={:.3f}".format(
                avg_detail["read_s"], avg_detail["resample_s"], avg_detail["crop_s"],
                avg_detail["concat_s"], avg_detail["silence_s"], avg_detail["activity_s"],
                avg_detail["build_total_s"], avg_detail["gen_total_s"],
                avg_detail["gen_attempts"], avg_detail["attempts"], attempt_avg
            )
        )
        clean_files_per_build = perf_detail["build_files_clean"] / max(
            perf_detail["gen_attempts_clean"], 1
        )
        noise_files_per_build = perf_detail["build_files_noise"] / max(
            perf_detail["gen_attempts_noise"], 1
        )
        activity_call_avg = perf_detail["activity_s"] / max(
            perf_detail["activity_calls"], 1
        )
        print(
            "Perf gen counts avg (per file): gen_clean={:.2f} gen_noise={:.2f} "
            "empty_reads={:.2f} clipped_reads={:.2f} short_clean={:.2f} "
            "short_noise={:.2f} low_act_clean={:.2f} low_act_noise={:.2f}".format(
                avg_detail["gen_attempts_clean"], avg_detail["gen_attempts_noise"],
                avg_detail["read_empty"], avg_detail["clipped"],
                avg_detail["short_clean"], avg_detail["short_noise"],
                avg_detail["low_activity_clean"], avg_detail["low_activity_noise"]
            )
        )
        print(
            "Perf gen build avg: clean_files/build={:.2f} noise_files/build={:.2f} "
            "activity_call_avg={:.3f}".format(
                clean_files_per_build, noise_files_per_build, activity_call_avg
            )
        )

    train_csv = cfg.get('train_csv', os.path.join(log_dir, 'train_metadata.csv'))
    with open(train_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['clean_path', 'noise_path', 'noisy_path'])
        writer.writerows(train_rows)

    if params.get('eval_stride', 0) > 0:
        eval_csv = cfg.get('eval_csv', os.path.join(log_dir, 'eval_metadata.csv'))
        with open(eval_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['clean_path', 'noise_path', 'noisy_path'])
            writer.writerows(eval_rows)


if __name__ == '__main__':

    main_body()
