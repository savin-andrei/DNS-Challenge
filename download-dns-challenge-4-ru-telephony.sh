#!/bin/zsh

# ***** Datasets for ICASSP 2022 DNS Challenge 4 - Main (Real-Time) Track *****

# NOTE: Before downloading, make sure you have enough space
# on your local storage!

# In all, you will need about 1TB to store the UNPACKED data.
# Archived, the same data takes about 550GB total.

# Please comment out the files you don't need before launching
# the script.

# NOTE: By default, the script *DOES NOT* DOWNLOAD ANY FILES!
# Please scroll down and edit this script to pick the
# downloading method that works best for you.

# -------------------------------------------------------------
# The directory structure of the unpacked data is:

# datasets_fullband 892G
# +-- impulse_responses 5.9G
# +-- noise_fullband 58G
# \-- clean_fullband 12G
#     +-- russian_speech 12G

BLOB_NAMES=(

    clean_fullband/datasets_fullband.clean_fullband.russian_speech_000_0.00_4.31.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.russian_speech_001_4.31_NA.tar.bz2

    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2

    noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2

    datasets_fullband.impulse_responses_000.tar.bz2
)

###############################################################

AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

OUTPUT_PATH="./datasets_fullband"

mkdir -p $OUTPUT_PATH/{clean_fullband,noise_fullband}

for BLOB in ${BLOB_NAMES[@]}
do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"

    # DRY RUN: print HTTP response and Content-Length
    # WITHOUT downloading the files
    curl -s -I "$URL" | head -n 2

    # Actually download the files: UNCOMMENT when ready to download
     curl "$URL" -o "$OUTPUT_PATH/$BLOB"

    # Same as above, but using wget
    # wget "$URL" -O "$OUTPUT_PATH/$BLOB"

    # Same, + unpack files on the fly
    # curl "$URL" | tar -C "$OUTPUT_PATH" -f - -x -j
done
