#!/bin/zsh

# Root directory where downloaded archives live
INPUT_PATH="./datasets_fullband"
# Destination root for extracted files
EXTRACT_OUTPUT="./datasets_fullband"

if [[ ! -d "$INPUT_PATH" ]]; then
  echo "INPUT_PATH does not exist: $INPUT_PATH"
  exit 1
fi

mkdir -p "$EXTRACT_OUTPUT"

find "$INPUT_PATH" -type f -name "*.tar.bz2" | while read -r archive; do
  echo "Extract: $archive -> $EXTRACT_OUTPUT"
  tar -xjf "$archive" -C "$EXTRACT_OUTPUT"
done
