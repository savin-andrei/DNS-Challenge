"""
Generate DNS-compatible RIR_table_simple.csv from an OpenSLR28 directory.

This script targets the CSV format expected by noisyspeech_synthesizer_singleprocess.py:
wavfile,channel,T60_WB,C50_WB,isRealRIR
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List


CSV_HEADER = ["wavfile", "channel", "T60_WB", "C50_WB", "isRealRIR"]


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    out = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _resolve_from_rir_list(list_file: Path, root: Path, token: str) -> Path:
    token = token.strip()
    if not token:
        return None

    candidate = Path(token)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # Most rir_list files use relative paths.
    candidate = (list_file.parent / token).resolve()
    if candidate.exists():
        return candidate

    candidate = (root / token).resolve()
    if candidate.exists():
        return candidate

    return None


def _collect_from_rir_lists(root: Path) -> List[Path]:
    wavs = []
    for list_file in sorted(root.rglob("rir_list")):
        with list_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                token = text.split()[0]
                resolved = _resolve_from_rir_list(list_file, root, token)
                if resolved is not None and resolved.suffix.lower() == ".wav":
                    wavs.append(resolved)
    return _dedupe_paths(wavs)


def _looks_like_noise(path: Path) -> bool:
    low = str(path).lower()
    name = path.name.lower()
    if "noise" not in low:
        return False

    # AIR and *_rir* names should be treated as RIRs even if parent dirs include "noise".
    if name.startswith("air_") or "rir" in name:
        return False
    if "real_rirs" in low or "simulated_rirs" in low:
        return False
    return True


def _collect_by_scan(root: Path, include_noise: bool) -> List[Path]:
    wavs = []
    for path in sorted(root.rglob("*.wav")):
        if include_noise:
            wavs.append(path.resolve())
            continue
        if _looks_like_noise(path):
            continue
        wavs.append(path.resolve())
    return _dedupe_paths(wavs)


def _infer_is_real(path: Path, default_is_real: int) -> int:
    low = str(path).lower()
    name = path.name.lower()
    if "simulated" in low:
        return 0
    if "real_rirs" in low or name.startswith("air_"):
        return 1
    return default_is_real


def _format_path(path: Path, path_mode: str, relative_to: Path) -> str:
    if path_mode == "absolute":
        return str(path.resolve())

    base = relative_to.resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return os.path.relpath(str(resolved), str(base))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DNS RIR_table_simple.csv from SLR28."
    )
    parser.add_argument(
        "--slr28-root",
        required=True,
        help="Path to SLR28 root (for example: .../impulse_responses/SLR28).",
    )
    parser.add_argument(
        "--output",
        default="datasets/acoustic_params/RIR_table_simple.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--default-t60",
        type=float,
        default=0.60,
        help="Fallback T60_WB value for rows with unknown RT60.",
    )
    parser.add_argument(
        "--default-c50",
        type=float,
        default=0.0,
        help="C50_WB value written to CSV (not used by the current synthesizer path).",
    )
    parser.add_argument(
        "--default-is-real",
        type=int,
        choices=[0, 1],
        default=1,
        help="isRealRIR fallback for files that do not match heuristics.",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include files that look like non-RIR noise wavs.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="Write wavfile paths as absolute or relative paths.",
    )
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Base directory for relative path mode; defaults to --slr28-root.",
    )
    parser.add_argument(
        "--no-compat-skip-row",
        action="store_true",
        help=(
            "Do not add the legacy compatibility row after header. "
            "Keep default behavior for this DNS repo."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    slr28_root = Path(args.slr28_root).expanduser().resolve()
    if not slr28_root.exists():
        raise FileNotFoundError("SLR28 root does not exist: {}".format(slr28_root))

    relative_to = (
        Path(args.relative_to).expanduser().resolve()
        if args.relative_to
        else slr28_root
    )

    wavs = _collect_from_rir_lists(slr28_root)
    source = "rir_list"
    if not wavs:
        wavs = _collect_by_scan(slr28_root, include_noise=args.include_noise)
        source = "recursive scan"

    if not wavs:
        raise RuntimeError("No .wav files found under {}".format(slr28_root))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    real_count = 0
    synthetic_count = 0

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)

        # The current DNS parser skips one row + slices [1:], so this row preserves full data.
        if not args.no_compat_skip_row:
            writer.writerow(
                ["__skip__", 1, "{:.3f}".format(args.default_t60), "{:.3f}".format(args.default_c50), args.default_is_real]
            )

        for path in wavs:
            is_real = _infer_is_real(path, args.default_is_real)
            if is_real == 1:
                real_count += 1
            else:
                synthetic_count += 1

            writer.writerow(
                [
                    _format_path(path, args.path_mode, relative_to),
                    1,
                    "{:.3f}".format(args.default_t60),
                    "{:.3f}".format(args.default_c50),
                    is_real,
                ]
            )
            rows_written += 1

    print("Source: {}".format(source))
    print("CSV: {}".format(output))
    print("Rows (RIR wavs): {}".format(rows_written))
    print("isRealRIR=1: {}, isRealRIR=0: {}".format(real_count, synthetic_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
