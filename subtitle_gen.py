#!/usr/bin/env python3
"""
Generate YouTube-compatible SRT subtitle files from music and lyrics.

Usage:
    python subtitle_gen.py audio.mp3 lyrics.txt
    python subtitle_gen.py audio.wav lyrics.txt -o output.srt --language ja --model small
"""
import argparse
import sys
from pathlib import Path


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms, s = 0, s + 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            f.write(f"{idx}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n♪ {text}\n\n")
            idx += 1


def is_suno_tag(line: str) -> bool:
    return line.startswith("[") and line.endswith("]")


def load_lyrics(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not is_suno_tag(line.strip())
        ]


def fix_overlaps(segments: list[dict], min_gap: float = 0.05) -> list[dict]:
    """Ensure segments don't overlap, preserving at least min_gap seconds between them."""
    for i in range(len(segments) - 1):
        next_start = segments[i + 1]["start"]
        if segments[i]["end"] > next_start - min_gap:
            segments[i]["end"] = max(segments[i]["start"] + 0.1, next_start - min_gap)
    return segments


def align_lyrics(
    audio_path: Path,
    lyrics_lines: list[str],
    model_name: str,
    language: str | None,
) -> list[dict]:
    try:
        import stable_whisper
    except ImportError:
        print("Error: stable-whisper is not installed.", file=sys.stderr)
        print("Run: pip install stable-whisper", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Whisper model '{model_name}'...")
    model = stable_whisper.load_model(model_name)

    print("Aligning lyrics to audio (this may take a few minutes)...")
    # Pass lyrics as a list so each line becomes one subtitle segment
    result = model.align(str(audio_path), lyrics_lines, language=language)

    return [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in result.segments]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate YouTube-compatible SRT subtitles from music and lyrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Lyrics file format:
  - One lyric line per subtitle entry
  - Lines starting with # are treated as comments and ignored
  - Empty lines are ignored

Model selection guide:
  tiny/base  : Fast, lower accuracy — good for testing
  small      : Balanced speed and accuracy (recommended for English)
  medium     : Good accuracy for Japanese/other languages
  large-v3   : Best accuracy, slowest (recommended for Japanese)

Examples:
  %(prog)s song.mp3 lyrics.txt
  %(prog)s song.wav lyrics.txt -o output.srt --language ja
  %(prog)s song.mp3 lyrics.txt --model large-v3 --language ja
        """,
    )
    parser.add_argument("audio", help="Audio file path (WAV or MP3)")
    parser.add_argument("lyrics", help="Lyrics text file (one line per subtitle)")
    parser.add_argument("-o", "--output", help="Output SRT file path (default: <audio>.srt)")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code, e.g. ja, en, ko (default: auto-detect)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    lyrics_path = Path(args.lyrics)
    output_path = Path(args.output) if args.output else audio_path.with_suffix(".srt")

    for p, label in [(audio_path, "Audio"), (lyrics_path, "Lyrics")]:
        if not p.exists():
            print(f"Error: {label} file not found: {p}", file=sys.stderr)
            return 1

    lyrics_lines = load_lyrics(lyrics_path)
    if not lyrics_lines:
        print("Error: No lyrics found in file.", file=sys.stderr)
        return 1

    print(f"Audio:    {audio_path.name}")
    print(f"Lyrics:   {len(lyrics_lines)} lines")
    print(f"Model:    {args.model}")
    print(f"Language: {args.language or 'auto-detect'}")
    print(f"Output:   {output_path}")
    print()

    segments = align_lyrics(audio_path, lyrics_lines, args.model, args.language)
    segments = fix_overlaps(segments)

    write_srt(segments, output_path)

    print(f"\nDone — {len(segments)} subtitle entries written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
