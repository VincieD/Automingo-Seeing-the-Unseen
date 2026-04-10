"""
Video Frame Extractor - Sincroniza eventos JSON con frames de video MP4
Extrae frames en timestamps de eventos y genera dataset organizado por carpetas
"""

import json
import cv2
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import argparse


EVENT_OFFSETS = {
    "traffic_light": [-2, -1, 0, 1, 2],
    "leading_braking": [-2, -1, 0, 1, 2],
    "cut_in": [-3, -1.5, 0, 1.5, 3],
    "construction_site": [-2, -1, 0, 1, 2],
    "crossing_object": [-2, -1, 0, 1, 2],
    "lateral_parked_car": [-1, -0.5, 0, 0.5, 1],
    "pedestrian": [-2, -1, 0, 1, 2],
    "merging_lane": [-2, -1, 0, 1, 2],
    "intersection_road": [-2, -1, 0, 1, 2],
    "roundabout": [-2, -1, 0, 1, 2],
    "speed_limit_adaptation": [-1, -0.5, 0, 0.5, 1],
}

# Offsets por defecto si el evento no está en el diccionario
DEFAULT_OFFSETS = [-2, -1, 0, 1, 2]

# Etiquetas que se ignoran completamente (no se extraen frames)
NO_KEEP_LABELS = {"noKeep", "no_keep", "discard_event"}


def format_offset(offset_sec: float) -> str:
    """
    Convierte un offset a string que ordena correctamente de forma alfabética.
    Usamos un valor desplazado (bias) para que todos sean positivos y ordenen bien.

    Bias de +1000 (en décimas) para que cualquier offset negativo razonable sea positivo:
      -3.0  → (−30 + 1000) = 0970  → 't0970'  (el más pequeño)
      -1.5  → (−15 + 1000) = 0985  → 't0985'
       0.0  → (  0 + 1000) = 1000  → 't1000'
      +1.5  → ( 15 + 1000) = 1015  → 't1015'
      +3.0  → ( 30 + 1000) = 1030  → 't1030'  (el más grande)
    Ordenación alfabética == ordenación numérica. ✓
    """
    val = int(round(offset_sec * 10)) + 1000
    return f"t{val:04d}"


def read_existing_counters(output_dir: Path) -> Dict[str, int]:
    """
    Lee la carpeta de output y devuelve los contadores actuales por clase.
    Ejemplo: si ya existe cut_in_7, devuelve {'cut_in': 7}
    Así la siguiente ejecución empieza desde cut_in_8.
    """
    counters = {}
    if not output_dir.exists():
        return counters

    for class_dir in output_dir.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name  # ej: 'cut_in', 'empty', 'pedestrian'

        max_index = 0
        for seq_dir in class_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            # Busca el patrón label_N al final del nombre de carpeta
            match = re.search(rf"^{re.escape(label)}_(\d+)$", seq_dir.name)
            if match:
                idx = int(match.group(1))
                if idx > max_index:
                    max_index = idx

        if max_index > 0:
            counters[label] = max_index
            print(f"  ↳ Found existing '{label}': {max_index} sequences already saved")

    return counters


def get_offsets_for_event(label: str) -> List[float]:
    """
    Devuelve la lista de offsets (en segundos) para un tipo de evento dado.
    Hace matching flexible: busca si alguna clave está contenida en el label
    o viceversa (ej: 'cut_in_1' matchea con 'cut_in').
    """
    # Primero intenta match exacto
    if label in EVENT_OFFSETS:
        return EVENT_OFFSETS[label]

    # Luego intenta match parcial (ej: 'cut_in_1' -> 'cut_in')
    label_lower = label.lower()
    for key in EVENT_OFFSETS:
        if key in label_lower or label_lower.startswith(key):
            return EVENT_OFFSETS[key]

    # Si no encuentra nada, usa los offsets por defecto
    print(
        f"  ⚠ No specific offsets found for '{label}', using default {DEFAULT_OFFSETS}"
    )
    return DEFAULT_OFFSETS


def parse_gopro_filename(filename: str) -> Optional[datetime]:
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month, day, hour, minute, second)
    return None


def extract_video_metadata(video_path: Path) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": 0.0,
    }
    if metadata["fps"] > 0:
        metadata["duration_seconds"] = metadata["frame_count"] / metadata["fps"]
    cap.release()
    return metadata


def calculate_video_start_time(video_path: Path, session_data: Dict) -> datetime:
    filename = video_path.name
    video_start = parse_gopro_filename(filename)
    if video_start:
        print(
            f"✓ Video start time detected from filename: {video_start.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return video_start

    recording_start_str = session_data.get("recording_start_datetime")
    if recording_start_str:
        try:
            video_start = datetime.strptime(recording_start_str, "%Y-%m-%d %H:%M:%S")
            print(
                f"✓ Using recording_start_datetime from JSON: {video_start.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return video_start
        except ValueError:
            try:
                video_start = datetime.fromisoformat(
                    recording_start_str.replace("Z", "+00:00")
                )
                print(
                    f"✓ Using recording_start_datetime from JSON: {video_start.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                return video_start
            except:
                pass

    print("\n⚠️  Cannot detect video start time automatically")
    print(f"Video file: {filename}")
    print("\nPlease enter the EXACT time when you pressed REC on the GoPro:")
    print("Format: YYYY-MM-DD HH:MM:SS")
    print("Example: 2026-02-10 14:30:28")

    while True:
        user_input = input("\nVideo start time: ").strip()
        try:
            video_start = datetime.strptime(user_input, "%Y-%m-%d %H:%M:%S")
            return video_start
        except ValueError:
            print("❌ Invalid format. Please use: YYYY-MM-DD HH:MM:SS")


def extract_frame(video_path: Path, frame_number: int) -> Optional[any]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def process_video_with_events(
    video_path: Path,
    json_path: Path,
    output_dir: Path,
    num_empty_frames: int = 50,
    extract_video_metadata_only: bool = False,
    prev_dataset_dir: Optional[Path] = None,  # ← NUEVO: carpeta dataset anterior
) -> None:
    print("\n" + "=" * 70)
    print("  VIDEO FRAME EXTRACTOR - Event Dataset Generator")
    print("=" * 70)

    print(f"\n📄 Loading events from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    total_events = len(
        [
            e
            for e in session_data["events"]
            if e["label"] not in ("SESSION_START", *NO_KEEP_LABELS)
        ]
    )
    print(f"✓ Loaded {total_events} events (noKeep events ignored)")

    print(f"\n🎥 Analyzing video: {video_path}")
    video_metadata = extract_video_metadata(video_path)
    print(f"✓ Resolution: {video_metadata['width']}x{video_metadata['height']}")
    print(f"✓ FPS: {video_metadata['fps']:.2f}")
    print(
        f"✓ Duration: {video_metadata['duration_seconds']:.2f}s ({video_metadata['frame_count']} frames)"
    )

    if extract_video_metadata_only:
        return

    video_start_time = calculate_video_start_time(video_path, session_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    empty_dir = output_dir / "empty"
    empty_dir.mkdir(exist_ok=True)

    print("\n" + "-" * 70)
    print("EXTRACTING EVENT FRAMES")
    print("-" * 70)

    event_frames = []
    manual_empty_frames = []
    discard_frames = []
    stats = {}

    # ── NUEVO: leer contadores del dataset previo (carpeta anonimizada) ──────
    event_counters = {}
    if prev_dataset_dir is not None:
        print(f"\n🔍 Reading counters from PREVIOUS dataset: {prev_dataset_dir}")
        event_counters = read_existing_counters(prev_dataset_dir)
        if event_counters:
            print(f"✓ Will continue numbering after previous dataset")
        else:
            print(f"✓ No existing sequences found in previous dataset")
    else:
        # Comportamiento original: leer del propio directorio de salida
        print("\n🔍 Checking existing output directory for previous runs...")
        event_counters = read_existing_counters(output_dir)
        if event_counters:
            print(f"✓ Resuming from existing dataset (continuing sequence numbering)")
        else:
            print(f"✓ Fresh output directory, starting from 1")

    fps = video_metadata["fps"]
    total_frames = video_metadata["frame_count"]

    for event in session_data["events"]:
        label = event["label"]

        # ── Ignorar SESSION_START y etiquetas noKeep ──────────────────────────
        if label == "SESSION_START":
            continue

        if label in NO_KEEP_LABELS:
            print(f"⏭️  Skipping noKeep event at {event['timestamp']:.2f}s")
            continue

        timestamp = event["timestamp"]
        base_frame = int(timestamp * fps)

        if label == "discard":
            discard_frames.append(base_frame)
            print(f"🚫 Discard zone marked at {timestamp:.2f}s (frame {base_frame})")
            continue

        if label == "empty":
            manual_empty_frames.append(base_frame)
            continue

        event_counters[label] = event_counters.get(label, 0) + 1
        event_index = event_counters[label]

        class_dir = output_dir / label
        class_dir.mkdir(exist_ok=True)

        event_dir = class_dir / f"{label}_{event_index}"
        event_dir.mkdir(exist_ok=True)

        # ── NUEVO: obtener offsets específicos para este tipo de evento ──
        time_offsets = get_offsets_for_event(label)

        print(
            f"\n🎯 Processing {label}_{event_index} at {timestamp:.2f}s  |  offsets: {time_offsets}"
        )

        for offset_sec in time_offsets:
            new_time = timestamp + offset_sec
            frame_number = int(new_time * fps)

            if frame_number < 0 or frame_number >= total_frames:
                print(f"  ⚠ Skipping offset {offset_sec}s (out of bounds)")
                continue

            frame = extract_frame(video_path, frame_number)

            if frame is not None:
                filename = (
                    f"{label}_{event_index}"
                    f"_{format_offset(offset_sec)}"
                    f"_frame{frame_number:06d}.jpg"
                )
                output_path = event_dir / filename
                cv2.imwrite(str(output_path), frame)
                event_frames.append(frame_number)
                print(f"   ✓ {offset_sec:+g}s → frame {frame_number}")
                stats[label] = stats.get(label, 0) + 1
            else:
                print(f"   ✗ Failed extracting frame {frame_number}")

    print("\n" + "-" * 70)
    print("EXTRACTING EMPTY SEQUENCES (Manual + Random)")
    print("-" * 70)

    buffer_seconds = 3.0
    buffer_frames = int(buffer_seconds * fps)

    # Solo contamos los eventos procesados EN ESTE RUN (no los del dataset previo)
    # stats tiene exactamente los frames extraídos en este run por clase
    events_this_run = sum(
        count // len(get_offsets_for_event(label))
        for label, count in stats.items()
        if label != "empty"
    )
    num_manual_empties = len(manual_empty_frames)
    num_discards = len(discard_frames)

    # El contador de empty arranca desde el dataset previo para no colisionar
    empty_sequence_count = event_counters.get("empty", 0)
    # Target: tantas secuencias empty como eventos de este run (50/50)
    target_empty_sequences = empty_sequence_count + events_this_run

    print(f"\n📊 Empty frames strategy:")
    print(f"  - Manually marked by analyst: {num_manual_empties}")
    print(f"  - Discard zones (to exclude): {num_discards}")
    print(f"  - Starting empty counter at: {empty_sequence_count}")
    print(f"  - Target total empty sequences: {target_empty_sequences}")
    print(
        f"  - Random sequences to generate: {max(0, target_empty_sequences - empty_sequence_count - num_manual_empties)}"
    )

    empty_class_dir = output_dir / "empty"
    empty_class_dir.mkdir(exist_ok=True)

    forbidden_frames = set()

    for ef in event_frames:
        for offset in range(-buffer_frames, buffer_frames + 1):
            f = ef + offset
            if 0 <= f < total_frames:
                forbidden_frames.add(f)

    for ef in manual_empty_frames:
        for offset in range(-buffer_frames, buffer_frames + 1):
            f = ef + offset
            if 0 <= f < total_frames:
                forbidden_frames.add(f)

    print(f"\n🚫 Processing discard zones (excluded from random empties):")
    for ef in discard_frames:
        for offset in range(-buffer_frames, buffer_frames + 1):
            f = ef + offset
            if 0 <= f < total_frames:
                forbidden_frames.add(f)
        t = ef / fps
        print(f"  - Excluding zone around frame {ef} (t={t:.2f}s) ± {buffer_seconds}s")

    offsets_seconds = [-2, -1, 0, 1, 2]  # offsets fijos para secuencias empty
    window_radius_frames = int(2 * fps)

    # PASO 1: empties marcados manualmente
    print("\n" + "-" * 40)
    print("STEP 1: Processing manually marked empties")
    print("-" * 40)

    for idx_offset, base_frame in enumerate(manual_empty_frames):
        idx = empty_sequence_count + idx_offset + 1
        timestamp = base_frame / fps
        event_dir = empty_class_dir / f"empty_{idx}"
        event_dir.mkdir(exist_ok=True)
        print(f"\n⭕ Processing manually marked empty_{idx} at {timestamp:.2f}s")

        for offset_sec in offsets_seconds:
            new_time = timestamp + offset_sec
            frame_number = int(new_time * fps)

            if frame_number < 0 or frame_number >= total_frames:
                print(f"  ⚠ Skipping offset {offset_sec}s (out of bounds)")
                continue

            frame = extract_frame(video_path, frame_number)
            if frame is not None:
                filename = (
                    f"empty_{idx}"
                    f"_{format_offset(offset_sec)}"
                    f"_frame{frame_number:06d}"
                    f"_manual.jpg"
                )
                cv2.imwrite(str(event_dir / filename), frame)
                print(f"   ✓ {offset_sec:+g}s → frame {frame_number} (manual)")
            else:
                print(f"   ✗ Failed extracting frame {frame_number}")

        empty_sequence_count += 1
        stats["empty"] = stats.get("empty", 0) + len(offsets_seconds)

    # PASO 2: empties aleatorios
    print("\n" + "-" * 40)
    print("STEP 2: Generating random empty sequences")
    print("-" * 40)

    current_frame = window_radius_frames

    while (
        empty_sequence_count < target_empty_sequences
        and current_frame < total_frames - window_radius_frames
    ):
        sequence_frames = []
        valid_sequence = True

        for offset_sec in offsets_seconds:
            frame_number = current_frame + int(offset_sec * fps)
            if (
                frame_number in forbidden_frames
                or frame_number < 0
                or frame_number >= total_frames
            ):
                valid_sequence = False
                break
            sequence_frames.append(frame_number)

        if valid_sequence:
            empty_sequence_count += 1
            event_dir = empty_class_dir / f"empty_{empty_sequence_count}"
            event_dir.mkdir(exist_ok=True)
            print(f"\n🟢 empty_{empty_sequence_count}")

            for offset_sec, frame_number in zip(offsets_seconds, sequence_frames):
                frame = extract_frame(video_path, frame_number)
                if frame is not None:
                    filename = (
                        f"empty_{empty_sequence_count}"
                        f"_{format_offset(offset_sec)}"
                        f"_frame{frame_number:06d}"
                        f"_random.jpg"
                    )
                    cv2.imwrite(str(event_dir / filename), frame)
                    print(f"   ✓ {offset_sec:+g}s → frame {frame_number} (random)")

            stats["empty"] = stats.get("empty", 0) + len(offsets_seconds)
            current_frame += window_radius_frames * 2 + 1
        else:
            current_frame += int(1 * fps)

    if empty_sequence_count < target_empty_sequences:
        print(f"\n⚠ Could not generate fully balanced empty dataset.")
        print(f"Generated {empty_sequence_count} instead of {target_empty_sequences}")

    # Resumen final
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Dataset Statistics:")
    print("-" * 70)

    if num_discards > 0:
        print(f"  🚫 Discard zones excluded: {num_discards} (not extracted)")
        print("-" * 70)

    total_extracted = sum(stats.values())
    for label, count in sorted(stats.items()):
        percentage = (count / total_extracted) * 100 if total_extracted > 0 else 0
        if label == "empty":
            manual_count = num_manual_empties * len(offsets_seconds)
            random_count = count - manual_count
            print(f"  {label:25s} : {count:4d} frames ({percentage:5.1f}%)")
            print(
                f"    ↳ Manual:  {manual_count:4d} frames ({num_manual_empties} sequences)"
            )
            print(
                f"    ↳ Random:  {random_count:4d} frames ({random_count // len(offsets_seconds)} sequences)"
            )
        else:
            print(f"  {label:25s} : {count:4d} frames ({percentage:5.1f}%)")

    print("-" * 70)
    print(f"  {'TOTAL':25s} : {total_extracted:4d} frames")
    print("=" * 70 + "\n")

    metadata_file = output_dir / "dataset_metadata.json"
    metadata = {
        "video_file": video_path.name,
        "video_start_time": video_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_metadata": video_metadata,
        "session_data": {
            "session_id": session_data["session_id"],
            "total_events": total_events,
            "total_duration": session_data["total_duration_seconds"],
        },
        "extraction_stats": stats,
        "total_frames_extracted": total_extracted,
        "prev_dataset_dir": str(prev_dataset_dir) if prev_dataset_dir else None,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Metadata saved to: {metadata_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video based on event labels JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_frames.py video.mp4 session.json
  python extract_frames.py video.mp4 session.json -o my_dataset
  python extract_frames.py video.mp4 session.json --prev-dataset dataset_anonimizado/
  python extract_frames.py video.mp4 session.json --info-only
        """,
    )

    parser.add_argument("video", type=str, help="Path to MP4 video file")
    parser.add_argument("json", type=str, help="Path to events JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dataset_prueba_majadahonda",
        help="Output directory (default: dataset_prueba_majadahonda)",
    )
    parser.add_argument(
        "-n",
        "--num-empty",
        type=int,
        default=50,
        help="Number of empty frames to extract (default: 50)",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show video info, do not extract frames",
    )
    # ── NUEVO argumento ────────────────────────────────────────────────────────
    parser.add_argument(
        "--prev-dataset",
        type=str,
        default=None,
        help=(
            "Path to a previously generated (e.g. anonymized) dataset directory. "
            "Counters will start AFTER the highest IDs found there, so merged "
            "datasets have no collisions. Example: --prev-dataset dataset_batch1/"
        ),
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    json_path = Path(args.json)
    output_dir = Path(args.output)
    prev_dataset_dir = Path(args.prev_dataset) if args.prev_dataset else None

    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return

    if not json_path.exists():
        print(f"❌ Error: JSON file not found: {json_path}")
        return

    if prev_dataset_dir is not None and not prev_dataset_dir.exists():
        print(f"❌ Error: Previous dataset directory not found: {prev_dataset_dir}")
        return

    try:
        process_video_with_events(
            video_path,
            json_path,
            output_dir,
            num_empty_frames=args.num_empty,
            extract_video_metadata_only=args.info_only,
            prev_dataset_dir=prev_dataset_dir,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
