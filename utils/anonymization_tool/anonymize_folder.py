"""
blur_dataset.py
Recorre un dataset con estructura:
  <root>/
    <scenario>/          (ej: construction_site, crossing_object, cut_in...)
      <scenario_N>/      (ej: construction_site_1, construction_site_2...)
        <sequence>/      (ej: construction_site_1_t-1s_frameXXXXXX...)
          *.jpg / *.png

Aplica egoblur-gen2 a cada imagen y replica la estructura en el output.

Uso:
  python blur_dataset.py \
    --dataset_root /ruta/a/dataset \
    --output_root  /ruta/a/output \
    --face_model   model_ego_blur_face/ego_blur_face_gen2/ego_blur_face_gen2.jit \
    --lp_model     model_ego_blur_lp/ego_blur_lp_gen2/ego_blur_lp_gen2.jit \
    [--camera_name camera-rgb] \
    [--face_threshold 0.1] \
    [--workers 4] \
    [--dry_run]
"""

import subprocess
import sys
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ── Extensiones de imagen soportadas ──────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def setup_logging(log_file: Path | None = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def find_images(dataset_root: Path) -> list[Path]:
    """Encuentra todas las imágenes bajo dataset_root recursivamente."""
    images = [
        p
        for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def build_output_path(image_path: Path, dataset_root: Path, output_root: Path) -> Path:
    """Calcula la ruta de salida replicando la estructura del dataset."""
    relative = image_path.relative_to(dataset_root)
    # Siempre guardamos como .png (egoblur suele exigirlo)
    output_path = output_root / relative.with_suffix(".png")
    return output_path


def blur_image(
    image_path: Path,
    output_path: Path,
    face_model: str,
    lp_model: str,
    camera_name: str,
    face_threshold: float,
    lp_threshold: float,
    dry_run: bool,
) -> tuple[Path, bool, str]:
    """
    Llama a egoblur-gen2 para una imagen.
    Devuelve (image_path, éxito, mensaje_error).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "egoblur-gen2",
        "--camera_name",
        camera_name,
        "--face_model_path",
        face_model,
        "--lp_model_path",
        lp_model,
        "--input_image_path",
        str(image_path),
        "--output_image_path",
        str(output_path),
        "--face_model_score_threshold",
        str(face_threshold),
        "--lp_model_score_threshold",
        str(lp_threshold),
    ]

    if dry_run:
        logging.debug(f"[DRY RUN] {' '.join(cmd)}")
        return image_path, True, ""

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 min por imagen como máximo
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            return image_path, False, err
        return image_path, True, ""
    except subprocess.TimeoutExpired:
        return image_path, False, "TIMEOUT (>120s)"
    except FileNotFoundError:
        return image_path, False, "egoblur-gen2 no encontrado en PATH"
    except Exception as e:
        return image_path, False, str(e)


def main():

    DATASET_ROOT = Path("final_events_9")  # <- raíz del dataset
    OUTPUT_ROOT = Path("final_events_blurry_9")  # <- donde se guardará el output

    FACE_MODEL = "model_ego_blur_face/ego_blur_face_gen2/ego_blur_face_gen2.jit"
    LP_MODEL = "model_ego_blur_lp/ego_blur_lp_gen2/ego_blur_lp_gen2.jit"

    CAMERA_NAME = "camera-rgb"
    FACE_THRESHOLD = 0.1
    LP_THRESHOLD = 0.1

    WORKERS = 4
    SKIP_EXISTING = True  # True = salta imágenes cuyo output ya existe
    DRY_RUN = False  # True = muestra comandos sin ejecutarlos
    LOG_FILE = None

    # Convertir a Path por si acaso
    dataset_root = Path(DATASET_ROOT)
    output_root = Path(OUTPUT_ROOT)
    face_model = FACE_MODEL
    lp_model = LP_MODEL
    camera_name = CAMERA_NAME
    face_threshold = FACE_THRESHOLD
    lp_threshold = LP_THRESHOLD
    workers = WORKERS
    skip_existing = SKIP_EXISTING
    dry_run = DRY_RUN
    log_file = Path(LOG_FILE) if LOG_FILE else None

    setup_logging(log_file)

    # ── Validaciones básicas ───────────────────────────────────────────────────
    if not dataset_root.exists():
        logging.error(f"dataset_root no existe: {dataset_root}")
        sys.exit(1)
    if not Path(face_model).exists():
        logging.warning(f"face_model no encontrado: {face_model}")
    if not Path(lp_model).exists():
        logging.warning(f"lp_model no encontrado: {lp_model}")

    # ── Descubrimiento de imágenes ─────────────────────────────────────────────
    logging.info(f"Buscando imágenes en: {dataset_root}")
    all_images = find_images(dataset_root)

    if not all_images:
        logging.error("No se encontraron imágenes. Revisa la ruta y las extensiones.")
        sys.exit(1)

    logging.info(f"Imágenes encontradas: {len(all_images)}")

    if not all_images:
        logging.info("Todas las imágenes ya están procesadas. ¡Nada que hacer!")
        sys.exit(0)

    # ── Procesamiento ──────────────────────────────────────────────────────────
    failed: list[tuple[Path, str]] = []
    success_count = 0

    def task(img: Path):
        out = build_output_path(img, dataset_root, output_root)
        return blur_image(
            image_path=img,
            output_path=out,
            face_model=face_model,
            lp_model=lp_model,
            camera_name=camera_name,
            face_threshold=face_threshold,
            lp_threshold=lp_threshold,
            dry_run=dry_run,
        )

    logging.info(f"Procesando con {workers} worker(s)...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task, img): img for img in all_images}
        with tqdm(total=len(all_images), unit="img", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                img_path, ok, err = future.result()
                if ok:
                    success_count += 1
                    pbar.set_postfix({"ok": success_count, "fail": len(failed)})
                else:
                    failed.append((img_path, err))
                    logging.warning(f"FALLO: {img_path} → {err}")
                pbar.update(1)

    # ── Resumen final ──────────────────────────────────────────────────────────
    total = success_count + len(failed)
    logging.info("=" * 60)
    logging.info(
        f"COMPLETADO: {success_count}/{total} imágenes procesadas correctamente"
    )

    if failed:
        logging.warning(f"FALLIDAS: {len(failed)} imágenes")
        fail_log = output_root / "failed_images.txt"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_log, "w") as f:
            for p, err in failed:
                f.write(f"{p}\t{err}\n")
        logging.warning(f"Lista de fallos guardada en: {fail_log}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
