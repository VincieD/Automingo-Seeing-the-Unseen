import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image
from pathlib import Path
import argparse
import os
from PIL import Image as PILImage
import glob
import re
from dotenv import load_dotenv


def get_sorted_images_from_scene(scene_id, base_images_dir="YOUR_BLURRED_IMAGES_DIR"):
    """
    Encuentra y ordena automáticamente las imágenes de una escena.

    Args:
        scene_id: ID de la escena (ej: "construction_site_2")
        base_images_dir: Directorio base donde están las carpetas de escenas

    Returns:
        Lista de 5 rutas de imágenes ordenadas por timestamp
    """
    scene_path = os.path.join(base_images_dir, scene_id)

    if not os.path.exists(scene_path):
        raise ValueError(f"❌ No se encuentra la carpeta de escena: {scene_path}")

    image_patterns = [
        os.path.join(scene_path, "*.png"),
        os.path.join(scene_path, "*.jpg"),
        os.path.join(scene_path, "*.jpeg"),
    ]

    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(pattern))
    all_images = list(set(os.path.normpath(img) for img in all_images))

    if len(all_images) == 0:
        raise ValueError(f"❌ No se encontraron imágenes en: {scene_path}")

    all_images.sort()

    if len(all_images) < 5:
        print(
            f"⚠️  ADVERTENCIA: Solo se encontraron {len(all_images)} imágenes en {scene_id}"
        )
        print(f"    Se necesitan exactamente 5. Rellenando con la última imagen...")
        while len(all_images) < 5:
            all_images.append(all_images[-1])

    selected_images = all_images[:5]

    print(f"✅ {scene_id}: {len(all_images)} imágenes únicas, usando primeras 5")

    return selected_images


def parse_time_span(df_row):
    """
    Genera el time_span. Si no existe en el Excel, usa default [-2, -1, 0, 1, 2]
    """
    # Si tu Excel tiene una columna time_span, úsala
    if "time_span" in df_row and not pd.isna(df_row["time_span"]):
        ts_str = str(df_row["time_span"]).strip().replace("[", "").replace("]", "")
        return [int(x.strip()) for x in ts_str.split(",")]

    # Default
    return [-2, -1, 0, 1, 2]


def excel_to_am_dataset(
    excel_path,
    images_base_dir="real_events_anon_blurred_v2",
    hf_repo_name=None,
    hf_token=None,
    upload=False,
):
    """
    Convierte Excel del formato AM a dataset de HuggingFace con train/validation splits

    Columnas esperadas en Excel:
    - scene_id: ID de la escena (nombre de carpeta)
    - situation: Descripción de la situación
    - question: Pregunta
    - answer_short: Respuesta corta (ground_truth_answer)
    - answer_reasoning: Razonamiento (ground_truth_reasoning)
    - Wrong answer 1: Distractor 1
    - Wrong answer 2: Distractor 2
    - Wrong answer 3: Distractor 3
    """

    print(f"\n{'='*70}")
    print(f"🚀 CREANDO DATASET AM DESDE EXCEL")
    print(f"{'='*70}")

    # Leer Excel
    print(f"\n📊 Leyendo Excel: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"   Filas encontradas: {len(df)}")
    print(f"   Columnas: {list(df.columns)}")

    # Verificar columnas requeridas
    required_base = [
        "scene_id",
        "situation",
        "question",
        "answer_short",
        "answer_reasoning",
    ]
    missing = [col for col in required_base if col not in df.columns]

    if missing:
        print(f"\n❌ ERROR: Faltan columnas requeridas: {missing}")
        print(f"   Columnas disponibles: {list(df.columns)}")
        raise ValueError(f"Faltan columnas: {missing}")

    # Verificar si tiene columnas de validation (wrong answers)
    has_validation_cols = all(
        col in df.columns
        for col in ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"]
    )

    if has_validation_cols:
        print(f"\n✅ Detectadas columnas de VALIDATION (wrong answers)")
        print(
            f"   Se crearán splits: TRAIN (sin wrong answers) + VALIDATION (con wrong answers)"
        )
    else:
        print(f"\n⚠️  No se detectaron columnas de wrong answers")
        print(f"   Solo se creará split de TRAIN")

    # Preparar datos para TRAIN
    train_data = {
        "scene_id": [],
        "situation": [],
        "question": [],
        "ground_truth_answer": [],
        "ground_truth_reasoning": [],
        "time_span": [],
        "distractor_1": [],
        "distractor_2": [],
        "distractor_3": [],
        "image_1": [],
        "image_2": [],
        "image_3": [],
        "image_4": [],
        "image_5": [],
    }

    # Preparar datos para VALIDATION (si aplica)
    val_data = None
    if has_validation_cols:
        val_data = {
            "scene_id": [],
            "situation": [],
            "question": [],
            "ground_truth_answer": [],
            "ground_truth_reasoning": [],
            "time_span": [],
            "distractor_1": [],
            "distractor_2": [],
            "distractor_3": [],
            "image_1": [],
            "image_2": [],
            "image_3": [],
            "image_4": [],
            "image_5": [],
        }

    print(f"\n{'='*70}")
    print(f"🔄 PROCESANDO FILAS...")
    print(f"{'='*70}\n")

    skipped_rows = 0

    for idx, row in df.iterrows():
        try:
            scene_id = str(row["scene_id"]).strip()

            print(f"[{idx+1}/{len(df)}] Procesando: {scene_id}")

            # Obtener imágenes automáticamente
            images = get_sorted_images_from_scene(scene_id, images_base_dir)

            # Datos comunes
            common_data = {
                "scene_id": scene_id,
                "situation": str(row["situation"]),
                "question": str(row["question"]),
                "ground_truth_answer": str(row["answer_short"]),
                "ground_truth_reasoning": str(row["answer_reasoning"]),
                "time_span": parse_time_span(row),
            }

            # Agregar a TRAIN
            for key, value in common_data.items():
                train_data[key].append(value)

            # En train, los distractors van vacíos
            train_data["distractor_1"].append("")
            train_data["distractor_2"].append("")
            train_data["distractor_3"].append("")

            for i, img_path in enumerate(images, 1):
                train_data[f"image_{i}"].append(img_path)

            # Agregar a VALIDATION si tiene wrong answers
            if val_data is not None:
                for key, value in common_data.items():
                    val_data[key].append(value)

                # Agregar wrong answers
                val_data["distractor_1"].append(str(row["Wrong answer 1"]))
                val_data["distractor_2"].append(str(row["Wrong answer 2"]))
                val_data["distractor_3"].append(str(row["Wrong answer 3"]))

                for i, img_path in enumerate(images, 1):
                    val_data[f"image_{i}"].append(img_path)

        except Exception as e:
            print(f"   ❌ ERROR procesando fila {idx+1}: {e}")
            skipped_rows += 1
            continue

    print(f"\n{'='*70}")
    print(f"📊 RESUMEN DE PROCESAMIENTO")
    print(f"{'='*70}")
    print(f"✅ Filas procesadas exitosamente: {len(train_data['scene_id'])}")
    if skipped_rows > 0:
        print(f"⏭️  Filas saltadas por errores: {skipped_rows}")

    if len(train_data["scene_id"]) == 0:
        raise ValueError("❌ No se procesó ninguna fila válida")

    # Definir schemas (ambos tienen las mismas columnas ahora)
    train_features = Features(
        {
            "scene_id": Value("string"),
            "situation": Value("string"),
            "question": Value("string"),
            "ground_truth_answer": Value("string"),
            "ground_truth_reasoning": Value("string"),
            "time_span": Sequence(Value("int32")),
            "distractor_1": Value("string"),
            "distractor_2": Value("string"),
            "distractor_3": Value("string"),
            "image_1": Image(),
            "image_2": Image(),
            "image_3": Image(),
            "image_4": Image(),
            "image_5": Image(),
        }
    )

    val_features = Features(
        {
            "scene_id": Value("string"),
            "situation": Value("string"),
            "question": Value("string"),
            "ground_truth_answer": Value("string"),
            "ground_truth_reasoning": Value("string"),
            "time_span": Sequence(Value("int32")),
            "distractor_1": Value("string"),
            "distractor_2": Value("string"),
            "distractor_3": Value("string"),
            "image_1": Image(),
            "image_2": Image(),
            "image_3": Image(),
            "image_4": Image(),
            "image_5": Image(),
        }
    )

    # Crear datasets
    print(f"\n{'='*70}")
    print(f"🏗️  CREANDO DATASETS...")
    print(f"{'='*70}\n")

    train_dataset = Dataset.from_dict(train_data, features=train_features)
    print(f"✅ TRAIN dataset creado: {len(train_dataset)} ejemplos")
    print(f"   - Distractors en train: vacíos ('')")

    datasets = {"train": train_dataset}

    if val_data is not None:
        val_dataset = Dataset.from_dict(val_data, features=val_features)
        print(f"✅ VALIDATION dataset creado: {len(val_dataset)} ejemplos")
        print(f"   - Distractors en validation: poblados con wrong answers")
        datasets["validation"] = val_dataset

    # Crear DatasetDict
    dataset_dict = DatasetDict(datasets)

    # Guardar localmente
    local_path = "AM_dataset"
    print(f"\n💾 Guardando dataset localmente en: {local_path}")
    dataset_dict.save_to_disk(local_path)

    # Mostrar resumen final
    print(f"\n{'='*70}")
    print(f"📦 DATASET FINAL")
    print(f"{'='*70}")
    for split_name, split_data in dataset_dict.items():
        print(f"\n{split_name.upper()}:")
        print(f"  - Ejemplos: {len(split_data)}")
        print(f"  - Columnas: {split_data.column_names}")

    # Subir a HuggingFace si se solicita
    if upload:
        if not hf_repo_name:
            raise ValueError(
                "❌ Debes especificar --repo-name para subir a HuggingFace"
            )

        print(f"\n{'='*70}")
        print(f"☁️  SUBIENDO A HUGGINGFACE HUB")
        print(f"{'='*70}\n")
        print(f"Repositorio: {hf_repo_name}")

        dataset_dict.push_to_hub(hf_repo_name, token=hf_token)

        print(f"\n✅ ¡Dataset subido exitosamente!")
        print(f"   URL: https://huggingface.co/datasets/{hf_repo_name}")
    else:
        print(
            f"\nℹ️  Para subir a HuggingFace Hub, usa: --upload --repo-name tu_usuario/dataset"
        )

    return dataset_dict


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Crea dataset AM desde Excel con train/validation splits"
    )

    parser.add_argument("excel_path", help="Ruta al archivo Excel con los datos")
    parser.add_argument(
        "--images-dir",
        default="real_events_anon_blurred_v2",
        help="Directorio base donde están las carpetas de escenas (default: real_events_anon_blurred_v2)",
    )
    parser.add_argument(
        "--repo-name",
        help="Nombre del repositorio en HuggingFace (formato: usuario/nombre_dataset)",
    )
    parser.add_argument(
        "--token", help="Token de HuggingFace (o usa variable HF_TOKEN)"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Subir el dataset a HuggingFace Hub"
    )

    args = parser.parse_args()

    # Obtener token
    hf_token = args.token or os.environ.get("HF_TOKEN")

    if args.upload and not hf_token:
        print("⚠️  Advertencia: No se proporcionó token de HuggingFace")
        print("   Intentando usar credenciales de 'hf auth login'...")

    # Ejecutar
    dataset_dict = excel_to_am_dataset(
        excel_path=args.excel_path,
        images_base_dir=args.images_dir,
        hf_repo_name=args.repo_name,
        hf_token=hf_token,
        upload=args.upload,
    )

    print(f"\n{'='*70}")
    print(f"🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
