import os
import json
import time
import argparse
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory, abort
from flask_cors import CORS

app = Flask(__name__, static_folder=None)
CORS(app)


IMAGES_ROOT = None
STATE_FILE = "labeling_state.json"
DATASET_TAG = None
state_lock = threading.Lock()


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"cases": [], "assignments": {}, "completed": {}, "labels": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def extract_dataset_tag(state_filename: str) -> str:
    """
    Extrae un tag corto del nombre del fichero de estado para usarlo en el Excel.
    Ejemplos:
      labeling_state_events_4.json  →  events_4
      labeling_state_events_1.json  →  events_1
      labeling_state.json           →  (vacío, no se añade sufijo)
      my_custom_state.json          →  my_custom_state
    """
    stem = Path(state_filename).stem

    prefix = "labeling_state_"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix) :]
    if stem == "labeling_state":
        return ""
    return stem


FOLDER_TO_SITUATION = {
    "traffic_light": "Traffic light",
    "leading_braking": "Leading braking",
    "cut_in": "Cut in",
    "construction_site": "Construction site",
    "crossing_object": "Crossing object",
    "lateral_parked_car": "Lateral parked car",
    "pedestrian": "Pedestrian",
    "vulnerable": "Vulnerable",
    "merging_lane": "Merging lane",
    "intersection_road": "Intersection road",
    "intersection": "Intersection",
    "roundabout": "Roundabout",
    "speed_limit_adaptation": "Speed limit adaptation",
}


def folder_to_prelabel(folder_name: str) -> str:
    key = folder_name.lower().replace(" ", "_").replace("-", "_")
    return FOLDER_TO_SITUATION.get(key, folder_name.replace("_", " ").title())


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_cases(root: Path):
    """
    Escanea root buscando la estructura:
      root / <situacion> / <escena> / imagen1.jpg ...

    Devuelve lista de dicts con sceneId, prelabel, folderPath, images.
    """
    cases = []
    for situation_dir in sorted(root.iterdir()):
        if not situation_dir.is_dir():
            continue
        prelabel = folder_to_prelabel(situation_dir.name)
        for scene_dir in sorted(situation_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            imgs = sorted(
                p for p in scene_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
            )
            if not imgs:
                continue
            rel_imgs = [str(p.relative_to(root)).replace("\\", "/") for p in imgs]
            cases.append(
                {
                    "sceneId": scene_dir.name,
                    "prelabel": prelabel,
                    "folderPath": str(scene_dir.relative_to(root)).replace("\\", "/"),
                    "images": rel_imgs,
                }
            )
    return cases


def assign_cases(state, labeler: str, batch_size: int = 0):
    """
    Sistema de lotes (batches):
    - batch_size=0 y labeler nuevo → asigna todos los casos libres.
    - batch_size>0 → reserva exactamente ese número de casos libres.
    - labeler ya existente + batch_size=0 → devuelve sus casos actuales sin cambios.
    Devuelve (lista_de_casos, casos_libres_restantes_tras_asignacion).
    """
    state["assignments"].setdefault(labeler, [])
    state["completed"].setdefault(labeler, [])
    state["labels"].setdefault(labeler, [])

    taken = set()
    for idxs in state["assignments"].values():
        taken.update(idxs)
    free_before = [i for i in range(len(state["cases"])) if i not in taken]

    if batch_size > 0:
        new_batch = free_before[:batch_size]
        state["assignments"][labeler] = state["assignments"][labeler] + new_batch
        save_state(state)
    elif not state["assignments"][labeler]:
        state["assignments"][labeler] = free_before
        save_state(state)

    current_idxs = state["assignments"][labeler]

    taken_after = set()
    for idxs in state["assignments"].values():
        taken_after.update(idxs)
    free_after = len([i for i in range(len(state["cases"])) if i not in taken_after])

    completed_ids = set(state["completed"].get(labeler, []))
    assigned_cases = []
    for i in current_idxs:
        if i < len(state["cases"]):
            c = dict(state["cases"][i])
            c["done"] = c["sceneId"] in completed_ids
            assigned_cases.append(c)
    return assigned_cases, free_after


@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok", "server": "servidor_colaborativo"})


@app.route("/api/dataset_tag")
def get_dataset_tag():
    """El cliente consulta aquí el tag del dataset para usarlo en el nombre del Excel."""
    return jsonify({"dataset_tag": DATASET_TAG or ""})


@app.route("/api/cases")
def get_cases():
    labeler = request.args.get("labeler", "").strip()
    batch_size = int(request.args.get("batch", 0))

    if not labeler:
        return jsonify({"error": "labeler param required"}), 400

    with state_lock:
        state = load_state()
        if not state["cases"]:
            if IMAGES_ROOT is None:
                return (
                    jsonify({"error": "Server not configured with --images path"}),
                    500,
                )
            state["cases"] = scan_cases(IMAGES_ROOT)
            save_state(state)

        taken = set()
        for idxs in state["assignments"].values():
            taken.update(idxs)
        own = set(state["assignments"].get(labeler, []))

        cases, free_after = assign_cases(state, labeler, batch_size)

    return jsonify(
        {
            "labeler": labeler,
            "cases": cases,
            "total_all": len(state["cases"]),
            "assigned": len(cases),
            "completed": len([c for c in cases if c["done"]]),
            "free_cases": free_after,
            "dataset_tag": DATASET_TAG or "",
        }
    )


@app.route("/api/rescan")
def rescan():
    """Fuerza un nuevo escaneo (útil si añades más carpetas)."""
    with state_lock:
        state = load_state()
        if IMAGES_ROOT is None:
            return jsonify({"error": "No images root configured"}), 500
        state["cases"] = scan_cases(IMAGES_ROOT)
        state["assignments"] = {}
        state["completed"] = {}
        state["labels"] = {}
        save_state(state)
    return jsonify({"status": "rescanned", "total": len(state["cases"])})


@app.route("/api/complete", methods=["POST"])
def mark_complete():
    data = request.get_json(silent=True) or {}
    labeler = data.get("labeler", "").strip()
    scene_id = data.get("sceneId", "").strip()
    label = data.get("label")

    if not labeler or not scene_id:
        return jsonify({"error": "labeler and sceneId required"}), 400

    with state_lock:
        state = load_state()
        state["completed"].setdefault(labeler, [])
        if scene_id not in state["completed"][labeler]:
            state["completed"][labeler].append(scene_id)
        if label:
            state["labels"].setdefault(labeler, [])
            existing = next(
                (
                    i
                    for i, l in enumerate(state["labels"][labeler])
                    if l.get("sceneId") == scene_id
                ),
                None,
            )
            if existing is not None:
                state["labels"][labeler][existing] = label
            else:
                state["labels"][labeler].append(label)
        save_state(state)

    return jsonify({"status": "ok"})


@app.route("/api/status")
def status():
    with state_lock:
        state = load_state()
    summary = {}
    for labeler, idxs in state["assignments"].items():
        done = len(state["completed"].get(labeler, []))
        summary[labeler] = {
            "assigned": len(idxs),
            "completed": done,
            "remaining": len(idxs) - done,
        }
    return jsonify({"total_cases": len(state["cases"]), "labelers": summary})


@app.route("/images/<path:filepath>")
def serve_image(filepath):
    if IMAGES_ROOT is None:
        abort(404)
    safe = Path(IMAGES_ROOT / filepath).resolve()
    if not str(safe).startswith(str(IMAGES_ROOT.resolve())):
        abort(403)
    if not safe.exists():
        abort(404)
    return send_file(safe)


@app.route("/")
@app.route("/index.html")
def serve_app():
    html_path = Path(__file__).parent / "labeling_app.html"
    if not html_path.exists():
        return "<h1>Error</h1><p>labeling_app.html not found next to server.py</p>", 404
    return send_file(html_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Servidor de labelización colaborativa"
    )
    parser.add_argument(
        "--images", required=True, help="Ruta a la carpeta raíz de imágenes"
    )
    parser.add_argument(
        "--state",
        default="labeling_state.json",
        help=(
            "Fichero JSON de estado (default: labeling_state.json). "
            "Si existe, se carga el progreso anterior. "
            "Si no existe, se crea nuevo. "
            "Ejemplo: --state labeling_state_events_4.json"
        ),
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--host", default="0.0.0.0", help="0.0.0.0 para que accedan desde la red local"
    )
    args = parser.parse_args()

    # Fijar variables globales
    STATE_FILE = args.state
    DATASET_TAG = extract_dataset_tag(args.state)

    IMAGES_ROOT = Path(args.images).resolve()
    if not IMAGES_ROOT.exists():
        print(f"❌ Error: la carpeta '{IMAGES_ROOT}' no existe.")
        exit(1)

    # Cargar o crear state
    with state_lock:
        state = load_state()
        if os.path.exists(STATE_FILE):
            print(
                f"✅ Estado cargado desde '{STATE_FILE}': {len(state['cases'])} casos."
            )
        else:
            print(f"🆕 Fichero '{STATE_FILE}' no existe, se creará uno nuevo.")

        if not state["cases"]:
            print("🔍 Escaneando imágenes...")
            state["cases"] = scan_cases(IMAGES_ROOT)
            save_state(state)
            print(f"   {len(state['cases'])} casos encontrados.")

    tag_info = f' · Dataset tag: "{DATASET_TAG}"' if DATASET_TAG else ""

    import socket

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "TU_IP_LOCAL"

    print(
        f"""
╔══════════════════════════════════════════════════════╗
║       Servidor de Labelización Colaborativa          ║
╠══════════════════════════════════════════════════════╣
║  Local:        http://localhost:{args.port:<5}               ║
║  Red local:    http://{local_ip}:{args.port:<5}         ║
║  Imágenes en:  {str(IMAGES_ROOT)[:40]:<40}  ║
║  State:        {STATE_FILE:<40}  ║
╚══════════════════════════════════════════════════════╝
  Dataset tag: "{DATASET_TAG or '(ninguno)'}"
  Para ver el estado:  http://localhost:{args.port}/api/status
  Para re-escanear:    http://localhost:{args.port}/api/rescan
"""
    )

    app.run(host=args.host, port=args.port, debug=False)
