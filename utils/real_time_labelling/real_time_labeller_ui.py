#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Etiquetado de Eventos de Conducción - Versión GUI
Interfaz visual con botones grandes para uso táctil en el coche
"""

import json
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional


def format_datetime(dt: datetime) -> str:
    """
    Formatea un datetime de manera legible
    Formato: YYYY-MM-DD HH:MM:SS
    Ejemplo: 2026-02-10 09:37:38
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# CONFIGURACIÓN DE EVENTOS
# ============================================================================
EVENT_CONFIG = {
    # Fila 1 - Traffic events
    "traffic_light": {
        "description": "🚦 Traffic light",
        "color": "#E74C3C",
        "row": 0,
        "col": 0,
    },
    "leading_braking": {
        "description": "🚗 Leading braking",
        "color": "#E67E22",
        "row": 0,
        "col": 1,
    },
    "cut_in": {"description": "↘️ Cut in", "color": "#F39C12", "row": 0, "col": 2},
    # Fila 2 - Infrastructure
    "construction_site": {
        "description": "🚧 Construction site",
        "color": "#F4D03F",
        "row": 1,
        "col": 0,
    },
    "crossing_object": {
        "description": "⚠️ Crossing object",
        "color": "#DC7633",
        "row": 1,
        "col": 1,
    },
    "lateral_parked_car": {
        "description": "🅿️ Lateral parked car",
        "color": "#D68910",
        "row": 1,
        "col": 2,
    },
    # Fila 3 - Vulnerable users & maneuvers
    "vulnerable": {
        "description": "🚶 Vulnerable",
        "color": "#27AE60",
        "row": 2,
        "col": 0,
    },
    "merging_lane": {
        "description": "🔀 Merging lane",
        "color": "#16A085",
        "row": 2,
        "col": 1,
    },
    "intersection": {
        "description": "✖️ Intersection road",
        "color": "#2980B9",
        "row": 2,
        "col": 2,
    },
    # Fila 4 - Navigation, Empty & Discard
    "roundabout": {
        "description": "🔄 Roundabout",
        "color": "#5DADE2",
        "row": 3,
        "col": 0,
    },
    "speed_limit_adaptation": {
        "description": "🔢 Speed limit adaptation",
        "color": "#C0392B",
        "row": 3,
        "col": 1,
    },
    "empty": {
        "description": "⭕ EMPTY (No event)",
        "color": "#95A5A6",
        "row": 3,
        "col": 2,
    },
    # Fila 5 - Special: Discard zone
    "NoKeep": {
        "description": "🚫 NoKeep (Exclude zone)",
        "color": "#34495E",
        "row": 4,
        "col": 0,
        "span": 3,
    },
    # Comentados (no se usan)
    #'lane_change': {'description': '↔️ Lane change', 'color': "#57C265", 'row': 5, 'col': 0},
    #'overtake': {'description': '🏎️ Overtake', 'color': "#5BBC47", 'row': 5, 'col': 1},
    #'emergency_vehicle': {'description': '🚑 Emergency vehicle', 'color': '#EF5350', 'row': 5, 'col': 2},
    #'tunnel': {'description': '🚇 Tunnel', 'color': '#90A4AE', 'row': 5, 'col': 2},
}


# ============================================================================
# CLASE PRINCIPAL CON GUI
# ============================================================================
class DrivingLabelerGUI:
    """
    Sistema de etiquetado visual con interfaz de botones grandes
    """

    def __init__(self):
        self.events: List[Dict] = []
        self.session_start_time: Optional[float] = None
        self.recording_start_time: Optional[float] = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_recording = False
        self.event_count = 0

        # Contador individual por tipo de evento
        self.event_counters: Dict[str, int] = {
            label: 0 for label in EVENT_CONFIG.keys()
        }

        # Directorio de salida
        self.output_dir = Path("labeled_sessions")
        self.output_dir.mkdir(exist_ok=True)

        # Crear ventana principal
        self.root = tk.Tk()
        self.root.title("🚗 Etiquetador de Conducción")
        self.root.configure(bg="#1E1E2E")  # Fondo oscuro moderno

        # Pantalla completa opcional (comentar si no quieres)
        # self.root.attributes('-fullscreen', True)

        # Tamaño mínimo
        self.root.geometry("800x600")

        # Variables de UI
        self.status_label: Optional[tk.Label] = None
        self.timer_label: Optional[tk.Label] = None
        self.event_counter_label: Optional[tk.Label] = None
        self.start_button: Optional[tk.Button] = None
        self.event_buttons: Dict[str, tk.Button] = {}

        # Crear interfaz
        self.create_ui()

        # Timer para actualizar el contador
        self.update_timer()

    def create_ui(self):
        """Crea la interfaz de usuario"""

        # ===== PANEL SUPERIOR: INFORMACIÓN =====
        top_frame = tk.Frame(self.root, bg="#2A2A3E", height=80)
        top_frame.pack(fill=tk.X, padx=6, pady=6)
        top_frame.pack_propagate(False)

        # Título
        title = tk.Label(
            top_frame,
            text="🚗 DRIVING EVENT LABELER",
            font=("Arial", 20, "bold"),
            bg="#2A2A3E",
            fg="#FFFFFF",
        )
        title.pack(pady=(8, 3))

        # ID de sesión
        session_info = tk.Label(
            top_frame,
            text=f"Session: {self.session_id}",
            font=("Arial", 11),
            bg="#2A2A3E",
            fg="#A0A0B0",
        )
        session_info.pack()

        # Status (esperando SESSION_START)
        self.status_label = tk.Label(
            top_frame,
            text="⏳ Press START to sync with GoPro",
            font=("Arial", 13, "bold"),
            bg="#2A2A3E",
            fg="#FFB84D",
        )
        self.status_label.pack(pady=5)

        # ===== PANEL CENTRAL: CONTROLES Y TIMER =====
        control_frame = tk.Frame(self.root, bg="#1E1E2E")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Botón START (grande y visible)
        self.start_button = tk.Button(
            control_frame,
            text="▶️ START RECORDING",
            font=("Arial", 16, "bold"),
            bg="#66BB6A",
            fg="white",
            activebackground="#4CAF50",
            relief=tk.FLAT,
            bd=0,
            height=1,
            command=self.start_recording,
        )
        self.start_button.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        # Timer
        self.timer_label = tk.Label(
            control_frame,
            text="00:00:00",
            font=("Arial", 32, "bold"),
            bg="#1E1E2E",
            fg="#42A5F5",
        )
        self.timer_label.pack(side=tk.LEFT, padx=20)

        # Contador de eventos
        self.event_counter_label = tk.Label(
            control_frame,
            text="Events: 0",
            font=("Arial", 16, "bold"),
            bg="#1E1E2E",
            fg="#FF6B6B",
        )
        self.event_counter_label.pack(side=tk.LEFT, padx=10)

        # ===== PANEL DE EVENTOS: BOTONES GRANDES =====
        events_container = tk.Frame(self.root, bg="#1E1E2E")
        events_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Título de sección
        events_title = tk.Label(
            events_container,
            text="AVAILABLE EVENTS",
            font=("Arial", 14, "bold"),
            bg="#1E1E2E",
            fg="#FFFFFF",
        )
        events_title.pack(pady=(0, 10))

        # Grid de botones
        button_frame = tk.Frame(events_container, bg="#1E1E2E")
        button_frame.pack(expand=True)

        # Crear botones según configuración
        for label, config in EVENT_CONFIG.items():
            colspan = config.get("span", 1)  # Por defecto 1 columna

            btn = tk.Button(
                button_frame,
                text=self.get_button_text(label),
                font=("Arial", 13, "bold"),
                bg=config["color"],
                fg="white",
                disabledforeground="white",  # texto cuando está DISABLED
                activeforeground="white",
                activebackground=self.darken_color(config["color"]),
                relief=tk.FLAT,
                bd=0,
                width=25,
                height=5,
                state=tk.DISABLED,  # Deshabilitado hasta START
                command=lambda l=label: self.register_event(l),
                cursor="hand2",
            )
            btn.grid(
                row=config["row"],
                column=config["col"],
                columnspan=colspan,  # Soporte para botones anchos
                padx=5,
                pady=5,
                sticky="nsew",
            )
            self.event_buttons[label] = btn

        # Configurar grid para que se expanda
        for i in range(5):  # 5 filas
            button_frame.grid_rowconfigure(i, weight=1)
        for i in range(3):  # 3 columnas
            button_frame.grid_columnconfigure(i, weight=1)

        # ===== PANEL INFERIOR: ACCIONES =====
        bottom_frame = tk.Frame(self.root, bg="#2A2A3E", height=80)
        bottom_frame.pack(fill=tk.X, padx=12, pady=12)
        bottom_frame.pack_propagate(False)

        # Botón UNDO
        undo_btn = tk.Button(
            bottom_frame,
            text="↶ UNDO LAST",
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            activebackground="#F57C00",
            relief=tk.FLAT,
            bd=0,
            height=2,
            command=self.undo_last_event,
            cursor="hand2",
        )
        undo_btn.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        # Botón FINALIZAR
        finish_btn = tk.Button(
            bottom_frame,
            text="⏹️ FINISH SESSION",
            font=("Arial", 12, "bold"),
            bg="#EF5350",
            fg="white",
            activebackground="#E53935",
            relief=tk.FLAT,
            bd=0,
            height=2,
            command=self.finish_session,
            cursor="hand2",
        )
        finish_btn.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH, expand=True)

        # Log de eventos recientes
        log_label = tk.Label(
            bottom_frame,
            text="Recent events will appear here",
            font=("Arial", 10),
            bg="#2A2A3E",
            fg="#A0A0B0",
            anchor="w",
        )
        log_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        self.log_label = log_label

        # Bind ESC para salir
        self.root.bind("<Escape>", lambda e: self.finish_session())

    def get_button_text(self, label: str) -> str:
        """
        Genera el texto del botón incluyendo el contador
        Formato: "🚦 Traffic light\n[3 times]"
        """
        config = EVENT_CONFIG[label]
        count = self.event_counters[label]

        if count == 0:
            return config["description"]
        elif count == 1:
            return f"{config['description']}\n[{count} time]"
        else:
            return f"{config['description']}\n[{count} times]"

    def update_button_text(self, label: str):
        """
        Actualiza el texto del botón con el contador actual
        """
        btn = self.event_buttons[label]
        btn.config(text=self.get_button_text(label))

    def darken_color(self, hex_color: str, factor: float = 0.8) -> str:
        """Oscurece un color hexadecimal para el efecto hover"""
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        dark_rgb = tuple(int(c * factor) for c in rgb)
        return f"#{dark_rgb[0]:02x}{dark_rgb[1]:02x}{dark_rgb[2]:02x}"

    def start_recording(self):
        """Marca el inicio de la grabación (SESSION_START)"""
        if self.is_recording:
            messagebox.showinfo("Info", "Recording already active")
            return

        self.session_start_time = time.time()
        self.recording_start_time = time.time()
        self.is_recording = True

        # Cambiar UI
        self.status_label.config(
            text="🎬 RECORDING ACTIVE - Logging events", fg="#66BB6A"
        )
        self.start_button.config(
            state=tk.DISABLED, bg="#78909C", text="✓ RECORDING STARTED"
        )

        # Habilitar botones de eventos
        for btn in self.event_buttons.values():
            btn.config(state=tk.NORMAL)

        # Registrar evento SESSION_START
        self.events.append(
            {
                "timestamp": 0.0,
                "label": "SESSION_START",
                "description": "Inicio de grabación sincronizado",
                "datetime": format_datetime(datetime.now()),
                "absolute_time": datetime.now().isoformat(),
            }
        )

        print(f"🎬 SESSION_START registered - {datetime.now()}")
        self.log_label.config(text="🎬 SESSION_START registered")

    def register_event(self, label: str):
        """Registra un evento de conducción"""
        if not self.is_recording:
            messagebox.showwarning("Warning", "Start recording first with START button")
            return

        current_time = time.time()
        timestamp = current_time - self.recording_start_time
        event_info = EVENT_CONFIG[label]
        now = datetime.now()

        event = {
            "timestamp": round(timestamp, 2),
            "label": label,
            "description": event_info["description"],
            "datetime": format_datetime(now),
        }

        self.events.append(event)
        self.event_count += 1

        # Incrementar contador individual de este tipo de evento
        self.event_counters[label] += 1

        # Actualizar texto del botón con el contador
        self.update_button_text(label)

        # Actualizar contador total
        self.event_counter_label.config(text=f"Events: {self.event_count}")

        # Feedback visual con información del contador
        count_text = f"(Total: {self.event_counters[label]})"
        print(
            f"✓ [{timestamp:7.2f}s] {event_info['description']} ({label}) {count_text}"
        )
        self.log_label.config(
            text=f"✓ [{timestamp:7.2f}s] {event_info['description']} {count_text}"
        )

        # Efecto visual en botón (flash)
        btn = self.event_buttons[label]
        original_bg = btn.cget("bg")
        btn.config(bg="#FFFFFF")
        self.root.after(100, lambda: btn.config(bg=original_bg))

    def undo_last_event(self):
        """Elimina el último evento registrado"""
        if len(self.events) <= 1:  # Solo SESSION_START
            messagebox.showinfo("Info", "No events to remove")
            return

        removed = self.events.pop()
        self.event_count -= 1

        # Decrementar contador individual del tipo de evento removido
        removed_label = removed["label"]
        if removed_label in self.event_counters:
            self.event_counters[removed_label] -= 1
            # Actualizar texto del botón
            self.update_button_text(removed_label)

        # Actualizar contador total
        self.event_counter_label.config(text=f"Events: {self.event_count}")

        print(f"↶ Removed: [{removed['timestamp']:7.2f}s] {removed['description']}")
        self.log_label.config(text=f"↶ Removed: {removed['description']}")

    def update_timer(self):
        """Actualiza el timer cada segundo"""
        if self.is_recording and self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.timer_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        # Programar siguiente actualización
        self.root.after(1000, self.update_timer)

    def finish_session(self):
        """Finaliza la sesión y guarda los datos"""
        if not self.is_recording:
            if messagebox.askyesno(
                "Confirm", "Recording not started. Exit without saving?"
            ):
                self.root.destroy()
            return

        # Confirmar
        if not messagebox.askyesno(
            "Finish Session", f"Save session with {self.event_count} events?"
        ):
            return

        # Preparar datos (incluir contadores individuales)
        session_data = {
            "session_id": self.session_id,
            "start_datetime": format_datetime(
                datetime.fromtimestamp(self.session_start_time)
            ),
            "recording_start_datetime": format_datetime(
                datetime.fromtimestamp(self.recording_start_time)
            ),
            "total_duration_seconds": round(time.time() - self.recording_start_time, 2),
            "total_events": self.event_count,
            "event_counters": self.event_counters,  # Añadir contadores individuales
            "events": self.events,
            "event_config_used": EVENT_CONFIG,
        }

        # Guardar JSON
        output_file = self.output_dir / f"{self.session_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        # Mensaje de éxito con breakdown de eventos
        breakdown = "\n".join(
            [
                f"{label}: {count}"
                for label, count in self.event_counters.items()
                if count > 0
            ]
        )

        messagebox.showinfo(
            "Session Saved",
            f"✅ Session saved successfully\n\n"
            f"📁 File: {output_file}\n"
            f"📊 Total Events: {self.event_count}\n"
            f"⏱️ Duration: {session_data['total_duration_seconds']:.2f}s\n\n"
            f"Event Breakdown:\n{breakdown}",
        )

        print(f"\n✅ Session saved: {output_file}")
        print(f"📊 Total events: {self.event_count}")
        print(f"\nEvent breakdown:")
        for label, count in self.event_counters.items():
            if count > 0:
                print(f"  {label}: {count}")

        # Cerrar aplicación
        self.root.destroy()

    def run(self):
        """Inicia la aplicación"""
        print("\n" + "=" * 70)
        print("  DRIVING EVENT LABELER - GUI VERSION")
        print("=" * 70)
        print(f"\nSession: {self.session_id}")
        print(f"Output: {self.output_dir / f'{self.session_id}.json'}")
        print("\nGraphical interface started. Use on-screen buttons.")
        print("=" * 70 + "\n")

        self.root.mainloop()


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
def main():
    """Función principal"""
    app = DrivingLabelerGUI()
    app.run()


if __name__ == "__main__":
    main()
