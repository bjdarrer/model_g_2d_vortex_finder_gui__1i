#!/usr/bin/env python3
"""
Model G 2D Vortex Finder GUI

A Tkinter GUI wrapper for Brendan's solver:
    model_g_2d_xy_safe_plus_vort__1a.py

Main features
-------------
- Edit the main CLI parameters from the original solver.
- SAFE vs VORT mode switching.
- Good support for vortex investigation: rotseed, omega_drive, quiver, vorticity limits.
- Command preview, live log, stop button.
- Optional MP4 export.
- Auto-patches a sidecar runner with --no_mp4 support when needed.
- Vortex diagnostics panel that reads the latest checkpoint and estimates vortex strength/center.

- Written by Brendan Darrer aided by ChatGPT 5.4 date: 8th March 2026 ~01:00 GMT
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT 5.4 writing it and Brendan guiding it to produce a clean code.

===========================
ChatGPT 5.4:

I’m putting together a standalone GUI wrapper around your existing solver so you can tune the vortex-related parameters interactively, 
run SAFE vs VORT mode, preview the command, and optionally export MP4 from the same window.

I’ve checked your current solver. One practical point: the attached script always writes an MP4 at the end, so to make “Export MP4” 
truly optional in the GUI I’m wiring the GUI to a lightly patched runner version rather than just a plain subprocess wrapper around the original file. 

model_g_2d_xy_safe_plus_vort__1a

The GUI file is built and the auto-patcher is working: it successfully generated a sidecar runner from your attached solver with a 
new --no_mp4 switch, so the MP4 checkbox now has real effect rather than just being cosmetic.

Thought for 3m 53s

Done — I created a standalone GUI wrapper for your solver.

The GUI is here: model_g_2d_vortex_finder_gui__1a.py

I also generated a patched runner here: model_g_2d_xy_safe_plus_vort__1a__gui_runner.py

I based it directly on the argument structure and run-flow in your attached model_g_2d_xy_safe_plus_vort__1a.py solver. 

model_g_2d_xy_safe_plus_vort__1a

What the GUI gives you:

SAFE vs VORT mode switch

editable fields for the main parameters you listed

quick vortex controls for rotseed, quiver, omega_drive, swirl_amp, swirl_sigma, nu, vortlim

presets like SAFE baseline, VORT seed test, and Driven vortex

live command preview

live log window

Run and Stop buttons

real Export MP4 on/off

The MP4 part needed a small workaround because your original solver always writes an MP4 at the end. The GUI auto-patches a runner 
with --no_mp4 support when needed, so the checkbox is genuinely functional.

Run it with:

python3 model_g_2d_vortex_finder_gui__1a.py

Best setup is to keep the GUI file in the same folder as your solver, though it also has a Browse button so you can point it at the 
solver manually.

A few practical notes:

In SAFE mode it sends --no_vort.

In VORT mode, rotseed gives the initial swirl and omega_drive adds persistent tangential driving.

If Export MP4 is unticked, the GUI uses the patched runner and passes --no_mp4.

A good first vortex test from the GUI is:

mode = VORT

rotseed = on

quiver = on

omega_drive = 0.0 first, then try 0.2 to 0.8

vary swirl_amp, swirl_sigma, nu, and vortlim

If you want, the next useful upgrade would be adding an automatic vortex strength / peak ωz readout inside the GUI.
..............

You’ve now got a usable vortex GUI with:

SAFE / VORT switching

parameter editing

optional MP4 export

command preview

diagnostics

working layout

The next most useful upgrade would probably be either:

a live ωz / vortex-strength readout during the run, or

an Open latest output folder button  -----> DONE
===========================
"""

from __future__ import annotations

import os
import re
import sys
import shlex
import queue
import threading
import subprocess
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -----------------------------------------------------------------------------
# Defaults copied from Brendan's attached solver
# -----------------------------------------------------------------------------
DEFAULTS = {
    # domain/grid
    "Lx": 20.0,
    "Ly": 20.0,
    "nx": 160,
    "ny": 160,
    # time
    "Tfinal": 40.0,
    "segment_dt": 0.5,
    "nt_anim": 480,
    # solver
    "method": "RK23",
    "max_step": 0.01,
    "atol": 1e-6,
    "rtol": 1e-6,
    # model G
    "a": 14.0,
    "b": 29.0,
    "Dg": 1.0,
    "Dx": 1.0,
    "Dy": 12.0,
    "pcoef": 1.0,
    "qcoef": 1.0,
    "gcoef": 0.1,
    "scoef": 0.0,
    "ucross": 0.0,
    # forcing
    "Tseed": 10.0,
    "seed_sigma_space": 2.0,
    "seed_sigma_time": 3.0,
    "seed_center_x": "",
    "seed_center_y": "",
    # vortical motion
    "no_vort": False,
    "alphaG": 0.02,
    "alphaX": 0.02,
    "alphaY": 0.02,
    "cs2": 1.0,
    "nu": 0.25,
    # rotational seed / drive
    "rotseed": False,
    "swirl_amp": 1.0,
    "swirl_sigma": 6.0,
    "swirl_cx": "",
    "swirl_cy": "",
    "omega_drive": 0.0,
    # viz
    "zlim": 1.0,
    "vortlim": "",
    "quiver": True,
    "quiver_stride": 8,
    # GUI-only
    "export_mp4": True,
}

FIELD_SPECS = [
    ("Domain / grid", [
        ("Lx", "float"), ("Ly", "float"), ("nx", "int"), ("ny", "int"),
    ]),
    ("Time", [
        ("Tfinal", "float"), ("segment_dt", "float"), ("nt_anim", "int"),
    ]),
    ("Solver", [
        ("method", "str"), ("max_step", "float"), ("atol", "float"), ("rtol", "float"),
    ]),
    ("Model G", [
        ("a", "float"), ("b", "float"), ("Dg", "float"), ("Dx", "float"), ("Dy", "float"),
        ("pcoef", "float"), ("qcoef", "float"), ("gcoef", "float"), ("scoef", "float"), ("ucross", "float"),
    ]),
    ("Forcing / seed", [
        ("Tseed", "float"), ("seed_sigma_space", "float"), ("seed_sigma_time", "float"),
        ("seed_center_x", "optfloat"), ("seed_center_y", "optfloat"),
    ]),
    ("Vortical motion", [
        ("alphaG", "float"), ("alphaX", "float"), ("alphaY", "float"), ("cs2", "float"), ("nu", "float"),
    ]),
    ("Rotational seed / drive", [
        ("swirl_amp", "float"), ("swirl_sigma", "float"), ("swirl_cx", "optfloat"), ("swirl_cy", "optfloat"), ("omega_drive", "float"),
    ]),
    ("Visualisation", [
        ("zlim", "float"), ("vortlim", "optfloat"), ("quiver_stride", "int"),
    ]),
]

PRESETS = {
    "SAFE baseline": {
        "no_vort": True,
        "rotseed": False,
        "omega_drive": 0.0,
        "quiver": False,
        "vortlim": "",
    },
    "VORT seed test": {
        "no_vort": False,
        "rotseed": True,
        "swirl_amp": 1.0,
        "swirl_sigma": 6.0,
        "omega_drive": 0.0,
        "quiver": True,
        "quiver_stride": 8,
    },
    "Driven vortex": {
        "no_vort": False,
        "rotseed": True,
        "swirl_amp": 1.0,
        "swirl_sigma": 6.0,
        "omega_drive": 0.5,
        "quiver": True,
        "quiver_stride": 8,
        "vortlim": 0.5,
    },
}


class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        if self.winfo_ismapped():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if not self.winfo_ismapped():
            return
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class VortexFinderGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Model G 2D Vortex Finder GUI")
        self._set_initial_geometry()
        self.root.minsize(900, 700)

        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.generated_runner: Path | None = None
        self.run_buttons: list[ttk.Button] = []
        self.stop_buttons: list[ttk.Button] = []
        self.last_run_context: dict[str, object] | None = None
        self.live_diag_after_id: str | None = None
        self.live_diag_interval_ms = 1500

        self.vars: dict[str, tk.Variable] = {}
        self._build_vars()
        self._build_ui()
        self._auto_detect_solver()
        self._update_mode_state()
        self._update_command_preview()
        self._set_diagnostics_empty("No diagnostics yet")
        self.root.after(100, self._poll_log_queue)

    def _set_initial_geometry(self):
        """Choose a startup size that fits on the current screen."""
        try:
            sw = max(800, int(self.root.winfo_screenwidth()))
            sh = max(700, int(self.root.winfo_screenheight()))
        except Exception:
            sw, sh = 1200, 900

        width = min(1160, max(900, int(sw * 0.88)))
        height = min(820, max(640, int(sh * 0.76)))
        x = max(0, (sw - width) // 2)
        y = max(0, (sh - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_vars(self):
        for key, value in DEFAULTS.items():
            if isinstance(value, bool):
                self.vars[key] = tk.BooleanVar(value=value)
            elif isinstance(value, int):
                self.vars[key] = tk.IntVar(value=value)
            elif isinstance(value, float):
                self.vars[key] = tk.StringVar(value=str(value))
            else:
                self.vars[key] = tk.StringVar(value=value)

        self.vars["solver_path"] = tk.StringVar(value="")
        self.vars["python_exec"] = tk.StringVar(value=sys.executable or "python3")
        self.vars["preset"] = tk.StringVar(value="VORT seed test")
        self.vars["command_preview"] = tk.StringVar(value="")
        self.vars["status"] = tk.StringVar(value="Ready")

        self.vars["diag_status"] = tk.StringVar(value="No diagnostics yet")
        self.vars["diag_mode"] = tk.StringVar(value="—")
        self.vars["diag_time"] = tk.StringVar(value="—")
        self.vars["diag_peak_abs"] = tk.StringVar(value="—")
        self.vars["diag_max"] = tk.StringVar(value="—")
        self.vars["diag_min"] = tk.StringVar(value="—")
        self.vars["diag_peak_loc"] = tk.StringVar(value="—")
        self.vars["diag_center"] = tk.StringVar(value="—")
        self.vars["diag_ckpt"] = tk.StringVar(value="—")
        self.vars["diag_live"] = tk.StringVar(value="—")

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.LabelFrame(outer, text="Solver / run setup", padding=10)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="Python").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.vars["python_exec"], width=35).grid(row=0, column=1, sticky="ew", padx=(5, 12))

        ttk.Label(top, text="Solver script").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.vars["solver_path"], width=70).grid(row=0, column=3, sticky="ew", padx=(5, 5))
        ttk.Button(top, text="Browse…", command=self._choose_solver).grid(row=0, column=4, sticky="ew")

        ttk.Label(top, text="Preset").grid(row=1, column=0, sticky="w", pady=(8, 0))
        preset_box = ttk.Combobox(top, textvariable=self.vars["preset"], values=list(PRESETS.keys()), state="readonly", width=22)
        preset_box.grid(row=1, column=1, sticky="w", padx=(5, 12), pady=(8, 0))
        preset_box.bind("<<ComboboxSelected>>", lambda _e: self.apply_preset())

        ttk.Checkbutton(top, text="Export MP4", variable=self.vars["export_mp4"], command=self._update_command_preview).grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Restore defaults", command=self.restore_defaults).grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Copy command", command=self.copy_command).grid(row=1, column=4, sticky="ew", pady=(8, 0))
        ttk.Button(top, text="Open output folder", command=self.open_latest_output_folder).grid(row=1, column=5, sticky="ew", padx=(8, 0), pady=(8, 0))

        toolbar = ttk.Frame(top)
        toolbar.grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 0))
        top_run_btn = ttk.Button(toolbar, text="Run simulation", command=self.run_simulation)
        top_run_btn.pack(side="left")
        top_stop_btn = ttk.Button(toolbar, text="Stop", command=self.stop_simulation, state="disabled")
        top_stop_btn.pack(side="left", padx=(8, 0))
        ttk.Label(toolbar, text="A second Run/Stop pair is also shown at the bottom.").pack(side="left", padx=(12, 0))
        self.run_buttons.append(top_run_btn)
        self.stop_buttons.append(top_stop_btn)

        top.columnconfigure(3, weight=1)

        center = ttk.PanedWindow(outer, orient="horizontal")
        center.pack(fill="both", expand=True)

        left_wrap = ttk.Frame(center)
        right_wrap = ttk.Frame(center)
        center.add(left_wrap, weight=2)
        center.add(right_wrap, weight=1)

        self.left_scroll = ScrollableFrame(left_wrap)
        self.left_scroll.pack(fill="both", expand=True)
        self._build_parameter_panels(self.left_scroll.inner)

        self.right_scroll = ScrollableFrame(right_wrap)
        self.right_scroll.pack(fill="both", expand=True)
        self._build_right_panel(self.right_scroll.inner)

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(10, 0))
        bottom.columnconfigure(2, weight=1)

        bottom_run_btn = ttk.Button(bottom, text="Run simulation", command=self.run_simulation)
        bottom_run_btn.grid(row=0, column=0, sticky="w")
        bottom_stop_btn = ttk.Button(bottom, text="Stop", command=self.stop_simulation, state="disabled")
        bottom_stop_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.run_buttons.append(bottom_run_btn)
        self.stop_buttons.append(bottom_stop_btn)

        status_frame = ttk.Frame(bottom)
        status_frame.grid(row=0, column=2, sticky="e")
        ttk.Label(status_frame, text="Status:").pack(side="left", padx=(12, 6))
        ttk.Label(status_frame, textvariable=self.vars["status"], anchor="e", justify="right").pack(side="left")

        for key, var in self.vars.items():
            if key not in {"status", "command_preview", "diag_status", "diag_mode", "diag_time", "diag_peak_abs", "diag_max", "diag_min", "diag_peak_loc", "diag_center", "diag_ckpt", "diag_live"}:
                try:
                    var.trace_add("write", lambda *_args: self._on_any_change())
                except Exception:
                    pass

    def _build_parameter_panels(self, parent):
        mode_frame = ttk.LabelFrame(parent, text="Mode / quick vortex controls", padding=10)
        mode_frame.pack(fill="x", pady=(0, 10))

        self.mode_var = tk.StringVar(value="vort")
        ttk.Radiobutton(mode_frame, text="SAFE (no vortical motion)", value="safe", variable=self.mode_var, command=self._mode_changed).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="VORT (vortical motion enabled)", value="vort", variable=self.mode_var, command=self._mode_changed).grid(row=0, column=1, sticky="w", padx=(15, 0))

        self.chk_rotseed = ttk.Checkbutton(mode_frame, text="rotseed", variable=self.vars["rotseed"], command=self._update_command_preview)
        self.chk_rotseed.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.chk_quiver = ttk.Checkbutton(mode_frame, text="quiver", variable=self.vars["quiver"], command=self._update_command_preview)
        self.chk_quiver.grid(row=1, column=1, sticky="w", pady=(8, 0))

        hint = ttk.Label(
            mode_frame,
            text="Typical vortex search: VORT + rotseed + quiver, then vary omega_drive, swirl_amp, swirl_sigma, nu and vortlim.",
            foreground="#444444",
            wraplength=780,
        )
        hint.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        highlight_names = {"swirl_amp", "swirl_sigma", "swirl_cx", "swirl_cy", "omega_drive", "alphaG", "alphaX", "alphaY", "cs2", "nu", "quiver_stride", "vortlim"}

        for section_title, fields in FIELD_SPECS:
            frame = ttk.LabelFrame(parent, text=section_title, padding=10)
            frame.pack(fill="x", pady=(0, 10))
            for i, (name, _kind) in enumerate(fields):
                r = i // 3
                c = (i % 3) * 2
                ttk.Label(frame, text=name).grid(row=r, column=c, sticky="w", padx=(0, 6), pady=4)
                entry = ttk.Entry(frame, textvariable=self.vars[name], width=18)
                entry.grid(row=r, column=c + 1, sticky="w", padx=(0, 18), pady=4)
                if name in highlight_names:
                    entry.configure(style="Vortex.TEntry")
            for cc in range(0, 6, 2):
                frame.columnconfigure(cc + 1, weight=1)

    def _build_right_panel(self, parent):
        cmd_frame = ttk.LabelFrame(parent, text="Command preview", padding=10)
        cmd_frame.pack(fill="x")
        self.cmd_text = tk.Text(cmd_frame, height=7, wrap="word")
        self.cmd_text.pack(fill="both", expand=True)

        notes_frame = ttk.LabelFrame(parent, text="Notes", padding=10)
        notes_frame.pack(fill="x", pady=(10, 10))
        notes = (
            "SAFE mode sends --no_vort.\n\n"
            "VORT mode enables the velocity fields; rotseed gives an initial swirl and omega_drive adds a persistent tangential drive.\n\n"
            "If Export MP4 is unticked, the GUI auto-builds a patched runner with --no_mp4 and uses that instead of the original solver."
        )
        ttk.Label(notes_frame, text=notes, wraplength=330, justify="left").pack(fill="x")

        diag_frame = ttk.LabelFrame(parent, text="Vortex diagnostics", padding=10)
        diag_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(diag_frame, textvariable=self.vars["diag_status"], wraplength=330, justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        diag_rows = [
            ("Detected mode", "diag_mode"),
            ("Checkpoint time", "diag_time"),
            ("Live readout", "diag_live"),
            ("|ωz| peak", "diag_peak_abs"),
            ("ωz max", "diag_max"),
            ("ωz min", "diag_min"),
            ("Peak ωz location", "diag_peak_loc"),
            ("Weighted center", "diag_center"),
            ("Checkpoint", "diag_ckpt"),
        ]
        for i, (label, key) in enumerate(diag_rows, start=1):
            ttk.Label(diag_frame, text=label).grid(row=i, column=0, sticky="nw", padx=(0, 8), pady=2)
            ttk.Label(diag_frame, textvariable=self.vars[key], wraplength=235, justify="left").grid(row=i, column=1, sticky="w", pady=2)
        diag_btns = ttk.Frame(diag_frame)
        diag_btns.grid(row=len(diag_rows) + 1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(diag_btns, text="Refresh diagnostics", command=self.refresh_diagnostics).pack(side="left")
        ttk.Button(diag_btns, text="Open latest output folder", command=self.open_latest_output_folder).pack(side="left", padx=(8, 0))
        ttk.Label(diag_frame, text="Live diagnostics auto-refresh during a run. Tip: the whole right panel now scrolls if the window is too short.", wraplength=330, justify="left").grid(row=len(diag_rows) + 2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        diag_frame.columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(parent, text="Run log", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(0, 6))
        self.log_text = tk.Text(log_frame, wrap="word", height=12)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    # ------------------------------------------------------------------
    # State / preview helpers
    # ------------------------------------------------------------------
    def _on_any_change(self):
        self._update_mode_state()
        self._update_command_preview()

    def _mode_changed(self):
        self.vars["no_vort"].set(self.mode_var.get() == "safe")
        self._update_mode_state()
        self._update_command_preview()

    def _update_mode_state(self):
        safe = bool(self.vars["no_vort"].get())
        self.mode_var.set("safe" if safe else "vort")
        if safe:
            self.chk_rotseed.state(["disabled"])
            self.chk_quiver.state(["disabled"])
        else:
            self.chk_rotseed.state(["!disabled"])
            self.chk_quiver.state(["!disabled"])

    def _update_command_preview(self):
        try:
            cmd = self.build_command(validate=False)
            preview = shlex.join(cmd)
        except Exception as e:
            preview = f"Command preview unavailable: {e}"
        self.vars["command_preview"].set(preview)
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", preview)

    def _append_log(self, text: str):
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _poll_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log(msg)
        self.root.after(100, self._poll_log_queue)

    # ------------------------------------------------------------------
    # Solver detection / patching
    # ------------------------------------------------------------------
    def _auto_detect_solver(self):
        candidates = [
            Path.cwd() / "model_g_2d_xy_safe_plus_vort__1a.py",
            Path(__file__).resolve().parent / "model_g_2d_xy_safe_plus_vort__1a.py",
            Path("/mnt/data/model_g_2d_xy_safe_plus_vort__1a.py"),
        ]
        for p in candidates:
            if p.exists():
                self.vars["solver_path"].set(str(p))
                return

    def _choose_solver(self):
        path = filedialog.askopenfilename(
            title="Choose the base solver script",
            filetypes=[("Python files", "*.py"), ("All files", "*")],
        )
        if path:
            self.vars["solver_path"].set(path)
            self._update_command_preview()

    def _ensure_no_mp4_runner(self, source_path: Path) -> Path:
        text = source_path.read_text(encoding="utf-8")
        if "--no_mp4" in text:
            return source_path

        marker = 'ap.add_argument("--quiver_stride", type=int, default=8)'
        if marker not in text:
            raise RuntimeError("Could not find quiver_stride argument in the solver, so auto-patching failed.")

        text = text.replace(
            marker,
            marker + '\nap.add_argument("--no_mp4", action="store_true", help="Skip MP4 assembly at end")',
            1,
        )

        pattern = re.compile(
            r'\n\s*# assemble MP4\n\s*print\("\[Video\] Writing MP4:", mp4_path\)\n\s*fps = .*?\n\s*with imageio\.get_writer\(mp4_path, fps=fps\) as writer:\n\s*for i in range\(args\.nt_anim\):\n\s*img = imageio\.imread\(os\.path\.join\(frames_dir, f"frame_\{i:04d\}\.png"\)\)\n\s*writer\.append_data\(img\)\n\s*print\("\[Done\] MP4 saved:", mp4_path\)',
            re.DOTALL,
        )

        replacement = '''
    # assemble MP4
    if args.no_mp4:
        print("[Video] Skipped MP4 assembly because --no_mp4 was requested")
    else:
        print("[Video] Writing MP4:", mp4_path)
        fps = max(8, int(args.nt_anim / max(1.0, args.Tfinal / 2.0)))
        with imageio.get_writer(mp4_path, fps=fps) as writer:
            for i in range(args.nt_anim):
                img = imageio.imread(os.path.join(frames_dir, f"frame_{i:04d}.png"))
                writer.append_data(img)
        print("[Done] MP4 saved:", mp4_path)'''

        new_text, nsubs = pattern.subn(replacement, text, count=1)
        if nsubs != 1:
            raise RuntimeError("Could not patch the MP4 assembly block cleanly.")

        # Also fix the checkpoint temp-file naming so live diagnostics can follow runs
        # cleanly when the patched runner is used.
        new_text = new_text.replace(
            'tmp = ckpt_path + ".tmp"',
            'tmp = ckpt_path + ".tmp.npz"',
            1,
        )

        runner_path = source_path.with_name(source_path.stem + "__gui_runner.py")
        runner_path.write_text(new_text, encoding="utf-8")
        self.generated_runner = runner_path
        return runner_path

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _set_diagnostics_empty(self, message: str = "No diagnostics yet"):
        self.vars["diag_status"].set(message)
        for key in ["diag_mode", "diag_time", "diag_live", "diag_peak_abs", "diag_max", "diag_min", "diag_peak_loc", "diag_center", "diag_ckpt"]:
            self.vars[key].set("—")

    def _format_num(self, value: float, digits: int = 6) -> str:
        if value is None or not np.isfinite(value):
            return "—"
        return f"{float(value):.{digits}g}"

    def _format_xy(self, xy: tuple[float, float] | None) -> str:
        if xy is None:
            return "—"
        x, y = xy
        if not (np.isfinite(x) and np.isfinite(y)):
            return "—"
        return f"({x:.4g}, {y:.4g})"

    def _expected_checkpoint_path(self, vals: dict[str, object], base_dir: Path) -> Path:
        base_name = "model_g_2d_xy_safe_plus_vort__1a"
        run_name = base_name + ("_novort" if bool(vals["no_vort"]) else "_vort")
        return base_dir / f"out_{run_name}" / "checkpoint_2d_plus_vort.npz"

    def _candidate_checkpoint_paths(self, ckpt_path: Path) -> list[Path]:
        # The current solver writes via:
        #   tmp = ckpt_path + ".tmp"
        #   np.savez_compressed(tmp, ...)
        # which actually creates "...tmp.npz" because NumPy appends ".npz" when needed.
        # So live diagnostics should look for both the intended checkpoint name and the
        # temporary filename that really appears on disk.
        candidates = [
            ckpt_path,
            Path(str(ckpt_path) + ".tmp"),
            Path(str(ckpt_path) + ".tmp.npz"),
        ]
        seen: set[str] = set()
        out: list[Path] = []
        for p in candidates:
            s = str(p)
            if s not in seen:
                out.append(p)
                seen.add(s)
        return out

    def _resolve_checkpoint_path(self, ckpt_path: Path) -> Path:
        existing = []
        for p in self._candidate_checkpoint_paths(ckpt_path):
            try:
                if p.exists():
                    existing.append(p)
            except Exception:
                pass
        if not existing:
            return ckpt_path
        try:
            return max(existing, key=lambda p: p.stat().st_mtime)
        except Exception:
            return existing[0]

    def _compute_vortex_diagnostics(self, ckpt_path: Path, vals: dict[str, object]) -> dict[str, object]:
        d = np.load(ckpt_path, allow_pickle=True)
        y_curr = np.asarray(d["y_curr"]).ravel()
        t_curr = float(d["t_curr"])

        nx = int(vals["nx"])
        ny = int(vals["ny"])
        Lx = float(vals["Lx"])
        Ly = float(vals["Ly"])
        N = nx * ny

        if y_curr.size == 3 * N:
            with_vort = False
            ux = np.zeros((ny, nx), dtype=float)
            uy = np.zeros((ny, nx), dtype=float)
        elif y_curr.size >= 5 * N:
            with_vort = True
            ux = y_curr[3 * N:4 * N].reshape(ny, nx)
            uy = y_curr[4 * N:5 * N].reshape(ny, nx)
        else:
            raise ValueError(
                f"Checkpoint size {y_curr.size} does not match nx*ny={N}. "
                "Try refreshing with the same grid settings that were used for the run."
            )

        x = np.linspace(-Lx / 2.0, Lx / 2.0, nx)
        y = np.linspace(-Ly / 2.0, Ly / 2.0, ny)
        dx = float(x[1] - x[0]) if nx > 1 else 1.0
        dy = float(y[1] - y[0]) if ny > 1 else 1.0

        def gradx(u):
            gx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
            gx[:, 0] = 0.0
            gx[:, -1] = 0.0
            return gx

        def grady(u):
            gy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
            gy[0, :] = 0.0
            gy[-1, :] = 0.0
            return gy

        vort = gradx(uy) - grady(ux)
        vmax = float(np.max(vort))
        vmin = float(np.min(vort))
        abs_vort = np.abs(vort)
        peak_abs = float(np.max(abs_vort))
        peak_idx = np.unravel_index(int(np.argmax(abs_vort)), abs_vort.shape)
        peak_loc = (float(x[peak_idx[1]]), float(y[peak_idx[0]]))

        center = None
        if peak_abs > 0.0:
            weights = np.where(abs_vort >= 0.5 * peak_abs, abs_vort, 0.0)
            wsum = float(np.sum(weights))
            if wsum > 0.0:
                Xg, Yg = np.meshgrid(x, y, indexing="xy")
                center = (
                    float(np.sum(weights * Xg) / wsum),
                    float(np.sum(weights * Yg) / wsum),
                )

        return {
            "mode": "VORT" if with_vort else "SAFE",
            "t_curr": t_curr,
            "peak_abs": peak_abs,
            "vmax": vmax,
            "vmin": vmin,
            "peak_loc": peak_loc,
            "center": center if center is not None else peak_loc,
            "ckpt_path": str(ckpt_path),
        }

    def _update_live_readout(self):
        mode = self.vars["diag_mode"].get()
        tval = self.vars["diag_time"].get()
        peak = self.vars["diag_peak_abs"].get()
        center = self.vars["diag_center"].get()
        if mode == "—" or tval == "—" or peak == "—":
            self.vars["diag_live"].set("Waiting for checkpoint…")
            return
        summary = f"{mode} | t={tval} | |ωz|={peak}"
        if center not in ("—", "", None):
            summary += f" | center {center}"
        self.vars["diag_live"].set(summary)

    def _schedule_live_diagnostics(self):
        self._cancel_live_diagnostics()
        self.live_diag_after_id = self.root.after(self.live_diag_interval_ms, self._live_diag_tick)

    def _cancel_live_diagnostics(self):
        if self.live_diag_after_id is not None:
            try:
                self.root.after_cancel(self.live_diag_after_id)
            except Exception:
                pass
            self.live_diag_after_id = None

    def _live_diag_tick(self):
        self.live_diag_after_id = None
        try:
            if self.process is not None and self.process.poll() is None:
                self.refresh_diagnostics(silent=True)
                self._schedule_live_diagnostics()
        except Exception:
            # Keep the GUI responsive even if a checkpoint is temporarily unreadable.
            if self.process is not None and self.process.poll() is None:
                self._schedule_live_diagnostics()

    def refresh_diagnostics(self, silent: bool = False):
        try:
            if self.last_run_context is not None:
                vals = dict(self.last_run_context["vals"])
                ckpt_path = Path(self.last_run_context["ckpt_path"])
            else:
                vals = self.collect_values(validate=True)
                solver_path_raw = self.vars["solver_path"].get().strip()
                if not solver_path_raw:
                    raise ValueError("Please choose the solver script first.")
                solver_dir = Path(solver_path_raw).expanduser().resolve().parent
                ckpt_path = self._expected_checkpoint_path(vals, solver_dir)

            resolved_ckpt = self._resolve_checkpoint_path(ckpt_path)
            if not resolved_ckpt.exists():
                candidates = "\n".join(str(p) for p in self._candidate_checkpoint_paths(ckpt_path))
                self._set_diagnostics_empty(f"Checkpoint not found yet. Looked for:\n{candidates}")
                self.vars["diag_live"].set("Waiting for checkpoint…")
                return

            diag = self._compute_vortex_diagnostics(resolved_ckpt, vals)
            if resolved_ckpt == ckpt_path:
                self.vars["diag_status"].set("Diagnostics loaded from latest checkpoint")
            else:
                self.vars["diag_status"].set(f"Diagnostics loaded from temporary checkpoint: {resolved_ckpt.name}")
            self.vars["diag_mode"].set(diag["mode"])
            self.vars["diag_time"].set(self._format_num(diag["t_curr"]))
            self.vars["diag_peak_abs"].set(self._format_num(diag["peak_abs"]))
            self.vars["diag_max"].set(self._format_num(diag["vmax"]))
            self.vars["diag_min"].set(self._format_num(diag["vmin"]))
            self.vars["diag_peak_loc"].set(self._format_xy(diag["peak_loc"]))
            self.vars["diag_center"].set(self._format_xy(diag["center"]))
            self.vars["diag_ckpt"].set(str(diag["ckpt_path"]))
            self._update_live_readout()
        except Exception as e:
            if silent:
                self.vars["diag_status"].set(f"Live refresh waiting: {e}")
                self.vars["diag_live"].set("Live refresh waiting for checkpoint…")
            else:
                self._set_diagnostics_empty(f"Diagnostics unavailable: {e}")

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def restore_defaults(self):
        for key, default in DEFAULTS.items():
            var = self.vars[key]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(default))
            elif isinstance(var, tk.IntVar):
                var.set(int(default))
            else:
                var.set(str(default))
        self.vars["preset"].set("VORT seed test")
        self._update_mode_state()
        self._update_command_preview()

    def apply_preset(self):
        preset = PRESETS.get(self.vars["preset"].get(), {})
        for key, value in preset.items():
            var = self.vars[key]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            elif isinstance(var, tk.IntVar):
                var.set(int(value))
            else:
                var.set(str(value))
        self._update_mode_state()
        self._update_command_preview()

    def _parse_number(self, name: str, kind: str):
        raw_value = self.vars[name].get()
        raw = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
        if kind == "optfloat":
            return None if raw in ("", None) else float(raw)
        if kind == "str":
            return raw if isinstance(raw, str) else str(raw)
        raise ValueError(f"Unknown kind: {kind}")

    def collect_values(self, validate: bool = True) -> dict[str, object]:
        vals: dict[str, object] = {}
        for _section, fields in FIELD_SPECS:
            for name, kind in fields:
                try:
                    vals[name] = self._parse_number(name, kind)
                except Exception as e:
                    if validate:
                        raise ValueError(f"Invalid value for {name}: {e}")
                    vals[name] = self.vars[name].get()

        vals["no_vort"] = bool(self.vars["no_vort"].get())
        vals["rotseed"] = bool(self.vars["rotseed"].get())
        vals["quiver"] = bool(self.vars["quiver"].get())
        vals["export_mp4"] = bool(self.vars["export_mp4"].get())

        if vals["seed_center_x"] is None or vals["seed_center_y"] is None:
            if (vals["seed_center_x"] is None) != (vals["seed_center_y"] is None):
                raise ValueError("seed_center_x and seed_center_y must either both be blank or both be set.")
        if vals["swirl_cx"] is None or vals["swirl_cy"] is None:
            if (vals["swirl_cx"] is None) != (vals["swirl_cy"] is None):
                raise ValueError("swirl_cx and swirl_cy must either both be blank or both be set.")

        if int(vals["nx"]) < 8 or int(vals["ny"]) < 8:
            raise ValueError("nx and ny should both be at least 8.")
        if float(vals["Tfinal"]) <= 0:
            raise ValueError("Tfinal must be > 0.")
        if int(vals["nt_anim"]) < 2:
            raise ValueError("nt_anim must be at least 2.")
        if float(vals["segment_dt"]) <= 0 or float(vals["max_step"]) <= 0:
            raise ValueError("segment_dt and max_step must be > 0.")
        if int(vals["quiver_stride"]) < 1:
            raise ValueError("quiver_stride must be >= 1.")
        return vals

    def build_command(self, validate: bool = True) -> list[str]:
        solver_path_raw = self.vars["solver_path"].get().strip()
        if not solver_path_raw:
            raise ValueError("Please choose the solver script first.")
        source_path = Path(solver_path_raw).expanduser().resolve()
        if validate and not source_path.exists():
            raise ValueError(f"Solver script not found: {source_path}")

        vals = self.collect_values(validate=validate)
        python_exec = self.vars["python_exec"].get().strip() or sys.executable or "python3"

        solver_to_run = source_path
        if not vals["export_mp4"]:
            solver_to_run = self._ensure_no_mp4_runner(source_path)

        cmd = [python_exec, str(solver_to_run)]
        ordered_fields = [
            "Lx", "Ly", "nx", "ny",
            "Tfinal", "segment_dt", "nt_anim",
            "method", "max_step", "atol", "rtol",
            "a", "b", "Dg", "Dx", "Dy", "pcoef", "qcoef", "gcoef", "scoef", "ucross",
            "Tseed", "seed_sigma_space", "seed_sigma_time",
            "alphaG", "alphaX", "alphaY", "cs2", "nu",
            "swirl_amp", "swirl_sigma", "omega_drive",
            "zlim", "quiver_stride",
        ]
        for key in ordered_fields:
            cmd.extend([f"--{key}", str(vals[key])])

        if vals["seed_center_x"] is not None and vals["seed_center_y"] is not None:
            cmd.extend(["--seed_center", str(vals["seed_center_x"]), str(vals["seed_center_y"])])
        if vals["swirl_cx"] is not None and vals["swirl_cy"] is not None:
            cmd.extend(["--swirl_cx", str(vals["swirl_cx"]), "--swirl_cy", str(vals["swirl_cy"])])
        if vals["vortlim"] is not None:
            cmd.extend(["--vortlim", str(vals["vortlim"])])

        if vals["no_vort"]:
            cmd.append("--no_vort")
        if vals["rotseed"] and not vals["no_vort"]:
            cmd.append("--rotseed")
        if vals["quiver"] and not vals["no_vort"]:
            cmd.append("--quiver")
        if not vals["export_mp4"]:
            cmd.append("--no_mp4")
        return cmd

    # ------------------------------------------------------------------
    # Run / stop / logging
    # ------------------------------------------------------------------
    def run_simulation(self):
        if self.process is not None:
            # Treat a dead process as finished even if the wait-thread has not
            # yet cleared self.process on the Tk thread.
            if self.process.poll() is None:
                messagebox.showinfo("Already running", "A simulation is already running. See the Run log / status line for its PID and output folder.")
                return
            self._on_process_finished(self.process.returncode if self.process.returncode is not None else -1)

        try:
            vals = self.collect_values(validate=True)
            cmd = self.build_command(validate=True)
        except Exception as e:
            messagebox.showerror("Cannot run", str(e))
            return

        solver_dir = Path(self.vars["solver_path"].get().strip()).expanduser().resolve().parent
        ckpt_path = self._expected_checkpoint_path(vals, solver_dir)
        out_dir = ckpt_path.parent
        self.last_run_context = {
            "vals": vals,
            "solver_dir": str(solver_dir),
            "ckpt_path": str(ckpt_path),
            "out_dir": str(out_dir),
        }

        self.log_text.delete("1.0", "end")
        self._append_log("Running command:\n" + shlex.join(cmd) + "\n\n")
        self._append_log(f"[GUI] Working directory: {solver_dir}\n")
        self._append_log(f"[GUI] Expected output folder: {out_dir}\n")
        self._append_log("[GUI] Note: solver stdout is captured into this Run log, not your launching terminal.\n\n")
        self.vars["status"].set("Running…")
        self._set_diagnostics_empty("Waiting for a checkpoint from the current run…")
        self.vars["diag_live"].set("Waiting for checkpoint…")
        for btn in self.run_buttons:
            btn.configure(state="disabled")
        for btn in self.stop_buttons:
            btn.configure(state="normal")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(solver_dir),
            )
        except Exception as e:
            self._cancel_live_diagnostics()
            self.process = None
            self.vars["status"].set("Launch failed")
            for btn in self.run_buttons:
                btn.configure(state="normal")
            for btn in self.stop_buttons:
                btn.configure(state="disabled")
            messagebox.showerror("Launch failed", str(e))
            return

        self.vars["status"].set(f"Running (PID {self.process.pid})")
        self._append_log(f"[GUI] Started solver PID: {self.process.pid}\n\n")
        threading.Thread(target=self._reader_thread, daemon=True).start()
        threading.Thread(target=self._wait_thread, daemon=True).start()
        self._schedule_live_diagnostics()
    def _reader_thread(self):
        assert self.process is not None
        try:
            for line in self.process.stdout:
                self.log_queue.put(line)
        except Exception as e:
            self.log_queue.put(f"\n[GUI] Log reader error: {e}\n")

    def _wait_thread(self):
        assert self.process is not None
        code = self.process.wait()
        self.root.after(0, lambda: self._on_process_finished(code))

    def _on_process_finished(self, code: int):
        self._cancel_live_diagnostics()
        self.process = None
        for btn in self.run_buttons:
            btn.configure(state="normal")
        for btn in self.stop_buttons:
            btn.configure(state="disabled")

        if code == 0:
            self.vars["status"].set("Finished")
            self._append_log("\n[GUI] Simulation finished successfully.\n")
            self.refresh_diagnostics()
        else:
            self.vars["status"].set(f"Stopped / failed (exit {code})")
            self._append_log(f"\n[GUI] Simulation exited with code {code}.\n")
            if self.last_run_context is not None and Path(self.last_run_context["ckpt_path"]).exists():
                self.refresh_diagnostics()

    def stop_simulation(self):
        if self.process is None:
            return
        try:
            self.process.terminate()
            self._cancel_live_diagnostics()
            self._append_log("\n[GUI] Stop requested…\n")
            self.vars["status"].set("Stopping…")
        except Exception as e:
            messagebox.showerror("Stop failed", str(e))

    def _guess_output_folder(self) -> Path | None:
        if self.last_run_context is not None:
            out_dir = Path(self.last_run_context["out_dir"])
            if out_dir.exists():
                return out_dir
        try:
            vals = self.collect_values(validate=True)
            solver_path_raw = self.vars["solver_path"].get().strip()
            if not solver_path_raw:
                return None
            solver_dir = Path(solver_path_raw).expanduser().resolve().parent
            return self._expected_checkpoint_path(vals, solver_dir).parent
        except Exception:
            return None

    def open_latest_output_folder(self):
        out_dir = self._guess_output_folder()
        if out_dir is None:
            messagebox.showerror("Open output folder", "Could not determine the output folder yet.")
            return
        if not out_dir.exists():
            messagebox.showinfo("Open output folder", f"Output folder does not exist yet:\n{out_dir}")
            return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(out_dir)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out_dir)])
            elif os.name == "nt":
                os.startfile(str(out_dir))
            else:
                messagebox.showinfo("Open output folder", f"Please open this folder manually:\n{out_dir}")
                return
            self.vars["status"].set("Opened output folder")
        except Exception as e:
            messagebox.showerror("Open output folder", f"Could not open folder:\n{out_dir}\n\n{e}")

    def copy_command(self):
        text = self.vars["command_preview"].get()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.vars["status"].set("Command copied")


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Vortex.TEntry")
    VortexFinderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
