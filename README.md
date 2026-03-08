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

....................................

ChatGPT 5.4:

I’m putting together a standalone GUI wrapper around your existing solver so you can tune the vortex-related parameters interactively, run SAFE vs VORT mode, preview the command, and optionally export MP4 from the same window.

I’ve checked your current solver. One practical point: the attached script always writes an MP4 at the end, so to make “Export MP4” truly optional in the GUI I’m wiring the GUI to a lightly patched runner version rather than just a plain subprocess wrapper around the original file. 

model_g_2d_xy_safe_plus_vort__1a

The GUI file is built and the auto-patcher is working: it successfully generated a sidecar runner from your attached solver with a new --no_mp4 switch, so the MP4 checkbox now has real effect rather than just being cosmetic.

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

The MP4 part needed a small workaround because your original solver always writes an MP4 at the end. 

The GUI auto-patches a runner with --no_mp4 support when needed, so the checkbox is genuinely functional.

Run it with:

python3 model_g_2d_vortex_finder_gui__1a.py

Best setup is to keep the GUI file in the same folder as your solver, though it also has a Browse 

button so you can point it at the solver manually.

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

If you want, the next useful upgrade would be adding an automatic vortex strength / peak ωz 

readout inside the GUI.

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
