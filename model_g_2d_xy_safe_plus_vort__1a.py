"""
Model G Particle 2D (proper x–y) — SAFE base + optional vortical motion + quiver-by-speed


- Written by Brendan Darrer aided by ChatGPT5 date: 27th February 2026 18:43 GMT
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT 5.2 writing it and Brendan guiding it to produce a clean code.

Install:
  pip3 install numpy scipy matplotlib imageio imageio[ffmpeg]

...........NOTES on design of this code.......................
Brendan: "It seems that code "model_g_2d_xy_vortical_plus__4f.py" (attached), when run for no vortical motion, does not produce the soliton particle well. It's kind of fuzzy and disappears towards the end of the animation. Can you remake it again from "model_g_2d_xy_safe__2a.py" attached here. "model_g_2d_xy_safe__2a.py" is the Model G in 2D only, that works well. Can you adapt it to include vortical motion and no vortical motion options, similar to what you did in "model_g_2d_xy_vortical_plus__4f.py" (attached), but with corrections made as described here? Also, can you include a constant z-axis (not a changing one) for the 4 plots; and make the quiver arrows in it (the 4th plot), all the same size and the magnitude of the arrows indicated by color?"
Thought for 3m 6s

Yep — I rebuilt it from your working baseline model_g_2d_xy_safe__2a.py, and then added vortical motion as an option, while ensuring that --no_vort runs the same “safe” reaction–diffusion evolution (so it shouldn’t go fuzzy / disappear like the spectral version did).

I also added:

Constant (“fixed”) z-scale for all 4 panels using --zlim (and optional --vortlim for the vorticity panel)

Quiver arrows all the same length, with magnitude encoded by color (--quiver)

New script

Download: model_g_2d_xy_safe_plus_vort__1a.py

1) SAFE mode (no vortical motion) — should match your good soliton behaviour
python3 model_g_2d_xy_safe_plus_vort__1a.py --no_vort
2) Vortical motion ON (adds ux,uy evolution + scalar advection)
python3 model_g_2d_xy_safe_plus_vort__1a.py --rotseed --quiver --quiver_stride 8
3) Fixed scales (constant z-axis)

Potentials fixed by:

--zlim 1.0

Vorticity fixed by:

--vortlim 0.5

Example:

python3 model_g_2d_xy_safe_plus_vort__1a.py --rotseed --quiver --zlim 1.0 --vortlim 0.5

Notes on what’s different vs the old __4f.py

The “SAFE” path is truly SAFE now: when --no_vort is passed, it integrates only (pG,pX,pY) with the same finite-difference Dirichlet formulation (no hidden velocity/advection/vorticity side-effects).

The vortical version uses the same finite-difference operators/BC style, so it’s much closer in “feel” to the safe code than the periodic spectral solver.

If you run this and want the vortical mode to “spin” more persistently, use --omega_drive (it’s off by default).
..................................


Built from: model_g_2d_xy_safe__2a.py (finite-difference, Dirichlet BC, solve_ivp segmented checkpoints)
Adds:
  - Optional vortical motion (ux,uy evolved) and scalar advection
  - --no_vort to reproduce SAFE behaviour (no velocity, no advection)
  - Fixed (constant) color limits for all 4 panels via --zlim and --vortlim
  - Quiver with constant-length arrows; speed encoded by color

Outputs under out_<run_name>/:
  - frames/frame_XXXX.png
  - <run_name>.mp4
  - checkpoint_2d_plus_vort.npz (auto-resume)
  
================ step-by-step, plain-English walkthrough ==============================
=========================
QUICK STEP-BY-STEP MAP
=========================
STEP 1  Parse CLI options (grid, time, parameters, plotting switches)
STEP 2  Set output paths (frames/, checkpoint, mp4)
STEP 3  Build the x–y grid (X,Y mesh + dx,dy)
STEP 4  Compute background steady state (G0,X0,Y0)
STEP 5  Define finite-difference operators + boundary clamping
STEP 6  Define the forcing pulse χ(x,y,t)
STEP 7  Pack/unpack fields for solve_ivp
STEP 8  Define RHS (SAFE and VORT) = diffusion + reactions (+ advection/velocity)
STEP 9  Render 4 panels (pX,pY,pG, ωz) + quiver-by-speed
STEP 10 Checkpoint save/load
STEP 11 Segmented integration + render + MP4 assembly


Below is a step-by-step, plain-English walkthrough of what model_g_2d_xy_safe_plus_vort__1a.py is doing. 

model_g_2d_xy_safe_plus_vort__1a

1) What the program is for

It simulates a 2D Model-G “particle/soliton” on an x–y grid using three fields (potentials):

pG (G-potential)

pX (X-potential)

pY (Y-potential)

Optionally it also simulates a 2D velocity field:

ux, uy (fluid velocity components)

So you can run it in two modes:

SAFE mode (--no_vort): only pG, pX, pY evolve (no flow, no advection).

VORT mode (default): pG, pX, pY evolve and they are carried around by ux,uy (“advection”), while ux,uy evolve too.

2) Command-line options (CLI)

At the top it defines lots of --args so you can control:

grid size (--nx --ny) and physical size (--Lx --Ly)

total time (--Tfinal), solver settings (--method, --max_step, tolerances)

Model-G parameters (a, b, Dg, Dx, Dy, ...)

seeding pulse (the “kick” that creates the particle) (--Tseed, --seed_sigma_*)

vortical options (--no_vort, --rotseed, viscosity --nu, etc.)

plotting controls (--zlim, --vortlim, --quiver, --quiver_stride) 

model_g_2d_xy_safe_plus_vort__1a

3) Output folders and checkpointing

It creates an output directory like:

out_model_g_2d_xy_safe_plus_vort__1a_novort/ (SAFE)
or

out_model_g_2d_xy_safe_plus_vort__1a_vort/ (VORT)

Inside it writes:

frames/frame_XXXX.png (animation frames)

a final MP4 video

a checkpoint file checkpoint_2d_plus_vort.npz so it can resume after a crash or interruption. 

model_g_2d_xy_safe_plus_vort__1a

4) The grid (the “world”)

It builds a 2D grid:

x from -Lx/2 to +Lx/2

y from -Ly/2 to +Ly/2

makes X, Y mesh arrays so every point knows its (x,y)

computes dx, dy spacing

This is the computational “canvas” where the particle forms. 

model_g_2d_xy_safe_plus_vort__1a

5) The “background” steady state (G0, X0, Y0)

The code computes constants G0, X0, Y0 which are the homogeneous equilibrium (the uniform background solution).

Important idea:

The simulation variables pG, pX, pY are perturbations around that background.

The total fields are G0+pG, X0+pX, Y0+pY. 

model_g_2d_xy_safe_plus_vort__1a

6) Finite-difference operators + boundary handling

This is the “SAFE” style you liked.

It defines:

laplacian_2d(u): approximate ∇²u using central differences (with np.roll)

gradx(u), grady(u): approximate ∂u/∂x, ∂u/∂y

_clamp_dirichlet_edges(u): forces boundaries to 0

Conceptually:

diffusion uses the Laplacian

advection and vorticity use gradients

edges are “Dirichlet” (pinned/zero) so nothing weird wraps around at borders. 

model_g_2d_xy_safe_plus_vort__1a

7) The seeding pulse χ(x,y,t)

A soliton particle doesn’t usually pop out of nowhere; you “kick” the system.

chi_xy_t(t) builds a Gaussian blob in space (centered at (0,0) by default), multiplied by a Gaussian in time centered at Tseed.

It returns a negative forcing term, so around t ≈ Tseed it perturbs the fields and helps the particle nucleate. 

model_g_2d_xy_safe_plus_vort__1a

8) Packing/unpacking state for the ODE solver

solve_ivp expects a single 1D vector.

So the code “packs” the 2D arrays into one long 1D vector:

SAFE: [pG, pX, pY]

VORT: [pG, pX, pY, ux, uy]

and unpacks them back into 2D arrays inside the RHS functions. 

model_g_2d_xy_safe_plus_vort__1a

9) The Model-G chemistry/reaction part

reaction_terms(pG,pX,pY,forcing) computes the local nonlinear reactions (no diffusion, no advection—just the pointwise kinetics).

It:

forms Xtot = pX+X0 and Ytot = pY+Y0

builds nonlinear terms like X^3 and X^2 Y

returns dG, dX, dY from the Model-G style equations plus the forcing χ.

This is the part that can create/maintain a localized structure when combined with diffusion. 

model_g_2d_xy_safe_plus_vort__1a

10) SAFE mode RHS (no vortical motion)

rhs_no_vort(t, yflat) does:

unpack pG,pX,pY

compute forcing χ(t)

compute reactions rG,rX,rY

add diffusion: D * laplacian(p)

return [dpGdt, dpXdt, dpYdt]

So SAFE mode is basically:

diffusion + nonlinear reactions + time/space forcing
and nothing is carried around by fluid flow. 

model_g_2d_xy_safe_plus_vort__1a

11) VORT mode RHS (adds flow + advection)

rhs_with_vort(t, yflat) does everything SAFE does, plus:

(a) Advection of pG,pX,pY

It computes terms like:

advG = ux * ∂pG/∂x + uy * ∂pG/∂y

and subtracts them from the time derivatives:

dpGdt = diffusion - advG + reaction

Meaning:

the flow field “moves” the scalar fields around.

(b) Velocity evolution

velocity_rhs(ux,uy,pG,pX,pY) updates the flow using a “toy compressible fluid” model:

it builds a “density” rho from the potentials (coupling)

uses gradients of log(rho) like a pressure force

includes convection (u·∇)u

includes viscosity (nu) via Laplacians

optional swirl forcing (--omega_drive) that keeps spinning if desired

It returns dux/dt, duy/dt. 

model_g_2d_xy_safe_plus_vort__1a

12) Rendering: the four plots

For each frame it makes a 2×2 figure:

pX/10 (scaled so it fits nice in the same range)

pY

pG

fluid vorticity ωz = ∂uy/∂x − ∂ux/∂y

Fixed “z-axis”

--zlim forces the same color limits for pX,pY,pG every frame:

always [-zlim, +zlim]

So the colors don’t “auto rescale” over time. 

model_g_2d_xy_safe_plus_vort__1a

Quiver (same arrow length, color shows speed)

If --quiver is enabled:

it samples the velocity on a coarse grid (stride)

computes speed = sqrt(U^2 + V^2)

normalizes direction so arrows are equal length

uses speed as the color of arrows

So:

arrow direction = flow direction

arrow color = how fast the flow is there 

model_g_2d_xy_safe_plus_vort__1a

13) Time integration strategy (segmented solve_ivp)

Instead of one huge solve:

it integrates in chunks of length --segment_dt (default 0.5)

For each segment:

determine which animation frame times fall in this segment

call solve_ivp(...) with t_eval at those frame times

render those frames from the returned solution

update t_curr, y_curr

checkpoint progress

This makes it:

more robust (less likely to lose everything)

able to resume mid-run easily. 

model_g_2d_xy_safe_plus_vort__1a

14) Building the MP4

After all frames are written, it reads them back in order and writes an MP4 using imageio / ffmpeg.
==================================================================================
"""

import os
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio.v2 as imageio

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
# domain/grid
ap.add_argument("--Lx", type=float, default=20.0)
ap.add_argument("--Ly", type=float, default=20.0)
ap.add_argument("--nx", type=int, default=160)
ap.add_argument("--ny", type=int, default=160)
# time
ap.add_argument("--Tfinal", type=float, default=40.0)
ap.add_argument("--segment_dt", type=float, default=0.5)
ap.add_argument("--nt_anim", type=int, default=480)
# solver (used in both modes)
ap.add_argument("--method", type=str, default="RK23", help="solve_ivp method, e.g. RK23, BDF")
ap.add_argument("--max_step", type=float, default=0.01)
ap.add_argument("--atol", type=float, default=1e-6)
ap.add_argument("--rtol", type=float, default=1e-6)

# Model-G params (eqs17 style)
ap.add_argument("--a", type=float, default=14.0)
ap.add_argument("--b", type=float, default=29.0)
ap.add_argument("--Dg", type=float, default=1.0)
ap.add_argument("--Dx", type=float, default=1.0)
ap.add_argument("--Dy", type=float, default=12.0)
ap.add_argument("--pcoef", type=float, default=1.0)
ap.add_argument("--qcoef", type=float, default=1.0)
ap.add_argument("--gcoef", type=float, default=0.1)
ap.add_argument("--scoef", type=float, default=0.0)
ap.add_argument("--ucross", type=float, default=0.0)

# forcing
ap.add_argument("--Tseed", type=float, default=10.0)
ap.add_argument("--seed_sigma_space", type=float, default=2.0)
ap.add_argument("--seed_sigma_time", type=float, default=3.0)
ap.add_argument("--seed_center", type=float, nargs=2, default=None)

# vortical motion (optional)
ap.add_argument("--no_vort", action="store_true", help="Disable vortical motion (SAFE mode: ux=uy=0, no advection)")
ap.add_argument("--alphaG", type=float, default=0.02)
ap.add_argument("--alphaX", type=float, default=0.02)
ap.add_argument("--alphaY", type=float, default=0.02)
ap.add_argument("--cs2", type=float, default=1.0)
ap.add_argument("--nu", type=float, default=0.25)

# rotational seed/drive
ap.add_argument("--rotseed", action="store_true", help="Initialize a swirling velocity field (only if vort enabled)")
ap.add_argument("--swirl_amp", type=float, default=1.0)
ap.add_argument("--swirl_sigma", type=float, default=6.0)
ap.add_argument("--swirl_cx", type=float, default=None)
ap.add_argument("--swirl_cy", type=float, default=None)
ap.add_argument("--omega_drive", type=float, default=0.0, help="If nonzero, apply persistent tangential body-force drive")

# viz
ap.add_argument("--zlim", type=float, default=1.0, help="Fixed color limits for pX,pY,pG panels: [-zlim,+zlim]")
ap.add_argument("--vortlim", type=float, default=None, help="Fixed color limits for vorticity: [-vortlim,+vortlim]. Default: auto per-frame")
ap.add_argument("--quiver", action="store_true", help="Overlay quiver arrows on vorticity panel")
ap.add_argument("--quiver_stride", type=int, default=8)

args = ap.parse_args()

# ---------------- Paths ----------------
base_name = "model_g_2d_xy_safe_plus_vort__1a"
run_name = base_name + ("_novort" if args.no_vort else "_vort")
out_dir = f"out_{run_name}"
frames_dir = os.path.join(out_dir, "frames")
ckpt_path = os.path.join(out_dir, "checkpoint_2d_plus_vort.npz")
mp4_path = os.path.join(out_dir, f"{run_name}.mp4")
final_png = os.path.join(out_dir, "final_snapshot.png")
os.makedirs(frames_dir, exist_ok=True)

# ---------------- Grid ----------------
Lx, Ly = args.Lx, args.Ly
nx, ny = args.nx, args.ny
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
X, Y = np.meshgrid(x, y, indexing="xy")
dx = x[1] - x[0]
dy = y[1] - y[0]
N = nx * ny

# ---------------- Params & homogeneous state ----------------
a = args.a
b = args.b
p_par = args.pcoef
q_par = args.qcoef
g_par = args.gcoef
s_par = args.scoef
u_cross = args.ucross

G0 = (a + g_par*0.0) / (q_par - g_par*p_par)
X0 = (p_par*a + q_par*0.0) / (q_par - g_par*p_par)
Y0 = ((s_par*X0**2 + b) * X0 / (X0**2 + u_cross)) if (X0**2 + u_cross) != 0 else 0.0
print(f"Homogeneous state: G0={G0:.6g}, X0={X0:.6g}, Y0={Y0:.6g}")

# ---------------- Operators (Dirichlet edges) ----------------
def _clamp_dirichlet_edges(u):
    u[:, 0] = 0.0; u[:, -1] = 0.0; u[0, :] = 0.0; u[-1, :] = 0.0

def laplacian_2d(u):
    dudxx = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / (dx*dx)
    dudyy = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / (dy*dy)
    out = dudxx + dudyy
    _clamp_dirichlet_edges(out)
    return out

def gradx(u):
    gx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dx)
    gx[:, 0] = 0.0; gx[:, -1] = 0.0
    return gx

def grady(u):
    gy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dy)
    gy[0, :] = 0.0; gy[-1, :] = 0.0
    return gy

def vorticity(ux, uy):
    return gradx(uy) - grady(ux)

# ---------------- Forcing chi(x,y,t) ----------------
def bell(sig, z):
    return np.exp(- (z/sig)**2 / 2.0)

Tseed = args.Tseed
seed_sigma_space = args.seed_sigma_space
seed_sigma_time = args.seed_sigma_time
seed_centers = [tuple(args.seed_center)] if args.seed_center is not None else [(0.0, 0.0)]

def chi_xy_t(t):
    spatial = np.zeros((ny, nx))
    for (xc, yc) in seed_centers:
        spatial += np.exp(-((X-xc)**2 + (Y-yc)**2) / (2*seed_sigma_space**2))
    return -spatial * bell(seed_sigma_time, t - Tseed)

# ---------------- Packing ----------------
def pack(pG, pX, pY, ux=None, uy=None):
    if ux is None:
        return np.concatenate([pG.ravel(), pX.ravel(), pY.ravel()])
    return np.concatenate([pG.ravel(), pX.ravel(), pY.ravel(), ux.ravel(), uy.ravel()])

def unpack(yflat, with_vort):
    pG = yflat[0:N].reshape(ny, nx)
    pX = yflat[N:2*N].reshape(ny, nx)
    pY = yflat[2*N:3*N].reshape(ny, nx)
    if not with_vort:
        return pG, pX, pY, None, None
    ux = yflat[3*N:4*N].reshape(ny, nx)
    uy = yflat[4*N:5*N].reshape(ny, nx)
    return pG, pX, pY, ux, uy

# ---------------- RHS ----------------
def reaction_terms(pG, pX, pY, forcing):
    Xtot = pX + X0
    Ytot = pY + Y0
    nonlinear_s  = s_par * (Xtot**3 - X0**3)
    nonlinear_xy = (Xtot**2 * Ytot - X0**2 * Y0)

    dG = - q_par * pG + g_par * pX
    dX = p_par * pG - (1.0 + b) * pX + u_cross * pY - nonlinear_s + nonlinear_xy + forcing
    dY = b * pX - u_cross * pY + (-nonlinear_xy + nonlinear_s)
    return dG, dX, dY

def velocity_rhs(ux, uy, pG, pX, pY):
    # compressible (isothermal) toy flow, as in your vortical code, but with FD operators + Dirichlet edges
    rho0 = 1.0
    rho = rho0 + args.alphaG*(G0+pG) + args.alphaX*(X0+pX) + args.alphaY*(Y0+pY)
    # pressure gradient
    rx = gradx(np.log(rho + 1e-12))
    ry = grady(np.log(rho + 1e-12))

    ux_x = gradx(ux); ux_y = grady(ux)
    uy_x = gradx(uy); uy_y = grady(uy)

    convx = ux*ux_x + uy*ux_y
    convy = ux*uy_x + uy*uy_y

    lap_ux = laplacian_2d(ux)
    lap_uy = laplacian_2d(uy)

    divu = ux_x + uy_y
    divx = gradx(divu)
    divy = grady(divu)

    visc_x = lap_ux + (1.0/3.0)*divx
    visc_y = lap_uy + (1.0/3.0)*divy

    dux = -convx - args.cs2*rx + args.nu*visc_x
    duy = -convy - args.cs2*ry + args.nu*visc_y

    # optional persistent rotational drive (keeps system spinning)
    if args.omega_drive != 0.0:
        cx = args.swirl_cx if args.swirl_cx is not None else 0.0
        cy = args.swirl_cy if args.swirl_cy is not None else 0.0
        dxg = X - cx
        dyg = Y - cy
        r2 = dxg*dxg + dyg*dyg
        sig = args.swirl_sigma if args.swirl_sigma > 0 else 1.0
        gauss = np.exp(-r2 / (2.0 * sig*sig))
        dux += (-args.omega_drive * dyg) * gauss
        duy += ( args.omega_drive * dxg) * gauss

    _clamp_dirichlet_edges(dux)
    _clamp_dirichlet_edges(duy)
    return dux, duy

def rhs_no_vort(t, yflat):
    pG, pX, pY, _, _ = unpack(yflat, with_vort=False)

    forcing = chi_xy_t(t)
    rG, rX, rY = reaction_terms(pG, pX, pY, forcing)

    dpGdt = args.Dg*laplacian_2d(pG) + rG
    dpXdt = args.Dx*laplacian_2d(pX) + rX
    dpYdt = args.Dy*laplacian_2d(pY) + rY

    _clamp_dirichlet_edges(dpGdt)
    _clamp_dirichlet_edges(dpXdt)
    _clamp_dirichlet_edges(dpYdt)
    return pack(dpGdt, dpXdt, dpYdt)

def rhs_with_vort(t, yflat):
    pG, pX, pY, ux, uy = unpack(yflat, with_vort=True)

    forcing = chi_xy_t(t)
    rG, rX, rY = reaction_terms(pG, pX, pY, forcing)

    # scalar advection
    advG = ux*gradx(pG) + uy*grady(pG)
    advX = ux*gradx(pX) + uy*grady(pX)
    advY = ux*gradx(pY) + uy*grady(pY)

    dpGdt = args.Dg*laplacian_2d(pG) - advG + rG
    dpXdt = args.Dx*laplacian_2d(pX) - advX + rX
    dpYdt = args.Dy*laplacian_2d(pY) - advY + rY

    duxdt, duydt = velocity_rhs(ux, uy, pG, pX, pY)

    for arr in (dpGdt, dpXdt, dpYdt, duxdt, duydt):
        _clamp_dirichlet_edges(arr)

    return pack(dpGdt, dpXdt, dpYdt, duxdt, duydt)

# ---------------- Rendering ----------------
def render_frame(yflat, t, fpath):
    with_vort = not args.no_vort
    pG, pX, pY, ux, uy = unpack(yflat, with_vort=with_vort)

    # 4th panel is fluid vorticity (ωz); if no_vort, show zeros
    if ux is None:
        vort = np.zeros((ny, nx))
        ux0 = np.zeros((ny, nx))
        uy0 = np.zeros((ny, nx))
    else:
        vort = vorticity(ux, uy)
        ux0, uy0 = ux, uy

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"Model G 2D — t={t:.3f}  ({'SAFE' if args.no_vort else 'VORT'})", fontsize=13)

    # fixed color limits (constant “z-axis”)
    vmin, vmax = -abs(args.zlim), abs(args.zlim)

    def imshow_panel(ax, field, title, cmap):
        im = ax.imshow(field, origin="lower",
                       extent=[x[0], x[-1], y[0], y[-1]],
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    imshow_panel(axs[0, 0], pX/10.0, "pX/10 (X scaled)", "plasma")
    imshow_panel(axs[0, 1], pY,       "pY (Y)",          "cividis")
    imshow_panel(axs[1, 0], pG,       "pG (G)",          "viridis")

    # vorticity panel with fixed (optional) limits
    if args.vortlim is None:
        vamp = max(float(np.max(np.abs(vort))), 1e-9)
        vvmin, vvmax = -vamp, vamp
    else:
        vvmin, vvmax = -abs(args.vortlim), abs(args.vortlim)

    imv = axs[1, 1].imshow(vort, origin="lower",
                           extent=[x[0], x[-1], y[0], y[-1]],
                           vmin=vvmin, vmax=vvmax, cmap="coolwarm")
    axs[1, 1].set_title("Fluid vorticity ωz")
    axs[1, 1].set_xlabel("x"); axs[1, 1].set_ylabel("y")
    plt.colorbar(imv, ax=axs[1, 1], fraction=0.046, pad=0.04)

    if args.quiver:
        qs = max(1, int(args.quiver_stride))
        U = ux0[::qs, ::qs]
        V = uy0[::qs, ::qs]
        speed = np.hypot(U, V)
        eps = 1e-12
        Uh = U / (speed + eps)
        Vh = V / (speed + eps)

        # constant arrow length in data units
        arrow_len = 0.7 * min(dx, dy) * qs
        scale = 1.0 / max(arrow_len, 1e-9)

        qv = axs[1, 1].quiver(
            X[::qs, ::qs], Y[::qs, ::qs],
            Uh, Vh,
            speed,  # color by magnitude
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.003,
            headwidth=3.5,
            headlength=4.5,
        )
        # (optional) colorbar for speed could be added, but we keep one colorbar per panel.

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(fpath, dpi=120)
    plt.close(fig)

# ---------------- Checkpointing ----------------
def save_ckpt(t_curr, y_curr, next_frame_idx, frames_done):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp = ckpt_path + ".tmp"
    np.savez_compressed(
        tmp,
        t_curr=float(t_curr),
        y_curr=y_curr,
        next_frame_idx=int(next_frame_idx),
        frames_done=np.array(sorted(list(frames_done)), dtype=np.int32),
    )
    if os.path.exists(tmp):
        os.replace(tmp, ckpt_path)

def load_ckpt():
    if not os.path.exists(ckpt_path):
        return None
    d = np.load(ckpt_path, allow_pickle=True)
    frames_done = set(int(v) for v in np.array(d["frames_done"]).ravel().tolist())
    return {
        "t_curr": float(d["t_curr"]),
        "y_curr": d["y_curr"],
        "next_frame_idx": int(d["next_frame_idx"]),
        "frames_done": frames_done,
    }

# ---------------- Main segmented integration ----------------
def main():
    frame_times = np.linspace(0.0, args.Tfinal, args.nt_anim)

    # initial state
    pG0 = np.zeros((ny, nx))
    pX0 = np.zeros((ny, nx))
    pY0 = np.zeros((ny, nx))

    if args.no_vort:
        y0 = pack(pG0, pX0, pY0)
    else:
        ux0 = np.zeros((ny, nx))
        uy0 = np.zeros((ny, nx))
        if args.rotseed:
            cx = args.swirl_cx if args.swirl_cx is not None else 0.0
            cy = args.swirl_cy if args.swirl_cy is not None else 0.0
            dxg = X - cx
            dyg = Y - cy
            r2 = dxg*dxg + dyg*dyg
            vtheta = args.swirl_amp * np.exp(-r2 / (2.0 * args.swirl_sigma**2))
            R = np.sqrt(r2) + 1e-12
            ux0 = -vtheta * (dyg / R)
            uy0 =  vtheta * (dxg / R)
            _clamp_dirichlet_edges(ux0)
            _clamp_dirichlet_edges(uy0)
        y0 = pack(pG0, pX0, pY0, ux0, uy0)

    ck = load_ckpt()
    if ck is None:
        t_curr = 0.0
        y_curr = y0
        next_frame_idx = 0
        frames_done = set()
        print("[Start] Fresh run")
    else:
        t_curr = ck["t_curr"]
        y_curr = ck["y_curr"]
        next_frame_idx = ck["next_frame_idx"]
        frames_done = ck["frames_done"]
        print(f"[Resume] t={t_curr:.3f}, next_frame={next_frame_idx}/{args.nt_anim}, frames_done={len(frames_done)}")

    # render any frames due
    while next_frame_idx < args.nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
        tframe = frame_times[next_frame_idx]
        fpath = os.path.join(frames_dir, f"frame_{next_frame_idx:04d}.png")
        if next_frame_idx not in frames_done:
            render_frame(y_curr, tframe, fpath)
            frames_done.add(next_frame_idx)
        next_frame_idx += 1
        save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

    rhs = rhs_no_vort if args.no_vort else rhs_with_vort

    t_start_wall = time.time()

    while t_curr < args.Tfinal - 1e-12:
        t_seg_end = min(args.Tfinal, t_curr + args.segment_dt)
        print(f"[Integrate] {t_curr:.3f} -> {t_seg_end:.3f}")

        t_eval = frame_times[(frame_times > t_curr + 1e-12) & (frame_times <= t_seg_end + 1e-12)]
        sol = solve_ivp(
            rhs,
            (t_curr, t_seg_end),
            y_curr,
            method=args.method,
            atol=args.atol,
            rtol=args.rtol,
            max_step=args.max_step,
            t_eval=t_eval if t_eval.size > 0 else None,
        )
        if sol.status < 0:
            print("[WARN] Segment failure:", sol.message)

        if sol.t.size > 0:
            for k, tframe in enumerate(sol.t):
                fidx = np.searchsorted(frame_times, tframe)
                if fidx < args.nt_anim and abs(frame_times[fidx] - tframe) < 1e-9:
                    if fidx not in frames_done:
                        render_frame(sol.y[:, k], tframe, os.path.join(frames_dir, f"frame_{fidx:04d}.png"))
                        frames_done.add(fidx)
                        save_ckpt(tframe, sol.y[:, k], fidx + 1, frames_done)

        # advance to segment end
        y_curr = sol.y[:, -1] if sol.y.ndim == 2 else sol.y
        t_curr = sol.t[-1] if sol.t.size > 0 else t_seg_end

        # catch up any frames
        while next_frame_idx < args.nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
            tframe = frame_times[next_frame_idx]
            fpath = os.path.join(frames_dir, f"frame_{next_frame_idx:04d}.png")
            if next_frame_idx not in frames_done:
                render_frame(y_curr, tframe, fpath)
                frames_done.add(next_frame_idx)
            next_frame_idx += 1
            save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

        elapsed = time.time() - t_start_wall
        print(f"  -> Reached t={t_curr:.3f}/{args.Tfinal}, frames={len(frames_done)}/{args.nt_anim}, wall={elapsed:.1f}s")
        render_frame(y_curr, t_curr, final_png)
        save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

    # assemble MP4
    print("[Video] Writing MP4:", mp4_path)
    fps = max(8, int(args.nt_anim / max(1.0, args.Tfinal / 2.0)))
    with imageio.get_writer(mp4_path, fps=fps) as writer:
        for i in range(args.nt_anim):
            img = imageio.imread(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            writer.append_data(img)
    print("[Done] MP4 saved:", mp4_path)

if __name__ == "__main__":
    main()
