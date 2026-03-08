"""
Model G Particle 2D (proper x–y) — SAFE base + optional vortical motion + quiver-by-speed

- Written by Brendan Darrer aided by ChatGPT5 date: 27th February 2026
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT 5.2 writing it and Brendan guiding it to produce a clean code.

Tested for: Ubuntu 24.04.3 LTS on i7-4790 (Optiplex 7020/9020), Python 3.10+

Install:
  pip3 install numpy scipy matplotlib imageio imageio[ffmpeg]


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
"""

import os
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio.v2 as imageio

# STEP 1: Read command-line options (grid size, time, parameters, plotting, etc.)
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

# STEP 2: Create output paths (frames folder, checkpoint file, MP4 name)
# ---------------- Paths ----------------
base_name = "model_g_2d_xy_safe_plus_vort__1a"
run_name = base_name + ("_novort" if args.no_vort else "_vort")
out_dir = f"out_{run_name}"
frames_dir = os.path.join(out_dir, "frames")
ckpt_path = os.path.join(out_dir, "checkpoint_2d_plus_vort.npz")
mp4_path = os.path.join(out_dir, f"{run_name}.mp4")
final_png = os.path.join(out_dir, "final_snapshot.png")
os.makedirs(frames_dir, exist_ok=True)

# STEP 3: Build the 2D x–y grid (arrays X,Y and spacings dx,dy)
# ---------------- Grid ----------------
Lx, Ly = args.Lx, args.Ly
nx, ny = args.nx, args.ny
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
X, Y = np.meshgrid(x, y, indexing="xy")
dx = x[1] - x[0]
dy = y[1] - y[0]
N = nx * ny

# STEP 4: Set Model-G parameters and compute the uniform background state (G0, X0, Y0)
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

# STEP 5: Define finite-difference operators (gradients, Laplacian) and Dirichlet boundary clamping
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

# STEP 6: Define the seeding/forcing pulse χ(x,y,t) that nucleates the soliton
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

# STEP 7: Pack/unpack 2D fields into a 1D vector for the ODE solver (solve_ivp)
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

# STEP 8: Define the time-derivative functions (RHS) for SAFE mode and VORT mode
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

# STEP 9: Render a 2×2 figure per frame (pX,pY,pG, and fluid vorticity + optional quiver)
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

# STEP 10: Save/load checkpoints so runs can resume (t, state vector, rendered frames)
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

# STEP 11: Integrate in time in segments, render frames at requested times, then assemble MP4
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
