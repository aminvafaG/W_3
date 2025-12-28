# JupyterLite Dashboard (Pure Browser, No Server)

This project publishes a **dynamic dashboard** that runs entirely in your browser (Pyodide) with **no server**.  
It uses **pure JupyterLite** (no Voila/Voici).

## How to use

1. Push this repository to GitHub.
2. Ensure **GitHub Pages** is enabled for the repo (Settings → Pages → Build and Deploy: GitHub Actions).
3. The included workflow builds the site and deploys it to Pages.

Once deployed, open your dashboard at:

```
https://<YOUR_GH_USERNAME>.github.io/<YOUR_REPO>/lab/index.html?path=App.ipynb
https://aminvafaG.github.io/W_3/lab/index.html?path=App.ipynb

```

> Tip: The notebook hides code by default. If you need to inspect code, open the first cell and set `SHOW_CODE = True`, then **Run All**.

## What’s inside?

- `content/App.ipynb` — the dashboard (filters + plots).  
- `content/utils.py`   — data synthesis and helper functions (uses keyword-only args & dataclasses to mimic MATLAB `arguments`).  
- `jupyter-lite.json`  — minimal app config.  
- `.github/workflows/jupyterlite.yml` — GitHub Actions to build & deploy.

## Features

- Generates a **synthetic dataset** of V1-like units:
  - Per unit: `layer` (SG/G/IG), `effect` (MUL/MXH), control & laser tuning curves, OSI, HBW, FR_mean.
  - Tuning curves based on a **von Mises** profile; MXH laser effect produces center-suppression and flank facilitation.
- **Interactive filters**:
  - Layer (multi-select), OSI range, HBW range, effect type, and toggle “show mean curve”.
- **Plots**:
  - Scatter: OSI vs HBW for filtered units.
  - Overlay: all filtered tuning curves (Control solid, Laser dashed) + optional mean.
  - Detail: single-unit panel populated from the filtered set.

## Notes

- The notebook is pre-executed during the build so the dashboard appears immediately on load.
- Everything executes **in the browser**. No backend server, no Python on the host.
