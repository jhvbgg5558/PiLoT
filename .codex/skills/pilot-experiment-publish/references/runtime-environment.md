# PiLoT Current Runtime Environment

This file is the quick environment reference for new Codex windows working on `jhvbgg5558/PiLoT`. Unless the user says otherwise, use this environment for local DOM/DSM experiments.

## Correct WSL Distribution

Use `Ubuntu-22.04`, not the default `Ubuntu-20.04` distro. From Windows PowerShell:

```powershell
wsl -d Ubuntu-22.04
```

## Repository Path

```text
Windows: D:\aiproject\PiLoT_work
WSL:     /mnt/d/aiproject/PiLoT_work
Branch:  feature/dom-dsm-renderer
```

Do not confuse this with `D:\aiproject\Pilot` / `/mnt/d/aiproject/Pilot`.

## Python Environment

Use the project-local command path:

```bash
cd /mnt/d/aiproject/PiLoT_work
./.conda/pilot22/bin/python
```

`pilot22` is a Linux symlink into the Ubuntu-22.04 ext4 filesystem:

```text
/mnt/d/aiproject/PiLoT_work/.conda/pilot22 -> /home/farsee2/pilot22
/mnt/d/aiproject/PiLoT_work/.conda/pilot22/bin/python -> python3.8
real Python: /home/farsee2/pilot22/bin/python3.8
```

If `/home/farsee2/pilot22/bin/python3.8` does not exist, the terminal is probably in a different WSL distro.

## Expected Runtime

```text
Python: 3.8.20
PyTorch: 2.4.1+cu124
Torch CUDA runtime: 12.4
GPU: NVIDIA GeForce GTX 1080
GPU compute capability: sm_61
```

## Sanity Check

```bash
cd /mnt/d/aiproject/PiLoT_work
readlink .conda/pilot22
readlink -f .conda/pilot22/bin/python
./.conda/pilot22/bin/python - <<'PY'
import os, sys, torch
print("sys.executable:", sys.executable)
print("real executable:", os.path.realpath(sys.executable))
print("sys.prefix:", sys.prefix)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

## CUDA Extension Note

The current GTX 1080 is `sm_61`. The prebuilt `direct_abs_cost_cuda` binary previously detected only `sm_86`, so CUDA optimizer/loss results are not trustworthy unless a rebuilt `sm_61` extension is present and validated.
