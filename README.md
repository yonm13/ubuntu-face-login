# 🔓 ubuntu-face-login

Face recognition login for Ubuntu using IR cameras and FaceNet-512. Authenticates in ~1 second on CPU — no GPU required.

Point your face at the camera, get authenticated. Falls back to password if it doesn't work. That's it.

## What it does

ubuntu-face-login is a PAM module that authenticates Linux users by face recognition. It plugs into `sudo`, GDM login, and polkit privilege prompts as a `sufficient` auth method — if face auth fails or times out, you get the normal password prompt.

The pipeline runs entirely on CPU in about 1 second:

```
IR camera frame
  → YuNet face detection (~5ms)
  → Landmark-based liveness check
  → FaceNet-512 embedding (~15ms)
  → L2 distance matching against enrolled faces
  → PAM success/failure
```

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ IR Camera │───▶│  YuNet Face  │───▶│   Liveness    │───▶│  FaceNet-512 │
│ + Emitter │    │  Detection   │    │    Check      │    │  Embedding   │
└──────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                               │
                   ┌──────────┐    ┌──────────────┐            │
                   │   PAM    │◀───│  L2 Distance  │◀───────────┘
                   │ success  │    │   Matching    │
                   └──────────┘    └──────────────┘
```

## Hardware

### IR cameras (recommended)

IR cameras work in complete darkness, resist printed-photo attacks, and produce consistent images regardless of ambient lighting. Most modern laptops with Windows Hello cameras have a paired IR sensor.

**Known working laptops:**
- ThinkPad X1 Carbon (Gen 7+), T14s, T480s, X280
- Dell XPS 13/15 (models with IR camera option)
- HP EliteBook 840/850 series
- Framework Laptop (with IR camera module)

Your IR camera usually shows up as a separate `/dev/video*` device outputting greyscale formats (`GREY`, `Y8`, `Y10`). Run `v4l2-ctl --list-devices` to see what you have.

Many IR cameras need their emitter explicitly activated. ubuntu-face-login handles this automatically via UVC ioctls or [linux-enable-ir-emitter](https://github.com/EmixamPP/linux-enable-ir-emitter). If your emitter doesn't fire, see [Troubleshooting](#ir-emitter-not-working).

### RGB webcams (works, less secure)

Any USB or built-in webcam works. RGB mode is less secure — it's easier to fool with a high-quality printed photo. Fine for convenience use (sudo on your personal laptop), not for anything sensitive.

### GPU not required

Both models run via ONNX Runtime on CPU. YuNet detection takes ~5ms, FaceNet-512 embedding takes ~15ms. Total pipeline including camera capture is ~1 second. There is no benefit to installing CUDA for this.

## Quick start

```bash
# Clone
git clone https://github.com/yonm13/ubuntu-face-login.git
cd ubuntu-face-login

# Run the setup wizard (recommended — installs, enrolls, and configures PAM)
python3 setup-wizard.py
```

Or manually:

```bash
# Install (downloads models, creates venv, sets up wrapper)
sudo ./install.sh

# Enroll your face (captures 20 samples by default)
sudo ubuntu-face-login enroll $(whoami)

# Test — should print ✅ with your username
sudo ubuntu-face-login auth --user $(whoami)

# Enable for sudo (interactive, safe — backs up PAM files)
sudo bash scripts/setup-pam.sh
```

The PAM setup script walks you through each step, asks you to verify `sudo whoami` works in another terminal, and offers to enable face auth for GDM login and polkit prompts too.

### Setup wizard

The recommended way to set up ubuntu-face-login is the GTK4 wizard:

```bash
python3 setup-wizard.py
```

It guides you through installation, live-camera face enrollment, an authentication test, and PAM configuration — all in one window. Per-service timeouts and match thresholds are configurable from the UI before applying.

### Install with GUI enrollment only

```bash
sudo ./install.sh --with-gui
```

Adds GTK4 dependencies for a graphical enrollment window with live camera preview.

### Uninstall

```bash
sudo ./uninstall.sh
```

Removes PAM configuration, installation directory, config files, and optionally user face data. Does not remove system packages (`libpam-python`, `v4l-utils`).

## Configuration

Configuration lives at `/etc/ubuntu-face-login/config.toml` (system-wide) or `~/.config/ubuntu-face-login/config.toml` (per-user override). See [`config.example.toml`](config.example.toml) for all options with defaults.

Key settings:

| Setting | Default | What it does |
|---------|---------|--------------|
| `camera.device` | auto-detect | Force a specific `/dev/video*` device |
| `camera.type` | auto-detect | `"ir"` or `"rgb"` — override auto-classification |
| `camera.ir_brightness_threshold` | `20` | Minimum brightness for IR frames (filters dark emitter-off frames) |
| `emitter.enabled` | `true` | Whether to activate IR emitter before capture |
| `auth.threshold` | `0.45` | Max L2 distance for a match. Lower = stricter. |
| `auth.timeout.sudo` | `3` | Seconds before face auth gives up for sudo |
| `auth.timeout.gdm-password` | `5` | Seconds before face auth gives up at login screen |
| `enrollment.samples` | `20` | Number of face captures during enrollment |
| `enrollment.min_confidence` | `0.6` | Minimum YuNet confidence to accept a sample |

### Per-service thresholds and timeouts

The setup wizard lets you configure both timeout and threshold independently per PAM service. Defaults are deliberately tighter for `sudo` than for the login screen:

| Service | Default timeout | Default threshold | Rationale |
|---------|----------------|-------------------|-----------|
| sudo | 2s | 0.40 | Stricter — called frequently, needs to be fast and secure |
| sudo-i | 2s | 0.40 | Same |
| gdm-password | 5s | 0.50 | More lenient — you're sitting at your own screen |
| polkit-1 | 5s | 0.45 | Middle ground |

These values are written directly into the PAM config line as `timeout=N threshold=X` and override the global config at auth time.

### Adjusting the threshold

The default threshold of `0.45` balances security and usability for FaceNet-512 embeddings. If you're getting false rejections, raise it to `0.50`–`0.55`. If you're concerned about false accepts (especially with RGB cameras), lower it to `0.40`.

The matcher uses a top-K average strategy: it averages the L2 distances to your 5 closest enrolled embeddings, which smooths out outlier samples.

## Security considerations

**This is a convenience feature, not bank-vault security.** It's equivalent to fingerprint unlock — fast and good enough for everyday use, with password as the real backstop.

What it does well:
- IR + liveness checks stop printed photos and phone-screen replay attacks
- Landmark validation rejects profile views, unusual angles, and partially visible faces
- Per-service timeouts limit the attack window
- `sufficient` PAM mode means failure always falls back to password — it never locks you out

What it doesn't do:
- **3D masks/models** — the liveness check uses 2D landmarks only. A realistic 3D-printed mask of your face would likely pass. This requires physical access and significant effort.
- **RGB mode** — no IR liveness signal. A high-quality printed photo held at the right distance could work. Use IR if your hardware supports it.
- **Identical twins** — FaceNet-512 may not distinguish identical twins reliably at the default threshold.
- **Encrypted storage** — enrolled embeddings (`.npy` files) are stored unencrypted. They're 512-dim vectors, not photos, but an attacker with disk access could potentially use them.

The PAM line is `auth sufficient` — if face auth fails, errors out, or times out, PAM continues to password authentication. You cannot get locked out by a face auth failure.

## How it works

### 1. Camera detection

`camera.py` enumerates `/dev/video*` devices via `v4l2-ctl`, inspects their pixel formats, and classifies each as IR (greyscale formats: `GREY`, `Y8`, `Y10`) or RGB (`YUYV`, `MJPG`). IR cameras are preferred automatically.

### 2. IR emitter activation

`emitter.py` activates the IR emitter via UVC extension-unit ioctls. It tries three strategies in order:

1. Direct ioctl using config values (`emitter.unit`, `emitter.selector`, `emitter.control_data`)
2. Parse `linux-enable-ir-emitter` TOML config and issue the ioctl
3. Shell out to `linux-enable-ir-emitter run` CLI
4. Skip — log a warning and continue (camera may have a built-in always-on emitter)

### 3. Face detection

`detector.py` runs OpenCV's YuNet model (~5ms on CPU) to find the highest-confidence face in the frame. YuNet outputs a bounding box plus 5 landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner.

### 4. Liveness validation

Using the 5 landmarks from YuNet (no separate model needed):

- **Centered**: face center is within 25% of frame center
- **Frontal**: inter-eye distance is 20–60% of face width (rejects profiles)
- **Facing camera**: nose is between and below eyes (rejects rotated/tilted photos)

### 5. Embedding

`embedder.py` crops the face to 160×160, normalizes to `[-1, 1]`, and runs FaceNet-512 via ONNX Runtime (CPU). Produces a 512-dimensional float32 vector. Inference takes ~15ms.

### 6. Matching

`matcher.py` loads all enrolled `.npy` embeddings, computes L2 distance from the query embedding to each stored embedding per user, averages the top-K (K=5) closest distances, and returns the user whose average distance is below the threshold.

### 7. PAM integration

`pam_module.py` is loaded by `pam_python.so`. It calls the `ubuntu-face-login auth` wrapper script as a subprocess, which runs the full pipeline above. Exit code 0 → `PAM_SUCCESS`, anything else → `PAM_AUTH_ERR` (falls through to password).

## Troubleshooting

### Camera not detected

```bash
# List all video devices
v4l2-ctl --list-devices

# Check if ubuntu-face-login sees your camera
sudo ubuntu-face-login detect-cameras
```

If your camera doesn't appear, check that the kernel module is loaded (`lsmod | grep uvcvideo`) and that the device node exists in `/dev/`.

### IR emitter not working

The most common issue. Symptoms: authentication always fails, logs show "dark frame" warnings.

```bash
# Check logs
journalctl -t ubuntu_face_login --no-pager -n 30

# Install the IR emitter tool
pip install linux-enable-ir-emitter

# Auto-detect your emitter parameters
sudo linux-enable-ir-emitter configure

# Test it
sudo linux-enable-ir-emitter run
```

Once `linux-enable-ir-emitter configure` works, ubuntu-face-login will automatically read its TOML config. You can also copy the unit/selector/control_data values into your `config.toml` for direct ioctl activation (faster, no CLI dependency).

### False rejections (face not recognized)

```bash
# Check your current threshold
sudo ubuntu-face-login config
```

Raise the threshold in `/etc/ubuntu-face-login/config.toml`:

```toml
[auth]
threshold = 0.50  # default is 0.45
```

If rejections persist, re-enroll with more samples and varied lighting:

```bash
sudo ubuntu-face-login enroll $(whoami) --samples 30
```

### Authentication too slow

Check which camera is being selected — RGB cameras are usually fine, but if both IR and RGB are present and the IR emitter isn't working, the system may waste time waiting for bright frames.

```bash
# Force a specific camera in config.toml
[camera]
device = "/dev/video0"
type = "rgb"
```

### Check logs

```bash
# PAM authentication logs
journalctl -t ubuntu_face_login --no-pager -n 50

# Verbose output from direct test
sudo ubuntu-face-login auth --user $(whoami)
```

## Comparison to howdy

[howdy](https://github.com/boltgolt/howdy) is the established face auth tool for Linux. ubuntu-face-login exists because:

| | howdy | ubuntu-face-login |
|---|---|---|
| Face detection | dlib (HOG or CNN) | YuNet (OpenCV, ~5ms) |
| Embedding | dlib face_recognition (128-dim) | FaceNet-512 (512-dim) |
| Runtime | dlib + face_recognition (~100MB) | ONNX Runtime + OpenCV (~50MB) |
| IR emitter | Requires linux-enable-ir-emitter CLI | Direct UVC ioctl with CLI fallback |
| Liveness | None built-in | Landmark-based (centred, frontal, nose position) |
| GPU | Optional (dlib CUDA) | Not needed (CPU-only, 15ms inference) |
| Python | System Python | Isolated venv |

The main practical differences: ubuntu-face-login has built-in liveness checking, uses a higher-dimensional embedding space (512 vs 128), handles IR emitters natively without shelling out on every auth attempt, and runs in an isolated virtualenv so it doesn't conflict with system packages.

If howdy works well for you, there's no reason to switch. ubuntu-face-login is an alternative for people who want liveness checks, cleaner IR emitter handling, or have had dependency issues with dlib.

The `setup-pam.sh` script will offer to comment out howdy's PAM lines if both are installed, to avoid running two face auth systems on every `sudo`.

## Project structure

```
├── install.sh              # Main installer (run via pkexec from wizard or sudo directly)
├── uninstall.sh            # Clean removal
├── setup-wizard.py         # GTK4 setup wizard (install + enroll + PAM config in one UI)
├── config.example.toml     # Annotated default configuration
├── scripts/
│   ├── setup-pam.sh        # Interactive PAM configuration (CLI alternative to wizard)
│   ├── export-onnx.py      # Export FaceNet to ONNX (development)
│   └── convert-embeddings.py
├── src/facelogin/
│   ├── auth.py             # Authentication orchestrator + CLI (--timeout, --threshold, --service)
│   ├── camera.py           # V4L2 camera detection and capture
│   ├── config.py           # TOML config loader
│   ├── detector.py         # YuNet face detection + liveness
│   ├── embedder.py         # FaceNet-512 ONNX inference
│   ├── emitter.py          # IR emitter activation (UVC ioctl)
│   ├── enroll.py           # Pose-guided face enrollment
│   ├── matcher.py          # Embedding database + L2 matching
│   └── pam_module.py       # PAM bridge (parses timeout= and threshold= from PAM config line)
├── ui/
│   └── enroll_gtk.py       # Standalone GTK4 enrollment window (re-enrollment without full wizard)
└── models/                 # Downloaded by install.sh — not in repo
    ├── yunet.onnx
    └── facenet512.onnx
```

## Contributing

Contributions welcome. The codebase is straightforward Python — no framework, no magic.

Areas that would benefit from help:
- Testing on more laptop IR cameras (especially non-ThinkPad)
- Multi-face rejection (currently picks highest-confidence face)
- Encrypted embedding storage
- `--re-enroll` flag to clear old samples before capturing new ones

```bash
# Development setup
git clone https://github.com/yonm13/ubuntu-face-login.git
cd ubuntu-face-login
python3 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python-headless onnxruntime
```

## License

MIT — see [LICENSE](LICENSE).

Copyright 2025–2026.
