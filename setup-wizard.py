#!/usr/bin/env python3
"""Ubuntu Face Login — Setup Wizard.

A multi-page GTK4 wizard that guides the user through:
  1. Welcome + hardware detection
  2. System installation (via pkexec)
  3. Face enrollment (live camera UI)
  4. Authentication test
  5. PAM setup (via pkexec, with safety test step)
  6. Done

Run as a normal user — privileged steps use pkexec for the password dialog.

Usage:
    python3 setup-wizard.py
"""

from __future__ import annotations

import getpass
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

try:
    import gi
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw, Gdk, GdkPixbuf, GLib, Gtk
except (ImportError, ValueError) as exc:
    print(f"GTK4/libadwaita not available: {exc}\n"
          "Install with: sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1",
          file=sys.stderr)
    sys.exit(1)

# Find project root relative to this file
WIZARD_DIR = Path(__file__).resolve().parent
SRC_DIR = WIZARD_DIR / "src"
INSTALL_SH = WIZARD_DIR / "install.sh"
SETUP_PAM_SH = WIZARD_DIR / "scripts" / "setup-pam.sh"
WRAPPER = Path("/usr/local/bin/ubuntu-face-login")

# Add src to path so we can import facelogin when installed
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
elif Path("/opt/ubuntu-face-login/src").exists():
    sys.path.insert(0, "/opt/ubuntu-face-login/src")

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def is_installed() -> bool:
    return WRAPPER.exists()


def get_enrolled_count(user_id: str) -> int:
    """Return number of .npy embeddings saved for user_id."""
    try:
        from facelogin.config import get_config
        data_dir = Path(get_config().data.dir)
    except Exception:
        data_dir = Path("/var/lib/ubuntu-face-login")
    return len(list(data_dir.glob(f"{user_id}_*.npy")))


def detect_cameras_info() -> list[str]:
    """Return human-readable camera descriptions."""
    try:
        from facelogin.camera import detect_cameras
        cams = detect_cameras()
        return [f"{c.device} ({c.type.upper()}) — {c.name}" for c in cams]
    except Exception:
        return ["Camera detection unavailable before installation"]


# ── Reusable widgets ─────────────────────────────────────────────────────────

def _label(text: str, *, css: str = "", wrap: bool = False,
           align: Gtk.Align = Gtk.Align.START) -> Gtk.Label:
    lbl = Gtk.Label(label=text)
    lbl.set_halign(align)
    lbl.set_wrap(wrap)
    if css:
        lbl.add_css_class(css)
    return lbl


def _section(title: str, child: Gtk.Widget) -> Gtk.Box:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    box.append(_label(title, css="heading"))
    box.append(child)
    return box


def _scrolled_log() -> tuple[Gtk.ScrolledWindow, Gtk.TextView]:
    tv = Gtk.TextView()
    tv.set_editable(False)
    tv.set_monospace(True)
    tv.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    tv.add_css_class("card")
    sw = Gtk.ScrolledWindow()
    sw.set_child(tv)
    sw.set_vexpand(True)
    sw.set_min_content_height(220)
    return sw, tv


def _append_log(tv: Gtk.TextView, text: str) -> bool:
    buf = tv.get_buffer()
    buf.insert(buf.get_end_iter(), text)
    # Scroll to end
    adj = tv.get_parent().get_vadjustment()
    if adj:
        adj.set_value(adj.get_upper())
    return False


# ── Page base ─────────────────────────────────────────────────────────────────

class WizardPage(Gtk.Box):
    """Base class for wizard pages."""

    title: str = ""
    subtitle: str = ""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        self.set_margin_top(24)
        self.set_margin_bottom(24)
        self.set_margin_start(32)
        self.set_margin_end(32)

    def on_enter(self, wizard: "SetupWizard") -> None:
        """Called when the page becomes visible."""

    def can_advance(self) -> bool:
        return True


# ── Page 1: Welcome ──────────────────────────────────────────────────────────

class WelcomePage(WizardPage):
    title = "Welcome to Ubuntu Face Login"
    subtitle = "Set up face recognition for sudo and the login screen"

    def __init__(self) -> None:
        super().__init__()

        # Icon + description
        icon = Gtk.Image.new_from_icon_name("camera-web-symbolic")
        icon.set_pixel_size(64)
        icon.set_halign(Gtk.Align.CENTER)
        self.append(icon)

        desc = _label(
            "This wizard will:\n"
            "  • Install the face recognition engine\n"
            "  • Capture your face from multiple angles\n"
            "  • Enable face login for sudo and the lock screen\n\n"
            "You will need your password once for the installation step.\n"
            "Face recognition always falls back to your password if it fails.",
            wrap=True, align=Gtk.Align.CENTER
        )
        desc.set_justify(Gtk.Justification.CENTER)
        self.append(desc)

        self.append(Gtk.Separator())

        # Hardware detection
        self._cam_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.append(_section("Detected cameras:", self._cam_box))

        # Already installed notice
        self._installed_row = _label(
            "✓ Already installed — wizard will skip to enrollment.",
            css="success", wrap=True
        )
        self._installed_row.set_visible(False)
        self.append(self._installed_row)

    def on_enter(self, wizard: "SetupWizard") -> None:
        # Clear and re-populate camera list
        child = self._cam_box.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._cam_box.remove(child)
            child = next_child

        cams = detect_cameras_info()
        for cam in cams:
            row = _label(f"  • {cam}", wrap=True)
            self._cam_box.append(row)

        self._installed_row.set_visible(is_installed())


# ── Page 2: Installation ─────────────────────────────────────────────────────

class InstallPage(WizardPage):
    title = "Installing"
    subtitle = "Setting up system dependencies and models"

    def __init__(self) -> None:
        super().__init__()
        self._done = False
        self._success = False

        self._status = _label("Waiting to start…", css="title-4",
                               align=Gtk.Align.CENTER)
        self.append(self._status)

        self._spinner = Gtk.Spinner()
        self._spinner.set_halign(Gtk.Align.CENTER)
        self._spinner.set_size_request(48, 48)
        self.append(self._spinner)

        sw, self._log = _scrolled_log()
        self.append(_section("Output:", sw))

    def on_enter(self, wizard: "SetupWizard") -> None:
        if is_installed():
            self._status.set_text("Already installed — skipping.")
            self._done = True
            self._success = True
            GLib.timeout_add(600, wizard.next_page)
            return

        self._done = False
        self._success = False
        self._spinner.start()
        self._status.set_text("Running installer… (password prompt will appear)")
        threading.Thread(target=self._run_install, args=(wizard,), daemon=True).start()

    def _run_install(self, wizard: "SetupWizard") -> None:
        cmd = ["pkexec", "bash", str(INSTALL_SH)]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                GLib.idle_add(_append_log, self._log, line)
            proc.wait()
            ok = proc.returncode == 0
        except Exception as exc:
            GLib.idle_add(_append_log, self._log, f"\nError: {exc}\n")
            ok = False

        self._done = True
        self._success = ok
        GLib.idle_add(self._finish, wizard, ok)

    def _finish(self, wizard: "SetupWizard", ok: bool) -> bool:
        self._spinner.stop()
        if ok:
            self._status.set_text("✓ Installation complete")
            self._status.add_css_class("success")
            # Re-import with the installed path available
            if Path("/opt/ubuntu-face-login/src").exists():
                sys.path.insert(0, "/opt/ubuntu-face-login/src")
            GLib.timeout_add(800, wizard.next_page)
        else:
            self._status.set_text("✗ Installation failed — check output above")
            self._status.add_css_class("error")
            wizard.set_can_advance(False)
        return False

    def can_advance(self) -> bool:
        return self._success


# ── Page 3: Enrollment ────────────────────────────────────────────────────────

class EnrollPage(WizardPage):
    title = "Enroll Your Face"
    subtitle = "We'll capture your face from several angles"

    def __init__(self) -> None:
        super().__init__()
        self._enrolled = False
        self._camera_widget_built = False

        self._status_lbl = _label(
            "Press Start to begin. Follow the pose prompts.",
            wrap=True, align=Gtk.Align.CENTER
        )
        self.append(self._status_lbl)

        # Placeholder; real enrollment widget built on first on_enter
        # (needs facelogin package which may only be available after install)
        self._enroll_container = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=8
        )
        self.append(self._enroll_container)

    def on_enter(self, wizard: "SetupWizard") -> None:
        if self._camera_widget_built:
            return
        try:
            self._build_camera_widget(wizard)
        except ImportError as exc:
            self._status_lbl.set_text(
                f"Could not load face recognition modules: {exc}\n"
                "Make sure installation completed successfully."
            )

    def _build_camera_widget(self, wizard: "SetupWizard") -> None:
        from facelogin.enroll import DEFAULT_POSES, enroll_user
        from facelogin.config import get_config
        import copy
        import cv2
        import numpy as np

        self._poses = [copy.copy(p) for p in DEFAULT_POSES]
        self._total = sum(p.samples for p in self._poses)
        self._saved = 0
        self._running = False
        self._wizard = wizard
        self._sample_delay = 0.4

        cfg = get_config()
        self._user_id = getpass.getuser()

        # Camera feed
        self._picture = Gtk.Picture()
        self._picture.set_size_request(480, 270)
        self._picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self._enroll_container.append(self._picture)

        # Pose instruction
        self._pose_lbl = _label("", css="title-3", align=Gtk.Align.CENTER)
        self._pose_lbl.set_wrap(True)
        self._enroll_container.append(self._pose_lbl)

        # Status chips
        chip_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        chip_box.set_halign(Gtk.Align.CENTER)
        self._lbl_face = _label("Face: —", css="caption")
        self._lbl_live = _label("Liveness: —", css="caption")
        chip_box.append(self._lbl_face)
        chip_box.append(self._lbl_live)
        self._enroll_container.append(chip_box)

        # Progress
        self._progress = Gtk.ProgressBar()
        self._progress.set_show_text(True)
        self._progress.set_text(f"0 / {self._total} samples")
        self._enroll_container.append(self._progress)

        # Buttons
        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        btn_row.set_halign(Gtk.Align.CENTER)

        self._btn_start = Gtk.Button(label="Start")
        self._btn_start.add_css_class("suggested-action")
        self._btn_start.connect("clicked", self._on_start)

        self._btn_stop = Gtk.Button(label="Stop")
        self._btn_stop.connect("clicked", self._on_stop)
        self._btn_stop.set_sensitive(False)

        btn_row.append(self._btn_start)
        btn_row.append(self._btn_stop)
        self._enroll_container.append(btn_row)

        self._camera_widget_built = True

    # Enrollment thread callbacks
    def _on_start(self, _btn: Gtk.Button) -> None:
        from facelogin.enroll import enroll_user
        self._running = True
        self._saved = 0
        self._enrolled = False
        self._btn_start.set_sensitive(False)
        self._btn_stop.set_sensitive(True)
        self._progress.set_fraction(0.0)
        self._progress.set_text(f"0 / {self._total} samples")
        self._wizard.set_can_advance(False)
        threading.Thread(target=self._run, daemon=True).start()

    def _on_stop(self, _btn: Gtk.Button) -> None:
        self._running = False

    def _run(self) -> None:
        from facelogin.enroll import enroll_user
        try:
            saved = enroll_user(
                user_id=self._user_id,
                poses=self._poses,
                sample_delay=self._sample_delay,
                on_frame=self._on_frame_guard,
                on_sample=self._on_sample,
                on_pose=self._on_pose,
                on_pose_transition=self._on_pose_transition,
            )
        except Exception as exc:
            logger.exception("Enrollment error")
            GLib.idle_add(self._finish, 0, str(exc))
            return
        GLib.idle_add(self._finish, saved, None)

    def _on_frame_guard(self, frame, box, landmarks, confidence, valid, reason):
        if not self._running:
            raise KeyboardInterrupt
        import cv2
        import numpy as np
        display = frame.copy()
        if box is not None:
            x, y, w, h = box
            color = (0, 220, 0) if valid else (0, 60, 220)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            if landmarks:
                for pt in landmarks.values():
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 2, color, -1)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        pb = GdkPixbuf.Pixbuf.new_from_data(
            rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w2, h2, w2 * ch
        )
        texture = Gdk.Texture.new_for_pixbuf(pb)
        face_str = "Face: Yes" if box is not None else "Face: No"
        live_str = ("Liveness: ✓" if valid
                    else f"Liveness: {reason}" if box is not None else "Liveness: —")
        GLib.idle_add(self._ui_frame, texture, face_str, live_str)

    def _on_sample(self, index: int, total: int) -> None:
        GLib.idle_add(self._ui_progress, index, total)

    def _on_pose(self, idx: int, pose, n: int, total: int) -> None:
        GLib.idle_add(self._ui_pose, f"[{idx+1}/{total}]  {pose.instruction}")

    def _on_pose_transition(self, next_pose, seconds: int) -> None:
        if seconds == 0:
            GLib.idle_add(self._ui_pose, f"➜  {next_pose.instruction}")
        else:
            GLib.idle_add(
                self._ui_pose,
                f"✓ Pose done!   Next: {next_pose.instruction}  ({seconds}s…)"
            )

    def _ui_frame(self, texture, face_str, live_str) -> bool:
        self._picture.set_paintable(texture)
        self._lbl_face.set_text(face_str)
        self._lbl_live.set_text(live_str)
        return False

    def _ui_progress(self, index: int, total: int) -> bool:
        self._progress.set_fraction(index / max(total, 1))
        self._progress.set_text(f"{index} / {total} samples")
        return False

    def _ui_pose(self, instruction: str) -> bool:
        self._pose_lbl.set_text(instruction)
        return False

    def _finish(self, saved: int, error: Optional[str]) -> bool:
        self._running = False
        self._btn_start.set_sensitive(True)
        self._btn_stop.set_sensitive(False)

        if error:
            self._pose_lbl.set_text(f"Error: {error}")
            return False

        if saved >= self._total:
            self._enrolled = True
            self._pose_lbl.set_text(f"✓ {saved} samples saved — all poses complete!")
            self._wizard.set_can_advance(True)
        elif saved > 0:
            self._pose_lbl.set_text(
                f"{saved}/{self._total} samples saved. "
                "Press Start again to continue adding samples."
            )
            if saved >= 5:  # allow advance with partial enrollment
                self._enrolled = True
                self._wizard.set_can_advance(True)
        else:
            self._pose_lbl.set_text("No samples captured. Check camera and try again.")

        return False

    def can_advance(self) -> bool:
        return self._enrolled or get_enrolled_count(getpass.getuser()) > 0


# ── Page 4: Test ─────────────────────────────────────────────────────────────

class TestPage(WizardPage):
    title = "Test Face Recognition"
    subtitle = "Verify everything is working before changing login settings"

    def __init__(self) -> None:
        super().__init__()
        self._passed = False

        desc = _label(
            "Before enabling face login, let's confirm the system "
            "can recognize you correctly.",
            wrap=True
        )
        self.append(desc)

        self._result_lbl = _label("", css="title-3", align=Gtk.Align.CENTER)
        self._result_lbl.set_wrap(True)
        self.append(self._result_lbl)

        self._spinner = Gtk.Spinner()
        self._spinner.set_halign(Gtk.Align.CENTER)
        self._spinner.set_size_request(48, 48)
        self.append(self._spinner)

        self._btn = Gtk.Button(label="Test Now")
        self._btn.set_halign(Gtk.Align.CENTER)
        self._btn.add_css_class("suggested-action")
        self._btn.connect("clicked", self._on_test)
        self.append(self._btn)

        self._detail = _label("", css="caption", wrap=True)
        self.append(self._detail)

    def on_enter(self, wizard: "SetupWizard") -> None:
        self._wizard = wizard
        wizard.set_can_advance(False)
        self._result_lbl.set_text("Click 'Test Now' and look at the camera.")

    def _on_test(self, _btn: Gtk.Button) -> None:
        self._btn.set_sensitive(False)
        self._spinner.start()
        self._result_lbl.set_text("Looking for your face…")
        self._passed = False
        threading.Thread(target=self._run_test, daemon=True).start()

    def _run_test(self) -> None:
        try:
            result = subprocess.run(
                [str(WRAPPER), "auth", "--timeout", "7"],
                capture_output=True, text=True, timeout=12
            )
            ok = result.returncode == 0
            output = (result.stdout + result.stderr).strip()
        except Exception as exc:
            ok = False
            output = str(exc)
        GLib.idle_add(self._finish, ok, output)

    def _finish(self, ok: bool, output: str) -> bool:
        self._spinner.stop()
        self._btn.set_sensitive(True)
        self._passed = ok

        if ok:
            self._result_lbl.set_text("✓ Recognized successfully!")
            self._result_lbl.remove_css_class("error")
            self._result_lbl.add_css_class("success")
            self._wizard.set_can_advance(True)
        else:
            self._result_lbl.set_text("✗ Not recognized — try again")
            self._result_lbl.remove_css_class("success")
            self._result_lbl.add_css_class("error")
            self._detail.set_text(output or "No face detected within timeout")

        return False

    def can_advance(self) -> bool:
        return self._passed


# ── Page 5: PAM Setup ─────────────────────────────────────────────────────────

PAM_TARGETS = [
    ("sudo",         "sudo (terminal privilege escalation)", True),
    ("sudo-i",       "sudo -i (root shell)", True),
    ("gdm-password", "Login screen and lock screen", True),
    ("polkit-1",     "GUI privilege dialogs (polkit)", False),
]

class PamPage(WizardPage):
    title = "Enable Face Login"
    subtitle = "Choose where to enable face recognition"

    def __init__(self) -> None:
        super().__init__()
        self._done = False

        desc = _label(
            "Face recognition will be added as a 'sufficient' auth method — "
            "your password always works as a fallback. "
            "Backups of all modified files will be saved.",
            wrap=True
        )
        self.append(desc)

        self.append(Gtk.Separator())

        # Checkboxes
        check_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self._checks: dict[str, Gtk.CheckButton] = {}
        for key, label, default in PAM_TARGETS:
            cb = Gtk.CheckButton(label=label)
            cb.set_active(default)
            self._checks[key] = cb
            check_box.append(cb)
        self.append(_section("Enable for:", check_box))

        # Howdy warning
        self._howdy_row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        howdy_lbl = _label(
            "⚠  Howdy is installed and active in common-auth.\n"
            "It may conflict — disable it?",
            wrap=True, css="warning"
        )
        self._howdy_check = Gtk.CheckButton(label="Disable Howdy in common-auth")
        self._howdy_check.set_active(True)
        self._howdy_row.append(howdy_lbl)
        self._howdy_row.append(self._howdy_check)
        self._howdy_row.set_visible(False)
        self.append(self._howdy_row)

        self.append(Gtk.Separator())

        # Apply button + spinner
        apply_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        apply_row.set_halign(Gtk.Align.CENTER)
        self._btn_apply = Gtk.Button(label="Apply")
        self._btn_apply.add_css_class("suggested-action")
        self._btn_apply.connect("clicked", self._on_apply)
        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(24, 24)
        apply_row.append(self._btn_apply)
        apply_row.append(self._spinner)
        self.append(apply_row)

        # Status + log
        self._status = _label("", css="title-4", align=Gtk.Align.CENTER)
        self.append(self._status)

        sw, self._log = _scrolled_log()
        sw.set_min_content_height(120)
        self.append(sw)

    def on_enter(self, wizard: "SetupWizard") -> None:
        self._wizard = wizard
        wizard.set_can_advance(False)

        # Detect howdy
        try:
            result = subprocess.run(
                ["grep", "-q", "howdy", "/etc/pam.d/common-auth"],
                capture_output=True
            )
            self._howdy_row.set_visible(result.returncode == 0)
        except Exception:
            pass

    def _on_apply(self, _btn: Gtk.Button) -> None:
        targets = [k for k, cb in self._checks.items() if cb.get_active()]
        disable_howdy = (
            self._howdy_row.get_visible() and self._howdy_check.get_active()
        )
        self._btn_apply.set_sensitive(False)
        self._spinner.start()
        self._status.set_text("Applying PAM configuration…")
        threading.Thread(
            target=self._run_pam, args=(targets, disable_howdy), daemon=True
        ).start()

    def _run_pam(self, targets: list[str], disable_howdy: bool) -> None:
        # Batch all privileged operations into one script → one pkexec prompt
        pam_line = (
            "auth sufficient pam_python.so "
            "/opt/ubuntu-face-login/pam_face.py"
        )

        script_lines = ["#!/bin/bash", "set -e", ""]

        for target in targets:
            pam_file = f"/etc/pam.d/{target}"
            bak_file = f"{pam_file}.ubuntu-face-login.bak"
            script_lines += [
                f'# --- {target} ---',
                f'cp "{pam_file}" "{bak_file}"',
                f'echo "Backed up {pam_file}"',
                # Only add if not already present
                f'if ! grep -q "ubuntu-face-login" "{pam_file}"; then',
                f'  sed -i \'/@include common-auth/i {pam_line}\' "{pam_file}"',
                f'  echo "✓ Added face auth to {pam_file}"',
                f'else',
                f'  echo "✓ Already configured: {pam_file}"',
                f'fi',
                '',
            ]

        if disable_howdy:
            script_lines += [
                '# --- disable howdy ---',
                r"sed -i 's|^auth.*pam_python.so /lib/security/howdy/pam.py|#&|' /etc/pam.d/common-auth",
                'echo "✓ Disabled Howdy in common-auth"',
                '',
            ]

        script = "\n".join(script_lines)

        # Write temp script, run once with pkexec
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", prefix="ufl-pam-", delete=False
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            os.chmod(script_path, 0o755)
            proc = subprocess.run(
                ["pkexec", "bash", script_path],
                capture_output=True, text=True,
            )
            output = (proc.stdout + proc.stderr).strip()
            ok = proc.returncode == 0
        except Exception as exc:
            output = str(exc)
            ok = False
        finally:
            os.unlink(script_path)

        GLib.idle_add(self._finish, ok, output)

    def _finish(self, ok: bool, log: str) -> bool:
        self._spinner.stop()
        _append_log(self._log, log)

        if ok:
            self._status.set_text("✓ PAM configuration applied!")
            self._status.add_css_class("success")
            self._done = True
            self._wizard.set_can_advance(True)
        else:
            self._status.set_text("✗ Some changes failed — check output above")
            self._status.add_css_class("error")
            self._btn_apply.set_sensitive(True)
        return False

    def can_advance(self) -> bool:
        return self._done


# ── Page 6: Done ─────────────────────────────────────────────────────────────

class DonePage(WizardPage):
    title = "Setup Complete"
    subtitle = "Face recognition is ready to use"

    def __init__(self) -> None:
        super().__init__()

        icon = Gtk.Image.new_from_icon_name("emblem-ok-symbolic")
        icon.set_pixel_size(72)
        icon.set_halign(Gtk.Align.CENTER)
        self.append(icon)

        msg = _label(
            "Ubuntu Face Login is active.\n\n"
            "Test it now: open a new terminal and run  sudo whoami\n"
            "Look at the camera — it should authenticate you in ~1 second.\n\n"
            "Your password always works as a fallback.",
            wrap=True, align=Gtk.Align.CENTER
        )
        msg.set_justify(Gtk.Justification.CENTER)
        self.append(msg)

        self.append(Gtk.Separator())

        tips = _label(
            "Useful commands:\n"
            "  ubuntu-face-login enroll <user>   — re-enroll or add more samples\n"
            "  ubuntu-face-login auth             — test authentication\n"
            "  sudo bash ~/ubuntu-face-login/uninstall.sh  — remove everything",
            wrap=True
        )
        tips.add_css_class("monospace")
        self.append(tips)


# ── Wizard window ─────────────────────────────────────────────────────────────

PAGES = [WelcomePage, InstallPage, EnrollPage, TestPage, PamPage, DonePage]


class SetupWizard(Adw.ApplicationWindow):

    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Ubuntu Face Login Setup")
        self.set_default_size(640, 680)

        self._pages = [cls() for cls in PAGES]
        self._current = 0
        self._can_advance = True

        # Header bar
        header = Adw.HeaderBar()
        header.set_show_end_title_buttons(False)

        self._back_btn = Gtk.Button(label="← Back")
        self._back_btn.connect("clicked", lambda _: self.prev_page())
        self._back_btn.set_sensitive(False)
        header.pack_start(self._back_btn)

        self._next_btn = Gtk.Button(label="Next →")
        self._next_btn.add_css_class("suggested-action")
        self._next_btn.connect("clicked", lambda _: self.next_page())
        header.pack_end(self._next_btn)

        # Title widget
        self._title_lbl = Adw.WindowTitle(
            title=self._pages[0].title,
            subtitle=self._pages[0].subtitle
        )
        header.set_title_widget(self._title_lbl)

        # Page indicator
        self._page_indicator = _label(
            f"Step 1 of {len(self._pages)}",
            css="caption", align=Gtk.Align.CENTER
        )

        # Stack
        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        for i, page in enumerate(self._pages):
            self._stack.add_named(page, f"page{i}")

        # Scrolled content (pages may be taller than window on small screens)
        scroll = Gtk.ScrolledWindow()
        scroll.set_child(self._stack)
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        # Root layout
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        root.append(header)
        root.append(scroll)
        root.append(self._page_indicator)
        self._page_indicator.set_margin_bottom(8)

        self.set_content(root)
        self._show_page(0)

    def _show_page(self, index: int) -> None:
        self._current = index
        page = self._pages[index]
        self._stack.set_visible_child_name(f"page{index}")
        self._title_lbl.set_title(page.title)
        self._title_lbl.set_subtitle(page.subtitle)
        self._page_indicator.set_text(f"Step {index + 1} of {len(self._pages)}")

        is_last = index == len(self._pages) - 1
        self._next_btn.set_label("Finish" if is_last else "Next →")
        self._back_btn.set_sensitive(index > 0)
        self._can_advance = True
        self.set_can_advance(page.can_advance())

        page.on_enter(self)

    def next_page(self) -> None:
        if self._current < len(self._pages) - 1:
            self._show_page(self._current + 1)
        else:
            self.close()

    def prev_page(self) -> None:
        if self._current > 0:
            self._show_page(self._current - 1)

    def set_can_advance(self, allowed: bool) -> None:
        self._can_advance = allowed
        self._next_btn.set_sensitive(allowed)


# ── Application ───────────────────────────────────────────────────────────────

class WizardApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id="com.ubuntu.facelogin.setup")

    def do_activate(self) -> None:
        win = SetupWizard(self)
        win.present()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    app = WizardApp()
    app.run(sys.argv)


if __name__ == "__main__":
    main()
