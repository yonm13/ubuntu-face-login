#!/usr/bin/env python3
"""GTK4 enrollment UI for Ubuntu Face Login.

Displays a live camera preview with face-detection status, pose
instructions, and a progress bar while capturing face embeddings
via :func:`facelogin.enroll.enroll_user`.

Falls back gracefully when GTK4 / PyGObject is not installed.

Usage::

    python ui/enroll_gtk.py
    python ui/enroll_gtk.py --user alice
"""

from __future__ import annotations

import argparse
import copy
import getpass
import logging
import sys
import threading
from typing import List, Optional

import cv2
import numpy as np

try:
    import gi
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gdk, GdkPixbuf, GLib, Gtk
except (ImportError, ValueError) as exc:
    print(
        f"GTK4 is not available ({exc}).\n"
        "Install PyGObject + GTK4 or use the CLI fallback:\n"
        "  python -m facelogin.enroll --cli <user_id>",
        file=sys.stderr,
    )
    sys.exit(1)

sys.path.insert(
    0,
    str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"),
)

from facelogin.config import get_config  # noqa: E402
from facelogin.enroll import DEFAULT_POSES, Pose, enroll_user  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_to_texture(frame: np.ndarray) -> Gdk.Texture:
    """Convert an OpenCV BGR ndarray to a GdkTexture for GTK4 Picture."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    pixbuf = GdkPixbuf.Pixbuf.new_from_data(
        rgb.tobytes(),
        GdkPixbuf.Colorspace.RGB,
        False, 8,
        w, h,
        w * ch,
    )
    return Gdk.Texture.new_for_pixbuf(pixbuf)


def _css(widget: Gtk.Widget, *classes: str) -> Gtk.Widget:
    for cls in classes:
        widget.add_css_class(cls)
    return widget


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(Gtk.Dialog):
    """Let the user tweak sample counts before starting enrollment."""

    def __init__(self, parent: Gtk.Window, poses: List[Pose], user_id: str) -> None:
        super().__init__(
            title="Enrollment Settings",
            transient_for=parent,
            modal=True,
        )
        self.set_default_size(360, -1)
        self._poses = [copy.copy(p) for p in poses]
        self._user_id = user_id
        self._spins: list[Gtk.SpinButton] = []

        self.add_button("Cancel", Gtk.ResponseType.CANCEL)
        ok = self.add_button("OK", Gtk.ResponseType.OK)
        ok.add_css_class("suggested-action")

        box = self.get_content_area()
        box.set_spacing(12)
        box.set_margin_top(16)
        box.set_margin_bottom(8)
        box.set_margin_start(16)
        box.set_margin_end(16)

        # User ID
        uid_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        uid_row.append(Gtk.Label(label="User ID:"))
        self._uid_entry = Gtk.Entry()
        self._uid_entry.set_text(user_id)
        self._uid_entry.set_hexpand(True)
        uid_row.append(self._uid_entry)
        box.append(uid_row)

        # Separator
        box.append(Gtk.Separator())

        # Per-pose sample counts
        box.append(_css(Gtk.Label(label="Samples per pose:"), "heading"))

        grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        for row_idx, pose in enumerate(self._poses):
            lbl = Gtk.Label(label=pose.instruction)
            lbl.set_xalign(0.0)
            lbl.set_hexpand(True)
            grid.attach(lbl, 0, row_idx, 1, 1)

            spin = Gtk.SpinButton.new_with_range(1, 20, 1)
            spin.set_value(pose.samples)
            self._spins.append(spin)
            grid.attach(spin, 1, row_idx, 1, 1)

        box.append(grid)

        # Delay
        box.append(Gtk.Separator())
        delay_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        delay_row.append(Gtk.Label(label="Delay between samples (s):"))
        self._delay_spin = Gtk.SpinButton.new_with_range(0.1, 2.0, 0.1)
        self._delay_spin.set_value(0.4)
        self._delay_spin.set_digits(1)
        delay_row.append(self._delay_spin)
        box.append(delay_row)

        self.show()

    def get_result(self) -> tuple[str, List[Pose], float]:
        """Return (user_id, poses_with_updated_counts, delay)."""
        uid = self._uid_entry.get_text().strip() or self._user_id
        for pose, spin in zip(self._poses, self._spins):
            pose.samples = int(spin.get_value())
        delay = self._delay_spin.get_value()
        return uid, self._poses, delay


# ---------------------------------------------------------------------------
# Main enrollment window
# ---------------------------------------------------------------------------

class EnrollWindow(Gtk.ApplicationWindow):

    def __init__(self, app: Gtk.Application, user_id: str) -> None:
        super().__init__(application=app, title="Ubuntu Face Login — Enrollment")
        self.set_default_size(700, 660)

        self._user_id = user_id
        self._poses: List[Pose] = [copy.copy(p) for p in DEFAULT_POSES]
        self._total_samples = sum(p.samples for p in self._poses)
        self._sample_delay = 0.4
        self._running = False
        self._current_pose_label = ""
        self._enroll_thread: Optional[threading.Thread] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        root.set_margin_top(12)
        root.set_margin_bottom(12)
        root.set_margin_start(12)
        root.set_margin_end(12)
        self.set_child(root)

        # Camera preview
        self._picture = Gtk.Picture()
        self._picture.set_size_request(640, 360)
        self._picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        root.append(self._picture)

        # --- Pose instruction (big, prominent) ---
        pose_frame = Gtk.Frame()
        pose_frame.set_margin_top(10)
        pose_frame.set_margin_bottom(4)

        self._lbl_pose = Gtk.Label(label="Press 'Start' to begin")
        self._lbl_pose.set_wrap(True)
        self._lbl_pose.set_justify(Gtk.Justification.CENTER)
        self._lbl_pose.add_css_class("title-2")
        self._lbl_pose.set_margin_top(10)
        self._lbl_pose.set_margin_bottom(10)
        self._lbl_pose.set_margin_start(16)
        self._lbl_pose.set_margin_end(16)
        pose_frame.set_child(self._lbl_pose)
        root.append(pose_frame)

        # --- Detection status row ---
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=24)
        status_box.set_halign(Gtk.Align.CENTER)
        status_box.set_margin_top(6)

        self._lbl_face = self._status_chip("Face: —")
        self._lbl_liveness = self._status_chip("Liveness: —")
        self._lbl_confidence = self._status_chip("Conf: —")
        status_box.append(self._lbl_face)
        status_box.append(self._lbl_liveness)
        status_box.append(self._lbl_confidence)
        root.append(status_box)

        # --- Progress ---
        prog_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        prog_box.set_margin_top(8)
        prog_box.set_margin_start(8)
        prog_box.set_margin_end(8)

        self._lbl_progress_text = Gtk.Label(label="0 / %d samples" % self._total_samples)
        self._lbl_progress_text.set_halign(Gtk.Align.START)
        self._lbl_progress_text.add_css_class("caption")
        prog_box.append(self._lbl_progress_text)

        self._progress = Gtk.ProgressBar()
        self._progress.set_fraction(0.0)
        prog_box.append(self._progress)
        root.append(prog_box)

        # --- Buttons ---
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        btn_box.set_halign(Gtk.Align.CENTER)
        btn_box.set_margin_top(10)

        self._btn_settings = Gtk.Button(label="⚙ Settings")
        self._btn_settings.connect("clicked", self._on_settings_clicked)

        self._btn_start = Gtk.Button(label="Start")
        self._btn_start.add_css_class("suggested-action")
        self._btn_start.connect("clicked", self._on_start_clicked)

        self._btn_cancel = Gtk.Button(label="Cancel")
        self._btn_cancel.add_css_class("destructive-action")
        self._btn_cancel.connect("clicked", self._on_cancel_clicked)
        self._btn_cancel.set_sensitive(False)

        btn_box.append(self._btn_settings)
        btn_box.append(self._btn_start)
        btn_box.append(self._btn_cancel)
        root.append(btn_box)

        # --- Status line ---
        self._lbl_status = Gtk.Label(label="Ready.")
        self._lbl_status.set_margin_top(6)
        self._lbl_status.add_css_class("dim-label")
        self._lbl_status.add_css_class("caption")
        root.append(self._lbl_status)

    @staticmethod
    def _status_chip(text: str) -> Gtk.Label:
        lbl = Gtk.Label(label=text)
        lbl.add_css_class("caption")
        lbl.add_css_class("dim-label")
        return lbl

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_settings_clicked(self, _btn: Gtk.Button) -> None:
        dlg = SettingsDialog(self, self._poses, self._user_id)

        def on_response(d: Gtk.Dialog, response: int) -> None:
            if response == Gtk.ResponseType.OK:
                uid, poses, delay = d.get_result()
                self._user_id = uid
                self._poses = poses
                self._sample_delay = delay
                self._total_samples = sum(p.samples for p in poses)
                self._lbl_progress_text.set_text(
                    f"0 / {self._total_samples} samples"
                )
                self._lbl_status.set_text(
                    f"User: {uid}  |  {self._total_samples} samples  |  delay {delay:.1f}s"
                )
            d.destroy()

        dlg.connect("response", on_response)

    def _on_start_clicked(self, _btn: Gtk.Button) -> None:
        self._running = True
        self._btn_start.set_sensitive(False)
        self._btn_settings.set_sensitive(False)
        self._btn_cancel.set_sensitive(True)
        self._progress.set_fraction(0.0)
        self._lbl_progress_text.set_text(f"0 / {self._total_samples} samples")
        self._lbl_status.set_text("Starting camera…")
        self._enroll_thread = threading.Thread(
            target=self._run_enrollment, daemon=True
        )
        self._enroll_thread.start()

    def _on_cancel_clicked(self, _btn: Gtk.Button) -> None:
        self._running = False
        self._btn_cancel.set_sensitive(False)
        self._lbl_status.set_text("Cancelling…")

    # ------------------------------------------------------------------
    # Enrollment callbacks (called on background thread)
    # ------------------------------------------------------------------

    def _on_frame(self, frame, box, landmarks, confidence, valid, reason) -> None:
        display = frame.copy()
        if box is not None:
            x, y, w, h = box
            color = (0, 220, 0) if valid else (0, 60, 220)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            if landmarks is not None:
                for pt in landmarks:
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 2, color, -1)

        texture = _frame_to_texture(display)
        face_str = "Face: Yes" if box is not None else "Face: No"
        if valid:
            live_str = "Liveness: ✓"
        else:
            live_str = f"Liveness: {reason}" if box is not None else "Liveness: —"
        conf_str = f"Conf: {confidence:.2f}" if confidence is not None else "Conf: —"

        GLib.idle_add(self._ui_update_frame, texture, face_str, live_str, conf_str)

    def _on_sample(self, index: int, total: int) -> None:
        GLib.idle_add(self._ui_update_progress, index, total)

    def _on_pose(self, pose_idx: int, pose: Pose, n: int, total_poses: int) -> None:
        instruction = f"[{pose_idx + 1}/{total_poses}]  {pose.instruction}"
        GLib.idle_add(self._ui_update_pose, instruction)

    # ------------------------------------------------------------------
    # Main-thread UI mutations (return False to auto-dequeue)
    # ------------------------------------------------------------------

    def _ui_update_frame(self, texture, face_str, live_str, conf_str) -> bool:
        self._picture.set_paintable(texture)
        self._lbl_face.set_text(face_str)
        self._lbl_liveness.set_text(live_str)
        self._lbl_confidence.set_text(conf_str)
        return False

    def _ui_update_progress(self, index: int, total: int) -> bool:
        self._progress.set_fraction(index / max(total, 1))
        self._lbl_progress_text.set_text(f"{index} / {total} samples")
        self._lbl_status.set_text(
            f"Sample {index}/{total} saved — {self._current_pose_label}"
        )
        return False

    def _ui_update_pose(self, instruction: str) -> bool:
        self._current_pose_label = instruction
        self._lbl_pose.set_text(instruction)
        return False

    def _ui_finish(self, saved: int) -> bool:
        self._running = False
        self._btn_start.set_sensitive(True)
        self._btn_settings.set_sensitive(True)
        self._btn_cancel.set_sensitive(False)

        if saved >= self._total_samples:
            self._lbl_pose.set_text("Enrollment complete! ✓")
            self._progress.set_fraction(1.0)
            self._lbl_progress_text.set_text(
                f"{saved} / {self._total_samples} samples"
            )
            self._lbl_status.set_text("All samples saved. You can close this window.")
            self._btn_start.set_label("Close")
            self._btn_start.disconnect_by_func(self._on_start_clicked)
            self._btn_start.connect("clicked", lambda _: self.close())
        elif saved > 0:
            self._lbl_pose.set_text("Stopped early")
            self._lbl_status.set_text(
                f"{saved}/{self._total_samples} samples saved. "
                "You can start again to add more."
            )
        else:
            self._lbl_pose.set_text("No face detected")
            self._lbl_status.set_text(
                "No samples captured. Check camera placement and try again."
            )
        return False

    def _ui_error(self, message: str) -> bool:
        self._running = False
        self._btn_start.set_sensitive(True)
        self._btn_settings.set_sensitive(True)
        self._btn_cancel.set_sensitive(False)
        self._lbl_pose.set_text("Error")
        self._lbl_status.set_text(f"Error: {message}")
        return False

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _on_frame_guard(self, frame, box, landmarks, confidence, valid, reason):
        if not self._running:
            raise KeyboardInterrupt("Cancelled by user")
        self._on_frame(frame, box, landmarks, confidence, valid, reason)

    def _run_enrollment(self) -> None:
        try:
            saved = enroll_user(
                user_id=self._user_id,
                poses=self._poses,
                sample_delay=self._sample_delay,
                on_frame=self._on_frame_guard,
                on_sample=self._on_sample,
                on_pose=self._on_pose,
            )
        except Exception as exc:
            logger.exception("Enrollment error")
            GLib.idle_add(self._ui_error, str(exc))
            return

        GLib.idle_add(self._ui_finish, saved)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class EnrollApp(Gtk.Application):
    def __init__(self, user_id: str) -> None:
        super().__init__(application_id="com.ubuntu.facelogin.enroll")
        self._user_id = user_id

    def do_activate(self) -> None:
        win = EnrollWindow(self, self._user_id)
        win.present()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="GTK4 enrollment UI for Ubuntu Face Login",
    )
    parser.add_argument(
        "--user",
        default=getpass.getuser(),
        help="User ID to enroll (default: current OS username)",
    )
    args = parser.parse_args()
    EnrollApp(user_id=args.user).run([])


if __name__ == "__main__":
    main()
