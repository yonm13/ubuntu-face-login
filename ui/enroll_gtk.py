#!/usr/bin/env python3
"""GTK4 enrollment UI for Ubuntu Face Login.

Displays a live camera preview with face-detection status, liveness
feedback, and a progress bar while capturing face embeddings via
:func:`facelogin.enroll.enroll_user`.

Falls back gracefully when GTK4 / PyGObject is not installed.

Usage::

    python -m ui.enroll_gtk
    python -m ui.enroll_gtk --user alice
"""

from __future__ import annotations

import argparse
import getpass
import logging
import sys
import threading
from typing import Optional

import cv2
import numpy as np

try:
    import gi

    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk, GLib, GdkPixbuf
except (ImportError, ValueError) as exc:
    print(
        f"GTK4 is not available ({exc}).\n"
        "Install PyGObject + GTK4 or use the CLI fallback:\n"
        "  python -m facelogin.enroll --cli <user_id>",
        file=sys.stderr,
    )
    sys.exit(1)

# Add project root to path so we can import facelogin
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from facelogin.enroll import enroll_user  # noqa: E402
from facelogin.config import get_config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def _frame_to_pixbuf(frame: np.ndarray) -> GdkPixbuf.Pixbuf:
    """Convert an OpenCV BGR frame to a GdkPixbuf (RGB, 8-bit)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, channels = rgb.shape
    rowstride = w * channels
    return GdkPixbuf.Pixbuf.new_from_data(
        rgb.tobytes(),
        GdkPixbuf.Colorspace.RGB,
        False,       # no alpha
        8,           # bits per channel
        w,
        h,
        rowstride,
    )


# ---------------------------------------------------------------------------
# Enrollment window
# ---------------------------------------------------------------------------

class EnrollWindow(Gtk.ApplicationWindow):
    """Main enrollment window with camera feed and progress tracking."""

    def __init__(self, app: Gtk.Application, user_id: str) -> None:
        super().__init__(application=app, title="Ubuntu Face Login — Enrollment")
        self.set_default_size(680, 640)

        self._user_id = user_id
        self._config = get_config()
        self._total_samples = self._config.enrollment.samples
        self._running = False
        self._enroll_thread: Optional[threading.Thread] = None

        self._build_ui()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self) -> None:
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
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

        # Status labels row
        status_grid = Gtk.Grid(column_spacing=16, row_spacing=4)
        status_grid.set_margin_top(8)
        status_grid.set_halign(Gtk.Align.CENTER)

        self._lbl_face = self._add_status_row(status_grid, 0, "Face Detected:", "—")
        self._lbl_liveness = self._add_status_row(status_grid, 1, "Liveness:", "—")
        self._lbl_confidence = self._add_status_row(status_grid, 2, "Confidence:", "—")
        root.append(status_grid)

        # User ID entry
        user_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        user_box.set_margin_top(8)
        user_box.set_halign(Gtk.Align.CENTER)
        user_label = Gtk.Label(label="User ID:")
        self._user_entry = Gtk.Entry()
        self._user_entry.set_text(self._user_id)
        self._user_entry.set_width_chars(24)
        user_box.append(user_label)
        user_box.append(self._user_entry)
        root.append(user_box)

        # Action buttons
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        btn_box.set_margin_top(8)
        btn_box.set_halign(Gtk.Align.CENTER)

        self._btn_start = Gtk.Button(label="Start Capture")
        self._btn_start.add_css_class("suggested-action")
        self._btn_start.connect("clicked", self._on_start_clicked)

        self._btn_cancel = Gtk.Button(label="Cancel")
        self._btn_cancel.add_css_class("destructive-action")
        self._btn_cancel.connect("clicked", self._on_cancel_clicked)
        self._btn_cancel.set_sensitive(False)

        btn_box.append(self._btn_start)
        btn_box.append(self._btn_cancel)
        root.append(btn_box)

        # Progress bar
        self._progress = Gtk.ProgressBar()
        self._progress.set_margin_top(8)
        self._progress.set_show_text(True)
        self._progress.set_text("0 / %d samples" % self._total_samples)
        self._progress.set_fraction(0.0)
        root.append(self._progress)

        # Status message
        self._lbl_status = Gtk.Label(label="Press 'Start Capture' to begin enrollment.")
        self._lbl_status.set_margin_top(4)
        self._lbl_status.add_css_class("dim-label")
        root.append(self._lbl_status)

    @staticmethod
    def _add_status_row(
        grid: Gtk.Grid, row: int, label_text: str, initial: str
    ) -> Gtk.Label:
        label = Gtk.Label(label=label_text)
        label.set_xalign(1.0)
        label.add_css_class("dim-label")
        grid.attach(label, 0, row, 1, 1)

        value = Gtk.Label(label=initial)
        value.set_xalign(0.0)
        value.set_width_chars(30)
        grid.attach(value, 1, row, 1, 1)
        return value

    # -- Callbacks from enroll_user (called on background thread) ------------

    def _on_frame(
        self,
        frame: np.ndarray,
        box,
        landmarks,
        confidence: Optional[float],
        valid: bool,
        reason: str,
    ) -> None:
        """Per-frame callback — push UI updates to the main thread."""
        # Draw bounding box overlay on a copy
        display = frame.copy()
        if box is not None:
            x, y, w, h = box
            color = (0, 255, 0) if valid else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        pixbuf = _frame_to_pixbuf(display)

        face_str = "Yes" if box is not None else "No"
        if valid:
            live_str = "Yes"
        else:
            live_str = f"No — {reason}" if box is not None else "—"
        conf_str = f"{confidence:.2f}" if confidence is not None else "—"

        GLib.idle_add(self._update_preview, pixbuf, face_str, live_str, conf_str)

    def _on_sample(self, index: int, total: int) -> None:
        """Per-sample callback — update progress bar on the main thread."""
        GLib.idle_add(self._update_progress, index, total)

    # -- Main-thread UI updates ----------------------------------------------

    def _update_preview(
        self,
        pixbuf: GdkPixbuf.Pixbuf,
        face_str: str,
        live_str: str,
        conf_str: str,
    ) -> bool:
        texture = Gtk.gdk_texture_new_for_pixbuf(pixbuf) if hasattr(Gtk, "gdk_texture_new_for_pixbuf") else None
        # GTK4 GtkPicture needs a GdkTexture — create from pixbuf
        try:
            from gi.repository import Gdk
            texture = Gdk.Texture.new_for_pixbuf(pixbuf)
            self._picture.set_paintable(texture)
        except Exception:
            # Fallback: set pixbuf directly if available
            pass

        self._lbl_face.set_text(face_str)
        self._lbl_liveness.set_text(live_str)
        self._lbl_confidence.set_text(conf_str)
        return False  # remove from idle queue

    def _update_progress(self, index: int, total: int) -> bool:
        frac = index / max(total, 1)
        self._progress.set_fraction(frac)
        self._progress.set_text(f"{index} / {total} samples")
        self._lbl_status.set_text(f"Sample {index}/{total} saved")
        return False

    def _finish_enrollment(self, saved: int) -> bool:
        """Called on the main thread when enrollment completes."""
        self._running = False
        self._btn_start.set_sensitive(True)
        self._btn_cancel.set_sensitive(False)
        self._user_entry.set_sensitive(True)

        if saved >= self._total_samples:
            self._lbl_status.set_text("Enrollment complete!")
            self._progress.set_fraction(1.0)
            self._progress.set_text(f"{saved} / {self._total_samples} samples")
            self._btn_start.set_label("Close")
            self._btn_start.disconnect_by_func(self._on_start_clicked)
            self._btn_start.connect("clicked", lambda _btn: self.close())
        elif saved > 0:
            self._lbl_status.set_text(
                f"Enrollment stopped — {saved}/{self._total_samples} samples saved."
            )
        else:
            self._lbl_status.set_text("No samples captured. Check camera and try again.")
        return False

    # -- Button handlers -----------------------------------------------------

    def _on_start_clicked(self, _btn: Gtk.Button) -> None:
        user_id = self._user_entry.get_text().strip()
        if not user_id:
            self._lbl_status.set_text("Please enter a user ID.")
            return

        self._user_id = user_id
        self._running = True
        self._btn_start.set_sensitive(False)
        self._btn_cancel.set_sensitive(True)
        self._user_entry.set_sensitive(False)
        self._progress.set_fraction(0.0)
        self._progress.set_text(f"0 / {self._total_samples} samples")
        self._lbl_status.set_text("Look at the camera...")

        self._enroll_thread = threading.Thread(
            target=self._run_enrollment, daemon=True
        )
        self._enroll_thread.start()

    def _on_cancel_clicked(self, _btn: Gtk.Button) -> None:
        self._running = False
        self._lbl_status.set_text("Cancelling...")
        self._btn_cancel.set_sensitive(False)

    # -- Background enrollment -----------------------------------------------

    def _run_enrollment(self) -> None:
        """Run enroll_user on a background thread."""
        try:
            saved = enroll_user(
                user_id=self._user_id,
                num_samples=self._total_samples,
                on_frame=self._on_frame_guard,
                on_sample=self._on_sample,
            )
        except Exception as exc:
            logger.exception("Enrollment failed")
            GLib.idle_add(self._show_error, str(exc))
            saved = 0

        GLib.idle_add(self._finish_enrollment, saved)

    def _on_frame_guard(self, frame, box, landmarks, confidence, valid, reason):
        """Wrapper that checks the running flag to support cancellation.

        enroll_user reads frames in a tight loop — when the user clicks
        Cancel, we need to stop.  The cleanest way without modifying
        enroll_user is to raise KeyboardInterrupt from the callback,
        which enroll_user already handles.
        """
        if not self._running:
            raise KeyboardInterrupt("Cancelled by user")
        self._on_frame(frame, box, landmarks, confidence, valid, reason)

    def _show_error(self, message: str) -> bool:
        self._running = False
        self._btn_start.set_sensitive(True)
        self._btn_cancel.set_sensitive(False)
        self._user_entry.set_sensitive(True)
        self._lbl_status.set_text(f"Error: {message}")
        return False


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class EnrollApp(Gtk.Application):
    """GTK4 application wrapper."""

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

    app = EnrollApp(user_id=args.user)
    app.run([])


if __name__ == "__main__":
    main()
