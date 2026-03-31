"""PAM bridge module — calls the face-auth script as a subprocess.

This module is loaded by ``pam_python.so`` and exposes the standard
PAM entry points.  It delegates actual authentication to the installed
``/usr/local/bin/ubuntu-face-login`` script (which runs ``auth.main``).

PAM configuration example (/etc/pam.d/sudo)::

    auth sufficient pam_python.so /usr/share/ubuntu-face-login/pam_module.py timeout=3

argv parameters:
    timeout=N   Override the default authentication timeout (seconds).
"""

from __future__ import annotations

import subprocess
import syslog

_AUTH_BINARY = "/usr/local/bin/ubuntu-face-login"
_SYSLOG_TAG = "ubuntu_face_login"
_SAFETY_MARGIN = 5  # extra seconds beyond script timeout for subprocess

# PAM constants (defined by pam_python, but we define fallbacks for testing)
PAM_SUCCESS = 0
PAM_AUTH_ERR = 7
PAM_IGNORE = 25


def _log(priority: int, msg: str) -> None:
    """Log to syslog with our tag."""
    syslog.openlog(_SYSLOG_TAG, syslog.LOG_PID, syslog.LOG_AUTH)
    syslog.syslog(priority, msg)
    syslog.closelog()


def _parse_timeout(argv: list[str]) -> int | None:
    """Extract timeout=N from PAM argv, or None if not present."""
    for arg in argv:
        if arg.startswith("timeout="):
            try:
                return int(arg.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
    return None


def _parse_threshold(argv: list[str]) -> float | None:
    """Extract threshold=X from PAM argv, or None if not present."""
    for arg in argv:
        if arg.startswith("threshold="):
            try:
                return float(arg.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
    return None


def _get_service(pamh) -> str:
    """Read the PAM service name from the handle, with safe fallback."""
    try:
        return pamh.get_item(pamh.PAM_SERVICE) or "default"
    except Exception:
        return "default"


def _send_msg(pamh, text: str) -> None:
    """Send a PAM_TEXT_INFO message back to the calling application.

    GDM shows this as status text below the login prompt.
    sudo prints it to the terminal.
    Failures are silently ignored — the message is informational only.
    """
    try:
        pamh.conversation(pamh.Message(pamh.PAM_TEXT_INFO, text))
    except Exception:
        pass


def pam_sm_authenticate(pamh, flags, argv):
    """PAM authentication entry point.

    Runs the face-auth script as a subprocess and returns PAM_SUCCESS
    on exit code 0, PAM_AUTH_ERR otherwise.
    """
    timeout = _parse_timeout(argv)
    threshold = _parse_threshold(argv)
    service = _get_service(pamh)

    cmd = [_AUTH_BINARY, "auth"]
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
    cmd.extend(["--service", service])

    script_timeout = timeout if timeout is not None else 10
    subprocess_timeout = script_timeout + _SAFETY_MARGIN

    _log(syslog.LOG_INFO, f"Starting auth for service={service} timeout={script_timeout} threshold={threshold}")
    _send_msg(pamh, "🔍  Face recognition in progress…")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=subprocess_timeout,
        )
        if result.returncode == 0:
            _log(syslog.LOG_INFO, f"Auth succeeded for service={service}")
            _send_msg(pamh, "✓  Face recognised")
            return PAM_SUCCESS
        else:
            _log(
                syslog.LOG_INFO,
                f"Auth failed for service={service}: exit={result.returncode}",
            )
            _send_msg(pamh, "✗  Face not recognised — falling back to password")
            return PAM_AUTH_ERR

    except subprocess.TimeoutExpired:
        _log(
            syslog.LOG_WARNING,
            f"Auth subprocess timed out after {subprocess_timeout}s for service={service}",
        )
        _send_msg(pamh, "✗  Face recognition timed out — falling back to password")
        return PAM_AUTH_ERR

    except FileNotFoundError:
        _log(syslog.LOG_ERR, f"Auth binary not found: {_AUTH_BINARY}")
        return PAM_AUTH_ERR

    except Exception as exc:
        _log(syslog.LOG_ERR, f"Unexpected error in pam_sm_authenticate: {exc}")
        return PAM_AUTH_ERR


def pam_sm_setcred(pamh, flags, argv):
    """PAM credential-setting entry point — always succeeds."""
    return PAM_SUCCESS
