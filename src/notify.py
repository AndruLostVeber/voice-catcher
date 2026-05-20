from __future__ import annotations

import sys
import threading

APP_NAME = "Voice Notes AI"


def _play_sound(success: bool = True) -> None:
    try:
        import winsound

        if success:
            winsound.MessageBeep(winsound.MB_OK)
        else:
            winsound.MessageBeep(winsound.MB_ICONHAND)
    except Exception:
        pass


def notify(title: str, message: str, success: bool = True) -> None:
    """Показать системное уведомление и проиграть звук. Никогда не падает."""
    def _run():
        _play_sound(success)
        try:
            from plyer import notification

            notification.notify(
                title=title,
                message=message[:240],
                app_name=APP_NAME,
                timeout=6,
            )
        except Exception:
            pass

    if sys.platform == "win32":
        threading.Thread(target=_run, daemon=True).start()
    else:
        _run()
