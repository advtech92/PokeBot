import numpy as np
import cv2
from mss import mss
import pygetwindow as gw


def capture_screen(window_title="RetroArch"):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"No windows found with title: {window_title}")
    retroarch_window = windows[0]
    left, top, right, bottom = retroarch_window.left, retroarch_window.top, retroarch_window.right, retroarch_window.bottom
    monitor = {'top': top, 'left': left, 'width': right-left, 'height': bottom-top}
    with mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
