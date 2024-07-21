from pynput.keyboard import Controller, Key
import time
import pygetwindow as gw

keyboard = Controller()


def press_key(key, duration=0.1):
    key_map = {
        'up': Key.up,
        'down': Key.down,
        'left': Key.left,
        'right': Key.right,
        'a': 'z',      # Change according to your game key mapping
        'b': 'x',      # Change according to your game key mapping
        'start': Key.enter,
        'select': Key.shift_r  # Right Shift
    }

    if key in key_map:
        keyboard.press(key_map[key])
        time.sleep(duration)
        keyboard.release(key_map[key])


def bring_window_to_front(window_title="RetroArch Gearboy 3.4.1"):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"No windows found with title: {window_title}")
    retroarch_window = windows[0]
    retroarch_window.activate()
