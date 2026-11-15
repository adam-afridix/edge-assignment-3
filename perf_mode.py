# perf_mode.py
# Windows 11 puts processes that are not in the foreground into "EcoQoS"
# (power throttling): they get moved to the slow efficiency cores at low clock.
# On my laptop (i5-13420H, 4 P-cores + 4 E-cores) this made the same model jump
# from ~2 ms to ~60 ms per image in the middle of a benchmark, so the timings
# were useless. This function turns the throttling off for the current process
# and raises its priority, so latency numbers are stable.
# On Linux / macOS it does nothing.

import sys


def high_performance():
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG),
                        ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        ProcessPowerThrottling = 4          # PROCESS_INFORMATION_CLASS value
        HIGH_PRIORITY_CLASS = 0x80

        state = PROCESS_POWER_THROTTLING_STATE(1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)  # 0 = never throttle
        k32 = ctypes.windll.kernel32
        # types must be set, otherwise the 64-bit handle gets cut to 32 bits
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        h = k32.GetCurrentProcess()
        ok = k32.SetProcessInformation(h, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state))
        k32.SetPriorityClass(h, HIGH_PRIORITY_CLASS)
        return bool(ok)
    except Exception:
        return False
