"""Keeps the background-removal worker from taking over the PC (Windows only).

The worker process runs inside a Windows Job Object, which the OS enforces:
  - a hard memory cap, so an oversized photo fails cleanly instead of paging
    the whole system to a standstill
  - a pinned CPU priority when you choose Normal or Low
  - kill-on-close, so if Odycentric exits or crashes the worker dies with it
    and nothing is left running in the background

How the worker gets into the job matters. A venv's python.exe is a launcher
that spawns the real interpreter as a child, and when that interpreter is the
Microsoft Store Python, Windows lets it break out of any job its parent was
in. So the worker joins the job itself, by name, as the first thing it does,
and the app refuses to hand it any work until contains() confirms it.
"""

import ctypes
import os
import subprocess
import uuid
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

GB = 1024**3

IDLE_PRIORITY = 0x40
BELOW_NORMAL_PRIORITY = 0x4000
NORMAL_PRIORITY = 0x20
HIGH_PRIORITY = 0x80

_JOB_EXTENDED_LIMITS = 9
_LIMIT_PRIORITY_CLASS = 0x20
_LIMIT_PROCESS_MEMORY = 0x100
_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x400
_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_ASSIGN_PROCESS = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_CREATE_NO_WINDOW = 0x08000000


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


kernel32.CreateJobObjectW.restype = wintypes.HANDLE
kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
kernel32.OpenJobObjectW.restype = wintypes.HANDLE
kernel32.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.SetInformationJobObject.restype = wintypes.BOOL
kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
kernel32.QueryInformationJobObject.restype = wintypes.BOOL
kernel32.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
kernel32.IsProcessInJob.restype = wintypes.BOOL
kernel32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.GetCurrentProcess.argtypes = []
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]
kernel32.SetPriorityClass.restype = wintypes.BOOL
kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]


kernel32.GetSystemPowerStatus.restype = wintypes.BOOL
kernel32.GetSystemPowerStatus.argtypes = [ctypes.POINTER(_PowerStatus)]


def _check(ok):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return ok


def _memory_status():
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    _check(kernel32.GlobalMemoryStatusEx(ctypes.byref(status)))
    return status


def free_memory():
    """Bytes of physical RAM available right now."""
    return _memory_status().ullAvailPhys


def total_memory():
    return _memory_status().ullTotalPhys


def on_battery():
    """True when the PC is running on battery, which slows the processor down."""
    status = _PowerStatus()
    return bool(kernel32.GetSystemPowerStatus(ctypes.byref(status))) and status.ACLineStatus == 0


def set_own_priority(priority):
    """Lets the app keep up with a High-priority worker, so Stop stays responsive."""
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority)


class Job:
    """A Windows Job Object with a memory cap, a priority ceiling and kill-on-close."""

    def __init__(self, memory_limit, priority=None):
        """priority pins the worker's priority class. Windows only lets a job pin
        Normal or lower without admin rights; for High the worker raises itself."""
        self.memory_limit = memory_limit
        self.name = f"Local\\Odycentric-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.handle = _check(kernel32.CreateJobObjectW(None, self.name))
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = (
            _LIMIT_PROCESS_MEMORY | _LIMIT_KILL_ON_JOB_CLOSE | _LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
        if priority is not None:
            limits.BasicLimitInformation.LimitFlags |= _LIMIT_PRIORITY_CLASS
            limits.BasicLimitInformation.PriorityClass = priority
        limits.ProcessMemoryLimit = memory_limit
        _check(kernel32.SetInformationJobObject(
            self.handle, _JOB_EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)))

    def start(self, args, cwd):
        """Start a hidden process with pipes. It must call join(job.name) itself;
        check contains() on the pid it reports before trusting it."""
        return subprocess.Popen(
            args, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW)

    def contains(self, pid):
        """True if the process with this id is running under this job's limits."""
        handle = _check(kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid))
        try:
            result = wintypes.BOOL()
            _check(kernel32.IsProcessInJob(handle, self.handle, ctypes.byref(result)))
            return bool(result.value)
        finally:
            kernel32.CloseHandle(handle)

    def peak_memory(self):
        """Highest memory any process in the job has committed, in bytes."""
        limits = _ExtendedLimits()
        _check(kernel32.QueryInformationJobObject(
            self.handle, _JOB_EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits), None))
        return limits.PeakProcessMemoryUsed

    def close(self):
        """Release the job. Kill-on-close means this also ends the worker."""
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None


def join(job_name):
    """Put the calling process into the named job. The worker calls this on itself."""
    job = _check(kernel32.OpenJobObjectW(_JOB_OBJECT_ASSIGN_PROCESS, False, job_name))
    try:
        _check(kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()))
    finally:
        kernel32.CloseHandle(job)
