"""In-process Windows GPU APIs, via ctypes (no subprocess, no extra
packages):

  - enumerate_dxgi_adapters(): DXGI (CreateDXGIFactory1 /
    IDXGIFactory1::EnumAdapters1 / IDXGIAdapter1::GetDesc1). The same
    API Direct3D applications and Task Manager use. Gives the PCI
    vendor ID, the adapter LUID, and three 64-bit memory figures --
    unlike WMI's AdapterRAM, which is a 32-bit field that misreports
    anything above ~4 GB. Measured at ~13 ms on Computer 1, versus
    several seconds for a PowerShell/WMI query (process startup).

  - GPUCounterSampler: PDH performance counters ("GPU Engine" and
    "GPU Adapter Memory" -- the counters Task Manager's GPU tab reads).
    The query is opened once and kept; each sample() is a single
    PdhCollectQueryData call (~5 ms on Computer 1), so periodic GPU
    utilization monitoring never spawns a process.

Everything here returns empty/None on any failure (non-Windows, old
Windows without these counters, a driver that doesn't publish them).
A failure means "could not measure", never "no GPU" -- callers must
keep those two apart.
"""

from __future__ import annotations

import ctypes
import dataclasses
import re
import sys
import threading
from typing import Dict, List, Optional

# DXGI_ADAPTER_FLAG_SOFTWARE: the "Microsoft Basic Render Driver" (a
# CPU rasterizer present on every Windows machine). Not hardware.
_DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
_DXGI_ERROR_NOT_FOUND = 0x887A0002

_PDH_FMT_DOUBLE = 0x00000200
_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2

_ENGINE_INSTANCE_RE = re.compile(r"luid_(0x[0-9a-fA-F]+)_(0x[0-9a-fA-F]+).*?engtype_(.+)$")
_ADAPTER_INSTANCE_RE = re.compile(r"luid_(0x[0-9a-fA-F]+)_(0x[0-9a-fA-F]+)")


@dataclasses.dataclass
class DXGIAdapter:
    description: str
    vendor_id: int
    device_id: int
    dedicated_video_memory: int
    dedicated_system_memory: int
    shared_system_memory: int
    luid: str  # "luid:0x00000000_0x000160E5" -- see format_luid()
    is_software: bool


def format_luid(high: int, low: int) -> str:
    """Canonical GPU id used across this project. Matches the
    "luid_0x..._0x..." fragment of PDH instance names (case-normalized)."""
    return f"luid:0x{high & 0xFFFFFFFF:08X}_0x{low & 0xFFFFFFFF:08X}"


def _luid_from_counter_parts(high_hex: str, low_hex: str) -> str:
    return format_luid(int(high_hex, 16), int(low_hex, 16))


# ---------------------------------------------------------------------
# DXGI
# ---------------------------------------------------------------------


def enumerate_dxgi_adapters() -> List[DXGIAdapter]:
    if sys.platform != "win32":
        return []
    try:
        return _enumerate_dxgi_adapters()
    except Exception:
        return []


def _enumerate_dxgi_adapters() -> List[DXGIAdapter]:
    from ctypes import wintypes

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class DXGI_ADAPTER_DESC1(ctypes.Structure):
        _fields_ = [
            ("Description", wintypes.WCHAR * 128),
            ("VendorId", wintypes.UINT),
            ("DeviceId", wintypes.UINT),
            ("SubSysId", wintypes.UINT),
            ("Revision", wintypes.UINT),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", LUID),
            ("Flags", wintypes.UINT),
        ]

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    # IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
    iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))

    def vcall(obj, index, restype, *argtypes):
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])

    # COM vtable slots: IUnknown 0-2, IDXGIObject 3-6, IDXGIFactory 7-11,
    # IDXGIFactory1::EnumAdapters1 = 12. IDXGIAdapter1::GetDesc1 = 10.
    release = 2
    enum_adapters1 = 12
    get_desc1 = 10

    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory:
        return []

    adapters: List[DXGIAdapter] = []
    try:
        index = 0
        while index < 64:  # hard stop; no real machine has this many
            adapter = ctypes.c_void_p()
            hr = vcall(factory, enum_adapters1, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(
                factory, index, ctypes.byref(adapter)
            )
            if (hr & 0xFFFFFFFF) == _DXGI_ERROR_NOT_FOUND or hr != 0:
                break
            try:
                desc = DXGI_ADAPTER_DESC1()
                if vcall(adapter, get_desc1, ctypes.c_long, ctypes.POINTER(DXGI_ADAPTER_DESC1))(
                    adapter, ctypes.byref(desc)
                ) == 0:
                    adapters.append(
                        DXGIAdapter(
                            description=desc.Description.strip(),
                            vendor_id=int(desc.VendorId),
                            device_id=int(desc.DeviceId),
                            dedicated_video_memory=int(desc.DedicatedVideoMemory),
                            dedicated_system_memory=int(desc.DedicatedSystemMemory),
                            shared_system_memory=int(desc.SharedSystemMemory),
                            luid=format_luid(desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart),
                            is_software=bool(desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE),
                        )
                    )
            finally:
                vcall(adapter, release, ctypes.c_ulong)(adapter)
            index += 1
    finally:
        vcall(factory, release, ctypes.c_ulong)(factory)
    return adapters


# ---------------------------------------------------------------------
# PDH GPU counters
# ---------------------------------------------------------------------


@dataclasses.dataclass
class GPUCounterSample:
    utilization_percent: Optional[float] = None
    dedicated_memory_used_bytes: Optional[int] = None
    shared_memory_used_bytes: Optional[int] = None


def aggregate_engine_utilization(samples: Dict[str, float]) -> Dict[str, float]:
    """Turns raw "GPU Engine" counter instances into one utilization
    figure per adapter.

    Instance names look like
    "pid_12552_luid_0x00000000_0x000160E5_phys_0_eng_0_engtype_3D".
    Utilization is summed per (adapter, engine type) across processes,
    then the busiest engine type is taken and capped at 100 -- the same
    rule Task Manager uses for its headline GPU % (engines run in
    parallel, so summing different engine types would overstate load).
    Pure function, unit tested without any GPU."""
    per_engine: Dict[tuple, float] = {}
    for name, value in samples.items():
        match = _ENGINE_INSTANCE_RE.search(name or "")
        if not match or value is None or value != value or value < 0:  # value != value: NaN
            continue
        key = (_luid_from_counter_parts(match.group(1), match.group(2)), match.group(3))
        per_engine[key] = per_engine.get(key, 0.0) + float(value)

    per_adapter: Dict[str, float] = {}
    for (luid, _engtype), total in per_engine.items():
        per_adapter[luid] = max(per_adapter.get(luid, 0.0), total)
    return {luid: min(total, 100.0) for luid, total in per_adapter.items()}


def aggregate_adapter_memory(samples: Dict[str, float]) -> Dict[str, int]:
    """"GPU Adapter Memory" instances ("luid_0x..._0x..._phys_0") ->
    bytes per adapter. Pure function."""
    out: Dict[str, int] = {}
    for name, value in samples.items():
        match = _ADAPTER_INSTANCE_RE.search(name or "")
        if not match or value is None or value != value or value < 0:
            continue
        luid = _luid_from_counter_parts(match.group(1), match.group(2))
        out[luid] = out.get(luid, 0) + int(value)
    return out


class GPUCounterSampler:
    """Keeps one PDH query open for the life of the worker. The engine
    utilization counter is a rate: the first sample() after opening has
    no previous data point, so it reports no utilization (absent, not
    0) -- subsequent calls report the average since the previous call.

    Thread-safe; cheap to call; never raises."""

    _ENGINE = r"\GPU Engine(*)\Utilization Percentage"
    _DEDICATED = r"\GPU Adapter Memory(*)\Dedicated Usage"
    _SHARED = r"\GPU Adapter Memory(*)\Shared Usage"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._query = None
        self._counters: Dict[str, object] = {}
        self._primed = False
        self._failed = sys.platform != "win32"

    @property
    def supported(self) -> bool:
        return not self._failed

    def _open(self) -> bool:
        if self._query is not None:
            return True
        if self._failed:
            return False
        try:
            from ctypes import wintypes

            self._pdh = ctypes.WinDLL("pdh")
            query = ctypes.c_void_p()
            if self._pdh.PdhOpenQueryW(None, None, ctypes.byref(query)) != 0:
                self._failed = True
                return False
            for path in (self._ENGINE, self._DEDICATED, self._SHARED):
                counter = ctypes.c_void_p()
                if self._pdh.PdhAddEnglishCounterW(query, ctypes.c_wchar_p(path), None, ctypes.byref(counter)) == 0:
                    self._counters[path] = counter
            if not self._counters:
                self._pdh.PdhCloseQuery(query)
                self._failed = True
                return False
            self._query = query
            self._wintypes = wintypes
            return True
        except Exception:
            self._failed = True
            return False

    def _read_array(self, counter, fmt: int) -> Dict[str, float]:
        wintypes = self._wintypes

        class _Value(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("value", ctypes.c_double if fmt == _PDH_FMT_DOUBLE else ctypes.c_longlong)]

        class _Item(ctypes.Structure):
            _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _Value)]

        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if (status & 0xFFFFFFFF) != _PDH_MORE_DATA or size.value == 0:
            return {}
        buf = (ctypes.c_byte * size.value)()
        if self._pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size), ctypes.byref(count), buf) != 0:
            return {}
        items = ctypes.cast(buf, ctypes.POINTER(_Item))
        out: Dict[str, float] = {}
        for i in range(count.value):
            item = items[i]
            if item.FmtValue.CStatus in (0, 1):  # PDH_CSTATUS_VALID_DATA / NEW_DATA
                out[item.szName or ""] = float(item.FmtValue.value)
        return out

    def sample(self) -> Dict[str, GPUCounterSample]:
        """Returns {gpu_id: GPUCounterSample}. Empty dict when counters
        are unavailable on this machine."""
        with self._lock:
            try:
                if not self._open():
                    return {}
                if self._pdh.PdhCollectQueryData(self._query) != 0:
                    return {}
                result: Dict[str, GPUCounterSample] = {}

                def _entry(luid: str) -> GPUCounterSample:
                    return result.setdefault(luid, GPUCounterSample())

                if self._DEDICATED in self._counters:
                    for luid, used in aggregate_adapter_memory(
                        self._read_array(self._counters[self._DEDICATED], _PDH_FMT_LARGE)
                    ).items():
                        _entry(luid).dedicated_memory_used_bytes = used
                if self._SHARED in self._counters:
                    for luid, used in aggregate_adapter_memory(
                        self._read_array(self._counters[self._SHARED], _PDH_FMT_LARGE)
                    ).items():
                        _entry(luid).shared_memory_used_bytes = used
                if self._ENGINE in self._counters and self._primed:
                    for luid, pct in aggregate_engine_utilization(
                        self._read_array(self._counters[self._ENGINE], _PDH_FMT_DOUBLE)
                    ).items():
                        _entry(luid).utilization_percent = round(pct, 2)
                self._primed = True
                return result
            except Exception:
                return {}

    def close(self) -> None:
        with self._lock:
            if self._query is not None:
                try:
                    self._pdh.PdhCloseQuery(self._query)
                except Exception:
                    pass
                self._query = None
