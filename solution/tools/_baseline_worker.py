"""Run the official baseline and measure memory inside the real Python worker."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--memory-report', required=True, type=Path)
    args = parser.parse_args()
    upstream = Path(__file__).resolve().parents[2] / 'more-than-a-hack-2027-main repo'
    sys.argv = [str(upstream / 'starter/baseline.py'), '--out', args.out]
    runpy.run_path(sys.argv[0], run_name='__main__')
    peak = None
    if sys.platform == 'win32':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool',
                    'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile')]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if not query(current(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        peak = round(counters.peak_working_set / (1024 * 1024), 2)
    args.memory_report.write_text(json.dumps({'peak_working_set_mib': peak,
        'measurement': 'Windows PeakWorkingSetSize inside the baseline Python process'
                       if peak is not None else 'unavailable on this platform'}) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
