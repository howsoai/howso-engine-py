from __future__ import annotations

LOG_FILE = "howso_stacktrace.txt"


#: Seconds to wait on the isolated date/time support check before giving up.
DATE_FEATURE_TIMEOUT = 60


#: Seconds each phase of the CPU availability probe runs for.
CPU_PROBE_SECONDS = 0.5


#: Most workers the CPU availability probe will run at once.
CPU_PROBE_MAX_WORKERS = 16


#: Hashes per clock check. `hashlib` releases the GIL but the loop around it
#: does not, so checking the clock every iteration would serialize the workers
#: and understate the parallelism actually available.
CPU_PROBE_BATCH = 8


#: Fraction of the normal-priority result the lower-priority process must
#: reach before it is judged to be getting less compute.
LOW_PRIORITY_MIN_RATIO = 0.75


#: Seconds to wait on the lower-priority probe process before giving up. It has
#: to spawn a fresh interpreter and import this module before it can measure.
LOW_PRIORITY_TIMEOUT = 120


#: Fraction of CPU time the hypervisor must take before it is worth mentioning
#: at all. Small amounts are normal on any shared host.
STEAL_REPORT_FRACTION = 0.01


#: Fraction above which stolen time is treated as a real shortfall.
STEAL_WARN_FRACTION = 0.10


#: Win32 CREATE_* flag placing a new process below Normal priority. Spelled out
#: because `subprocess` only defines it when running on Windows.
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


#: Marks the probe's result line, so it can be picked out of whatever else the
#: child's imports may write to stdout.
PROBE_SENTINEL = "__HOWSO_IV_PROBE__"
