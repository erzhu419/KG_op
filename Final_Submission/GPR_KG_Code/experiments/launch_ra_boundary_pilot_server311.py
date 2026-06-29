"""Launch the RA-boundary RZDT pilot as a detached Windows process."""

import os
import subprocess
import time


CODE_DIR = r"C:\Users\erzhu419\GPR_KG_Code"
SCRIPT = os.path.join(
    CODE_DIR, "experiments", "run_ra_boundary_pilot_server311.cmd")
LOG_DIR = os.path.join(CODE_DIR, "results", "logs")
PID_PATH = os.path.join(
    LOG_DIR, "ra_boundary_pilot_gprkg_20260526.launcher.pid")
MARKER_PATH = os.path.join(
    LOG_DIR, "ra_boundary_pilot_gprkg_20260526.launcher.log")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000
FLAGS = (
    DETACHED_PROCESS
    | CREATE_NEW_PROCESS_GROUP
    | CREATE_BREAKAWAY_FROM_JOB
    | CREATE_NO_WINDOW
)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(MARKER_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n--- RA boundary pilot launch at {time.ctime()} ---\n")
        f.write(f"script={SCRIPT}\n")

    proc = subprocess.Popen(
        ["cmd.exe", "/c", SCRIPT],
        cwd=CODE_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=FLAGS,
        close_fds=True,
    )
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))
    print(f"launched launcher_pid={proc.pid}")
    print(f"pid_file={PID_PATH}")
    print(f"launcher_log={MARKER_PATH}")
    print(
        "run_logs="
        + os.path.join(LOG_DIR, "ra_boundary_pilot_gprkg_20260526.out.log")
        + " ; "
        + os.path.join(LOG_DIR, "ra_boundary_pilot_gprkg_20260526.err.log")
    )

    time.sleep(8)
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -like '*run_rzdt_checkpointed.py*' "
                "-or $_.CommandLine -like '*run_ra_boundary_pilot_server311.cmd*' } | "
                "Select-Object ProcessId,Name,CommandLine"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


if __name__ == "__main__":
    main()
