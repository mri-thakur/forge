"""Run a long job so it outlives the terminal, editor, or coding agent that started it.

    python scripts/detach.py --name wikitext -- python -m forge train --backend torch ...
    python scripts/detach.py --status
    python scripts/detach.py --stop wikitext

Output is appended to logs/<name>.log; logs/<name>.json records the command, process,
and exit code. On Windows the job is created through WMI, so it runs outside the job
object of whatever started it and keeps going when that session ends. Sleep, shutdown,
or a crash still stop it; training then continues from its last checkpoint by rerunning
the recorded command with --resume.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
STILL_ACTIVE = 259


def now():
    return datetime.now().isoformat(timespec="seconds")


def load(name):
    path = LOGS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save(name, record):
    temporary = LOGS / f"{name}.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LOGS / f"{name}.json")


def alive(pid):
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def state(record):
    status = record.get("status", "unknown")
    if status == "running" and not alive(record["pid"]):
        return "stopped"  # killed, slept, or shut down before it could record an exit
    return status


def create_outside_job(command_line):
    """Start a process via WMI so it is not a member of the caller's job object.

    Agents and editors often run commands inside a job that kills every member when
    the session closes; an ordinary child process would die with it.
    """
    script = (
        "$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly"
        " -Property @{ShowWindow = [uint16]0};"
        " $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments"
        " @{CommandLine = $env:DETACH_COMMAND; CurrentDirectory = $env:DETACH_CWD;"
        " ProcessStartupInformation = $startup};"
        " exit $result.ReturnValue"
    )
    env = {**os.environ, "DETACH_COMMAND": command_line, "DETACH_CWD": str(ROOT)}
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script], env=env, check=True
    )


def launch(name, command):
    LOGS.mkdir(exist_ok=True)
    previous = load(name)
    if previous and state(previous) in ("running", "launching"):
        sys.exit(f"{name} is already {state(previous)}; see logs/{name}.log")
    # The detached process may get a different PATH, so a bare "python" would not
    # necessarily be this environment's interpreter.
    if command[0].lower() in ("python", "python.exe", "python3"):
        command = [sys.executable, *command[1:]]
    save(name, {"command": command, "cwd": str(ROOT), "launched": now(), "status": "launching"})
    wrapper = [sys.executable, str(Path(__file__).resolve()), "--run", name, "--", *command]
    if os.name == "nt":
        create_outside_job(subprocess.list2cmdline(wrapper))
    else:
        subprocess.Popen(
            wrapper,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    for _ in range(60):
        try:
            record = load(name)
        except (OSError, ValueError):
            record = {}
        if record.get("status") == "running":
            print(f"{name} started (pid {record['pid']}); output: logs/{name}.log")
            return
        time.sleep(0.5)
    save(name, {**load(name), "status": "failed to start"})
    sys.exit(f"{name} did not start within 30 s")


def exempt_from_power_throttling(pid):
    """Windows treats a windowless background process as low priority work (EcoQoS:
    efficiency cores, lower clocks, coarse timers); a detached benchmark measured 8x
    slower than the same run in a terminal. Opt the process out."""
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [(field, wintypes.ULONG) for field in ("Version", "ControlMask", "StateMask")]

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x0200, False, pid)  # PROCESS_SET_INFORMATION
    if not handle:
        return False
    # Take control of EXECUTION_SPEED and IGNORE_TIMER_RESOLUTION, both switched off.
    state = State(1, 0x1 | 0x4, 0)
    ok = kernel32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))
    kernel32.CloseHandle(handle)
    return bool(ok)


def descendants(root):
    """PIDs of every process below `root`, from a Windows process snapshot."""
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    children = {}
    entry = Entry(dwSize=ctypes.sizeof(Entry))
    more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
    while more:
        children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
        more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    kernel32.CloseHandle(snapshot)
    found, pending = set(), [root]
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in found and child != root:
                found.add(child)
                pending.append(child)
    return found


def keep_exempt(finished):
    """The exemption is not inherited, so apply it to each new descendant."""
    exempted = set()
    while not finished.is_set():
        for pid in descendants(os.getpid()) - exempted:
            if exempt_from_power_throttling(pid):
                exempted.add(pid)
        finished.wait(1)


def run(name, command):
    """The detached wrapper: run the command, then record how it ended."""
    finished = threading.Event()
    if os.name == "nt":
        exempt_from_power_throttling(os.getpid())
        threading.Thread(target=keep_exempt, args=(finished,), daemon=True).start()
    record = load(name)
    record.update(pid=os.getpid(), started=now(), status="running")
    save(name, record)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with (LOGS / f"{name}.log").open("a", encoding="utf-8") as log:
        log.write(f"# {now()} start: {subprocess.list2cmdline(command)}\n")
        log.flush()
        code = subprocess.call(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"# {now()} exit code {code}\n")
    finished.set()
    record.update(finished=now(), exit_code=code, status="finished" if code == 0 else "failed")
    save(name, record)


def stop(name):
    record = load(name)
    if state(record) != "running":
        sys.exit(f"{name} is not running")
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(record["pid"]), "/T", "/F"], check=True, capture_output=True
        )
    else:
        os.killpg(record["pid"], signal.SIGTERM)  # the wrapper leads its own session
    record.update(finished=now(), status="cancelled")
    save(name, record)
    with (LOGS / f"{name}.log").open("a", encoding="utf-8") as log:
        log.write(f"# {now()} cancelled with --stop\n")
    print(f"{name} cancelled")


def show_status():
    paths = sorted(LOGS.glob("*.json")) if LOGS.exists() else []
    if not paths:
        print("no jobs")
    for path in paths:
        record = load(path.stem)
        current = state(record)
        print(f"{path.stem}: {current} (launched {record.get('launched')})")
        log = LOGS / f"{path.stem}.log"
        if log.exists():
            lines = log.read_bytes()[-4096:].decode("utf-8", "replace").splitlines()
            if lines:
                print(f"  last output: {lines[-1][:160]}")
        if current == "stopped":
            print("  ended without recording an exit; to continue training, add --resume:")
            print(f"  {subprocess.list2cmdline(record['command'])} --resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", help="job name; output goes to logs/<name>.log")
    parser.add_argument("--status", action="store_true", help="list jobs and their state")
    parser.add_argument("--stop", metavar="NAME", help="terminate a running job")
    parser.add_argument("--run", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.status:
        show_status()
    elif args.stop:
        stop(args.stop)
    elif args.run:
        run(args.run, command)
    elif args.name and command:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
            parser.error("--name may contain only letters, digits, '.', '_' and '-'")
        launch(args.name, command)
    else:
        parser.error("give --name NAME -- COMMAND ..., --status, or --stop NAME")


if __name__ == "__main__":
    main()
