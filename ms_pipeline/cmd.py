from typing import List, Optional, Dict
from pathlib import Path

import datetime
import re
import subprocess
import sys


def run_cmd(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    log_path: Optional[Path] = None,
) -> None:
    def quote(s):
        if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", s):
            return s
        return "'" + s.replace("'", "'\"'\"'") + "'"

    cmd_str = " ".join([quote(x) for x in cmd])

    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    msg = f"[{now}] $ {cmd_str}"
    print(msg)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    if dry_run:
        return

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []

    with proc.stdout:
        for line in iter(proc.stdout.readline, ''):
            print(line, end="")        # stream to terminal
            sys.stdout.flush()        # force immediate display
            output_lines.append(line)

            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)

    returncode = proc.wait()

    if returncode != 0:
        output = "".join(output_lines)
        raise RuntimeError(
            f"Command failed with exit code {returncode}\n"
            f"Command: {cmd_str}\n"
            f"Output:\n{output}"
        )