#!/usr/bin/env python3
"""Count exact live trainer worker command lines without matching this probe."""
from pathlib import Path


def main() -> None:
    count = 0
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if (len(args) > 2 and Path(args[0].decode(errors="ignore")).name == "python"
                and args[1:3] == [b"-u", b"scripts/train_jevany_streaming.py"]):
            count += 1
    print(count)


if __name__ == "__main__":
    main()
