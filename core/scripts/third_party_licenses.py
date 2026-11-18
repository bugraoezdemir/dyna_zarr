"""Write THIRD_PARTY_LICENSES.txt: the licences of everything dyna-zarr-core's wheels contain.

    python scripts/third_party_licenses.py        (from core/)

The wheels link every crate of the normal dependency graph and the C / C++
libraries that zarrs' ``*-sys`` crates compile from source (``C_LIBRARIES``):
c-blosc, LZ4, Snappy, zlib, zstd and the mimalloc allocator.  MIT, BSD and Apache licences ask for their
notices in binary distributions, so each component's licence files are listed and
their full texts written out, each distinct text once.  blosc-src ships no licence
file, so c-blosc's text is kept in ``licenses/c-blosc`` (from github.com/Blosc/c-blosc).
Adapted from micro-reader-codecs' script.  Run it again whenever Cargo.lock
changes; CI checks the file is current.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # core/
LICENSE_NAME = re.compile(r"^(licen[cs]e|copying|notice|copyright|authors)", re.IGNORECASE)
#: crate -> (bundled library, the folder in the crate holding its licence files, or a
#: folder of this repository when the crate ships none)
C_LIBRARIES = {
    "blosc-src": ("c-blosc", None),
    "lz4-sys": ("LZ4", "liblz4/lib"),
    "snappy_src": ("Snappy (C++)", "snappy"),
    "libz-sys": ("zlib", "src/zlib"),
    "zstd-sys": ("libzstd", "zstd"),
    # the module's allocator; libmimalloc-sys builds mimalloc v2 unless its v3 feature is on
    "libmimalloc-sys": ("mimalloc", "c_src/mimalloc/v2"),
}
#: crates that ship no licence file -> licence files of this repository standing for theirs
OWN_FILES = {"blosc-src": ["licenses/c-blosc/LICENSE.txt"],
             # rc-zip (the zip reader under zarrs_zip) ships none; its repository's texts
             # (github.com/bearcove/rc-zip, Apache-2.0 OR MIT)
             "rc-zip": ["licenses/rc-zip/LICENSE-APACHE", "licenses/rc-zip/LICENSE-MIT"]}
#: crates that ship no licence file -> the crate of the same authors and licence (same
#: repository) whose files stand for theirs
SAME_AS = {"zarrs_plugin": "zarrs"}
#: the workspace's own crates (MIT, this repository's LICENSE)
OWN_CRATES = {"dyna-zarr-transforms", "dyna-zarr-arrays"}
#: the targets the wheels are built for (.github/workflows/build-core.yml)
TARGETS = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "x86_64-apple-darwin",
           "aarch64-apple-darwin", "x86_64-pc-windows-msvc"]
RULE = "\n" + "=" * 78 + "\n\n"


def cargo(*args) -> str:
    return subprocess.run(["cargo", *args], cwd=HERE, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def license_files(directory: str, depth: int = 1) -> list:
    found = []
    for root, dirs, files in os.walk(directory):
        if root[len(directory):].count(os.sep) >= depth:
            dirs[:] = []
        for name in sorted(files):
            if LICENSE_NAME.match(name) and not name.endswith((".rs", ".py", ".toml")):
                found.append(os.path.join(root, name))
    return sorted(found)


class Texts:
    """Each distinct licence text once, numbered in order of first use."""

    def __init__(self):
        self.numbers: dict = {}

    def cite(self, files, directory) -> str:
        lines = []
        for f in files:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = fh.read().strip()
            number = self.numbers.setdefault(text, len(self.numbers) + 1)
            relative = os.path.relpath(f, directory).replace(os.sep, "/")
            lines.append(f"  {relative}: licence text [{number}]")
        return "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    out_path = os.path.join(HERE, "THIRD_PARTY_LICENSES.txt")
    meta = json.loads(cargo("metadata", "--format-version", "1"))
    root = "dyna-zarr-core"
    linked = set()                    # (name, version): a crate may be linked twice
    # the crates of every wheel's target (windows-sys, libc, ...), so the file covers
    # every wheel and is the same wherever it is generated
    targets = [arg for t in TARGETS for arg in ("--target", t)]
    for line in cargo("tree", "-e", "normal", *targets, "--prefix", "none",
                      "--format", "{p}", "-p", root).splitlines():
        parts = line.replace(" (*)", "").split()
        if len(parts) >= 2 and parts[0] != root and parts[0] not in OWN_CRATES:
            linked.add((parts[0], parts[1].lstrip("v")))
    packages = {(p["name"], p["version"]): p for p in meta["packages"]}
    versions = dict(sorted(linked))
    texts, sections, missing = Texts(), [], []
    for name, version in sorted(linked):
        pkg = packages[(name, version)]
        directory = os.path.dirname(pkg["manifest_path"])
        files = license_files(directory)
        head = (f"{name} {version}\n  licence: {pkg.get('license') or '(see files)'}\n"
                f"  source: {pkg.get('repository') or 'crates.io'}\n")
        if not files and name in SAME_AS:
            sibling = packages[(SAME_AS[name], versions[SAME_AS[name]])]
            directory = os.path.dirname(sibling["manifest_path"])
            files = license_files(directory)
            head += f"  (ships no licence file; same authors and licence as {SAME_AS[name]})\n"
        if not files and name in OWN_FILES:
            files, directory = [os.path.join(HERE, f) for f in OWN_FILES[name]], HERE
            head += "  (ships no licence file; the texts from its source, kept in this repository)\n"
        if not files:
            missing.append(name)
        sections.append(head + texts.cite(files, directory))
        if name in C_LIBRARIES:
            library, sub = C_LIBRARIES[name]
            if sub is None:
                c_files, c_dir = [os.path.join(HERE, f) for f in OWN_FILES[name]], HERE
            else:
                c_dir = directory
                c_files = license_files(os.path.join(directory, sub), depth=1)
            if not c_files:
                print(f"no licence file found for the bundled {library}", file=sys.stderr)
                return 1
            sections.append(f"{library} (compiled into the wheel by {name})\n"
                            + texts.cite(c_files, c_dir))
    body = "".join(f"{RULE}[{number}]\n\n{text}\n" for text, number in texts.numbers.items())
    out = (f"Third-party software in {root} binary distributions\n"
           f"({len(linked)} Rust crates and the C / C++ libraries c-blosc, LZ4, Snappy,\n"
           "zlib, zstd and mimalloc).\n"
           "Generated by scripts/third_party_licenses.py from Cargo.lock.  Each component\n"
           "lists its licence files; their full texts follow, each distinct text once.\n"
           + RULE + "\n".join(sections) + RULE + "LICENCE TEXTS\n" + body)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out)
    print(f"{out_path}: {len(linked)} crates, {len(texts.numbers)} distinct licence texts, "
          f"{len(out) // 1024} KB" + (f"; no licence file in: {', '.join(missing)}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
