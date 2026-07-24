#!/usr/bin/env python3
"""Generate static manifests for all interactive mesh result sections."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_MESH_DIRECTORY = PROJECT_ROOT / "assets" / "meshes" / "single"
CONTROL_MESH_DIRECTORY = PROJECT_ROOT / "assets" / "meshes" / "control"
MULTI_MESH_DIRECTORY = PROJECT_ROOT / "assets" / "meshes" / "multi"
CONTROL_FILE_PATTERN = re.compile(
    r"^(?P<name>.+)_pred_(?P<budget>\d+)\.obj$",
    re.IGNORECASE,
)


def natural_sort_key(value: str) -> list[str | int]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def discover_single_meshes() -> list[str]:
    return sorted(
        (
            path.name
            for path in SINGLE_MESH_DIRECTORY.iterdir()
            if path.is_file() and path.suffix.casefold() == ".obj"
        ),
        key=natural_sort_key,
    )


def discover_control_sets() -> list[dict[str, object]]:
    grouped_meshes: dict[str, list[dict[str, object]]] = defaultdict(list)

    for path in CONTROL_MESH_DIRECTORY.iterdir():
        if not path.is_file():
            continue

        match = CONTROL_FILE_PATTERN.match(path.name)
        if not match:
            continue

        grouped_meshes[match.group("name")].append(
            {
                "budget": int(match.group("budget")),
                "file": path.name,
            }
        )

    return [
        {
            "name": name,
            "meshes": sorted(meshes, key=lambda mesh: int(mesh["budget"])),
        }
        for name, meshes in sorted(
            grouped_meshes.items(),
            key=lambda item: natural_sort_key(item[0]),
        )
    ]


def discover_multi_meshes() -> list[str]:
    return sorted(
        (
            path.name
            for path in MULTI_MESH_DIRECTORY.iterdir()
            if path.is_file() and path.suffix.casefold() == ".glb"
        ),
        key=natural_sort_key,
    )


def write_manifest(directory: Path, payload: dict[str, object]) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.dumps(payload, ensure_ascii=False, indent=2)
    manifest_path.write_text(f"{manifest}\n", encoding="utf-8")


def main() -> None:
    single_meshes = discover_single_meshes()
    control_sets = discover_control_sets()
    multi_meshes = discover_multi_meshes()

    write_manifest(SINGLE_MESH_DIRECTORY, {"meshes": single_meshes})
    write_manifest(CONTROL_MESH_DIRECTORY, {"sets": control_sets})
    write_manifest(MULTI_MESH_DIRECTORY, {"meshes": multi_meshes})

    control_mesh_count = sum(len(control_set["meshes"]) for control_set in control_sets)
    print(
        "Updated mesh manifests: "
        f"{len(single_meshes)} single-generation OBJ files; "
        f"{control_mesh_count} mesh-complexity OBJ files across "
        f"{len(control_sets)} pages; "
        f"{len(multi_meshes)} part-wise GLB files."
    )


if __name__ == "__main__":
    main()
