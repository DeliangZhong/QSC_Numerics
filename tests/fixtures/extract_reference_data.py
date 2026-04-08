"""Extract reference spectral data from Mathematica .mx files to JSON.

Usage:
    python3 tests/fixtures/extract_reference_data.py

Requires wolframscript to be installed and on PATH.
Extracts (g, Delta) pairs for 10 representative states spanning all Delta_0 values.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "reference"
    / "qsc"
    / "local operators N4 SYM"
    / "data"
)
NUMERICAL_DIR = DATA_DIR / "numerical"
OUTPUT_FILE = Path(__file__).resolve().parent / "reference_spectral_data.json"

# 10 representative states spanning all Delta_0 values
FILENAMES = [
    "numerical_spectral_data_Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1.mx",
    "numerical_spectral_data_Delta03_b10_b20_f12_f22_f31_f41_a10_a20_sol1.mx",
    "numerical_spectral_data_Delta04_b10_b20_f11_f21_f31_f41_a12_a20_sol1.mx",
    "numerical_spectral_data_Delta04_b10_b20_f12_f22_f32_f42_a10_a20_sol1.mx",
    "numerical_spectral_data_Delta04_b10_b20_f13_f23_f31_f41_a10_a20_sol1.mx",
    "numerical_spectral_data_Delta05_b10_b20_f12_f22_f31_f41_a12_a20_sol1.mx",
    "numerical_spectral_data_Delta05_b10_b20_f13_f21_f31_f41_a12_a20_sol1.mx",
    "numerical_spectral_data_Delta011by2_b10_b20_f13_f22_f32_f42_a11_a20_sol1.mx",
    "numerical_spectral_data_Delta011by2_b10_b20_f13_f23_f32_f41_a11_a20_sol1.mx",
    "numerical_spectral_data_Delta06_b10_b20_f12_f22_f32_f42_a11_a21_sol1.mx",
]


def parse_state_id_mma(filename: str) -> str:
    """Parse filename into a Mathematica-compatible state ID expression.

    Returns e.g. '{2,0,0,1,1,1,1,0,0,1}' or '{11/2,0,0,3,2,2,2,1,0,1}'.
    """
    m = re.match(
        r"numerical_spectral_data_Delta0(\d+(?:by\d+)?)_"
        r"b1(\d+)_b2(\d+)_f1(\d+)_f2(\d+)_f3(\d+)_f4(\d+)_a1(\d+)_a2(\d+)_sol(\d+)\.mx",
        filename,
    )
    if not m:
        raise ValueError(f"Cannot parse filename: {filename}")

    delta_str = m.group(1)
    # Handle half-integer: "11by2" -> "11/2"
    if "by" in delta_str:
        num, den = delta_str.split("by")
        delta_mma = f"{num}/{den}"
    else:
        delta_mma = delta_str

    parts = [delta_mma] + [m.group(i) for i in range(2, 11)]
    return "{" + ",".join(parts) + "}"


def parse_state_id_python(filename: str) -> dict:
    """Parse filename into a Python dict with quantum numbers."""
    m = re.match(
        r"numerical_spectral_data_Delta0(\d+(?:by\d+)?)_"
        r"b1(\d+)_b2(\d+)_f1(\d+)_f2(\d+)_f3(\d+)_f4(\d+)_a1(\d+)_a2(\d+)_sol(\d+)\.mx",
        filename,
    )
    if not m:
        raise ValueError(f"Cannot parse filename: {filename}")

    delta_str = m.group(1)
    if "by" in delta_str:
        num, den = delta_str.split("by")
        delta0 = int(num) / int(den)
    else:
        delta0 = int(delta_str)

    return {
        "Delta0": delta0,
        "nb": [int(m.group(2)), int(m.group(3))],
        "nf": [int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))],
        "na": [int(m.group(8)), int(m.group(9))],
        "sol": int(m.group(10)),
    }


def extract_state(filename: str) -> list[list[float]]:
    """Extract (g, Delta) pairs from a single .mx file via wolframscript."""
    mx_file = NUMERICAL_DIR / filename
    if not mx_file.exists():
        print(f"  WARNING: {filename} not found, skipping")
        return []

    state_id_mma = parse_state_id_mma(filename)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    mma_code = (
        f'Get["{mx_file}"];'
        f"data = SpectralData[{state_id_mma}];"
        f"sorted = SortBy[data, First];"
        f'Export["{tmp_path}", N[sorted, 20], "JSON"];'
        f'Print["OK ", Length[sorted]];'
    )

    result = subprocess.run(
        ["wolframscript", "-code", mma_code],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"  ERROR running wolframscript: {result.stderr[:200]}")
        return []

    try:
        with open(tmp_path) as f:
            pairs = json.load(f)
        Path(tmp_path).unlink(missing_ok=True)
        return pairs
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ERROR reading JSON: {e}")
        return []


def main() -> None:
    print("Extracting reference spectral data from .mx files...")
    print(f"Data directory: {NUMERICAL_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    all_data: dict[str, dict] = {}

    for filename in FILENAMES:
        state_info = parse_state_id_python(filename)
        key = filename.removeprefix("numerical_spectral_data_").removesuffix(".mx")

        print(f"Extracting: {key} (Delta0={state_info['Delta0']})")

        pairs = extract_state(filename)

        if pairs:
            print(f"  Got {len(pairs)} data points, g in [{pairs[0][0]:.4f}, {pairs[-1][0]:.4f}]")
        else:
            print("  No data extracted")

        all_data[key] = {
            "quantum_numbers": state_info,
            "data": pairs,
        }
        print()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

    total_points = sum(len(v["data"]) for v in all_data.values())
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Total: {len(all_data)} states, {total_points} data points")


if __name__ == "__main__":
    main()
