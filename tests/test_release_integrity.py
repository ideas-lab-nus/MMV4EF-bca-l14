"""Safety checks for the public, data-free research-code release."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
MAX_UNEXPECTED_FILE_BYTES = 10 * 1024 * 1024

SKIP_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}

# These directories must remain absent from the public tree even when the
# target has not been initialized as a Git repository yet.
FORBIDDEN_RELEASE_DIRECTORIES = {
    ".agents",
    ".codex",
    ".ipython",
    ".notebook_runs",
    ".uv-cache",
    ".worktrees",
    "analysis_outputs",
    "getPIdata",
    "literature",
    "paper_figures",
    "pv_model_outputs",
    "tmp",
    "weather_data",
    "weather_model_outputs",
}

REQUIRED_PRODUCERS = (
    Path("scripts/reproduce_figures.py"),
    Path("scripts/make_aug23_mmv_motivation_figure.py"),
    Path("scripts/build_emulator_diagram.py"),
    Path("scripts/compare_daily_rollout_methods.py"),
    Path("fit_all_pmv_regression.py"),
    Path("scripts/build_observed_future_results.py"),
    Path("scripts/build_discussion_results.py"),
    Path("scripts/build_weight_sensitivity_results.py"),
    Path("scripts/build_pv_appendix_figure.py"),
    Path("notebooks/main_study_observed_lstm64.ipynb"),
    Path("notebooks/temperature_slack_sensitivity.ipynb"),
    Path("notebooks/train_ac_surrogate.ipynb"),
    Path("notebooks/train_nv_surrogate.ipynb"),
)

REQUIRED_STATIC_ASSETS = (Path("assets/ceiling_fan.png"),)

PV_MODEL = Path("models/pv/pv_appendix_b_best_model.csv")
THERMAL_MODELS = (
    Path("models/thermal/ac_model.pth"),
    Path("models/thermal/nv_model.pth"),
)
WEATHER_SCALER = Path("models/lstm64/scalers.json")
WEATHER_TARGETS = (
    "temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "solar",
)
WEATHER_SEEDS = (17, 29, 43)
WEATHER_MODELS = tuple(
    Path(
        "models/lstm64/checkpoints/univariate"
    )
    / target
    / f"seed_{seed}.pt"
    for target in WEATHER_TARGETS
    for seed in WEATHER_SEEDS
)

REQUIRED_MODELS = (PV_MODEL, *THERMAL_MODELS, WEATHER_SCALER, *WEATHER_MODELS)

TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cfg",
    ".cmd",
    ".gitignore",
    ".gitattributes",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

ACQUISITION_CODE_SUFFIXES = {
    ".bat",
    ".cmd",
    ".ipynb",
    ".ps1",
    ".py",
    ".sh",
}

# Constructed in pieces so this guardrail does not trigger on its own source.
PI_ACQUISITION_MARKERS = (
    "osi" + "soft",
    "aveva" + "pi",
    "pi" + "webapi",
    "pi" + "connect",
    "af" + "sdk",
    "pi" + "server",
    "pi" + "point",
    "get" + "pidata",
)

FORBIDDEN_DATA_SUFFIXES = (
    ".csv.gz",
    ".tsv.gz",
    ".parquet",
    ".feather",
    ".hdf5",
    ".sqlite3",
    ".jsonl",
    ".grib2",
    ".csv",
    ".tsv",
    ".h5",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".joblib",
    ".xlsx",
    ".xls",
    ".sqlite",
    ".db",
    ".nc",
    ".grib",
)

FORBIDDEN_CREDENTIAL_NAMES = {
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "gurobi.lic",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}
FORBIDDEN_CREDENTIAL_SUFFIXES = {".key", ".kdbx", ".p12", ".pem", ".pfx"}

PRIVATE_KEY_MARKER = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
)
KNOWN_TOKEN_MARKER = re.compile(
    r"(?:AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,})"
)
GUROBI_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:WLSACCESSID|WLSSECRET|LICENSEID)\s*=\s*\S+"
)
GENERIC_LITERAL_CREDENTIAL = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|"
    r"client[_-]?secret|password|passwd|auth[_-]?token)\b"
    r"\s*[:=]\s*([\"'])([^\"'\r\n]{8,})\1"
)
URL_WITH_USERINFO = re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.IGNORECASE)

WINDOWS_USER_PATH = re.compile(
    r"(?i)\b[A-Z]:(?:\\+|/+)(?:Users|Documents and Settings)(?:\\+|/+)"
    r"[^\\/\s\"']+"
)
UNIX_USER_PATH = re.compile(r"/(?:Users|home)/[^/\s\"']+")
ENCODED_WINDOWS_USER_PATH = re.compile(
    r"(?i)\b[A-Z]%3A(?:%5C|/)+(?:Users|Documents%20and%20Settings)(?:%5C|/)+"
)


def _git_release_candidates() -> list[Path] | None:
    """Return tracked and committable files, excluding ignored local inputs."""

    if not (ROOT / ".git").exists():
        return None

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=False,
            capture_output=True,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    return [
        ROOT / Path(raw.decode("utf-8", errors="surrogateescape"))
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def release_files() -> list[Path]:
    git_files = _git_release_candidates()
    if git_files is not None:
        return sorted(path for path in git_files if path.is_file())

    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in SKIP_DIRECTORIES for part in path.relative_to(ROOT).parts)
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_placeholder(value: str) -> bool:
    normalized = value.strip().upper()
    return (
        not normalized
        or normalized.startswith(("${", "<", "YOUR_"))
        or any(
            marker in normalized
            for marker in ("CHANGEME", "EXAMPLE", "PLACEHOLDER", "REDACTED")
        )
    )


class ReleaseIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.files = release_files()

    def test_required_figure_producers_exist(self) -> None:
        missing = [str(path) for path in REQUIRED_PRODUCERS if not (ROOT / path).is_file()]
        self.assertFalse(missing, "Missing figure/model producer files: " + ", ".join(missing))

    def test_required_static_assets_exist(self) -> None:
        missing = [
            str(path) for path in REQUIRED_STATIC_ASSETS if not (ROOT / path).is_file()
        ]
        self.assertFalse(missing, "Missing required static assets: " + ", ".join(missing))

    def test_no_forbidden_release_directories(self) -> None:
        failures = sorted(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_dir()
            and path.name in FORBIDDEN_RELEASE_DIRECTORIES
            and ".git" not in path.relative_to(ROOT).parts
        )
        self.assertFalse(
            failures,
            "Private/generated directories must not be present in the release:\n"
            + "\n".join(failures),
        )

    def test_required_model_artifacts_exist(self) -> None:
        missing = [str(path) for path in REQUIRED_MODELS if not (ROOT / path).is_file()]
        self.assertFalse(missing, "Missing required model artifacts: " + ", ".join(missing))

        empty = [
            str(path)
            for path in REQUIRED_MODELS
            if (ROOT / path).is_file() and (ROOT / path).stat().st_size == 0
        ]
        self.assertFalse(empty, "Empty model artifacts: " + ", ".join(empty))

    def test_pv_artifact_is_one_coefficient_row(self) -> None:
        if not (ROOT / PV_MODEL).is_file():
            self.skipTest(f"Required artifact is missing: {PV_MODEL}")

        with (ROOT / PV_MODEL).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1, "The PV model CSV must contain one fitted-model row")
        required_columns = {"a1", "a2", "a3"}
        self.assertTrue(
            required_columns.issubset(rows[0]),
            f"PV model CSV must contain coefficients {sorted(required_columns)}",
        )

    def test_model_json_is_valid_and_confined_to_models(self) -> None:
        failures: list[str] = []
        for path in self.files:
            if path.suffix.lower() != ".json":
                continue
            rel = path.relative_to(ROOT)
            if not rel.parts or rel.parts[0] != "models":
                failures.append(f"{relative(path)} (JSON is only allowed under models/)")
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                failures.append(f"{relative(path)} ({error})")
                continue
            if not isinstance(value, dict):
                failures.append(f"{relative(path)} (top level must be an object)")

        self.assertFalse(failures, "Invalid or misplaced JSON:\n" + "\n".join(failures))

    def test_no_credential_filenames(self) -> None:
        failures: list[str] = []
        for path in self.files:
            name = path.name.lower()
            if name == ".env.example":
                continue
            if (
                name == ".env"
                or name.startswith(".env.")
                or name in FORBIDDEN_CREDENTIAL_NAMES
                or path.suffix.lower() in FORBIDDEN_CREDENTIAL_SUFFIXES
                or name.startswith("service-account")
            ):
                failures.append(relative(path))

        self.assertFalse(
            failures,
            "Credential-like files must not be released:\n" + "\n".join(failures),
        )

    def test_no_pi_historian_acquisition_code(self) -> None:
        failures: list[str] = []
        for path in self.files:
            if (
                path.resolve() == THIS_FILE
                or path.suffix.lower() not in ACQUISITION_CODE_SUFFIXES
            ):
                continue

            normalized_name = re.sub(
                r"[^a-z0-9]", "", path.relative_to(ROOT).as_posix().casefold()
            )
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            normalized_text = re.sub(r"[^a-z0-9]", "", text.casefold())

            if any(
                marker in normalized_name or marker in normalized_text
                for marker in PI_ACQUISITION_MARKERS
            ):
                failures.append(relative(path))

        self.assertFalse(
            failures,
            "OSIsoft/AVEVA PI historian acquisition code must not be released:\n"
            + "\n".join(failures),
        )

    def test_no_secrets_or_absolute_user_paths_in_text(self) -> None:
        failures: list[str] = []
        for path in self.files:
            if path.resolve() == THIS_FILE or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            reasons: list[str] = []
            if PRIVATE_KEY_MARKER.search(text):
                reasons.append("private-key marker")
            if KNOWN_TOKEN_MARKER.search(text):
                reasons.append("token-shaped value")
            if GUROBI_CREDENTIAL_ASSIGNMENT.search(text):
                reasons.append("Gurobi credential assignment")
            if URL_WITH_USERINFO.search(text):
                reasons.append("URL containing user information")
            if any(
                pattern.search(text)
                for pattern in (
                    WINDOWS_USER_PATH,
                    UNIX_USER_PATH,
                    ENCODED_WINDOWS_USER_PATH,
                )
            ):
                reasons.append("absolute user-home path")
            if any(
                not is_placeholder(match.group(2))
                for match in GENERIC_LITERAL_CREDENTIAL.finditer(text)
            ):
                reasons.append("literal credential assignment")

            if reasons:
                failures.append(f"{relative(path)}: {', '.join(sorted(set(reasons)))}")

        self.assertFalse(
            failures,
            "Potential secrets or local paths found (values intentionally omitted):\n"
            + "\n".join(failures),
        )

    def test_no_unexpected_data_or_tabular_files(self) -> None:
        failures: list[str] = []
        for path in self.files:
            rel = path.relative_to(ROOT)
            rel_lower = rel.as_posix().lower()
            if not rel_lower.endswith(FORBIDDEN_DATA_SUFFIXES):
                continue
            if rel == PV_MODEL:
                continue
            failures.append(relative(path))

        self.assertFalse(
            failures,
            "Unexpected data/tabular artifacts; only the fitted PV coefficient CSV is allowed:\n"
            + "\n".join(failures),
        )

    def test_binary_and_model_json_files_follow_exact_allowlist(self) -> None:
        allowed_models = set(REQUIRED_MODELS)
        failures = []
        for path in self.files:
            rel = path.relative_to(ROOT)
            if path.suffix.lower() in {".pt", ".pth", ".json"} and rel not in allowed_models:
                failures.append(relative(path))
            if path.suffix.lower() in {".png", ".jpg", ".pdf", ".zip", ".gz", ".exe", ".dll"}:
                if rel not in set(REQUIRED_STATIC_ASSETS):
                    failures.append(relative(path))
        self.assertFalse(failures, "Unapproved binary/model artifacts:\n" + "\n".join(failures))

    def test_notebooks_are_clean(self) -> None:
        failures: list[str] = []
        for path in self.files:
            if path.suffix.lower() != ".ipynb":
                continue
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                failures.append(f"{relative(path)}: invalid notebook JSON ({error})")
                continue

            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                if cell.get("execution_count") is not None:
                    failures.append(
                        f"{relative(path)}: cell {index} has an execution count"
                    )
                if cell.get("outputs"):
                    failures.append(f"{relative(path)}: cell {index} has saved outputs")

        self.assertFalse(
            failures,
            "Release notebooks must have cleared outputs and execution counts:\n"
            + "\n".join(failures),
        )

    def test_no_unexpected_large_files(self) -> None:
        failures = [
            f"{relative(path)} ({path.stat().st_size / (1024 * 1024):.1f} MiB)"
            for path in self.files
            if path.stat().st_size > MAX_UNEXPECTED_FILE_BYTES
        ]
        self.assertFalse(
            failures,
            f"Files larger than {MAX_UNEXPECTED_FILE_BYTES // (1024 * 1024)} MiB require "
            "explicit review:\n" + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
