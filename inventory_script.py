import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "_organized",
    "organizer_logs",
    "gfn.egg-info",
}


_EXT_KIND = {
    ".py": ("code", "python"),
    ".pyi": ("code", "python"),
    ".pyx": ("code", "python"),
    ".pxd": ("code", "python"),
    ".ipynb": ("code", "notebook"),
    ".cu": ("code", "cuda"),
    ".cuh": ("code", "cuda"),
    ".c": ("code", "c_cpp"),
    ".h": ("code", "c_cpp"),
    ".hpp": ("code", "c_cpp"),
    ".cpp": ("code", "c_cpp"),
    ".md": ("docs", "markdown"),
    ".rst": ("docs", "restructuredtext"),
    ".txt": ("docs", "text"),
    ".pdf": ("docs", "pdf"),
    ".tex": ("docs", "latex"),
    ".yaml": ("config", "yaml"),
    ".yml": ("config", "yaml"),
    ".json": ("config", "json"),
    ".toml": ("config", "toml"),
    ".ini": ("config", "ini"),
    ".cfg": ("config", "ini"),
    ".csv": ("data", "csv"),
    ".tsv": ("data", "tsv"),
    ".parquet": ("data", "parquet"),
    ".png": ("data", "image"),
    ".jpg": ("data", "image"),
    ".jpeg": ("data", "image"),
    ".gif": ("data", "image"),
    ".svg": ("data", "image"),
    ".webp": ("data", "image"),
    ".mp3": ("data", "audio"),
    ".wav": ("data", "audio"),
    ".flac": ("data", "audio"),
    ".mp4": ("data", "video"),
    ".mov": ("data", "video"),
    ".mkv": ("data", "video"),
    ".zip": ("archive", "zip"),
    ".tar": ("archive", "tar"),
    ".gz": ("archive", "gzip"),
    ".7z": ("archive", "7z"),
    ".rar": ("archive", "rar"),
    ".whl": ("artifact", "python_wheel"),
    ".exe": ("artifact", "binary"),
    ".dll": ("artifact", "binary"),
    ".so": ("artifact", "binary"),
    ".pyd": ("artifact", "binary"),
    ".pt": ("models", "pytorch"),
    ".pth": ("models", "pytorch"),
    ".ckpt": ("models", "checkpoint"),
    ".onnx": ("models", "onnx"),
    ".safetensors": ("models", "safetensors"),
    ".log": ("results", "log"),
}


_PURPOSE_HINTS = [
    ("test", "pruebas"),
    ("bench", "benchmark"),
    ("demo", "demo"),
    ("train", "entrenamiento"),
    ("eval", "evaluación"),
    ("infer", "inferencia"),
    ("viz", "visualización"),
    ("plot", "visualización"),
    ("config", "configuración"),
    ("readme", "documentación"),
    ("license", "documentación"),
    ("changelog", "documentación"),
]


@dataclass(frozen=True)
class FileDescription:
    path: Path
    ext: str
    size_bytes: int
    mtime_utc: dt.datetime
    kind_group: str
    kind_detail: str
    purpose: str
    content_main: str


def _setup_logger(log_dir: Path, verbose: bool) -> tuple[logging.Logger, Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"organizer_{ts}.log"
    report_path = log_dir / f"organizer_{ts}.jsonl"
    logger = logging.getLogger("file_organizer")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False
    return logger, log_path, report_path


def _write_report_event(report_path: Path, event: dict) -> None:
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _is_probably_text(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in {".py", ".pyi", ".pyx", ".pxd", ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".c", ".cpp", ".h", ".hpp", ".cu", ".cuh"}:
        return True
    return False


def _read_text_head(path: Path, max_bytes: int = 24_576) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
    except Exception:
        return ""
    try:
        return data.decode("utf-8")
    except Exception:
        try:
            return data.decode("latin-1", errors="ignore")
        except Exception:
            return ""


def _slugify_stem(stem: str, lowercase: bool) -> str:
    stem = unicodedata.normalize("NFKD", stem)
    stem = "".join(ch for ch in stem if not unicodedata.combining(ch))
    stem = stem.strip()
    stem = re.sub(r"[\s\-]+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_\.]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_.")
    if lowercase:
        stem = stem.lower()
    return stem or "file"


def _shorten_filename(stem: str, ext: str, max_len: int) -> str:
    candidate = f"{stem}{ext}"
    if len(candidate) <= max_len:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
    room = max(1, max_len - len(ext) - 11)
    return f"{stem[:room]}_{digest}{ext}"


def normalize_filename(name: str, lowercase: bool, max_len: int) -> str:
    p = Path(name)
    ext = p.suffix.lower()
    stem = p.stem
    clean_stem = _slugify_stem(stem, lowercase=lowercase)
    return _shorten_filename(clean_stem, ext, max_len=max_len)


def _safe_path_length(target: Path, max_total: int = 240) -> Path:
    target_str = str(target)
    if len(target_str) <= max_total:
        return target
    ext = target.suffix
    stem = target.stem
    digest = hashlib.sha1(target_str.encode("utf-8")).hexdigest()[:10]
    parent_str = str(target.parent)
    room = max(1, max_total - len(parent_str) - 1 - len(ext) - 11)
    new_name = f"{stem[:room]}_{digest}{ext}"
    return target.with_name(new_name)


def _size_bucket(size_bytes: int) -> str:
    if size_bytes < 1_000_000:
        return "small"
    if size_bytes < 50_000_000:
        return "medium"
    return "large"


def _guess_purpose(path: Path, text_head: str) -> str:
    lowered = str(path).lower()
    for needle, label in _PURPOSE_HINTS:
        if needle in lowered:
            return label
    if path.suffix.lower() == ".py" and "if __name__" in text_head:
        return "script"
    if path.suffix.lower() in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}:
        return "configuración"
    if path.suffix.lower() in {".md", ".rst", ".txt"}:
        return "documentación"
    return "general"


def _main_content(text_head: str, ext: str) -> str:
    if not text_head:
        return ""
    lines = [ln.strip() for ln in text_head.splitlines() if ln.strip()]
    if not lines:
        return ""
    if ext in {".md", ".rst"}:
        for ln in lines[:40]:
            if ln.startswith("#") or ln.startswith("="):
                return ln[:200]
    if ext == ".py":
        for ln in lines[:80]:
            if ln.startswith("def ") or ln.startswith("class ") or ln.startswith("import ") or ln.startswith("from "):
                return ln[:200]
    return lines[0][:200]


def describe_file(path: Path) -> FileDescription:
    st = path.stat()
    ext = path.suffix.lower()
    kind_group, kind_detail = _EXT_KIND.get(ext, ("other", ext.lstrip(".") or "noext"))
    head = _read_text_head(path) if _is_probably_text(path) else ""
    purpose = _guess_purpose(path, head)
    content_main = _main_content(head, ext)
    return FileDescription(
        path=path,
        ext=ext,
        size_bytes=int(st.st_size),
        mtime_utc=dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc),
        kind_group=kind_group,
        kind_detail=kind_detail,
        purpose=purpose,
        content_main=content_main,
    )


def _theme_for(desc: FileDescription) -> str:
    ext = desc.ext
    if desc.kind_group == "code":
        if ext in {".py", ".pyi", ".pyx", ".pxd"}:
            return "01_code/python"
        if ext in {".cu", ".cuh"}:
            return "01_code/cuda"
        if ext in {".c", ".cpp", ".h", ".hpp"}:
            return "01_code/c_cpp"
        if ext == ".ipynb":
            return "01_code/notebooks"
        return "01_code/other"
    if desc.kind_group == "docs":
        if ext in {".md", ".rst", ".txt"}:
            return "02_docs/text"
        return "02_docs/other"
    if desc.kind_group == "config":
        return "03_configs"
    if desc.kind_group == "data":
        if desc.kind_detail == "image":
            return "04_data/images"
        if desc.kind_detail == "audio":
            return "04_data/audio"
        if desc.kind_detail == "video":
            return "04_data/video"
        return "04_data/other"
    if desc.kind_group == "models":
        return "05_models"
    if desc.kind_group == "archive":
        return "06_archives"
    if desc.kind_group == "artifact":
        return "07_artifacts"
    if desc.kind_group == "results":
        return "08_results"
    return "09_misc"


def _build_destination(
    dest_root: Path,
    desc: FileDescription,
    normalized_name: str,
    by_date: bool,
    by_size: bool,
) -> Path:
    theme = _theme_for(desc)
    parts: list[str] = [theme]
    if by_date:
        ts = desc.mtime_utc
        parts.extend([str(ts.year), f"{ts.month:02d}"])
    if by_size:
        parts.append(_size_bucket(desc.size_bytes))
    dst_dir = dest_root.joinpath(*parts)
    dst_path = dst_dir / normalized_name
    return _safe_path_length(dst_path)


def _iter_files(root: Path, ignored_dirs: set[str], dynamic_ignored_prefixes: list[Path], logger: logging.Logger, report_path: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            cur_resolved = cur.resolve()
        except Exception:
            cur_resolved = cur
        if any(str(cur_resolved).startswith(str(p)) for p in dynamic_ignored_prefixes):
            continue
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    p = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in ignored_dirs:
                            continue
                        stack.append(p)
                        continue
                    if entry.is_file(follow_symlinks=False):
                        out.append(p)
        except PermissionError as e:
            logger.warning(f"PERMISSION | skip_dir={cur} | err={e}")
            _write_report_event(report_path, {"event": "permission_skip_dir", "path": str(cur), "error": str(e)})
        except OSError as e:
            logger.warning(f"OSERROR | skip_dir={cur} | err={e}")
            _write_report_event(report_path, {"event": "oserror_skip_dir", "path": str(cur), "error": str(e)})
    return out


def _hash_head(path: Path, head_bytes: int = 65_536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()


def _hash_full(path: Path, chunk: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _are_files_equal(a: Path, b: Path) -> bool:
    try:
        sa = a.stat()
        sb = b.stat()
    except Exception:
        return False
    if sa.st_size != sb.st_size:
        return False
    try:
        if _hash_head(a) != _hash_head(b):
            return False
    except Exception:
        return False
    try:
        return _hash_full(a) == _hash_full(b)
    except Exception:
        return False


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    ext = path.suffix
    for i in range(1, 10_000):
        candidate = path.with_name(f"{stem}_dup{i}{ext}")
        if not candidate.exists():
            return candidate
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return path.with_name(f"{stem}_{digest}{ext}")


def organize(
    root: Path,
    dest_root: Path,
    apply_changes: bool,
    lowercase: bool,
    by_date: bool,
    by_size: bool,
    dedupe: bool,
    copy_mode: bool,
    max_name_len: int,
    max_files: int | None,
    verbose: bool,
) -> tuple[Path, Path]:
    log_dir = root / "organizer_logs"
    logger, log_path, report_path = _setup_logger(log_dir, verbose=verbose)

    dynamic_ignored_prefixes = []
    try:
        dynamic_ignored_prefixes.append(dest_root.resolve())
    except Exception:
        dynamic_ignored_prefixes.append(dest_root)
    try:
        dynamic_ignored_prefixes.append(log_dir.resolve())
    except Exception:
        dynamic_ignored_prefixes.append(log_dir)

    logger.info(f"START | root={root} | dest_root={dest_root} | apply={apply_changes} | copy={copy_mode}")
    _write_report_event(
        report_path,
        {
            "event": "start",
            "root": str(root),
            "dest_root": str(dest_root),
            "apply": bool(apply_changes),
            "copy": bool(copy_mode),
            "lowercase": bool(lowercase),
            "by_date": bool(by_date),
            "by_size": bool(by_size),
            "dedupe": bool(dedupe),
        },
    )

    files = _iter_files(root, ignored_dirs=set(_DEFAULT_IGNORED_DIRS), dynamic_ignored_prefixes=dynamic_ignored_prefixes, logger=logger, report_path=report_path)
    files = [p for p in files if p != Path(__file__).resolve()]
    if max_files is not None:
        files = files[: max(0, int(max_files))]

    duplicates_dir = dest_root / "00_duplicates"

    processed = 0
    moved = 0
    skipped = 0
    errors = 0

    for src in files:
        processed += 1
        try:
            desc = describe_file(src)
            normalized_name = normalize_filename(src.name, lowercase=lowercase, max_len=max_name_len)
            dst = _build_destination(dest_root=dest_root, desc=desc, normalized_name=normalized_name, by_date=by_date, by_size=by_size)

            if src.resolve() == dst.resolve():
                skipped += 1
                logger.info(f"SKIP | already_ok | path={src}")
                _write_report_event(report_path, {"event": "skip", "reason": "already_ok", "path": str(src)})
                continue

            if dst.exists():
                if dedupe and _are_files_equal(src, dst):
                    dup_target = _next_available_path(_safe_path_length(duplicates_dir / normalized_name))
                    if apply_changes:
                        dup_target.parent.mkdir(parents=True, exist_ok=True)
                        if copy_mode:
                            shutil.copy2(src, dup_target)
                        else:
                            shutil.move(src, dup_target)
                    logger.warning(f"DUPLICATE | src={src} | existing={dst} | dup_target={dup_target} | apply={apply_changes}")
                    _write_report_event(
                        report_path,
                        {"event": "duplicate", "src": str(src), "existing": str(dst), "duplicate_target": str(dup_target), "apply": bool(apply_changes)},
                    )
                    moved += 1
                    continue
                dst = _next_available_path(dst)

            event = {
                "event": "move" if not copy_mode else "copy",
                "src": str(src),
                "dst": str(dst),
                "ext": desc.ext,
                "kind_group": desc.kind_group,
                "kind_detail": desc.kind_detail,
                "purpose": desc.purpose,
                "size_bytes": desc.size_bytes,
                "mtime_utc": desc.mtime_utc.isoformat(),
                "content_main": desc.content_main,
                "apply": bool(apply_changes),
            }
            _write_report_event(report_path, event)

            if apply_changes:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if copy_mode:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)

            logger.info(f"{'COPY' if copy_mode else 'MOVE'} | src={src} | dst={dst} | apply={apply_changes}")
            moved += 1
        except PermissionError as e:
            errors += 1
            logger.warning(f"PERMISSION | src={src} | err={e}")
            _write_report_event(report_path, {"event": "permission_error", "src": str(src), "error": str(e)})
        except OSError as e:
            errors += 1
            logger.warning(f"OSERROR | src={src} | err={e}")
            _write_report_event(report_path, {"event": "oserror", "src": str(src), "error": str(e)})
        except Exception as e:
            errors += 1
            logger.warning(f"ERROR | src={src} | err={e}")
            _write_report_event(report_path, {"event": "error", "src": str(src), "error": str(e)})

    summary = {"processed": processed, "moved_or_copied": moved, "skipped": skipped, "errors": errors}
    logger.info(f"DONE | {summary}")
    _write_report_event(report_path, {"event": "done", **summary})

    return log_path, report_path


def _classify_formula_file(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in [".py", ".pyx", ".pxd"]:
        return "Python"
    if ext in [".cu", ".cuh", ".c", ".h", ".hpp", ".cpp"]:
        return "CUDA"
    return None


def _extract_formulas_python(file_path: Path) -> list[dict]:
    formulas: list[dict] = []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines(True)
    except Exception:
        try:
            lines = file_path.read_text(encoding="latin-1", errors="ignore").splitlines(True)
        except Exception:
            return formulas

    current_context = "Global"
    context_pattern = re.compile(r"^\s*(class|def)\s+([a-zA-Z0-9_]+)")
    math_pattern = re.compile(
        r"("
        r"[\+\-\*\/@]"
        r"|np\."
        r"|math\."
        r"|torch\.(sin|cos|tan|exp|log|sqrt|rsqrt|pow|einsum|matmul|mm|bmm|addmm|sigmoid|tanh|softmax)\b"
        r"|\.(sum|mean|norm|var|std|matmul|mm|bmm|einsum)\("
        r")"
    )

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            m = context_pattern.match(stripped)
            if m:
                current_context = m.group(2)

        if stripped and not stripped.startswith("#") and "=" in stripped:
            full_stmt = stripped
            start_line = i + 1
            open_parens = full_stmt.count("(") - full_stmt.count(")")
            open_brackets = full_stmt.count("[") - full_stmt.count("]")
            open_braces = full_stmt.count("{") - full_stmt.count("}")

            j = i + 1
            while (open_parens > 0 or open_brackets > 0 or open_braces > 0 or full_stmt.rstrip().endswith("\\")) and j < len(lines):
                nxt = lines[j].strip()
                if nxt and not nxt.startswith("#"):
                    full_stmt += " " + nxt
                    open_parens += nxt.count("(") - nxt.count(")")
                    open_brackets += nxt.count("[") - nxt.count("]")
                    open_braces += nxt.count("{") - nxt.count("}")
                j += 1

            if (
                math_pattern.search(full_stmt)
                and len(full_stmt) > 15
                and "torch.backends." not in full_stmt
                and "torch.device(" not in full_stmt
            ):
                formulas.append({"line": start_line, "context": current_context, "code": full_stmt})

            i = j - 1

        i += 1

    return formulas


def _extract_formulas_cuda(file_path: Path) -> list[dict]:
    formulas: list[dict] = []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines(True)
    except Exception:
        try:
            lines = file_path.read_text(encoding="latin-1", errors="ignore").splitlines(True)
        except Exception:
            return formulas

    current_context = "Global"
    context_pattern = re.compile(r"(__global__|__device__|__host__)\s+.*\s+([a-zA-Z0-9_]+)\(")
    math_pattern = re.compile(r"([\+\-\*\/]|sin|cos|exp|log|pow|sqrt|rsqrt|fma)")

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("//"):
            i += 1
            continue

        m = context_pattern.search(lines[i])
        if m:
            current_context = m.group(2)

        if ("=" in stripped or "+=" in stripped or "*=" in stripped) and not stripped.startswith(("if", "for", "while", "return")):
            full_stmt = stripped
            start_line = i + 1
            j = i + 1
            while ";" not in full_stmt and j < len(lines):
                nxt = lines[j].strip()
                if nxt and not nxt.startswith("//"):
                    full_stmt += " " + nxt
                j += 1

            if math_pattern.search(full_stmt) and len(full_stmt) > 10:
                formulas.append({"line": start_line, "context": current_context, "code": full_stmt})

            i = j - 1

        i += 1

    return formulas


def run_inventory(root_dir: Path, output_path: Path) -> None:
    python_files: list[Path] = []
    cuda_files: list[Path] = []

    stack = [root_dir]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    p = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in _DEFAULT_IGNORED_DIRS:
                            continue
                        stack.append(p)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    kind = _classify_formula_file(p)
                    if kind == "Python":
                        python_files.append(p)
                    elif kind == "CUDA":
                        cuda_files.append(p)
        except Exception:
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as md:
        md.write("# Inventario de Fórmulas Matemáticas Core (Python vs CUDA)\n\n")
        md.write("Este documento lista las fórmulas matemáticas identificadas en el codebase, con referencias exactas a las líneas de código.\n\n")

        all_python_formulas: list[dict] = []
        all_cuda_formulas: list[dict] = []

        md.write("## Archivos Python\n\n")
        for p_file in sorted(python_files, key=lambda x: str(x).lower()):
            rel_path = os.path.relpath(p_file, root_dir)
            formulas = _extract_formulas_python(p_file)
            if not formulas:
                continue
            md.write(f"### {rel_path}\n\n")
            md.write("| Línea | Contexto | Fórmula |\n")
            md.write("| :--- | :--- | :--- |\n")
            for item in formulas:
                code_display = str(item["code"]).replace("|", r"\|")
                md.write(f"| {item['line']} | {item['context']} | `{code_display}` |\n")
                item["file"] = rel_path
                all_python_formulas.append(item)

            md.write("\n#### Fórmulas Listas para Usar (Python)\n")
            md.write("```python\n")
            for item in formulas:
                md.write(f"# {item['context']} (L{item['line']})\n")
                md.write(f"{item['code']}\n")
            md.write("```\n\n")

        md.write("## Archivos CUDA\n\n")
        for c_file in sorted(cuda_files, key=lambda x: str(x).lower()):
            rel_path = os.path.relpath(c_file, root_dir)
            formulas = _extract_formulas_cuda(c_file)
            if not formulas:
                continue
            md.write(f"### {rel_path}\n\n")
            md.write("| Línea | Contexto | Fórmula |\n")
            md.write("| :--- | :--- | :--- |\n")
            for item in formulas:
                code_display = str(item["code"]).replace("|", r"\|")
                md.write(f"| {item['line']} | {item['context']} | `{code_display}` |\n")
                item["file"] = rel_path
                all_cuda_formulas.append(item)

            md.write("\n#### Fórmulas Listas para Usar (CUDA)\n")
            md.write("```cpp\n")
            for item in formulas:
                md.write(f"// {item['context']} (L{item['line']})\n")
                md.write(f"{item['code']}\n")
            md.write("```\n\n")

        md.write("## Repositorio Global de Fórmulas (Listas para Usar)\n\n")

        md.write("### Colección Completa Python (por archivo)\n")
        current_file: str | None = None
        current_context: str | None = None
        for item in sorted(all_python_formulas, key=lambda x: (str(x.get("file", "")).lower(), str(x.get("context", "")).lower(), int(x.get("line", 0)))):
            if item["file"] != current_file:
                if current_file is not None:
                    md.write("```\n\n")
                current_file = item["file"]
                current_context = None
                md.write(f"#### {current_file}\n")
                md.write("```python\n")
            if item["context"] != current_context:
                current_context = item["context"]
                md.write(f"# Contexto: {current_context}\n")
            md.write(f"# L{item['line']}\n")
            md.write(f"{item['code']}\n")
        if current_file is not None:
            md.write("```\n\n")
        else:
            md.write("_Sin fórmulas Python detectadas._\n\n")

        md.write("### Colección Completa CUDA (por archivo)\n")
        current_file = None
        current_context = None
        for item in sorted(all_cuda_formulas, key=lambda x: (str(x.get("file", "")).lower(), str(x.get("context", "")).lower(), int(x.get("line", 0)))):
            if item["file"] != current_file:
                if current_file is not None:
                    md.write("```\n\n")
                current_file = item["file"]
                current_context = None
                md.write(f"#### {current_file}\n")
                md.write("```cpp\n")
            if item["context"] != current_context:
                current_context = item["context"]
                md.write(f"// Contexto: {current_context}\n")
            md.write(f"// L{item['line']}\n")
            md.write(f"{item['code']}\n")
        if current_file is not None:
            md.write("```\n")
        else:
            md.write("_Sin fórmulas CUDA detectadas._\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="inventory_script.py")
    sub = parser.add_subparsers(dest="command")

    org = sub.add_parser("organize")
    org.add_argument("--root", type=str, default=str(Path.cwd()))
    org.add_argument("--dest-root", type=str, default="")
    org.add_argument("--apply", action="store_true")
    org.add_argument("--copy", action="store_true")
    org.add_argument("--lowercase", action="store_true", default=True)
    org.add_argument("--no-lowercase", action="store_false", dest="lowercase")
    org.add_argument("--by-date", action="store_true")
    org.add_argument("--by-size", action="store_true")
    org.add_argument("--no-dedupe", action="store_false", dest="dedupe")
    org.add_argument("--dedupe", action="store_true", default=True)
    org.add_argument("--max-name-len", type=int, default=140)
    org.add_argument("--max-files", type=int, default=None)
    org.add_argument("--verbose", action="store_true")

    inv = sub.add_parser("inventory")
    inv.add_argument("--root", type=str, default=r"D:\ASAS\manifold_mini\manifold_working")
    inv.add_argument("--output", type=str, default="")

    if not argv:
        return argparse.Namespace(command="inventory", root=r"D:\ASAS\manifold_mini\manifold_working", output="")
    ns = parser.parse_args(argv)
    if ns.command is None:
        return argparse.Namespace(command="inventory", root=r"D:\ASAS\manifold_mini\manifold_working", output="")
    return ns


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if args.command == "inventory":
        root_dir = Path(args.root).resolve()
        if args.output:
            output_path = Path(args.output).resolve()
        else:
            output_path = root_dir / "gfn" / "core_cuda_formulas_inventory.md"
        run_inventory(root_dir=root_dir, output_path=output_path)
        return 0

    root = Path(args.root).resolve()
    dest_root = Path(args.dest_root).resolve() if args.dest_root else (root / "_organized").resolve()
    organize(
        root=root,
        dest_root=dest_root,
        apply_changes=bool(args.apply),
        lowercase=bool(args.lowercase),
        by_date=bool(args.by_date),
        by_size=bool(args.by_size),
        dedupe=bool(args.dedupe),
        copy_mode=bool(args.copy),
        max_name_len=int(args.max_name_len),
        max_files=args.max_files,
        verbose=bool(args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
