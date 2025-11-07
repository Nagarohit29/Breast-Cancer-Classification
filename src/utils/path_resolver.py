from pathlib import Path
import os


def resolve_breakhis_path(csv_path, data_dir: str):
    """Resolve CSV 'filename' entries from the BreaKHis CSVs to an actual file path.

    The CSVs contain paths like:
       BreaKHis_v1/histology_slides/breast/benign/.../100X/<file>.png

    But some users extract the archive so that `data/raw/` contains `benign/` and
    `malignant/` folders directly. This helper tries several fallbacks:
      1. If csv_path is absolute and exists, return it.
      2. If csv_path contains the segment 'breast', drop the prefix up to 'breast'
         and join the remainder under data_dir.
      3. If csv_path begins with 'BreaKHis' or 'BreaKHis_v1', drop that root and
         join under data_dir.
      4. If csv_path already starts with 'benign' or 'malignant', join under data_dir.
      5. As a last resort, attempt to find a file with the same basename anywhere
         under data_dir (recursive glob). This is slower but robust.

    Returns: a path string (not guaranteed to exist). The caller should handle
    missing files.
    """
    if not csv_path:
        return csv_path

    data_dir = str(data_dir)
    # Absolute path check
    try:
        if os.path.isabs(csv_path) and os.path.exists(csv_path):
            return csv_path
    except Exception:
        pass

    norm = str(csv_path).replace('\\', '/').strip()
    parts = [p for p in norm.split('/') if p]

    # If 'breast' in parts, use parts after it (drops BreaKHis_v1/histology_slides/breast)
    if 'breast' in parts:
        try:
            idx = parts.index('breast')
            rel = parts[idx+1:]
            candidate = os.path.join(data_dir, *rel)
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass

    # If starts with breakhis root, drop it
    if parts and parts[0].lower().startswith('breakhis'):
        rel = parts[1:]
        candidate = os.path.join(data_dir, *rel)
        if os.path.exists(candidate):
            return candidate

    # If path already starts with benign/malignant, join under data_dir
    if parts and parts[0].lower() in ('benign', 'malignant'):
        candidate = os.path.join(data_dir, *parts)
        if os.path.exists(candidate):
            return candidate

    # Try dropping known prefix 'BreaKHis_v1/histology_slides/breast' if present
    prefix = ['BreaKHis_v1', 'histology_slides', 'breast']
    if parts[:3] == prefix:
        candidate = os.path.join(data_dir, *parts[3:])
        if os.path.exists(candidate):
            return candidate

    # Fallback: try joining data_dir with the basename
    basename = parts[-1] if parts else os.path.basename(norm)
    candidate = os.path.join(data_dir, basename)
    if os.path.exists(candidate):
        return candidate

    # Final fallback: search recursively for basename under data_dir (may be slow)
    try:
        root = Path(data_dir)
        matches = list(root.rglob(basename))
        if matches:
            return str(matches[0])
    except Exception:
        pass

    # If nothing found, return a reasonable candidate path under data_dir using the last parts
    try:
        return os.path.join(data_dir, *parts[-4:]) if len(parts) >= 4 else os.path.join(data_dir, *parts)
    except Exception:
        return os.path.join(data_dir, basename)
