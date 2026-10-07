"""Image references -> storage keys.

Core stores where an image lives as a path in services_transactiondatacontainer.meta_data (front_url,
front_img, url, dict_of_paths, ...). We found three shapes, all pointing at a .zip:
    media/<...>/x.zip      relative to Django's media storage
    /<...>/x.zip           absolute path on a server or mount
    ./<...>/x.zip          relative to the folder the writing service ran in

storage_key() turns every shape into one relative key, e.g. "tun_nid_ocr/x.zip". The same key is a path
under the NFS root and an object name in an Azure / S3 / GCS bucket. Which leading folders to drop
(the mount point, "media/", ...) is configuration: `strip_prefixes`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")  # http://, https://, s3://, ...


def _clean(path: str) -> str:
    """Forward slashes only, no "./" at the start, no doubled slashes."""
    path = path.strip().replace("\\", "/")
    path = re.sub(r"/{2,}", "/", path)
    while path.startswith("./"):
        path = path[2:] # remove leading "./"
    return path


def storage_key(reference: str, strip_prefixes: Sequence[str] = ()) -> str:
    """The storage key of one image reference. Raises ValueError when the reference cannot be a key
    (empty, a URL, or a path with ".." that could leave the storage root).

    strip_prefixes: leading folders to drop, e.g. ["/mnt/nfs", "media"]. The longest matching one is
    dropped (only whole folders match: "/mnt/nfs" does not match "/mnt/nfs2/x.zip")."""
    if not reference or not reference.strip():
        raise ValueError("empty reference")
    if URL_SCHEME.match(reference.strip()):
        raise ValueError("reference is a URL, not a storage path")
    path = _clean(reference)
    for prefix in sorted((_clean(p).rstrip("/") for p in strip_prefixes), key=len, reverse=True):
        if prefix and path.startswith(prefix + "/"):
            path = path[len(prefix) + 1:]
            break
    if path.endswith("/"):
        raise ValueError("reference has no file name")
    parts = [part for part in path.split("/") if part not in ("", ".")]   # "/a/./b" -> ["a", "b"]
    if ".." in parts:
        raise ValueError("reference contains '..'")
    if not parts:
        raise ValueError("reference has no file name")
    return "/".join(parts)
