"""ColabFold/MMseqs2 client and the on-disk MSA cache.

One client rather than two, taking each existing implementation's strengths as
the plan sets out. The protocol below was read off the AF3 fork's
`msa_server.py`, which has the parsing right; the operational shape --
authentication, a cache keyed so it cannot collide, a download timeout that
fits a large tarball -- comes from Optimus's `mmseqs2.py`, with its two defects
corrected:

* **Its cache key collides.** It keys on `f"{prefix}_{mode}"` with the caller's
  directory as the prefix, so two sequences sharing a directory resolve to the
  same cached tarball. Here the key is a digest of the sequence *and* every
  parameter that changes the answer, so an alignment built with the
  environmental databases cannot be served for a request without them.
* **Its download timeout is 6.02 s**, which is the submit timeout, not a
  tarball's. Submit and poll keep the short timeout; the download gets its own.

**This module performs network egress.** Only sequences passed to it leave, so
the caller decides what goes out. For holo that is the target chains alone --
the binder needs no alignment, and a de novo binder has no homologs to find.

The cache stores *unpaired* alignments only. An unpaired MSA is a property of a
single sequence, so it is context-free and cacheable by digest. A paired MSA is
a property of the set of chains queried together and is not cacheable in
general.
"""

from __future__ import annotations

import hashlib
import io
import logging
import random
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "MsaError",
    "a3m_cache_path",
    "cache_key",
    "parse_a3m_blocks",
    "query_colabfold",
    "read_cached_a3m",
    "write_cached_a3m",
]

# ColabFold labels each query block with an "M-index" starting at 101; the
# response is keyed by it, which is how a multi-sequence query is unpacked.
_M_INDEX_BASE = 101

SUBMIT_TIMEOUT = 6.02
DOWNLOAD_TIMEOUT = 300.0
MAX_ATTEMPTS = 6


class MsaError(RuntimeError):
    """The server refused, failed, or returned something unusable."""


def cache_key(
    sequence: str,
    *,
    use_env: bool,
    use_filter: bool,
    paired: bool = False,
    pairing_strategy: str = "greedy",
) -> str:
    """A digest over the sequence and everything that changes the answer.

    Keying on the sequence alone would serve an alignment built against the
    environmental databases for a request that asked without them -- the same
    class of mistake as Optimus keying on the caller's directory.
    """
    parts = [
        sequence.strip().upper(),
        f"env={int(use_env)}",
        f"filter={int(use_filter)}",
        f"paired={int(paired)}",
    ]
    if paired:
        parts.append(f"pairing={pairing_strategy}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


def a3m_cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.a3m"


def read_cached_a3m(cache_dir: str | Path, key: str) -> str | None:
    path = a3m_cache_path(cache_dir, key)
    if not path.is_file():
        return None
    text = path.read_text()
    # A truncated write is worse than a miss: it would be folded against.
    return text if text.strip() else None


def write_cached_a3m(cache_dir: str | Path, key: str, a3m: str) -> Path:
    """Write to a temporary name and rename, so an interrupted write cannot
    leave a half-written alignment for the next run to read."""
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final = a3m_cache_path(directory, key)
    with tempfile.NamedTemporaryFile(
        "w", dir=directory, prefix=f".{key}.", suffix=".part", delete=False
    ) as handle:
        handle.write(a3m)
        temporary = Path(handle.name)
    temporary.replace(final)
    return final


def parse_a3m_blocks(text: str, n_queries: int) -> list[str]:
    """Split a ColabFold a3m into one block per query, in query order.

    ColabFold separates per-query blocks with NUL bytes and heads each with its
    M-index. Splitting on NUL first is what keeps one query's hits out of
    another's. The trailing newline matters: without it the last sequence of a
    block has no line ending and the next block's header is concatenated onto
    it when blocks from several files are joined.
    """
    blocks: dict[int, list[str]] = {}
    for raw in text.split("\x00"):
        raw = raw.strip()
        if not raw:
            continue
        lines = (raw + "\n").splitlines(keepends=True)
        header = lines[0].strip()
        if not header.startswith(">"):
            continue
        try:
            index = int(header[1:].split()[0])
        except (ValueError, IndexError):
            continue
        blocks.setdefault(index, []).extend(lines)
    return [
        "".join(blocks.get(_M_INDEX_BASE + offset, [])) for offset in range(n_queries)
    ]


def _headers(user_agent: str, auth_header: str | None) -> dict[str, str]:
    headers = {"User-Agent": user_agent}
    if auth_header:
        # Pre-formatted, e.g. "Authorization: Bearer ..." collapsed to a value.
        name, _, value = auth_header.partition(":")
        headers[name.strip()] = value.strip() if value else auth_header
    return headers


def query_colabfold(
    sequences: Sequence[str],
    *,
    paired: bool = False,
    use_env: bool = True,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    username: str | None = None,
    password: str | None = None,
    auth_header: str | None = None,
    user_agent: str = "rfd3-pipeline/1.0",
    sleep=time.sleep,
) -> list[str]:
    """Submit sequences and return one a3m string per input, in input order.

    Duplicates are collapsed into one query and the result fanned back out, so
    two identical target chains cost one submission.

    `sleep` is injected so the polling loop is testable without real waits.
    """
    import requests

    if not sequences:
        return []

    host = host_url.rstrip("/")
    headers = _headers(user_agent, auth_header)
    auth = (username, password) if username is not None else None

    if paired:
        endpoint, mode = "ticket/pair", f"pair{pairing_strategy}"
        if use_env:
            mode = f"{mode}-env"
    else:
        endpoint, mode = "ticket/msa", ("env" if use_env else "all")

    unique: list[str] = []
    seen: dict[str, int] = {}
    for sequence in sequences:
        if sequence not in seen:
            seen[sequence] = len(unique)
            unique.append(sequence)
    query = "".join(
        f">{_M_INDEX_BASE + offset}\n{sequence}\n" for offset, sequence in enumerate(unique)
    )

    def submit() -> dict:
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = requests.post(
                    f"{host}/{endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=SUBMIT_TIMEOUT,
                    headers=headers,
                    auth=auth,
                )
                return response.json()
            except Exception as error:  # noqa: BLE001 - retried, then surfaced
                last = error
                logger.warning("MSA submit failed (%s/%s): %s", attempt + 1, MAX_ATTEMPTS, error)
                sleep(5)
        raise MsaError(f"MSA submit failed after {MAX_ATTEMPTS} attempts: {last}")

    out = submit()
    # RATELIMIT is the public endpoint saying "later", not an error. It is also
    # the reason an authenticated or self-hosted server is the path off it.
    while out.get("status") in ("UNKNOWN", "RATELIMIT"):
        sleep(5 + random.randint(0, 5))
        out = submit()

    status = out.get("status")
    if status == "ERROR":
        raise MsaError("MSA server returned ERROR; check the sequences")
    if status == "MAINTENANCE":
        raise MsaError("MSA server is under maintenance")
    if "id" not in out:
        raise MsaError(f"MSA server returned no job id: {out!r}")

    job_id = out["id"]
    while out.get("status") in ("UNKNOWN", "RUNNING", "PENDING"):
        sleep(5 + random.randint(0, 5))
        try:
            out = requests.get(
                f"{host}/ticket/{job_id}", timeout=SUBMIT_TIMEOUT, headers=headers, auth=auth
            ).json()
        except Exception as error:  # noqa: BLE001 - polling is allowed to stumble
            logger.warning("MSA poll failed, retrying: %s", error)

    if out.get("status") != "COMPLETE":
        raise MsaError(f"MSA job {job_id} ended with status {out.get('status')!r}")

    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            # Its own timeout: a tarball is not a submit.
            download = requests.get(
                f"{host}/result/download/{job_id}",
                timeout=DOWNLOAD_TIMEOUT,
                headers=headers,
                auth=auth,
            )
            break
        except Exception as error:  # noqa: BLE001
            last = error
            logger.warning("MSA download failed (%s/%s): %s", attempt + 1, MAX_ATTEMPTS, error)
            sleep(5)
    else:
        raise MsaError(f"MSA download failed after {MAX_ATTEMPTS} attempts: {last}")

    merged = _extract(download.content, paired=paired, use_env=use_env, n_queries=len(unique))
    return [merged[seen[sequence]] for sequence in sequences]


def _extract(payload: bytes, *, paired: bool, use_env: bool, n_queries: int) -> list[str]:
    """The a3m files the tarball is expected to hold, concatenated per query."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            archive.extractall(root)

        if paired:
            wanted = [root / "pair.a3m"]
        else:
            wanted = [root / "uniref.a3m"]
            if use_env:
                wanted.append(root / "bfd.mgnify30.metaeuk30.smag30.a3m")

        per_query = [""] * n_queries
        found = False
        for path in wanted:
            if not path.is_file():
                continue
            found = True
            for offset, block in enumerate(parse_a3m_blocks(path.read_text(), n_queries)):
                per_query[offset] += block
        if not found:
            present = sorted(p.name for p in root.rglob("*") if p.is_file())
            raise MsaError(
                f"none of {[p.name for p in wanted]} in the MSA tarball; it held {present}"
            )
        return per_query
