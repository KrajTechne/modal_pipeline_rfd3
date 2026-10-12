"""Which launch form actually survives the client exiting?

    modal run --detach experiments/detach_probe.py --mode spawn
    # wait ~4 minutes, then:
    modal volume ls rfd3-pipeline detach_probe

Two campaign launches were lost to guessing at this, so it gets measured on a
CPU container for pennies instead of on a four-hour GPU job.

The shape mirrors `orchestrate` exactly, because the failure was specific to
that shape: a **parent** function that dispatches **children** via `.map()` and
only then writes its result. Both losses came from the parent dying while its
children lived.

* `--detach` with `.remote()` -- what was tried first. The app stayed alive and
  all 80 RFD3 designs landed, then nothing. Detached mode keeps only the *last
  triggered* call alive, and by the time the client died that was the child.
* `.spawn()` without `--detach` -- tried second. `modal run` makes an
  *ephemeral* app that tears down when the entrypoint returns, so the app was
  created and stopped inside the same minute and the parent never ran.
* `--detach` with `.spawn()` -- the untested combination, and the reason this
  file exists.

`parent` sleeps past any plausible client lifetime before writing, so a marker
on the volume can only mean it outlived this process.
"""

from __future__ import annotations

import modal

app = modal.App("detach-probe")

data = modal.Volume.from_name("rfd3-pipeline", create_if_missing=True)
DATA = "/data"
MARK = f"{DATA}/detach_probe"

image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={DATA: data}, timeout=20 * 60)
def child(index: int) -> int:
    """A dispatched child, as `run_rfd3` is to `orchestrate`."""
    import time

    time.sleep(5)
    return index * index


@app.function(image=image, volumes={DATA: data}, timeout=20 * 60)
def parent(mode: str) -> dict:
    """Dispatch children, wait out the client, then record having finished.

    The sleep is the whole experiment. A marker written before it would prove
    only that the parent *started*; written after, it can only mean the parent
    was still alive long after this client exited.
    """
    import json
    import pathlib
    import time

    started = time.time()
    squares = list(child.map(range(4)))

    # Longer than the client takes to exit, by a wide margin.
    time.sleep(180)

    root = pathlib.Path(MARK)
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "mode": mode,
        "children": squares,
        "parent_seconds": round(time.time() - started, 1),
        "verdict": "parent outlived the client",
    }
    (root / f"{mode}.json").write_text(json.dumps(record, indent=2))
    data.commit()
    return record


@app.local_entrypoint()
def main(mode: str = "spawn"):
    """`--mode spawn` returns at once; `--mode remote` blocks.

    Run the spawn form under `modal run --detach`. If a marker appears on the
    volume minutes after this process is gone, that launch form is safe for a
    campaign; if it never appears, it is not, and no GPU time was spent finding
    out.
    """
    if mode == "spawn":
        call = parent.spawn(mode)
        print(
            f"spawned parent, call id {call.object_id}\n"
            f"  this client is about to exit. check in ~4 minutes:\n"
            f"    modal volume ls rfd3-pipeline detach_probe\n"
            f"  a {mode}.json there means the parent survived."
        )
        return
    print(parent.remote(mode))
