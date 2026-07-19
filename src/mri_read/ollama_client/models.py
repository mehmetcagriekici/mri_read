"""Checking for / pulling local Ollama models."""

from __future__ import annotations

import json
import logging
import urllib.request

from mri_read.ollama_client.resolve import resolve_model

logger = logging.getLogger(__name__)


def model_present(host: str, model: str) -> bool:
    """Is `model` (or some tag of it) already downloaded?"""
    return resolve_model(host, model) is not None


def ensure_model(host: str, model: str) -> str:
    """Pull `model` into the local Ollama store if it's not there yet.

    Returns the exact tag to use for subsequent API calls: if `model` (or a
    same-base-name tag of it) is already present, that exact tag is returned
    unchanged; if a pull is triggered, `model` is returned as-is since Ollama
    pulls an untagged name to ":latest", which then matches exactly.

    This is why the Docker image stays small: weights are pulled at runtime
    into a persistent volume, not baked into the image.
    """
    resolved = resolve_model(host, model)
    if resolved is not None:
        return resolved
    logger.info("Pulling local model '%s' (one-time)...", model)
    # /api/pull streams progress lines; read to completion.
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/pull",
        data=json.dumps({"name": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=None) as r:
        for line in r:
            try:
                status = json.loads(line).get("status", "")
            except json.JSONDecodeError:
                continue
            if status:
                # A live-updating single-line progress meter, not a discrete
                # log event — print() (with \r) is the right tool here, not
                # logging, which would emit one line per update instead of
                # overwriting in place.
                print(f"  {status}", end="\r")
    print()
    logger.info("done.")
    return model
