"""Minimal async client for the Krea API: submit a job, poll it, download the results."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import aiohttp

log = logging.getLogger("ComfyUI-Krea-API")

API_BASE = os.environ.get("KREA_API_BASE", "https://api.krea.ai").rstrip("/")
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
USER_AGENT = "ComfyUI-Krea-API/1.0"

PENDING = {"backlogged", "queued", "scheduled", "processing", "sampling", "intermediate-complete"}
RATE_LIMIT_RETRIES = 6
TRANSIENT_RETRIES = 3
# Inline images up to this size; bigger ones go through POST /assets.
MAX_INLINE_BYTES = 6 * 1024 * 1024


class KreaError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("KREA_API_KEY", "").strip()
    if not key and CONFIG_FILE.is_file():
        try:
            key = str(json.loads(CONFIG_FILE.read_text()).get("api_key", "")).strip()
        except (OSError, ValueError):
            log.warning("Could not read %s", CONFIG_FILE)
    if not key:
        raise KreaError(
            "No Krea API key. Set the KREA_API_KEY environment variable or put "
            f'{{"api_key": "..."}} in {CONFIG_FILE}. Create a key at https://www.krea.ai/settings/api-tokens'
        )
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {api_key()}", "User-Agent": USER_AGENT}


def _error_text(status: int, body: str) -> str:
    """Krea errors look like {"error": "Validation failed", "details": [{field, message}]},
    {"error": {"code", "message"}} or {"message": ...}."""
    try:
        data = json.loads(body)
    except ValueError:
        return f"Krea API error (HTTP {status}): {body[:500]}"
    if not isinstance(data, dict):
        return f"Krea API error (HTTP {status}): {str(data)[:500]}"
    err = data.get("error")
    code = ""
    if isinstance(err, dict):
        code = err.get("code") or err.get("type") or ""
        msg = err.get("message") or err.get("detail") or json.dumps(err)[:500]
    else:
        msg = err or data.get("message") or data.get("detail") or json.dumps(data)[:500]
    details = data.get("details")
    if isinstance(details, list) and details:
        parts = []
        for d in details[:10]:
            if isinstance(d, dict):
                field = d.get("field") or ""
                parts.append(f"{field}: {d.get('message', '')}" if field else str(d.get("message", "")))
        if parts:
            msg = f"{msg}: " + "; ".join(parts)
    label = f"HTTP {status}, {code}" if code else f"HTTP {status}"
    return f"Krea API error ({label}): {msg}"


def _check_interrupt() -> None:
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


def _progress(node_id: str | None, text: str) -> None:
    if node_id is None:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:  # noqa: BLE001 - progress text is cosmetic
        pass


async def _sleep_interruptible(seconds: float) -> None:
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        _check_interrupt()
        await asyncio.sleep(min(0.5, left))


async def _request(session: aiohttp.ClientSession, method: str, path: str, **kwargs) -> dict:
    url = f"{API_BASE}{path}"
    rate_limited = transient = 0
    while True:
        _check_interrupt()
        try:
            async with session.request(method, url, headers=_headers(), **kwargs) as resp:
                body = await resp.text()
                if resp.status == 429 and rate_limited < RATE_LIMIT_RETRIES:
                    rate_limited += 1
                    wait = float(resp.headers.get("Retry-After") or min(60, 2**rate_limited))
                    log.info("Krea rate limit (429); retrying in %.0fs", wait)
                    await _sleep_interruptible(wait)
                    continue
                if resp.status >= 500 and transient < TRANSIENT_RETRIES:
                    transient += 1
                    await _sleep_interruptible(2**transient)
                    continue
                if resp.status >= 400:
                    raise KreaError(_error_text(resp.status, body))
                return json.loads(body) if body else {}
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            if transient >= TRANSIENT_RETRIES:
                raise KreaError(f"Could not reach Krea ({method} {path}): {e}") from e
            transient += 1
            await _sleep_interruptible(2**transient)


class Uploads:
    """Turns PNG bytes into something Krea accepts: a data URI, or an uploaded asset for big files."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.asset_ids: list[str] = []

    async def image_ref(self, png: bytes, data_uri: str) -> str:
        if len(png) <= MAX_INLINE_BYTES:
            return data_uri
        form = aiohttp.FormData()
        form.add_field("file", png, filename="comfyui.png", content_type="image/png")
        form.add_field("description", "Uploaded by ComfyUI-Krea-API")
        asset = await _request(self.session, "POST", "/assets", data=form)
        if asset.get("id"):
            self.asset_ids.append(asset["id"])
        return asset["image_url"]

    async def cleanup(self) -> None:
        for asset_id in self.asset_ids:
            try:
                await _request(self.session, "DELETE", f"/assets/{asset_id}")
            except KreaError as e:
                log.warning("Could not delete temporary Krea asset %s: %s", asset_id, e)


def _result_urls(job: dict) -> list[str]:
    urls = (job.get("result") or {}).get("urls") or []
    out = []
    for item in urls:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("type", "model") != "preview" and item.get("url"):
            out.append(item["url"])
    return out


async def run_job(
    session: aiohttp.ClientSession,
    path: str,
    payload: dict,
    *,
    node_id: str | None = None,
    timeout: float = 900,
    poll_interval: float = 2.0,
) -> tuple[str, list[str]]:
    job = await _request(session, "POST", path, json=payload)
    job_id = job.get("job_id")
    if not job_id:
        raise KreaError(f"Krea did not return a job id: {json.dumps(job)[:500]}")
    started = time.monotonic()
    status = job.get("status", "queued")
    while True:
        if status == "completed":
            urls = _result_urls(job)
            if not urls:
                raise KreaError(f"Krea job {job_id} completed without output URLs.")
            _progress(node_id, "Krea: downloading")
            return job_id, urls
        if status in ("failed", "cancelled"):
            err = job.get("error") or {}
            code, msg = err.get("code", status), err.get("message", "")
            if code == "content_policy":
                msg = msg or "The generation was rejected by content moderation."
            raise KreaError(f"Krea job {job_id} {status} ({code}): {msg}".rstrip(": "))
        if status not in PENDING:
            raise KreaError(f"Krea job {job_id} has unexpected status {status!r}.")
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            raise KreaError(f"Krea job {job_id} still {status} after {timeout:.0f}s; giving up polling.")
        _progress(node_id, f"Krea: {status} ({elapsed:.0f}s)")
        await _sleep_interruptible(poll_interval)
        job = await _request(session, "GET", f"/jobs/{job_id}")
        status = job.get("status")


async def download(session: aiohttp.ClientSession, url: str) -> bytes:
    for attempt in range(TRANSIENT_RETRIES + 1):
        _check_interrupt()
        try:
            async with session.get(url, headers={"User-Agent": USER_AGENT}) as resp:
                if resp.status in (401, 403) and url.startswith(API_BASE):
                    async with session.get(url, headers=_headers()) as authed:
                        authed.raise_for_status()
                        return await authed.read()
                resp.raise_for_status()
                return await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt >= TRANSIENT_RETRIES:
                raise KreaError(f"Could not download the Krea result {url}: {e}") from e
            await _sleep_interruptible(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def session(total_timeout: float = 300) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=total_timeout, sock_connect=30))
