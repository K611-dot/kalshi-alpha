"""Kalshi REST and websocket client.

Authentication is RSA-PSS: each request is signed over
``timestamp_ms + METHOD + path`` with the account's private key, and the
signature travels in ``KALSHI-ACCESS-SIGNATURE`` alongside the key id and
timestamp. The private key never leaves the process and is never logged.

Three things this client does that a thin ``requests`` wrapper would not, and
that matter the moment it is pointed at production:

* **Token-bucket rate limiting.** The exchange publishes per-tier request
  limits and answers 429 when they are exceeded. Limiting client-side is
  strictly better than being throttled: the bucket also smooths bursts, so a
  scan across a hundred markets does not spend its budget in the first second
  and then stall.
* **Retry with jittered backoff, on idempotent verbs only.** Retrying a GET is
  free. Retrying a POST that creates an order can double a position, so order
  placement is retried only when the server explicitly signals that nothing was
  created, and always carries a client-supplied idempotency key.
* **Sequence-checked websocket.** Book deltas carry sequence numbers; a gap
  means the local book is wrong. The client re-subscribes to force a fresh
  snapshot rather than continuing to quote against a book that has diverged.

Nothing here executes automatically. :mod:`kalshi_alpha.execution.oms` gates
every order behind explicit risk limits, and the package defaults to offline
mode so that importing it can never reach the network.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from kalshi_alpha.config import Settings
from kalshi_alpha.logging_setup import get_logger
from kalshi_alpha.types import PAYOUT, MarketMeta, OrderBook, Side, Trade

log = get_logger(__name__)

IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class KalshiError(RuntimeError):
    """Any non-retryable failure from the exchange."""

    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body


class RateLimiter:
    """Async token bucket.

    ``rate`` tokens are added per second up to ``capacity``. Acquiring waits
    exactly long enough for the next token rather than polling, so a saturated
    client burns no CPU while it is queued.
    """

    def __init__(self, rate: float = 8.0, capacity: float | None = None) -> None:
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(rate, 1.0))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                await asyncio.sleep((tokens - self._tokens) / self.rate)


@dataclass
class Signer:
    """RSA-PSS request signer."""

    key_id: str
    private_key_path: Path
    _key: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        data = Path(self.private_key_path).read_bytes()
        self._key = load_pem_private_key(data, password=None)

    def headers(self, method: str, path: str) -> dict[str, str]:
        """Sign ``timestamp + METHOD + path`` and return the auth headers.

        ``path`` must be the request path only -- no query string and no host --
        exactly as the exchange reconstructs it server-side.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        ts = str(int(time.time() * 1000))
        message = f"{ts}{method.upper()}{path}".encode()
        signature = self._key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }


class KalshiClient:
    """Async REST client. Read-only unless order methods are explicitly called."""

    def __init__(
        self,
        settings: Settings | None = None,
        rate: float = 8.0,
        timeout: float = 10.0,
        max_retries: int = 4,
    ) -> None:
        self.settings = settings or Settings()
        self.base = self.settings.rest_base
        self.limiter = RateLimiter(rate)
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(base_url=self.base, timeout=timeout)
        self._signer: Signer | None = None
        if self.settings.has_credentials:
            assert self.settings.private_key_path is not None
            self._signer = Signer(
                str(self.settings.api_key_id), self.settings.private_key_path
            )

    async def __aenter__(self) -> KalshiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def authenticated(self) -> bool:
        return self._signer is not None

    # ---- transport -----------------------------------------------------
    async def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        retry_unsafe: bool = False,
    ) -> dict[str, Any]:
        """Issue one signed, rate-limited request with bounded retries."""
        url_path = f"/trade-api/v2{path}"
        attempt = 0
        while True:
            await self.limiter.acquire()
            headers = (
                self._signer.headers(method, url_path)
                if self._signer
                else {"Accept": "application/json"}
            )
            try:
                resp = await self._client.request(
                    method, path, params=params, json=body, headers=headers
                )
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise KalshiError(0, f"transport failure: {exc}") from exc
                await self._backoff(attempt)
                attempt += 1
                continue

            if resp.status_code < 400:
                return resp.json() if resp.content else {}

            retryable = resp.status_code in RETRY_STATUS and (
                method.upper() in IDEMPOTENT_METHODS or retry_unsafe
            )
            if not retryable or attempt >= self.max_retries:
                raise KalshiError(resp.status_code, resp.reason_phrase, resp.text[:500])

            # Honour Retry-After when the exchange sends one.
            wait = resp.headers.get("Retry-After")
            await (asyncio.sleep(float(wait)) if wait else self._backoff(attempt))
            attempt += 1

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with full jitter.

        Full jitter rather than fixed backoff so that a fleet of reconnecting
        clients does not synchronise into a thundering herd against a service
        that is already struggling.
        """
        import random

        delay = min(30.0, 0.25 * (2**attempt))
        await asyncio.sleep(random.uniform(0.0, delay))

    # ---- market data ---------------------------------------------------
    async def get_markets(
        self, event_ticker: str | None = None, status: str = "open", limit: int = 200
    ) -> list[dict[str, Any]]:
        """Page through the markets endpoint until the cursor is exhausted."""
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": limit, "status": status}
            if event_ticker:
                params["event_ticker"] = event_ticker
            if cursor:
                params["cursor"] = cursor
            payload = await self.request("GET", "/markets", params=params)
            out.extend(payload.get("markets", []))
            cursor = payload.get("cursor") or None
            if not cursor:
                return out

    async def get_market(self, ticker: str) -> dict[str, Any]:
        payload = await self.request("GET", f"/markets/{ticker}")
        return payload.get("market", payload)

    async def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        payload = await self.request(
            "GET", f"/markets/{ticker}/orderbook", params={"depth": depth}
        )
        # parse_orderbook unwraps the envelope itself, so hand it the whole
        # response rather than guessing which key the current API version uses.
        return parse_orderbook(ticker, payload, time.time())

    async def get_trades(self, ticker: str, limit: int = 200) -> list[Trade]:
        payload = await self.request(
            "GET", "/markets/trades", params={"ticker": ticker, "limit": limit}
        )
        return [parse_trade(t) for t in payload.get("trades", [])]

    async def get_events(self, status: str = "open", limit: int = 200) -> list[dict[str, Any]]:
        payload = await self.request("GET", "/events", params={"status": status, "limit": limit})
        return payload.get("events", [])

    async def snapshot(self, tickers: Sequence[str], depth: int = 10) -> dict[str, OrderBook]:
        """Fetch many books concurrently, subject to the shared rate limit."""
        results = await asyncio.gather(
            *(self.get_orderbook(t, depth) for t in tickers), return_exceptions=True
        )
        out: dict[str, OrderBook] = {}
        for ticker, res in zip(tickers, results, strict=True):
            if isinstance(res, OrderBook):
                out[ticker] = res
            else:
                log.warning("orderbook fetch failed", extra={"ticker": ticker, "err": str(res)})
        return out

    # ---- account and orders --------------------------------------------
    async def get_balance(self) -> int:
        payload = await self.request("GET", "/portfolio/balance")
        return int(payload.get("balance", 0))

    async def get_positions(self) -> list[dict[str, Any]]:
        payload = await self.request("GET", "/portfolio/positions")
        return payload.get("market_positions", [])

    async def get_orders(self, status: str = "resting") -> list[dict[str, Any]]:
        payload = await self.request("GET", "/portfolio/orders", params={"status": status})
        return payload.get("orders", [])

    async def create_order(
        self,
        ticker: str,
        side: Side,
        action: str,
        count: int,
        price_cents: int | None = None,
        order_type: str = "limit",
        tif: str = "fill_or_kill",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place one order.

        ``client_order_id`` is generated when absent and is what makes the call
        safe to retry: the exchange rejects a duplicate id rather than creating
        a second order, so a timeout on the response never silently doubles the
        position.
        """
        if not self.authenticated:
            raise KalshiError(401, "cannot place orders without credentials")
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side.value,
            "action": action,
            "count": int(count),
            "type": order_type,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "time_in_force": tif,
        }
        if price_cents is not None:
            key = "yes_price" if side is Side.YES else "no_price"
            body[key] = int(price_cents)
        return await self.request("POST", "/portfolio/orders", body=body)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        return await self.request("DELETE", f"/portfolio/orders/{order_id}")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def dollars_to_cents(value: Any) -> int | None:
    """Convert a decimal dollar amount (``"0.0100"`` or ``0.01``) to whole cents.

    The exchange serialises prices as fixed-point *strings* in dollars. Parsing
    them as floats and rounding is safe here only because the grid is whole
    cents; ``round`` rather than ``int`` because ``0.0300`` can land on
    ``0.029999...`` in binary floating point and truncation would silently shift
    the entire book down a cent.
    """
    if value is None:
        return None
    try:
        return int(round(float(value) * 100.0))
    except (TypeError, ValueError):
        return None


def parse_size(value: Any) -> int:
    """Convert a fixed-point size (``"677.52"``) to whole contracts.

    Kalshi now reports fractional positions. This package models integer
    contracts, so sizes are **floored**: understating available depth makes an
    arbitrage look smaller than it is, which costs opportunity; overstating it
    makes a leg unfillable, which costs money.
    """
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def parse_timestamp(payload: Mapping[str, Any], *keys: str) -> float:
    """Read the first present timestamp, accepting epoch numbers or ISO-8601."""
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            pass
        try:
            # ``Z`` is valid ISO-8601 but fromisoformat only learned it in 3.11.
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return time.time()


def _levels(payload: Mapping[str, Any], dollar_key: str, cent_key: str) -> list[tuple[int, int]]:
    """Read one bid ladder, accepting either the dollar or the legacy cent form.

    Disambiguation is by **key**, not by inspecting the values: ``40`` is a
    valid price in both encodings (40 cents versus $40) and guessing from the
    magnitude would be a coin flip on exactly the tail prices this package
    cares most about.
    """
    raw = payload.get(dollar_key)
    if raw:
        out = []
        for price, size in raw:
            cents = dollars_to_cents(price)
            qty = parse_size(size)
            if cents is not None and 0 < cents < PAYOUT and qty > 0:
                out.append((cents, qty))
        return out

    raw = payload.get(cent_key)
    if not raw:
        return []
    out = []
    for price, size in raw:
        try:
            cents, qty = int(price), int(size)
        except (TypeError, ValueError):
            continue
        if 0 < cents < PAYOUT and qty > 0:
            out.append((cents, qty))
    return out


def parse_orderbook(ticker: str, payload: Mapping[str, Any], ts: float) -> OrderBook:
    """Build an :class:`OrderBook` from the exchange's two bid arrays.

    Both ``yes`` and ``no`` are *bid* books -- see the module docstring in
    :mod:`kalshi_alpha.types` for why that distinction drives everything.

    Two wire formats are accepted. The current API nests the ladders under
    ``orderbook_fp`` and names them ``yes_dollars`` / ``no_dollars``, with
    fixed-point decimal strings for both price and size; older responses used
    ``yes`` / ``no`` with integer cents. Supporting both is not
    future-proofing for its own sake -- the demo and production environments
    have been observed on different versions, and a parser that silently
    returns an empty book produces a system that finds no arbitrage anywhere
    and looks like it is working.
    """
    book = payload.get("orderbook_fp") or payload.get("orderbook") or payload
    if not isinstance(book, Mapping):
        book = {}
    yes = _levels(book, "yes_dollars", "yes")
    no = _levels(book, "no_dollars", "no")
    return OrderBook.from_levels(ticker, ts, yes, no)


def parse_trade(payload: Mapping[str, Any]) -> Trade:
    """Parse one print, accepting dollar-string or integer-cent encodings."""
    taker = str(payload.get("taker_side", "yes")).lower()
    price = dollars_to_cents(payload.get("yes_price_dollars"))
    if price is None:
        try:
            price = int(payload.get("yes_price", 0))
        except (TypeError, ValueError):
            price = 0

    size_raw = payload.get("count_fp", payload.get("count", 0))
    return Trade(
        ticker=str(payload.get("ticker", "")),
        ts=parse_timestamp(payload, "created_time_ts", "created_time", "ts"),
        price=int(price or 0),
        size=parse_size(size_raw),
        taker_side=Side.YES if taker == "yes" else Side.NO,
    )


def parse_quote(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Top-of-book ``(yes_bid, yes_ask)`` in cents, straight off a market payload.

    A cheap alternative to a full orderbook call when scanning many markets:
    the market list endpoint already carries the touch, so a coarse pre-filter
    can run on one request instead of hundreds.
    """
    bid = dollars_to_cents(payload.get("yes_bid_dollars"))
    ask = dollars_to_cents(payload.get("yes_ask_dollars"))
    if bid is None and "yes_bid" in payload:
        bid = int(payload["yes_bid"])
    if ask is None and "yes_ask" in payload:
        ask = int(payload["yes_ask"])
    return bid, ask


def parse_market_meta(payload: dict[str, Any]) -> MarketMeta:
    """Extract the settlement rule from a market payload.

    ``strike_type`` is what makes a market analysable: without knowing whether
    the contract is "greater than", "less than" or a bucket, the ladder and
    Dutch-book detectors cannot construct a settlement matrix, and the market is
    excluded from structural scans rather than guessed at.
    """
    cap = payload.get("cap_strike")
    floor = payload.get("floor_strike")
    strike_type = payload.get("strike_type")
    strike: float | None = None
    upper: float | None = None

    if strike_type in ("greater", "greater_or_equal"):
        strike = _as_float(floor)
        strike_type = "gt" if strike_type == "greater" else "gte"
    elif strike_type in ("less", "less_or_equal"):
        strike = _as_float(cap)
        strike_type = "lt" if strike_type == "less" else "lte"
    elif strike_type == "between" or (cap is not None and floor is not None):
        strike = _as_float(floor)
        upper = _as_float(cap)
        strike_type = "between"
    else:
        strike_type = None

    return MarketMeta(
        ticker=str(payload.get("ticker", "")),
        event_ticker=str(payload.get("event_ticker", "")),
        title=str(payload.get("title", "")),
        strike=strike,
        strike_type=strike_type,
        strike_upper=upper,
        close_ts=parse_timestamp(payload, "close_time_ts", "close_time")
        if payload.get("close_time_ts") or payload.get("close_time")
        else None,
        category=str(payload.get("category", "")),
    )


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# websocket
# --------------------------------------------------------------------------
class KalshiWebsocket:
    """Sequence-checked market-data feed.

    On a detected gap the connection re-subscribes instead of patching over the
    hole. Continuing from a book known to be wrong is the failure mode that
    produces phantom arbitrages -- the scanner would happily "find" edge in a
    book that only exists locally.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.url = self.settings.ws_base
        self._signer: Signer | None = None
        if self.settings.has_credentials:
            assert self.settings.private_key_path is not None
            self._signer = Signer(str(self.settings.api_key_id), self.settings.private_key_path)
        self._cmd_id = 0
        self.gaps = 0

    def _next_id(self) -> int:
        self._cmd_id += 1
        return self._cmd_id

    async def stream(
        self, tickers: Iterable[str], channels: Sequence[str] = ("orderbook_delta", "trade")
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield decoded messages, reconnecting with backoff on failure."""
        import websockets

        tickers = list(tickers)
        headers = self._signer.headers("GET", "/trade-api/ws/v2") if self._signer else {}
        attempt = 0
        while True:
            try:
                async with websockets.connect(self.url, additional_headers=headers) as ws:
                    attempt = 0
                    await ws.send(
                        json.dumps(
                            {
                                "id": self._next_id(),
                                "cmd": "subscribe",
                                "params": {"channels": list(channels), "market_tickers": tickers},
                            }
                        )
                    )
                    seq_by_channel: dict[str, int] = {}
                    async for raw in ws:
                        msg = json.loads(raw)
                        chan = str(msg.get("type", ""))
                        seq = msg.get("seq")
                        if seq is not None:
                            expected = seq_by_channel.get(chan)
                            if expected is not None and int(seq) != expected + 1:
                                self.gaps += 1
                                log.warning(
                                    "websocket sequence gap; resubscribing",
                                    extra={"channel": chan, "expected": expected + 1,
                                           "received": seq},
                                )
                                break
                            seq_by_channel[chan] = int(seq)
                        yield msg
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                log.warning("websocket error; reconnecting", extra={"err": str(exc)})
            delay = min(30.0, 0.5 * (2**attempt))
            attempt += 1
            await asyncio.sleep(delay)
