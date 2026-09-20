"""
OAuth Client ID Metadata Document (CIMD) support.

Implements ``draft-ietf-oauth-client-id-metadata-document-01``: a client
identifies itself with an ``https`` URL as its ``client_id``; the authorization
server fetches that URL to retrieve the client's metadata (the same shape as
RFC 7591 Dynamic Client Registration) and resolves it to an Application. See
``rfcs/draft-ietf-oauth-client-id-metadata-document-01.txt``.

The fetch is an outbound request to a client-controlled URL made inside the
authorization request flow, so the default fetcher is hardened against SSRF
(https only, resolve and validate the target IP once then connect to that same
IP, no redirects, tight timeouts, response size cap) and the resolver bounds
the cost of a flood of bad URLs with a per-URL failure backoff and an in-flight
concurrency cap. See ``docs/cimd.rst`` for the threat model.
"""

import contextlib
import hashlib
import ipaddress
import json
import logging
import re
import socket
import ssl
import threading
import time
from datetime import timedelta
from urllib.parse import urlparse

import urllib3
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.http.request import validate_host
from django.utils import timezone

from .models import get_application_model
from .settings import oauth2_settings


log = logging.getLogger(__name__)

# RFC 7591 grant_type name → DOT AbstractApplication.authorization_grant_type.
# Mirrors GRANT_TYPE_MAP in views/dynamic_client_registration.py; kept as a
# separate subset here because CIMD clients are public (confidential-only grants
# are intentionally absent) and device_code is out of scope for CIMD (its grant
# would also need DeviceGrant.client_id widened to hold a URL).
GRANT_TYPE_MAP = {
    "authorization_code": "authorization-code",
    "implicit": "implicit",
}
# Handled automatically by DOT alongside authorization_code, so it is absent from
# GRANT_TYPE_MAP above and dropped like any other grant this server does not register.
IGNORED_GRANT_TYPES = {"refresh_token"}

# token_endpoint_auth_method values a CIMD registration can be stored with. The
# spec forbids shared-secret methods (no shared secret can be established), and
# the asymmetric ones are not wired to CIMD yet, so "none" is the whole set.
SUPPORTED_AUTH_METHODS = ("none",)

# Cache-freshness lives on the model (cimd_expires_at, durable and authoritative
# per row); the failure backoff is ephemeral/best-effort, so it lives in the
# cache under this prefix.
BACKOFF_CACHE_PREFIX = "oauth2_provider:cimd:backoff:"
MAX_AGE_RE = re.compile(r"max-age\s*=\s*(\d+)", re.IGNORECASE)
# NAT64 well-known prefix (RFC 6052): 64:ff9b::/96 addresses embed an IPv4 in
# their low 32 bits, so they must be decoded before the public-IP check.
NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")

# The in-flight cap is a per-process BoundedSemaphore, rebuilt when the
# configured size changes (e.g. between tests). Across N server processes the
# real ceiling is size × N; see docs/cimd.rst.
_semaphore_lock = threading.Lock()
_semaphore = None
_semaphore_size = None


class CIMDError(Exception):
    """A client ID metadata document could not be resolved.

    The message is safe to log but is never returned to the client: a failed
    resolution simply looks like an unknown client to the OAuth flow.
    """


def is_cimd_client_id(client_id):
    """Return True if *client_id* looks like a CIMD URL.

    Cheap gate so ordinary (non-URL) client_ids skip all CIMD machinery. The
    scheme is matched case-insensitively (RFC 3986 section 3.1) to stay
    consistent with :func:`_validate_client_id_url`, where full validation
    happens.
    """
    return bool(client_id) and client_id[:8].lower() == "https://"


class AllowAllCIMDPermission:
    """Allow any CIMD client_id URL to register (default).

    Registration happens on the pre-auth authorize/token path, where no
    authenticated user exists, so unlike DCR the default is open.

    The interface mirrors DCR's request-first ``has_permission``. *request* is
    the oauthlib request the client_id arrived on; its ``headers`` carry the
    HTTP headers, for e.g. IP-bound policies.
    """

    def has_permission(self, request, client_id) -> bool:
        return True


class HostAllowlistCIMDPermission:
    """Allow only client_id URLs whose host matches ``CIMD_ALLOWED_HOSTS``.

    Entries use Django's ``ALLOWED_HOSTS`` syntax: an exact hostname,
    ``".example.com"`` for a domain and all its subdomains, or ``"*"``.
    An empty allowlist denies every host.
    """

    def has_permission(self, request, client_id) -> bool:
        host = urlparse(client_id).hostname
        return bool(host) and validate_host(host, oauth2_settings.CIMD_ALLOWED_HOSTS or [])


def _registration_permitted(request, client_id):
    """Run all CIMD_REGISTRATION_PERMISSION_CLASSES; return True if all pass.

    Fails closed: an empty setting denies all registration, matching the DCR
    permission-class semantics.
    """
    permission_classes = oauth2_settings.CIMD_REGISTRATION_PERMISSION_CLASSES
    if not permission_classes:
        return False
    return all(cls().has_permission(request, client_id) for cls in permission_classes)


def _validate_client_id_url(client_id):
    """Validate and parse the client_id URL per the CIMD spec (section 3).

    The URL MUST use https, contain a host, a valid port and a path, MUST NOT
    carry a userinfo or fragment component, and MUST NOT contain single-dot or
    double-dot path segments. Raises :class:`CIMDError` otherwise.
    """
    parsed = urlparse(client_id)
    if parsed.scheme != "https":
        raise CIMDError("client_id URL must use the https scheme")
    if not parsed.hostname:
        raise CIMDError("client_id URL must contain a host")
    try:
        _ = parsed.port  # an out-of-range port raises ValueError on access
    except ValueError as exc:
        raise CIMDError("client_id URL has an invalid port") from exc
    if parsed.username or parsed.password:
        raise CIMDError("client_id URL must not contain a userinfo component")
    if parsed.fragment:
        raise CIMDError("client_id URL must not contain a fragment component")
    if not parsed.path:
        raise CIMDError("client_id URL must contain a path component")
    if any(segment in (".", "..") for segment in parsed.path.split("/")):
        raise CIMDError("client_id URL must not contain dot path segments")
    return parsed


def _ip_is_public(ip_str):
    """Return True only for a globally routable address.

    ``ipaddress.is_global`` excludes private, loopback, link-local (including
    the 169.254.169.254 cloud-metadata address), CGNAT, multicast, reserved and
    unspecified ranges. But several IPv6 forms embed an IPv4 address that
    ``is_global`` can misjudge — IPv4-mapped ``::ffff:0:0/96`` (only fixed in
    newer CPython patch levels), 6to4, Teredo ``2001::/32``, and the NAT64
    well-known prefix ``64:ff9b::/96`` — so decode the embedded IPv4 and judge
    that instead. Otherwise a crafted AAAA record could smuggle an internal
    IPv4 (e.g. the cloud-metadata address) past the allowlist.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        teredo = ip.teredo
        if teredo is not None:
            # (server, de-obfuscated client) IPv4 pair; both must be public.
            return all(part.is_global for part in teredo)
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip in NAT64_PREFIX:
            # The low 32 bits of a 64:ff9b::/96 address are the embedded IPv4.
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            ip = embedded
    return ip.is_global


def _resolve_and_validate(hostname, port):
    """Resolve *hostname* and return its validated IPs, or raise CIMDError.

    Every resolved address must be public; if any is internal we refuse the
    whole host rather than cherry-pick a good one, so a split public/private
    result can't smuggle a connection to an internal service.
    """
    try:
        infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise CIMDError(f"could not resolve client_id host: {exc}") from exc
    ips = []
    for info in infos:
        ip = info[4][0]
        if not _ip_is_public(ip):
            raise CIMDError(f"client_id host resolves to a non-public address ({ip})")
        ips.append(ip)
    if not ips:
        raise CIMDError("client_id host did not resolve to any address")
    return ips


def _effective_max_age(cache_control):
    """Derive a cache lifetime (seconds) from a Cache-Control header value.

    Honours ``max-age`` when present, treats ``no-store``/``no-cache`` as the
    configured floor (we still persist the Application for the FK, but re-fetch
    soon), and otherwise defaults to the configured ceiling. The result is
    always clamped to [MIN_AGE, MAX_AGE].
    """
    floor = oauth2_settings.CIMD_METADATA_MIN_AGE_SECONDS
    ceiling = oauth2_settings.CIMD_METADATA_MAX_AGE_SECONDS
    age = ceiling
    if cache_control:
        lowered = cache_control.lower()
        if "no-store" in lowered or "no-cache" in lowered:
            age = floor
        else:
            match = MAX_AGE_RE.search(cache_control)
            if match:
                age = int(match.group(1))
    return max(floor, min(age, ceiling))


class SafeMetadataFetcher:
    """Default SSRF-hardened fetcher for CIMD documents.

    Override with the ``CIMD_METADATA_FETCHER`` setting to route through an
    egress proxy or apply site-specific policy. A fetcher's ``fetch(client_id)``
    must return ``(metadata_dict, max_age_seconds)`` or raise
    :class:`CIMDError`.
    """

    def fetch(self, client_id):
        parsed = _validate_client_id_url(client_id)
        port = parsed.port or 443
        ips = _resolve_and_validate(parsed.hostname, port)

        timeout = oauth2_settings.CIMD_FETCH_TIMEOUT_SECONDS
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        # parsed.netloc is the authority minus userinfo (already rejected), so it
        # carries the port and IPv6 brackets that the Host header needs.
        # The structured-suffix range is advertised because _read_document
        # accepts application/<subtype>+json, and a server honouring Accept
        # strictly could otherwise answer 406.
        headers = {"Host": parsed.netloc, "Accept": "application/json, application/*+json"}
        ssl_context = ssl.create_default_context()

        # One deadline shared across every IP attempt. A hostname can resolve to
        # many public IPs; giving each attempt its own total=timeout would let N
        # addresses hold a worker for N × timeout on the pre-auth path. Each
        # attempt instead gets only the remaining budget, so the whole fetch
        # stays bounded by CIMD_FETCH_TIMEOUT_SECONDS regardless of address count.
        deadline = time.monotonic() + timeout

        last_exc = None
        for ip in ips:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Connect to the validated IP (host=ip) while SNI, certificate
            # verification and the Host header all use the real hostname, so a
            # second DNS lookup can't rebind the connection to another address.
            # total=remaining bounds this attempt (connect + all reads) to the
            # shared deadline, so a slow-drip body can't hold the worker past it.
            pool = urllib3.HTTPSConnectionPool(
                host=ip,
                port=port,
                timeout=urllib3.Timeout(connect=remaining, read=remaining, total=remaining),
                retries=False,
                maxsize=1,
                ssl_context=ssl_context,
                server_hostname=parsed.hostname,
            )
            try:
                response = pool.urlopen(
                    "GET",
                    path,
                    headers=headers,
                    redirect=False,
                    preload_content=False,
                )
                try:
                    return self._read_document(response)
                finally:
                    response.release_conn()
            except urllib3.exceptions.HTTPError as exc:
                last_exc = exc
            finally:
                pool.close()
        raise CIMDError(f"could not fetch client_id document: {last_exc}")

    def _read_document(self, response):
        if response.status != 200:
            raise CIMDError(f"client_id document returned HTTP {response.status}")
        # Accept application/json and the RFC 6839 structured suffix
        # application/<subtype>+json (the spec permits AS-defined JSON types).
        media_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if media_type != "application/json" and not (
            media_type.startswith("application/") and media_type.endswith("+json")
        ):
            raise CIMDError(f"client_id document is not JSON (Content-Type: {media_type!r})")

        max_size = oauth2_settings.CIMD_MAX_DOCUMENT_SIZE
        body = response.read(max_size + 1)
        if len(body) > max_size:
            raise CIMDError("client_id document exceeds the maximum allowed size")

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise CIMDError("client_id document is not valid JSON") from exc
        if not isinstance(data, dict):
            raise CIMDError("client_id document must be a JSON object")

        return data, _effective_max_age(response.headers.get("Cache-Control"))


def _resolve_grant_type(grant_types):
    """Resolve an RFC 7591 grant_types list to a single DOT grant constant.

    RFC 7591 section 2.1 has the authorization server ignore the grant types it does not
    support, so an unsupported entry alongside a supported one is dropped instead of
    failing the whole document. Published clients rely on this: Claude's client metadata
    declares ``jwt-bearer`` next to the ``authorization_code`` its connector actually uses.

    ``Application`` stores a single grant, so one of the supported entries has to win.
    ``authorization_code`` does, because it is the grant a CIMD client is registered to
    run and the only one whose flow this resolver's redirect handling covers.
    """
    supported = [g for g in grant_types if g in GRANT_TYPE_MAP]
    if not supported:
        raise CIMDError("client metadata declares no grant_type this server supports")
    preferred = "authorization_code" if "authorization_code" in supported else supported[0]
    return GRANT_TYPE_MAP[preferred]


def _resolve_auth_method(metadata):
    """Resolve the token_endpoint_auth_method a CIMD registration is stored with.

    ``token_endpoint_auth_method`` carries the client's chosen method. When this
    server supports it, it wins: the spec (section 6.2) has the authorization server
    require client authentication of the registered type, so a client that chose an
    asymmetric method must not be quietly downgraded to a public one.

    When this server does *not* support the chosen method, the document may still
    name others it can use in ``token_endpoint_auth_methods_supported``, a property
    the spec permits (section 4.1: a document MAY define additional properties) and
    published clients rely on: ChatGPT's document at
    https://chatgpt.com/oauth/client.json chooses ``private_key_jwt`` and offers
    ``["none", "private_key_jwt"]``, expecting a server without ``private_key_jwt``
    to register it as the public client it can also be. The first offered method
    this server supports wins, so the client's own ordering decides.

    Shared-secret methods can never be negotiated this way, whatever a document
    offers, because :data:`SUPPORTED_AUTH_METHODS` does not contain any.
    """
    declared = metadata.get("token_endpoint_auth_method", "none")
    if declared in SUPPORTED_AUTH_METHODS:
        return declared

    offered = metadata.get("token_endpoint_auth_methods_supported")
    if isinstance(offered, list):
        for method in offered:
            if method in SUPPORTED_AUTH_METHODS:
                return method

    raise CIMDError(
        f"client metadata declares token_endpoint_auth_method {declared!r} and offers "
        f"{offered!r}; this server registers CIMD clients with {list(SUPPORTED_AUTH_METHODS)}"
    )


def _build_application_kwargs(metadata):
    """Convert a CIMD metadata document to public-Application field kwargs.

    Resolves the client authentication method with :func:`_resolve_auth_method`
    (only ``"none"`` can be stored today), rejects any ``client_secret``
    property, and requires at least one redirect URI. Returns kwargs; raises
    :class:`CIMDError` on invalid metadata.
    """
    # Only "none" can be stored today, so the resolved value is already implied by
    # the public client_type _fetch_validate_upsert sets; resolving is still what
    # rejects a document this server cannot register at all.
    _resolve_auth_method(metadata)
    # Spec: neither property may appear in a CIMD document (presence, not value).
    if "client_secret" in metadata or "client_secret_expires_at" in metadata:
        raise CIMDError("CIMD client metadata must not include client_secret or client_secret_expires_at")

    # Only redirect-based grants are supported, and those require registered
    # redirect URIs (RFC 7591 section 2), so an empty list would persist an
    # unusable Application row.
    redirect_uris = metadata.get("redirect_uris")
    if (
        not isinstance(redirect_uris, list)
        or not redirect_uris
        or not all(isinstance(u, str) for u in redirect_uris)
    ):
        raise CIMDError("redirect_uris must be a non-empty array of strings")

    grant_types = metadata.get("grant_types", ["authorization_code"])
    if not isinstance(grant_types, list) or not all(isinstance(g, str) for g in grant_types):
        raise CIMDError("grant_types must be an array of strings")

    client_name = metadata.get("client_name", "")
    if not isinstance(client_name, str):
        raise CIMDError("client_name must be a string")

    return {
        "name": client_name,
        "redirect_uris": " ".join(redirect_uris),
        "authorization_grant_type": _resolve_grant_type(grant_types),
    }


def _get_fetch_semaphore():
    """Return the in-flight fetch semaphore, or None when the cap is disabled."""
    global _semaphore, _semaphore_size
    size = oauth2_settings.CIMD_MAX_CONCURRENT_FETCHES
    if not size:
        return None
    with _semaphore_lock:
        if _semaphore is None or _semaphore_size != size:
            _semaphore = threading.BoundedSemaphore(size)
            _semaphore_size = size
        return _semaphore


@contextlib.contextmanager
def _fetch_slot():
    """Take an in-flight fetch slot without blocking.

    Yields True when a slot was taken (or the cap is disabled), False when the
    in-flight cap is already full, and releases the slot on exit if one was
    held. Non-blocking so a flood of distinct URLs fails fast rather than
    queuing and tying up workers.
    """
    semaphore = _get_fetch_semaphore()
    acquired = semaphore is None or semaphore.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired and semaphore is not None:
            semaphore.release()


def _fetch_validate_upsert(client_id):
    """Fetch, validate and upsert the Application for a CIMD *client_id*."""
    fetcher = oauth2_settings.CIMD_METADATA_FETCHER()
    metadata, max_age = fetcher.fetch(client_id)

    # Spec: the document's client_id MUST equal the URL it was fetched from.
    # This binds the metadata to its URL, so a document cannot claim to be a
    # different client (and cannot poison another URL's Application row).
    if metadata.get("client_id") != client_id:
        raise CIMDError("document client_id does not match the client_id URL")

    kwargs = _build_application_kwargs(metadata)

    Application = get_application_model()
    try:
        application = Application.objects.get(client_id=client_id)
        if application.registration_source != Application.RegistrationSource.CIMD:
            # A manually provisioned client happens to own this id; never let a
            # fetched document take it over.
            raise CIMDError("client_id URL collides with a non-CIMD application")
    except Application.DoesNotExist:
        application = Application(client_id=client_id)

    application.user = None
    application.client_type = Application.CLIENT_PUBLIC
    application.registration_source = Application.RegistrationSource.CIMD
    application.cimd_expires_at = timezone.now() + timedelta(seconds=max_age)
    for field, value in kwargs.items():
        setattr(application, field, value)

    try:
        # validate_unique=False: client_id is the only unique field and is
        # server-controlled (the validated URL). Letting full_clean pre-check it
        # would, under a concurrent first-sight race, surface the winner's row as
        # a ValidationError on the loser and fail a valid client; instead the DB
        # constraint plus the IntegrityError handler below are the sole arbiter.
        application.full_clean(exclude=["client_secret"], validate_unique=False)
    except ValidationError as exc:
        messages = "; ".join(exc.messages)
        raise CIMDError(f"invalid client metadata: {messages}") from exc

    try:
        application.save()
    except IntegrityError as exc:
        # Concurrent first-sight of the same URL: another request won the race.
        # Re-load the winner and re-apply the non-CIMD collision guard.
        try:
            application = Application.objects.get(client_id=client_id)
        except Application.DoesNotExist:
            raise CIMDError("client_id row vanished during a concurrent upsert") from exc
        if application.registration_source != Application.RegistrationSource.CIMD:
            raise CIMDError("client_id URL collides with a non-CIMD application")
    return application


def _backoff_cache_key(client_id):
    """Return the failure-backoff cache key for *client_id*.

    The client_id (untrusted, up to 255 chars) is hashed so the key can't exceed
    a cache backend's key-length limit (e.g. memcached's 250 bytes) and silently
    drop the backoff — or raise before the URL is ever validated.
    """
    digest = hashlib.sha256(client_id.encode("utf-8")).hexdigest()
    return BACKOFF_CACHE_PREFIX + digest


def resolve_cimd_application(client_id, request):
    """Resolve a CIMD *client_id* URL to a persisted Application, or None.

    Returns None (the caller then treats the client as unknown) when CIMD is
    disabled, the id is not a CIMD URL, registration is refused by the
    permission classes, the URL is in failure backoff, the in-flight cap is
    reached, or the document is missing or invalid. *request* is the oauthlib
    request the client_id arrived on; it is forwarded to the permission
    classes.
    """
    if not oauth2_settings.CIMD_ENABLED or not is_cimd_client_id(client_id):
        return None

    # Policy gate, checked before any fetch. A denial is not a fetch failure,
    # so it does not set the backoff: adding a host to the allowlist takes
    # effect on the very next request.
    if not _registration_permitted(request, client_id):
        log.info("CIMD registration refused by permission classes for %r", client_id)
        return None

    backoff_key = _backoff_cache_key(client_id)
    if cache.get(backoff_key):
        return None

    with _fetch_slot() as acquired:
        if not acquired:
            # Not backed off: this URL may be perfectly fine, just over capacity.
            log.warning("CIMD fetch skipped for %r: in-flight cap reached", client_id)
            return None
        try:
            return _fetch_validate_upsert(client_id)
        except CIMDError as exc:
            log.info("CIMD resolution failed for %r: %r", client_id, exc)
            cache.set(backoff_key, True, oauth2_settings.CIMD_FAILURE_BACKOFF_SECONDS)
            return None
        except Exception:
            # This runs on the pre-auth authorize/token endpoint against a
            # client-controlled URL, so an unexpected error must degrade to
            # "unknown client", never a 500. Back it off to stop cheap repeats.
            log.exception("Unexpected error resolving CIMD client %r", client_id)
            cache.set(backoff_key, True, oauth2_settings.CIMD_FAILURE_BACKOFF_SECONDS)
            return None


def refresh_if_stale(application, request):
    """Re-fetch a CIMD Application's metadata when its cache has expired.

    Returns the refreshed Application, or the original unchanged when it is not
    a CIMD application, is still fresh, or the re-fetch fails. Keeping the last
    good document on failure avoids locking a client out over a transient blip
    (the spec forbids caching an error as the authoritative result).
    """
    if (
        application.registration_source != application.RegistrationSource.CIMD
        or application.cimd_expires_at is None
    ):
        return application
    if timezone.now() <= application.cimd_expires_at:
        return application
    refreshed = resolve_cimd_application(application.client_id, request)
    return refreshed if refreshed is not None else application
