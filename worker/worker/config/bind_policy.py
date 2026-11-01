"""Bind-address policy for every worker listener (Health, StageExecution).

Neither service is authenticated, so (mirrors the orchestrator's
config.CheckBindAddress):
  - loopback (127.0.0.1, ::1, localhost) and private / link-local
    addresses (10/8, 172.16/12, 192.168/16, fc00::/7, ...) are allowed;
  - 0.0.0.0 / :: (every interface, public ones included) only when the
    caller allows it, and logged as a warning;
  - globally routable (public) addresses and hostnames are refused.
"""

from __future__ import annotations

import ipaddress
import logging

logger = logging.getLogger("worker")


class BindAddressError(ValueError):
    pass


def classify(host: str) -> str:
    """'loopback' | 'private' | 'all-interfaces' | 'public' | 'hostname'."""
    if host == "localhost":
        return "loopback"
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return "hostname"
    if ip.is_unspecified:
        return "all-interfaces"
    if ip.is_loopback:
        return "loopback"
    if ip.is_private or ip.is_link_local:
        return "private"
    return "public"


def check_bind_host(host: str, allow_all_interfaces: bool = False, setting: str = "execution_listen_host",
                    allow_flag: str = "execution_allow_all_interfaces") -> None:
    kind = classify(host)
    if kind == "hostname":
        raise BindAddressError(f"{setting} {host!r} must be an IP address (or 'localhost'), so its scope can be checked")
    if kind == "all-interfaces":
        if not allow_all_interfaces:
            raise BindAddressError(
                f"{setting} {host!r} listens on EVERY interface, including any public one; set it to this machine's "
                f"private LAN IP, or set {allow_flag}: true if a firewall restricts it"
            )
        logger.warning("%s %s listens on ALL interfaces: the service is unauthenticated; make sure a firewall limits "
                       "it to the private LAN", setting, host)
        return
    if kind == "public":
        raise BindAddressError(f"{setting} {host!r} is a public address; unauthenticated services must only listen on "
                               "loopback or a private network")
