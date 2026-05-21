#!/usr/bin/env python3
"""
CLI entry point for reMarkable MCP Server.

Usage:
    # As MCP server (default, uses cloud API)
    remarkable-mcp

    # Use SSH transport (direct connection via USB)
    remarkable-mcp --ssh

    # Convert one-time code to token (run once)
    remarkable-mcp --register <one-time-code>
"""

import argparse
import glob
import json
import os
import sys


def _ensure_macos_cairo_loadable() -> None:
    """Make libcairo discoverable on macOS before cairosvg is imported.

    ``remarkable_image`` renders pages with cairosvg, which dlopen()s libcairo
    at runtime. On macOS, Homebrew installs libcairo under /opt/homebrew/lib
    (Apple Silicon) or /usr/local/lib (Intel) -- neither is on dyld's default
    search path. Setting DYLD_LIBRARY_PATH fixes it, but launchers like the
    1Password ``op`` CLI are hardened binaries, and macOS strips DYLD_* from the
    environment when they exec a child -- so a DYLD_LIBRARY_PATH set in mcp.json
    never reaches us. Detect that, set the path ourselves, and re-exec once so
    dyld picks it up at launch (DYLD_* is only read when the process starts).

    Set REMARKABLE_CAIRO_LIB_DIR to override the library directory.
    """
    if sys.platform != "darwin":
        return
    # Re-exec only once -- guard against an exec loop.
    if os.environ.get("REMARKABLE_DYLD_REEXEC") == "1":
        return

    candidates = []
    override = os.environ.get("REMARKABLE_CAIRO_LIB_DIR")
    if override:
        candidates.append(override)
    candidates += ["/opt/homebrew/lib", "/usr/local/lib"]

    lib_dir = next(
        (d for d in candidates if glob.glob(os.path.join(d, "libcairo*.dylib"))),
        None,
    )
    if lib_dir is None:
        # Homebrew cairo isn't installed where we expect; nothing to do. The
        # image tool will surface a clear error if cairosvg can't load.
        return

    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if lib_dir in existing.split(os.pathsep):
        # Already discoverable (e.g. launched without a hardened wrapper).
        return

    os.environ["DYLD_LIBRARY_PATH"] = os.pathsep.join(p for p in (lib_dir, existing) if p)
    os.environ["REMARKABLE_DYLD_REEXEC"] = "1"
    # execv replaces this process image, preserving the stdio fds the MCP client
    # is connected to, and inherits the env we just set.
    os.execv(sys.executable, [sys.executable, __file__, *sys.argv[1:]])


def main():
    """Main entry point - handle CLI args or run MCP server."""
    _ensure_macos_cairo_loadable()
    parser = argparse.ArgumentParser(
        description="reMarkable MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register and get token (run once)
  uvx remarkable-mcp --register abcd1234

  # Run as MCP server (cloud API)
  uvx remarkable-mcp

  # Run with token from environment
  REMARKABLE_TOKEN="your-token" uvx remarkable-mcp

  # Run with USB web interface
  uvx remarkable-mcp --usb

  # Run with SSH transport (direct USB connection, requires dev mode)
  uvx remarkable-mcp --ssh

  # SSH with custom host (e.g., using SSH config)
  REMARKABLE_SSH_HOST="remarkable" uvx remarkable-mcp --ssh

  # Serve over HTTP for network access (standalone, always-on endpoint)
  # Clients connect to http://<host>:8000/mcp instead of spawning a process
  uvx remarkable-mcp --ssh --http --host 0.0.0.0 --port 8000

HTTP Transport Environment Variables (defaults for --host / --port):
  REMARKABLE_HTTP_HOST     HTTP bind host (default: 127.0.0.1)
  REMARKABLE_HTTP_PORT     HTTP port (default: 8000)

USB Web Interface Environment Variables:
  REMARKABLE_USB_HOST      USB web interface host (default: http://10.11.99.1)
  REMARKABLE_USB_TIMEOUT   Request timeout in seconds (default: 10)

SSH Environment Variables:
  REMARKABLE_SSH_HOST      SSH host (default: 10.11.99.1 for USB)
  REMARKABLE_SSH_USER      SSH user (default: root)
  REMARKABLE_SSH_PORT      SSH port (default: 22)
  REMARKABLE_SSH_PASSWORD  SSH password (optional, requires sshpass)

Security Note:
  For better security, set up SSH key authentication instead of using
  a password. See: https://github.com/SamMorrowDrums/remarkable-mcp/blob/main/docs/ssh-setup.md
""",
    )
    parser.add_argument(
        "--register",
        metavar="CODE",
        help="Register with reMarkable using a one-time code and print the token",
    )
    parser.add_argument(
        "--ssh",
        action="store_true",
        help="Use SSH transport instead of cloud API (requires developer mode)",
    )
    parser.add_argument(
        "--usb",
        action="store_true",
        help="Use USB web interface (connect via USB cable, enable in Storage Settings)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over HTTP (streamable-http) at /mcp instead of stdio, for network access",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("REMARKABLE_HTTP_HOST", "127.0.0.1"),
        help="HTTP bind host (default: 127.0.0.1; use 0.0.0.0 for LAN/mesh access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("REMARKABLE_HTTP_PORT", "8000")),
        help="HTTP port (default: 8000)",
    )

    args = parser.parse_args()

    if args.register:
        # Registration mode - convert one-time code to token
        # Only import what's needed for registration
        from remarkable_mcp.api import register_and_get_token

        try:
            print(f"Registering with reMarkable using code: {args.register}")
            token = register_and_get_token(args.register)
            print("\n✅ Successfully registered!\n")
            print("Your token (add to mcp.json env):")
            print("-" * 50)
            print(token)
            print("-" * 50)
            print("\nAdd to your .vscode/mcp.json:")
            print(
                json.dumps(
                    {
                        "servers": {
                            "remarkable": {
                                "command": "uvx",
                                "args": ["remarkable-mcp"],
                                "env": {"REMARKABLE_TOKEN": token},
                            }
                        }
                    },
                    indent=2,
                )
            )
        except Exception as e:
            print(f"❌ Registration failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Server mode. Connection mode (cloud / USB web / SSH) is selected via
        # env vars; transport (stdio / HTTP) is orthogonal and passed to run().
        if args.usb:
            os.environ["REMARKABLE_USE_USB_WEB"] = "1"
        elif args.ssh:
            os.environ["REMARKABLE_USE_SSH"] = "1"

        from remarkable_mcp.server import run

        run(http=args.http, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
