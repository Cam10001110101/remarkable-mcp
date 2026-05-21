"""
reMarkable SSH Client

Direct access to reMarkable tablet via SSH when connected over USB.
Default connection: root@10.11.99.1 (USB connection)

The tablet stores documents at:
/home/root/.local/share/remarkable/xochitl/

Each document is a folder with:
- {uuid}.metadata - JSON with visibleName, type, parent, etc.
- {uuid}.content - JSON with file info
- {uuid}/ - folder with .rm files (pages), .pdf, etc.
"""

import io
import json
import logging
import os
import shlex
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default SSH settings for USB connection
DEFAULT_SSH_HOST = "10.11.99.1"
DEFAULT_SSH_USER = "root"
DEFAULT_SSH_PORT = 22

# Document storage path on the tablet
XOCHITL_PATH = "/home/root/.local/share/remarkable/xochitl"


@dataclass
class Document:
    """Represents a document or folder on the reMarkable tablet."""

    id: str
    hash: str
    name: str
    doc_type: str  # "DocumentType" or "CollectionType"
    parent: str = ""
    deleted: bool = False
    pinned: bool = False
    synced: bool = True  # False means cloud-archived (not on device)
    last_modified: Optional[datetime] = None
    size: int = 0
    files: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    # SSH-specific: local path to the document folder
    local_path: Optional[str] = None

    @property
    def is_folder(self) -> bool:
        return self.doc_type == "CollectionType"

    @property
    def is_cloud_archived(self) -> bool:
        """True if document is archived to cloud (not on device)."""
        return not self.synced or self.parent == "trash"

    @property
    def VissibleName(self) -> str:
        """Compatibility with cloud client naming."""
        return self.name

    @property
    def ID(self) -> str:
        """Compatibility with cloud client naming."""
        return self.id

    @property
    def Parent(self) -> str:
        """Compatibility with cloud client naming."""
        return self.parent

    @property
    def Type(self) -> str:
        """Compatibility with cloud client naming."""
        return self.doc_type

    @property
    def ModifiedClient(self) -> Optional[datetime]:
        """Compatibility with cloud client naming."""
        return self.last_modified


# Alias for compatibility
Folder = Document


class SSHClient:
    """Client for accessing reMarkable tablet via SSH."""

    def __init__(
        self,
        host: str = DEFAULT_SSH_HOST,
        user: str = DEFAULT_SSH_USER,
        port: int = DEFAULT_SSH_PORT,
        password: Optional[str] = None,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self._documents: List[Document] = []
        self._documents_by_id: Dict[str, Document] = {}

    def _ssh_command(self, command: str, timeout: int = 30) -> str:
        """Execute a command on the tablet via SSH."""
        ssh_args = [
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(self.port),
            f"{self.user}@{self.host}",
            command,
        ]

        # If no password, use BatchMode for key-based auth
        if not self.password:
            ssh_args.insert(1, "-o")
            ssh_args.insert(2, "BatchMode=yes")
        else:
            # Use sshpass for password authentication
            ssh_args = ["sshpass", "-p", self.password] + ssh_args

        try:
            result = subprocess.run(
                ssh_args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(f"SSH command failed: {result.stderr}")
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SSH command timed out after {timeout}s")
        except FileNotFoundError as e:
            if self.password and "sshpass" in str(e):
                raise RuntimeError(
                    "sshpass not found. Install it with: "
                    "apt install sshpass (Debian/Ubuntu), "
                    "brew install hudochenkov/sshpass/sshpass (macOS), "
                    "or set up SSH key authentication instead."
                )
            raise RuntimeError("SSH client not found. Install openssh-client.")

    def _scp_download(self, remote_path: str, timeout: int = 60) -> bytes:
        """Download a file from the tablet via SSH cat (more reliable than SCP)."""
        # Use SSH + cat instead of SCP for binary file transfer
        # This avoids issues with /dev/stdout on various platforms
        ssh_args = [
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(self.port),
            f"{self.user}@{self.host}",
            f"cat '{remote_path}'",
        ]

        # If no password, use BatchMode for key-based auth
        if not self.password:
            ssh_args.insert(1, "-o")
            ssh_args.insert(2, "BatchMode=yes")
        else:
            # Use sshpass for password authentication
            ssh_args = ["sshpass", "-p", self.password] + ssh_args

        try:
            result = subprocess.run(
                ssh_args,
                capture_output=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(f"SSH cat failed: {result.stderr.decode()}")
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SSH cat timed out after {timeout}s")

    def _ssh_pipe(self, data: bytes, remote_command: str, timeout: int = 120) -> None:
        """Run a remote command with `data` piped to its stdin.

        This is how all writes reach the tablet. Streaming over stdin (vs.
        embedding the payload in the command string) avoids ARG_MAX limits and,
        critically, any dependency on a `base64` binary — the reMarkable's
        minimal userland ships `cat`/`mv` but NOT `base64`.
        """
        ssh_args = [
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(self.port),
            f"{self.user}@{self.host}",
            remote_command,
        ]

        # If no password, use BatchMode for key-based auth
        if not self.password:
            ssh_args.insert(1, "-o")
            ssh_args.insert(2, "BatchMode=yes")
        else:
            # Use sshpass for password authentication
            ssh_args = ["sshpass", "-p", self.password] + ssh_args

        try:
            result = subprocess.run(
                ssh_args,
                input=data,
                capture_output=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(f"SSH pipe failed: {result.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SSH pipe timed out after {timeout}s")

    def _scp_upload(self, data: bytes, remote_path: str, timeout: int = 120) -> None:
        """Upload raw bytes to a remote path via SSH (`cat > file` over stdin)."""
        self._ssh_pipe(data, f"cat > {shlex.quote(remote_path)}", timeout=timeout)

    def check_connection(self) -> bool:
        """Check if SSH connection to tablet is available."""
        try:
            self._ssh_command("echo ok", timeout=5)
            return True
        except Exception as e:
            logger.debug(f"SSH connection check failed: {e}")
            return False

    def get_meta_items(self, limit: Optional[int] = None) -> List[Document]:
        """
        Fetch documents and folders from the tablet via SSH.

        Args:
            limit: Maximum number of documents to fetch. If None, fetches all.

        Returns a list of Document objects.
        """
        # Return cached documents if available and no limit specified
        if self._documents and limit is None:
            return self._documents

        # If we have cached docs and limit is within cache, return slice
        if self._documents and limit is not None and len(self._documents) >= limit:
            return self._documents[:limit]

        # Read all metadata files in a single SSH command for efficiency
        # Output format: filename<TAB>content (JSON)
        try:
            # Use a single command to read all metadata files at once
            # This is MUCH faster than individual cat commands
            output = self._ssh_command(
                f"for f in {XOCHITL_PATH}/*.metadata; do "
                f'echo "===FILE===$(basename $f .metadata)"; cat "$f" 2>/dev/null; '
                f"done",
                timeout=60,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata: {e}")

        documents = []

        # Parse the output by splitting on the marker STRING rather than
        # requiring it at the start of a line. A metadata file written without
        # a trailing newline makes its closing brace abut the next marker
        # (`}===FILE===nextid`); a line-based parser would fold that marker into
        # the previous file's content and drop both files. Splitting on the
        # marker itself stays correct regardless of trailing newlines.
        for chunk in output.split("===FILE==="):
            chunk = chunk.strip()
            if not chunk:
                continue
            # First line of the chunk is the document id; the rest is its JSON.
            newline = chunk.find("\n")
            if newline == -1:
                continue
            doc_id = chunk[:newline].strip()
            content = chunk[newline + 1 :]
            if not doc_id:
                continue
            self._parse_and_add_document(doc_id, content, documents, limit)
            if limit is not None and len(documents) >= limit:
                break

        self._documents = documents
        self._documents_by_id = {d.id: d for d in documents}

        logger.info(f"Loaded {len(documents)} documents via SSH")
        return documents

    def _parse_and_add_document(
        self,
        doc_id: str,
        content: str,
        documents: List[Document],
        limit: Optional[int],
    ) -> None:
        """Parse metadata JSON and add document to list."""
        if limit is not None and len(documents) >= limit:
            return

        try:
            metadata = json.loads(content.strip())

            # Skip deleted documents
            if metadata.get("deleted", False):
                return

            # Parse last modified timestamp
            last_modified = None
            if "lastModified" in metadata:
                try:
                    ts = int(metadata["lastModified"]) / 1000
                    last_modified = datetime.fromtimestamp(ts)
                except (ValueError, TypeError):
                    pass

            doc = Document(
                id=doc_id,
                hash=doc_id,  # Use ID as hash for SSH
                name=metadata.get("visibleName", doc_id),
                doc_type=metadata.get("type", "DocumentType"),
                parent=metadata.get("parent", ""),
                deleted=metadata.get("deleted", False),
                pinned=metadata.get("pinned", False),
                synced=metadata.get("synced", True),
                last_modified=last_modified,
                size=0,
                tags=metadata.get("tags", []),
                local_path=f"{XOCHITL_PATH}/{doc_id}",
            )

            documents.append(doc)

        except Exception as e:
            logger.debug(f"Failed to parse metadata for {doc_id}: {e}")

    def get_doc(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        if not self._documents_by_id:
            self.get_meta_items()
        return self._documents_by_id.get(doc_id)

    def download(self, doc: Document) -> bytes:
        """
        Download a document's content as a zip file.

        Creates a zip archive with the same structure as the cloud API.
        """
        doc_path = f"{XOCHITL_PATH}/{doc.id}"

        # List files in the document folder
        try:
            output = self._ssh_command(f"find '{doc_path}' -type f 2>/dev/null || true")
        except Exception:
            output = ""

        file_list = [f.strip() for f in output.strip().split("\n") if f.strip()]

        # Also include the .content file if it exists
        content_file = f"{XOCHITL_PATH}/{doc.id}.content"
        try:
            self._ssh_command(f"test -f '{content_file}' && echo exists")
            file_list.append(content_file)
        except Exception:
            pass

        # Create zip archive
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for remote_path in file_list:
                try:
                    content = self._scp_download(remote_path)
                    # Use relative path in zip
                    rel_path = os.path.basename(remote_path)
                    if "/" in remote_path.replace(f"{XOCHITL_PATH}/{doc.id}", ""):
                        # Preserve subdirectory structure
                        rel_path = remote_path.replace(f"{XOCHITL_PATH}/{doc.id}/", "")
                    zf.writestr(rel_path, content)
                except Exception as e:
                    logger.debug(f"Failed to download {remote_path}: {e}")
                    continue

        zip_buffer.seek(0)
        return zip_buffer.read()

    def download_raw_file(self, doc: Document, extension: str) -> Optional[bytes]:
        """
        Download a raw file (PDF or EPUB) for a document.

        Args:
            doc: The document to download
            extension: File extension without dot (e.g., 'pdf', 'epub')

        Returns:
            Raw file bytes, or None if file doesn't exist
        """
        file_path = f"{XOCHITL_PATH}/{doc.id}.{extension}"

        try:
            # Check if file exists first
            self._ssh_command(f"test -f '{file_path}'", timeout=5)
            # Download the file
            return self._scp_download(file_path, timeout=120)
        except Exception as e:
            logger.debug(f"Raw file not found: {file_path}: {e}")
            return None

    def get_file_type(self, doc: Document) -> Optional[str]:
        """
        Get the file type (pdf, epub, etc.) for a document.

        Returns the extension without dot, or None if not a file-based document.
        """
        # Check cache first
        if hasattr(self, "_file_type_cache") and doc.id in self._file_type_cache:
            return self._file_type_cache[doc.id]

        content_file = f"{XOCHITL_PATH}/{doc.id}.content"

        try:
            content = self._scp_download(content_file, timeout=10)
            data = json.loads(content.decode("utf-8"))
            return data.get("fileType")
        except Exception:
            return None

    # =========================================================================
    # Write operations (delete + move) — SSH-only.
    # The reMarkable tablet's USB Web Interface does not expose move or delete
    # endpoints, so these are implemented by editing .metadata files directly
    # on the tablet's filesystem and restarting xochitl to pick up the change.
    # =========================================================================

    def create_folder(self, name: str, parent_id: str = "") -> Dict[str, Any]:
        """
        Create a new folder on the tablet via SSH.

        Writes a {uuid}.metadata file directly to xochitl's storage directory
        and restarts xochitl. This is the only way to create real folders on
        the device — the USB Web Interface does not expose a folder endpoint.

        Args:
            name: Folder name as it will appear on the tablet.
            parent_id: Parent folder UUID. Empty string places at root.

        Returns:
            Dict {"id": <new-uuid>, "name": ..., "parent": ..., "transport": "ssh"}.
        """
        import uuid as _uuid

        folder_id = str(_uuid.uuid4())
        now_ms = str(int(time.time() * 1000))
        metadata = {
            "createdTime": now_ms,
            "lastModified": now_ms,
            "lastOpened": now_ms,
            "lastOpenedPage": 0,
            "metadatamodified": False,
            "modified": False,
            "new": True,
            "parent": parent_id,
            "pinned": False,
            # On-device item: must NOT be synced=False, or is_cloud_archived
            # treats it as "not on device" and browse/read/recent hide it.
            "synced": True,
            "type": "CollectionType",
            "version": 0,
            "visibleName": name,
        }
        self._write_metadata(folder_id, metadata)
        self._restart_xochitl()
        self._documents = []
        self._documents_by_id = {}
        return {
            "id": folder_id,
            "name": name,
            "parent": parent_id,
            "transport": "ssh",
        }

    def _read_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Read a document's .metadata JSON from the tablet."""
        path = f"{XOCHITL_PATH}/{doc_id}.metadata"
        raw = self._ssh_command(f"cat {shlex.quote(path)}", timeout=15)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Could not parse metadata for {doc_id}: {e}")

    def _write_remote_json(self, remote_path: str, obj: Dict[str, Any]) -> None:
        """Atomically write a JSON file to the tablet (tempfile + mv).

        Used for both .metadata and .content. The payload streams over stdin
        (no `base64` — the tablet has none); writing to a tempfile then mv'ing
        means xochitl never observes a half-written file. Streaming also sides
        steps any shell quoting issues with arbitrary document names.
        """
        # Trailing newline keeps the file well-formed like xochitl's own writes,
        # so the listing's `cat`-per-file never abuts the next ===FILE=== marker.
        payload = (json.dumps(obj, indent=4) + "\n").encode("utf-8")
        tmp_path = f"{remote_path}.tmp"
        self._ssh_pipe(
            payload,
            f"cat > {shlex.quote(tmp_path)} && mv {shlex.quote(tmp_path)} "
            f"{shlex.quote(remote_path)}",
        )

    def _write_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Atomically write a document's .metadata JSON back to the tablet."""
        self._write_remote_json(f"{XOCHITL_PATH}/{doc_id}.metadata", metadata)

    def _restart_xochitl(self) -> None:
        """Restart xochitl so it picks up filesystem-level metadata changes.

        A run of write ops issues several restarts in quick succession; once
        systemd's start-limit (StartLimitBurst) trips, the unit enters a *failed*
        state and stays down — freezing the tablet UI — until `reset-failed`. So
        if the restart fails with a start-limit error, clear the failed state and
        start once more rather than leaving xochitl dead.
        """
        # xochitl is a systemd service; restart is brief (a few seconds of black
        # screen) but unavoidable — xochitl caches metadata in memory.
        start_limit_markers = (
            "attempted too often",
            "start request repeated too quickly",
        )
        try:
            self._ssh_command("systemctl restart xochitl", timeout=30)
        except RuntimeError as e:
            if not any(marker in str(e) for marker in start_limit_markers):
                raise
            # Start-limit tripped: clear the failed state, then start once more.
            self._ssh_command(
                "systemctl reset-failed xochitl.service && "
                "systemctl start xochitl.service",
                timeout=30,
            )

    def delete(self, doc_id: str) -> Dict[str, Any]:
        """
        Mark a document or folder as deleted (moves it to the tablet's trash).

        We set deleted=true in the .metadata rather than rm-ing the files,
        which is reversible: the user can still recover from the tablet's
        trash UI until they empty it.

        Args:
            doc_id: Document UUID.

        Returns:
            Dict {"id": doc_id, "deleted": True, "transport": "ssh"}.
        """
        metadata = self._read_metadata(doc_id)
        metadata["deleted"] = True
        metadata["metadatamodified"] = True
        metadata["lastModified"] = str(int(time.time() * 1000))
        self._write_metadata(doc_id, metadata)
        self._restart_xochitl()
        # Invalidate cache
        self._documents = []
        self._documents_by_id = {}
        return {"id": doc_id, "deleted": True, "transport": "ssh"}

    def move(self, doc_id: str, new_parent_id: str) -> Dict[str, Any]:
        """
        Move a document or folder to a new parent.

        Args:
            doc_id: UUID of the document/folder to move.
            new_parent_id: UUID of the destination folder. Empty string moves
                to root.

        Returns:
            Dict {"id": doc_id, "parent": new_parent_id, "transport": "ssh"}.
        """
        metadata = self._read_metadata(doc_id)
        metadata["parent"] = new_parent_id
        metadata["metadatamodified"] = True
        metadata["lastModified"] = str(int(time.time() * 1000))
        self._write_metadata(doc_id, metadata)
        self._restart_xochitl()
        # Invalidate cache
        self._documents = []
        self._documents_by_id = {}
        return {"id": doc_id, "parent": new_parent_id, "transport": "ssh"}

    def upload(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None,
        parent_id: str = "",
        page_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Upload a PDF or EPUB to the tablet via SSH (filesystem + xochitl restart).

        Unlike the USB web interface — which forces every upload into the
        "Clippings Import" folder — the SSH path writes the document straight
        into any folder by setting `parent` in the metadata.

        Writes three files into xochitl's storage ({uuid}.{ext}, {uuid}.metadata
        as DocumentType, {uuid}.content) then restarts xochitl to pick them up.

        Args:
            file_data: Raw file bytes.
            filename: Filename including extension; the extension sets fileType.
            content_type: Unused over SSH (kept for signature parity with the
                USB-web client); fileType is derived from the extension.
            parent_id: Destination folder UUID. Empty string places at root.
            page_count: Optional page count to seed .content. xochitl recomputes
                this on first open, so it is only a hint.

        Returns:
            Dict {"id", "name", "parent", "fileType", "transport": "ssh"}.
        """
        import uuid as _uuid

        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext not in ("pdf", "epub"):
            raise RuntimeError(
                f"SSH upload supports .pdf and .epub, got '.{ext}'. "
                "(.rmdoc archives must be uploaded via USB web mode.)"
            )

        doc_id = str(_uuid.uuid4())
        now_ms = str(int(time.time() * 1000))
        visible_name = filename.rsplit(".", 1)[0]

        # 1. The document payload itself (binary, piped over stdin).
        self._scp_upload(file_data, f"{XOCHITL_PATH}/{doc_id}.{ext}")

        # 2. .content — tells xochitl how to render the file. Minimal but
        #    sufficient; xochitl derives the rest (e.g. real page count) from
        #    the file on first open.
        content = {
            "fileType": ext,
            "formatVersion": 1,
            "lineHeight": -1,
            "margins": 125,
            "orientation": "portrait",
            "pageCount": page_count or 0,
            "textScale": 1,
            "extraMetadata": {},
        }
        self._write_remote_json(f"{XOCHITL_PATH}/{doc_id}.content", content)

        # 3. .metadata — makes it appear in the library under `parent`.
        metadata = {
            "createdTime": now_ms,
            "lastModified": now_ms,
            "lastOpened": now_ms,
            "lastOpenedPage": 0,
            "metadatamodified": False,
            "modified": False,
            "new": True,
            "parent": parent_id,
            "pinned": False,
            # On-device item: must NOT be synced=False, or is_cloud_archived
            # treats it as "not on device" and browse/read/recent hide it.
            "synced": True,
            "type": "DocumentType",
            "version": 0,
            "visibleName": visible_name,
        }
        self._write_metadata(doc_id, metadata)

        self._restart_xochitl()
        self._documents = []
        self._documents_by_id = {}
        return {
            "id": doc_id,
            "name": visible_name,
            "parent": parent_id,
            "fileType": ext,
            "transport": "ssh",
        }

    def create_notebook(
        self, name: str, pages: int = 1, parent_id: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new blank annotatable document on the tablet via SSH.

        Generates a blank PDF (shared with the USB-web backend via
        remarkable_mcp.pdf.blank_pdf_bytes) and uploads it. The result is
        annotatable with the pen and behaves like any other PDF. Because this
        uses the SSH path, the notebook can be placed directly into `parent_id`
        rather than landing in "Clippings Import".

        Args:
            name: Notebook name as it will appear on the tablet.
            pages: Number of blank pages (default 1).
            parent_id: Destination folder UUID. Empty string places at root.

        Returns:
            Dict {"name", "pages", "id", "parent", "transport": "ssh"}.
        """
        from remarkable_mcp.pdf import blank_pdf_bytes

        pdf_bytes = blank_pdf_bytes(pages)
        result = self.upload(
            pdf_bytes, filename=f"{name}.pdf", parent_id=parent_id, page_count=pages
        )
        return {
            "name": name,
            "pages": pages,
            "id": result["id"],
            "parent": parent_id,
            "transport": "ssh",
        }

    def create_rm_notebook(
        self,
        name: str,
        rm_pages: List[bytes],
        parent_id: str = "",
        template: str = "Blank",
    ) -> Dict[str, Any]:
        """Create a NATIVE .rm v6 notebook on the tablet via SSH (EXPERIMENTAL).

        Unlike `create_notebook` (which uploads a blank PDF), this writes real
        reMarkable stroke pages: one `{page_uuid}.rm` per entry in `rm_pages`,
        inside a `{doc_uuid}/` folder, plus a v6 notebook `.content` (with
        `cPages`) and a DocumentType `.metadata`. The strokes are native
        (selectable/erasable, real pen tool). The `.content` schema mirrors a
        real single-page notebook captured from a device. Restarts xochitl.
        """
        import uuid as _uuid

        if not rm_pages:
            raise RuntimeError("create_rm_notebook requires at least one .rm page")

        doc_id = str(_uuid.uuid4())
        now_ms = str(int(time.time() * 1000))

        # Page folder + one {page}.rm per page (binary, piped over stdin).
        self._ssh_command(f"mkdir -p {shlex.quote(f'{XOCHITL_PATH}/{doc_id}')}")
        page_entries = []
        for rm_bytes in rm_pages:
            page_id = str(_uuid.uuid4())
            self._scp_upload(rm_bytes, f"{XOCHITL_PATH}/{doc_id}/{page_id}.rm")
            page_entries.append(
                {
                    "id": page_id,
                    "idx": {"timestamp": "1:2", "value": "ba"},
                    "template": {"timestamp": "1:1", "value": template},
                }
            )

        content = {
            "cPages": {
                "lastOpened": {"timestamp": "0:0", "value": ""},
                "original": {"timestamp": "0:0", "value": -1},
                "pages": page_entries,
                "uuids": [{"first": str(_uuid.uuid4()), "second": 1}],
            },
            "coverPageNumber": 0,
            "customZoomCenterX": 0,
            "customZoomCenterY": 0,
            "customZoomOrientation": "portrait",
            "customZoomPageHeight": 1872,
            "customZoomPageWidth": 1404,
            "customZoomScale": 1,
            "documentMetadata": {},
            "extraMetadata": {},
            "fileType": "notebook",
            "fontName": "",
            "formatVersion": 2,
            "lineHeight": -1,
            "orientation": "portrait",
            "pageCount": len(rm_pages),
            "pageTags": [],
            "tags": [],
            "textAlignment": "justify",
            "textScale": 1,
            "zoomMode": "bestFit",
        }
        self._write_remote_json(f"{XOCHITL_PATH}/{doc_id}.content", content)

        metadata = {
            "createdTime": now_ms,
            "lastModified": now_ms,
            "lastOpened": now_ms,
            "lastOpenedPage": 0,
            "metadatamodified": False,
            "modified": False,
            "new": True,
            "parent": parent_id,
            "pinned": False,
            # On-device item: must NOT be synced=False (see is_cloud_archived).
            "synced": True,
            "type": "DocumentType",
            "version": 0,
            "visibleName": name,
        }
        self._write_metadata(doc_id, metadata)

        self._restart_xochitl()
        self._documents = []
        self._documents_by_id = {}
        return {
            "id": doc_id,
            "name": name,
            "pages": len(rm_pages),
            "parent": parent_id,
            "transport": "ssh",
        }

    def get_all_file_types(self) -> dict[str, Optional[str]]:
        """
        Get file types for all documents in a single SSH command.

        Returns a dict mapping document ID to file type (pdf, epub, or None).
        Much more efficient than calling get_file_type() for each document.
        """
        if hasattr(self, "_file_type_cache"):
            return self._file_type_cache

        self._file_type_cache: dict[str, Optional[str]] = {}

        try:
            # Read all .content files in a single command
            output = self._ssh_command(
                f"for f in {XOCHITL_PATH}/*.content; do "
                f'echo "===FILE===$(basename $f .content)"; cat "$f" 2>/dev/null; '
                f"done",
                timeout=60,
            )

            current_id = None
            current_content = []

            for line in output.split("\n"):
                if line.startswith("===FILE==="):
                    # Parse previous content
                    if current_id and current_content:
                        try:
                            data = json.loads("\n".join(current_content))
                            self._file_type_cache[current_id] = data.get("fileType")
                        except json.JSONDecodeError:
                            self._file_type_cache[current_id] = None

                    current_id = line.replace("===FILE===", "").strip()
                    current_content = []
                else:
                    current_content.append(line)

            # Don't forget the last one
            if current_id and current_content:
                try:
                    data = json.loads("\n".join(current_content))
                    self._file_type_cache[current_id] = data.get("fileType")
                except json.JSONDecodeError:
                    self._file_type_cache[current_id] = None

        except Exception as e:
            logger.warning(f"Failed to batch-load file types: {e}")

        return self._file_type_cache


def check_ssh_available(
    host: str = DEFAULT_SSH_HOST,
    user: str = DEFAULT_SSH_USER,
    port: int = DEFAULT_SSH_PORT,
) -> bool:
    """Check if SSH connection to reMarkable tablet is available."""
    client = SSHClient(host=host, user=user, port=port)
    return client.check_connection()


def create_ssh_client(
    host: Optional[str] = None,
    user: Optional[str] = None,
    port: Optional[int] = None,
    password: Optional[str] = None,
) -> SSHClient:
    """
    Create an SSH client with settings from environment or defaults.

    Environment variables:
    - REMARKABLE_SSH_HOST: SSH host (default: 10.11.99.1)
    - REMARKABLE_SSH_USER: SSH user (default: root)
    - REMARKABLE_SSH_PORT: SSH port (default: 22)
    - REMARKABLE_SSH_PASSWORD: SSH password (optional, key-based auth recommended)
    """
    return SSHClient(
        host=host or os.environ.get("REMARKABLE_SSH_HOST", DEFAULT_SSH_HOST),
        user=user or os.environ.get("REMARKABLE_SSH_USER", DEFAULT_SSH_USER),
        port=port or int(os.environ.get("REMARKABLE_SSH_PORT", str(DEFAULT_SSH_PORT))),
        password=password or os.environ.get("REMARKABLE_SSH_PASSWORD"),
    )
