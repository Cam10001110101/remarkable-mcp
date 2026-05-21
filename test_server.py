#!/usr/bin/env python3
"""
Tests for reMarkable MCP Server

Tests the 4 intent-based tools using FastMCP's testing capabilities.
"""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from remarkable_mcp.api import (
    get_item_path,
    get_items_by_id,
    register_and_get_token,
)
from remarkable_mcp.extract import (
    extract_text_from_document_zip,
    extract_text_from_rm_file,
    find_similar_documents,
)
from remarkable_mcp.responses import (
    make_error,
    make_response,
)
from remarkable_mcp.server import mcp

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _no_cairo_reexec(monkeypatch):
    """Stop the macOS cairo/DYLD guard from re-exec'ing the test process.

    cli.main() calls _ensure_macos_cairo_loadable(), which os.execv's the
    process on macOS when libcairo isn't on DYLD_LIBRARY_PATH. Setting the
    re-exec sentinel makes the guard a no-op for every test that calls main().
    """
    monkeypatch.setenv("REMARKABLE_DYLD_REEXEC", "1")


@pytest.fixture
def mock_document():
    """Create a mock Document object."""
    doc = Mock()
    doc.VissibleName = "Test Document"
    doc.ID = "doc-123"
    doc.Parent = ""
    doc.ModifiedClient = "2024-01-15T10:30:00Z"
    return doc


@pytest.fixture
def mock_folder():
    """Create a mock Folder object."""
    folder = Mock()
    folder.VissibleName = "Test Folder"
    folder.ID = "folder-456"
    folder.Parent = ""
    return folder


@pytest.fixture
def mock_collection(mock_document, mock_folder):
    """Create a mock collection of items."""
    return [mock_document, mock_folder]


@pytest.fixture
def sample_zip_file():
    """Create a sample reMarkable document zip for testing."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with zipfile.ZipFile(tmp.name, "w") as zf:
            # Add a sample text file
            zf.writestr("sample.txt", "This is sample text content")
            # Add a sample content json
            zf.writestr("metadata.content", '{"text": "Content metadata text"}')
        yield Path(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


# =============================================================================
# Test MCP Server Initialization
# =============================================================================


class TestMCPServerInitialization:
    """Test MCP server initialization and basic functionality."""

    def test_server_name(self):
        """Test that server has correct name."""
        assert mcp.name == "remarkable"

    @pytest.mark.asyncio
    async def test_tools_registered(self):
        """Test that all expected tools are registered."""
        tools = await mcp.list_tools()
        tool_names = [tool.name for tool in tools]

        expected_tools = [
            "remarkable_read",
            "remarkable_browse",
            "remarkable_recent",
            "remarkable_search",
            "remarkable_status",
            "remarkable_image",
            "remarkable_upload",
            "remarkable_create_folder",
            "remarkable_create_notebook",
            "remarkable_move",
            "remarkable_delete",
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Tool {tool_name} not found"

    @pytest.mark.asyncio
    async def test_tools_count(self):
        """Test that we have all read + write tools registered."""
        tools = await mcp.list_tools()
        assert len(tools) == 11, f"Expected 11 tools, got {len(tools)}"

    @pytest.mark.asyncio
    async def test_tool_schemas(self):
        """Test that tools have proper schemas."""
        tools = await mcp.list_tools()

        for tool in tools:
            assert tool.name, "Tool should have a name"
            assert tool.description, "Tool should have a description"
            assert hasattr(tool, "inputSchema"), "Tool should have inputSchema"

    @pytest.mark.asyncio
    async def test_all_tools_have_xml_docstrings(self):
        """Test that all tools have XML-structured documentation."""
        tools = await mcp.list_tools()

        for tool in tools:
            # Check for XML tags in description
            desc = tool.description
            assert "<usecase>" in desc, f"Tool {tool.name} missing <usecase> tag"


# =============================================================================
# Test Transport Selection (stdio vs streamable-http)
# =============================================================================


class TestTransportSelection:
    """Test that run() and the CLI select stdio vs HTTP transport correctly."""

    def test_run_defaults_to_stdio(self):
        """run() with no args runs stdio (mcp.run() with no transport)."""
        with patch("remarkable_mcp.server.mcp") as mock_mcp:
            from remarkable_mcp.server import run

            run()

            mock_mcp.run.assert_called_once_with()

    def test_run_http_sets_settings_and_transport(self):
        """run(http=True, ...) sets host/port and runs streamable-http."""
        with patch("remarkable_mcp.server.mcp") as mock_mcp:
            from remarkable_mcp.server import run

            run(http=True, host="0.0.0.0", port=9000)

            assert mock_mcp.settings.host == "0.0.0.0"
            assert mock_mcp.settings.port == 9000
            mock_mcp.run.assert_called_once_with(transport="streamable-http")

    def test_cli_default_runs_stdio(self, monkeypatch):
        """`remarkable-mcp` with no flags calls run() with http=False defaults."""
        import remarkable_mcp.cli as cli

        monkeypatch.setattr(sys, "argv", ["remarkable-mcp"])
        mock_run = Mock()
        monkeypatch.setattr("remarkable_mcp.server.run", mock_run)

        cli.main()

        mock_run.assert_called_once_with(http=False, host="127.0.0.1", port=8000)

    def test_cli_ssh_http_passthrough(self, monkeypatch):
        """`--ssh --http --host --port` sets SSH mode and passes HTTP args through."""
        import remarkable_mcp.cli as cli

        monkeypatch.setattr(
            sys,
            "argv",
            ["remarkable-mcp", "--ssh", "--http", "--host", "0.0.0.0", "--port", "9000"],
        )
        mock_run = Mock()
        monkeypatch.setattr("remarkable_mcp.server.run", mock_run)

        prev = os.environ.get("REMARKABLE_USE_SSH")
        try:
            cli.main()

            mock_run.assert_called_once_with(http=True, host="0.0.0.0", port=9000)
            assert os.environ["REMARKABLE_USE_SSH"] == "1"
        finally:
            if prev is None:
                os.environ.pop("REMARKABLE_USE_SSH", None)
            else:
                os.environ["REMARKABLE_USE_SSH"] = prev

    def test_cli_host_port_default_from_env(self, monkeypatch):
        """--host/--port default from REMARKABLE_HTTP_HOST/PORT env vars."""
        import remarkable_mcp.cli as cli

        monkeypatch.setattr(sys, "argv", ["remarkable-mcp", "--http"])
        monkeypatch.setenv("REMARKABLE_HTTP_HOST", "0.0.0.0")
        monkeypatch.setenv("REMARKABLE_HTTP_PORT", "7777")
        mock_run = Mock()
        monkeypatch.setattr("remarkable_mcp.server.run", mock_run)

        cli.main()

        mock_run.assert_called_once_with(http=True, host="0.0.0.0", port=7777)


class TestMacOSCairoGuard:
    """Test cli._ensure_macos_cairo_loadable (the macOS cairo/DYLD re-exec guard)."""

    def _fake_cairo_dir(self, tmp_path):
        (tmp_path / "libcairo.2.dylib").write_bytes(b"")
        return str(tmp_path)

    def test_reexecs_when_cairo_dir_missing_from_dyld_path(self, tmp_path, monkeypatch):
        """On macOS, when the cairo lib dir isn't on DYLD_LIBRARY_PATH, set it and re-exec."""
        import remarkable_mcp.cli as cli

        lib_dir = self._fake_cairo_dir(tmp_path)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("REMARKABLE_DYLD_REEXEC", raising=False)
        monkeypatch.setenv("REMARKABLE_CAIRO_LIB_DIR", lib_dir)
        monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)

        captured = {}

        def fake_execv(path, args):
            captured.update(path=path, args=args)

        monkeypatch.setattr(cli.os, "execv", fake_execv)

        cli._ensure_macos_cairo_loadable()

        assert captured, "expected a re-exec"
        assert lib_dir in os.environ["DYLD_LIBRARY_PATH"].split(os.pathsep)
        assert os.environ["REMARKABLE_DYLD_REEXEC"] == "1"
        assert captured["args"][1:] == [cli.__file__, *sys.argv[1:]]

    def test_no_reexec_when_dir_already_on_dyld_path(self, tmp_path, monkeypatch):
        """No re-exec when the cairo lib dir is already discoverable."""
        import remarkable_mcp.cli as cli

        lib_dir = self._fake_cairo_dir(tmp_path)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("REMARKABLE_DYLD_REEXEC", raising=False)
        monkeypatch.setenv("REMARKABLE_CAIRO_LIB_DIR", lib_dir)
        monkeypatch.setenv("DYLD_LIBRARY_PATH", lib_dir)

        called = []
        monkeypatch.setattr(cli.os, "execv", lambda *a: called.append(a))

        cli._ensure_macos_cairo_loadable()

        assert not called

    def test_no_reexec_on_non_darwin(self, monkeypatch):
        """The guard is a no-op off macOS."""
        import remarkable_mcp.cli as cli

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("REMARKABLE_DYLD_REEXEC", raising=False)

        called = []
        monkeypatch.setattr(cli.os, "execv", lambda *a: called.append(a))

        cli._ensure_macos_cairo_loadable()

        assert not called

    def test_no_reexec_when_sentinel_already_set(self, tmp_path, monkeypatch):
        """The sentinel prevents a second re-exec (no exec loop)."""
        import remarkable_mcp.cli as cli

        lib_dir = self._fake_cairo_dir(tmp_path)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("REMARKABLE_DYLD_REEXEC", "1")
        monkeypatch.setenv("REMARKABLE_CAIRO_LIB_DIR", lib_dir)
        monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)

        called = []
        monkeypatch.setattr(cli.os, "execv", lambda *a: called.append(a))

        cli._ensure_macos_cairo_loadable()

        assert not called


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_make_response(self):
        """Test response creation with hint."""
        data = {"key": "value"}
        result = make_response(data, "This is a hint")
        parsed = json.loads(result)

        assert parsed["key"] == "value"
        assert parsed["_hint"] == "This is a hint"

    def test_make_error(self):
        """Test error creation with suggestions."""
        result = make_error(
            error_type="test_error",
            message="Something went wrong",
            suggestion="Try this instead",
            did_you_mean=["option1", "option2"],
        )
        parsed = json.loads(result)

        assert parsed["_error"]["type"] == "test_error"
        assert parsed["_error"]["message"] == "Something went wrong"
        assert parsed["_error"]["suggestion"] == "Try this instead"
        assert parsed["_error"]["did_you_mean"] == ["option1", "option2"]

    def test_make_error_without_did_you_mean(self):
        """Test error creation without did_you_mean."""
        result = make_error(
            error_type="test_error", message="Error message", suggestion="Suggestion"
        )
        parsed = json.loads(result)

        assert "did_you_mean" not in parsed["_error"]

    def test_find_similar_documents(self):
        """Test fuzzy document matching."""
        docs = [
            Mock(VissibleName="Meeting Notes"),
            Mock(VissibleName="Project Plan"),
            Mock(VissibleName="Notes Daily"),
        ]

        # Exact partial match
        results = find_similar_documents("Notes", docs)
        assert "Meeting Notes" in results or "Notes Daily" in results

        # Fuzzy match
        results = find_similar_documents("Meating", docs, limit=3)
        assert len(results) <= 3

    def test_get_items_by_id(self, mock_collection):
        """Test building ID lookup dict."""
        items_by_id = get_items_by_id(mock_collection)

        assert "doc-123" in items_by_id
        assert "folder-456" in items_by_id

    def test_get_item_path(self, mock_document, mock_collection):
        """Test getting full item path."""
        items_by_id = get_items_by_id(mock_collection)
        path = get_item_path(mock_document, items_by_id)

        assert path == "/Test Document"

    def test_get_item_path_nested(self, mock_folder):
        """Test getting path for nested item."""
        # Create nested structure
        child_doc = Mock()
        child_doc.VissibleName = "Child Doc"
        child_doc.ID = "child-789"
        child_doc.Parent = mock_folder.ID

        items_by_id = {mock_folder.ID: mock_folder, child_doc.ID: child_doc}

        path = get_item_path(child_doc, items_by_id)
        assert path == "/Test Folder/Child Doc"


# =============================================================================
# Test Text Extraction
# =============================================================================


class TestTextExtraction:
    """Test text extraction functions."""

    def test_extract_text_from_document_zip(self, sample_zip_file):
        """Test extracting text from a zip file."""
        result = extract_text_from_document_zip(sample_zip_file)

        assert "typed_text" in result
        assert "highlights" in result
        assert "handwritten_text" in result
        assert "pages" in result

        # Should have extracted text from txt file
        assert any("sample text" in text.lower() for text in result["typed_text"])

    def test_extract_text_from_rm_file_no_rmscene(self):
        """Test graceful fallback when rmscene not available."""
        # Create a dummy file
        with tempfile.NamedTemporaryFile(suffix=".rm", delete=False) as tmp:
            tmp.write(b"dummy data")
            tmp_path = Path(tmp.name)

        try:
            # This should return empty list if rmscene fails
            result = extract_text_from_rm_file(tmp_path)
            assert isinstance(result, list)
        finally:
            tmp_path.unlink(missing_ok=True)


# =============================================================================
# Test remarkable_status Tool
# =============================================================================


class TestRemarkableStatus:
    """Test remarkable_status tool."""

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_status_authenticated(self, mock_get_rmapi):
        """Test status when authenticated."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = []

        result = await mcp.call_tool("remarkable_status", {})
        data = json.loads(result[0][0].text)

        assert data["authenticated"] is True
        assert "transport" in data
        assert "connection" in data
        assert data["status"] == "connected"
        assert "_hint" in data

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_status_not_authenticated(self, mock_get_rmapi):
        """Test status when not authenticated."""
        mock_get_rmapi.side_effect = RuntimeError("Failed to authenticate")

        result = await mcp.call_tool("remarkable_status", {})
        data = json.loads(result[0][0].text)

        assert data["authenticated"] is False
        assert "error" in data
        assert "_hint" in data
        # Hint should include registration instructions or SSH mode
        assert "register" in data["_hint"].lower() or "ssh" in data["_hint"].lower()


# =============================================================================
# Test remarkable_browse Tool
# =============================================================================


class TestRemarkableBrowse:
    """Test remarkable_browse tool."""

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_browse_root(self, mock_get_rmapi):
        """Test browsing root folder."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = []

        result = await mcp.call_tool("remarkable_browse", {"path": "/"})
        data = json.loads(result[0][0].text)

        assert data["mode"] == "browse"
        assert data["path"] == "/"
        assert "_hint" in data

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_browse_search_mode(self, mock_get_rmapi):
        """Test search mode."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client

        # Create mock items that have VissibleName
        mock_doc = Mock()
        mock_doc.VissibleName = "Test Document"
        mock_doc.ID = "doc-123"
        mock_doc.Parent = ""
        mock_doc.ModifiedClient = "2024-01-15"

        mock_client.get_meta_items.return_value = [mock_doc]

        result = await mcp.call_tool("remarkable_browse", {"query": "Test"})
        data = json.loads(result[0][0].text)

        assert data["mode"] == "search"
        assert data["query"] == "Test"
        assert "results" in data
        assert "_hint" in data

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_browse_error_handling(self, mock_get_rmapi):
        """Test error handling in browse."""
        mock_get_rmapi.side_effect = RuntimeError("Connection failed")

        result = await mcp.call_tool("remarkable_browse", {"path": "/"})
        data = json.loads(result[0][0].text)

        assert "_error" in data
        assert data["_error"]["type"] == "browse_failed"

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_browse_document_path_autoredirects_to_read(self, mock_get_rmapi):
        """Browsing a document (not folder) path auto-redirects to read.

        Regression test: remarkable_browse is async and must 'await' the
        internal remarkable_read call. When it was sync and called the async
        remarkable_read without await, json.loads received a coroutine and
        raised 'the JSON object must be str, bytes or bytearray, not coroutine'.
        """
        import io

        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client

        doc = Mock()
        doc.VissibleName = "My PDF"
        doc.ID = "pdf-123"
        doc.Parent = ""
        doc.ModifiedClient = "2024-01-15T10:30:00Z"
        doc.is_folder = False
        doc.is_cloud_archived = False
        doc.tags = []
        mock_client.get_meta_items.return_value = [doc]

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("pdf-123.content", '{"fileType": "pdf"}')
        mock_client.download.return_value = zip_buffer.getvalue()

        with patch("remarkable_mcp.tools.get_file_type", return_value="pdf"):
            result = await mcp.call_tool("remarkable_browse", {"path": "/My PDF"})
        data = json.loads(result[0][0].text)

        # The precise regression: browsing a document path must reach the
        # internal read via 'await', never leak a coroutine into json.loads.
        assert "coroutine" not in json.dumps(data), f"Got coroutine error: {data}"


# =============================================================================
# Test remarkable_recent Tool
# =============================================================================


class TestRemarkableRecent:
    """Test remarkable_recent tool."""

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_recent_default_limit(self, mock_get_rmapi):
        """Test getting recent documents with default limit."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = []

        result = await mcp.call_tool("remarkable_recent", {})
        data = json.loads(result[0][0].text)

        assert "count" in data
        assert "documents" in data
        assert "_hint" in data

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_recent_custom_limit(self, mock_get_rmapi):
        """Test getting recent documents with custom limit."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = []

        result = await mcp.call_tool("remarkable_recent", {"limit": 5})
        data = json.loads(result[0][0].text)

        assert "count" in data
        assert "documents" in data

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_recent_limit_clamped(self, mock_get_rmapi):
        """Test that limit is clamped to valid range."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = []

        # Test with limit > 50
        result = await mcp.call_tool("remarkable_recent", {"limit": 100})
        # Should not raise an error
        data = json.loads(result[0][0].text)
        assert "count" in data

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_recent_error_handling(self, mock_get_rmapi):
        """Test error handling in recent."""
        mock_get_rmapi.side_effect = RuntimeError("Connection failed")

        result = await mcp.call_tool("remarkable_recent", {})
        data = json.loads(result[0][0].text)

        assert "_error" in data
        assert data["_error"]["type"] == "recent_failed"

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_recent_include_preview_does_not_crash(self, mock_get_rmapi):
        """Test that include_preview=True works without AttributeError on download result.

        This is a regression test for the bug where client.download() returns bytes
        but the code called raw_doc.content (treating it like a requests.Response).
        """
        import io

        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client

        # Create a PDF document mock
        doc = Mock()
        doc.VissibleName = "My PDF"
        doc.ID = "pdf-123"
        doc.Parent = ""
        doc.ModifiedClient = "2024-01-15T10:30:00Z"
        doc.is_folder = False
        doc.tags = []

        mock_client.get_meta_items.return_value = [doc]

        # download() returns bytes (not a requests.Response)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("pdf-123.content", '{"fileType": "pdf"}')
        mock_client.download.return_value = zip_buffer.getvalue()

        # Simulate get_file_type returning "pdf"
        with patch("remarkable_mcp.tools.get_file_type", return_value="pdf"):
            result = await mcp.call_tool("remarkable_recent", {"include_preview": True})
        data = json.loads(result[0][0].text)

        # Should not crash with AttributeError; may return empty preview but no error
        assert "_error" not in data
        assert "documents" in data


# =============================================================================
# Test remarkable_read Tool
# =============================================================================


class TestRemarkableRead:
    """Test remarkable_read tool."""

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_read_document_not_found(self, mock_get_rmapi):
        """Test reading a non-existent document."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = []

        result = await mcp.call_tool("remarkable_read", {"document": "NonExistent"})
        data = json.loads(result[0][0].text)

        assert "_error" in data
        assert data["_error"]["type"] == "document_not_found"
        assert "suggestion" in data["_error"]

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_read_error_handling(self, mock_get_rmapi):
        """Test error handling in read."""
        mock_get_rmapi.side_effect = RuntimeError("Connection failed")

        result = await mcp.call_tool("remarkable_read", {"document": "Test"})
        data = json.loads(result[0][0].text)

        assert "_error" in data
        assert data["_error"]["type"] == "read_failed"

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_read_provides_suggestions(self, mock_get_rmapi, mock_document):
        """Test that read provides 'did you mean' suggestions."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = [mock_document]

        # Search for something similar but not exact
        result = await mcp.call_tool("remarkable_read", {"document": "Test Doc"})
        data = json.loads(result[0][0].text)

        # Should get a not found error with suggestions
        assert "_error" in data
        assert data["_error"]["type"] == "document_not_found"

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_read_notebook_empty_content_ocr_retry(self, mock_get_rmapi):
        """Test that remarkable_read correctly awaits the OCR auto-retry for empty notebooks.

        This is a regression test for the bug where the recursive call to
        remarkable_read() was missing 'await', causing a coroutine object to be
        passed to json.loads() with the error:
        'the JSON object must be str, bytes or bytearray, not coroutine'
        """
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client

        # Create a notebook document mock
        doc = Mock()
        doc.VissibleName = "Quick sheets"
        doc.ID = "notebook-123"
        doc.Parent = ""
        doc.ModifiedClient = "2024-01-15T10:30:00Z"
        doc.is_folder = False
        doc.tags = []

        mock_client.get_meta_items.return_value = [doc]

        # Create a minimal zip with no typed text (simulates a handwritten notebook)
        import io

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            # Add empty content file (no text field) to simulate notebook
            zf.writestr("notebook-123.content", '{"fileType": "notebook"}')
        zip_bytes = zip_buffer.getvalue()

        mock_client.download.return_value = zip_bytes

        # This should NOT raise "the JSON object must be str, bytes or bytearray, not coroutine"
        # Previously failed because remarkable_read() was called without 'await'
        result = await mcp.call_tool("remarkable_read", {"document": "Quick sheets"})
        data = json.loads(result[0][0].text)

        # Should return a valid response (not a coroutine error)
        assert (
            "_error" not in data
            or data["_error"]["type"] != "read_failed"
            or ("coroutine" not in data["_error"].get("message", ""))
        ), f"Got coroutine error: {data}"


# =============================================================================
# Test remarkable_image Tool
# =============================================================================


class TestRemarkableImage:
    """Test remarkable_image tool."""

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_image_document_not_found(self, mock_get_rmapi):
        """Test getting image from non-existent document."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = []

        result = await mcp.call_tool("remarkable_image", {"document": "NonExistent"})
        data = json.loads(result[0].text)

        assert "_error" in data
        assert data["_error"]["type"] == "document_not_found"
        assert "suggestion" in data["_error"]

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_image_error_handling(self, mock_get_rmapi):
        """Test error handling in image tool."""
        mock_get_rmapi.side_effect = RuntimeError("Connection failed")

        result = await mcp.call_tool("remarkable_image", {"document": "Test"})
        data = json.loads(result[0].text)

        assert "_error" in data
        assert data["_error"]["type"] == "image_failed"

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_image_provides_suggestions(self, mock_get_rmapi, mock_document):
        """Test that image tool provides 'did you mean' suggestions."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = [mock_document]

        # Search for something similar but not exact
        result = await mcp.call_tool("remarkable_image", {"document": "Test Doc"})
        data = json.loads(result[0].text)

        # Should get a not found error with suggestions
        assert "_error" in data
        assert data["_error"]["type"] == "document_not_found"

    @pytest.mark.asyncio
    async def test_image_compatibility_parameter_in_schema(self):
        """Test that remarkable_image tool has the compatibility parameter in its schema."""
        tools = await mcp.list_tools()
        image_tool = next(t for t in tools if t.name == "remarkable_image")

        # Check that compatibility parameter exists in the input schema
        assert "compatibility" in image_tool.inputSchema.get("properties", {})
        compat_schema = image_tool.inputSchema["properties"]["compatibility"]
        assert compat_schema.get("type") == "boolean"
        assert compat_schema.get("default") is False


# =============================================================================
# Test remarkable_search Tool
# =============================================================================


class TestRemarkableSearch:
    """Test remarkable_search tool."""

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_search_returns_content_for_matches(self, mock_get_rmapi):
        """Search reads each matching document and returns its content.

        Regression test: remarkable_search is async and must 'await' both the
        internal remarkable_browse and remarkable_read calls. When it was sync
        and called the async remarkable_read without await, every matched
        document came back with the error 'the JSON object must be str, bytes
        or bytearray, not coroutine' and the hint reported 'Found 0 documents'.
        """
        import io

        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client

        doc = Mock()
        doc.VissibleName = "Searchable Report"
        doc.ID = "rep-123"
        doc.Parent = ""
        doc.ModifiedClient = "2024-01-15T10:30:00Z"
        doc.is_folder = False
        doc.is_cloud_archived = False
        doc.tags = []
        mock_client.get_meta_items.return_value = [doc]

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("rep-123.content", '{"fileType": "pdf"}')
        mock_client.download.return_value = zip_buffer.getvalue()

        with patch("remarkable_mcp.tools.get_file_type", return_value="pdf"):
            result = await mcp.call_tool("remarkable_search", {"query": "Searchable"})
        data = json.loads(result[0][0].text)

        # The matched document must be found and read via 'await'. The original
        # bug surfaced as a per-document "...not coroutine" error and a
        # "Found 0 documents" hint despite a non-zero count.
        assert "_error" not in data
        assert data["count"] == 1
        assert "coroutine" not in json.dumps(data), f"Got coroutine error: {data}"


# =============================================================================
# Test Registration
# =============================================================================


class TestRegistration:
    """Test registration functionality."""

    @patch("requests.post")
    @patch("pathlib.Path.write_text")
    def test_register_and_get_token(self, mock_write_text, mock_post):
        """Test registration process."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test_device_token_12345"
        mock_post.return_value = mock_response

        token = register_and_get_token("test_code")

        # Should return JSON with devicetoken
        import json

        token_data = json.loads(token)
        assert token_data["devicetoken"] == "test_device_token_12345"
        assert "usertoken" in token_data

        # Verify API was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "webapp-prod.cloud.remarkable.engineering" in call_args[0][0]

    @patch("requests.post")
    def test_register_invalid_code(self, mock_post):
        """Test registration with invalid/expired code."""
        # Mock 400 response (invalid code)
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = ""
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="Registration failed"):
            register_and_get_token("invalid_code")


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestE2E:
    """End-to-end tests for MCP server."""

    def test_server_can_initialize(self):
        """Test that server can be initialized."""
        assert mcp is not None
        assert mcp.name == "remarkable"

    @pytest.mark.asyncio
    async def test_server_lists_all_tools(self):
        """Test that server can list all tools (e2e)."""
        tools = await mcp.list_tools()

        assert len(tools) == 11

        # Check each tool has required properties and starts with remarkable_
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert tool.name.startswith("remarkable_")

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_e2e_call_tool_flow(self, mock_get_rmapi):
        """Test end-to-end flow of calling a tool."""
        mock_client = Mock()
        mock_get_rmapi.return_value = mock_client
        mock_client.get_meta_items.return_value = []

        # Call status tool
        result = await mcp.call_tool("remarkable_status", {})

        # Verify we get valid JSON back
        data = json.loads(result[0][0].text)
        assert "authenticated" in data
        assert "_hint" in data

    @pytest.mark.asyncio
    async def test_tool_parameters_schema(self):
        """Test that tool parameters have proper schemas."""
        tools = await mcp.list_tools()

        # Check specific tools exist
        browse_tool = next(t for t in tools if t.name == "remarkable_browse")
        assert browse_tool is not None

        read_tool = next(t for t in tools if t.name == "remarkable_read")
        assert read_tool is not None

        recent_tool = next(t for t in tools if t.name == "remarkable_recent")
        assert recent_tool is not None

        status_tool = next(t for t in tools if t.name == "remarkable_status")
        assert status_tool is not None

    @pytest.mark.asyncio
    async def test_all_tools_return_json_with_hint(self):
        """Test that all tools return JSON with _hint field."""
        with patch("remarkable_mcp.tools.get_rmapi") as mock_get_rmapi:
            mock_client = Mock()
            mock_get_rmapi.return_value = mock_client
            mock_client.get_meta_items.return_value = []

            # Test status
            result = await mcp.call_tool("remarkable_status", {})
            data = json.loads(result[0][0].text)
            assert "_hint" in data

            # Test browse
            result = await mcp.call_tool("remarkable_browse", {"path": "/"})
            data = json.loads(result[0][0].text)
            assert "_hint" in data or "_error" in data

            # Test recent
            result = await mcp.call_tool("remarkable_recent", {})
            data = json.loads(result[0][0].text)
            assert "_hint" in data or "_error" in data


# =============================================================================
# Test Response Consistency
# =============================================================================


class TestResponseConsistency:
    """Test that responses follow consistent patterns."""

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_all_errors_have_required_fields(self, mock_get_rmapi):
        """Test that all error responses have required fields."""
        mock_get_rmapi.side_effect = RuntimeError("Test error")

        tools_to_test = [
            ("remarkable_status", {}),
            ("remarkable_browse", {"path": "/"}),
            ("remarkable_recent", {}),
            ("remarkable_read", {"document": "test"}),
        ]

        for tool_name, args in tools_to_test:
            result = await mcp.call_tool(tool_name, args)
            data = json.loads(result[0][0].text)

            # Either success with _hint or error with _error
            has_hint = "_hint" in data
            has_error = "_error" in data

            assert has_hint or has_error, f"Tool {tool_name} response missing _hint or _error"

            if has_error:
                assert "type" in data["_error"], f"Error in {tool_name} missing type"
                assert "message" in data["_error"], f"Error in {tool_name} missing message"
                assert "suggestion" in data["_error"], f"Error in {tool_name} missing suggestion"


# =============================================================================
# Test Capability Checking
# =============================================================================


class TestCapabilityChecking:
    """Test capability checking utilities."""

    def test_get_client_capabilities_without_context(self):
        """Test get_client_capabilities returns None without valid context."""
        from remarkable_mcp.capabilities import get_client_capabilities

        # Create mock context without session
        mock_ctx = Mock()
        mock_ctx.session = None

        result = get_client_capabilities(mock_ctx)
        assert result is None

    def test_get_client_capabilities_without_client_params(self):
        """Test get_client_capabilities returns None without client_params."""
        from remarkable_mcp.capabilities import get_client_capabilities

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = None

        result = get_client_capabilities(mock_ctx)
        assert result is None

    def test_get_client_capabilities_with_valid_context(self):
        """Test get_client_capabilities returns capabilities when available."""
        from mcp.types import ClientCapabilities, SamplingCapability

        from remarkable_mcp.capabilities import get_client_capabilities

        mock_caps = ClientCapabilities(sampling=SamplingCapability())

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.capabilities = mock_caps

        result = get_client_capabilities(mock_ctx)
        assert result is not None
        assert result.sampling is not None

    def test_client_supports_sampling_true(self):
        """Test client_supports_sampling returns True when sampling available."""
        from mcp.types import ClientCapabilities, SamplingCapability

        from remarkable_mcp.capabilities import client_supports_sampling

        mock_caps = ClientCapabilities(sampling=SamplingCapability())

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.capabilities = mock_caps

        result = client_supports_sampling(mock_ctx)
        assert result is True

    def test_client_supports_sampling_false(self):
        """Test client_supports_sampling returns False when sampling not available."""
        from mcp.types import ClientCapabilities

        from remarkable_mcp.capabilities import client_supports_sampling

        mock_caps = ClientCapabilities(sampling=None)

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.capabilities = mock_caps

        result = client_supports_sampling(mock_ctx)
        assert result is False

    def test_client_supports_elicitation(self):
        """Test client_supports_elicitation."""
        from mcp.types import ClientCapabilities, ElicitationCapability

        from remarkable_mcp.capabilities import client_supports_elicitation

        # Test with elicitation enabled
        mock_caps = ClientCapabilities(elicitation=ElicitationCapability())

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.capabilities = mock_caps

        assert client_supports_elicitation(mock_ctx) is True

        # Test with elicitation disabled
        mock_caps = ClientCapabilities(elicitation=None)
        mock_ctx.session.client_params.capabilities = mock_caps

        assert client_supports_elicitation(mock_ctx) is False

    def test_client_supports_roots(self):
        """Test client_supports_roots."""
        from mcp.types import ClientCapabilities, RootsCapability

        from remarkable_mcp.capabilities import client_supports_roots

        # Test with roots enabled
        mock_caps = ClientCapabilities(roots=RootsCapability())

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.capabilities = mock_caps

        assert client_supports_roots(mock_ctx) is True

        # Test with roots disabled
        mock_caps = ClientCapabilities(roots=None)
        mock_ctx.session.client_params.capabilities = mock_caps

        assert client_supports_roots(mock_ctx) is False

    def test_client_supports_experimental(self):
        """Test client_supports_experimental."""
        from mcp.types import ClientCapabilities

        from remarkable_mcp.capabilities import client_supports_experimental

        # Test with experimental feature present
        mock_caps = ClientCapabilities(experimental={"my_feature": {}})

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.capabilities = mock_caps

        assert client_supports_experimental(mock_ctx, "my_feature") is True
        assert client_supports_experimental(mock_ctx, "other_feature") is False

        # Test with no experimental features
        mock_caps = ClientCapabilities(experimental=None)
        mock_ctx.session.client_params.capabilities = mock_caps

        assert client_supports_experimental(mock_ctx, "my_feature") is False

    def test_get_client_info(self):
        """Test get_client_info."""
        from remarkable_mcp.capabilities import get_client_info

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.clientInfo = Mock()
        mock_ctx.session.client_params.clientInfo.name = "Test Client"
        mock_ctx.session.client_params.clientInfo.version = "1.0.0"
        mock_ctx.session.client_params.protocolVersion = "2024-11-05"

        result = get_client_info(mock_ctx)
        assert result is not None
        assert result["name"] == "Test Client"
        assert result["version"] == "1.0.0"
        assert result["protocol_version"] == "2024-11-05"

    def test_get_client_info_without_client_info(self):
        """Test get_client_info when clientInfo is None."""
        from remarkable_mcp.capabilities import get_client_info

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.clientInfo = None
        mock_ctx.session.client_params.protocolVersion = "2024-11-05"

        result = get_client_info(mock_ctx)
        assert result is not None
        assert result["name"] is None
        assert result["version"] is None
        assert result["protocol_version"] == "2024-11-05"

    def test_get_protocol_version(self):
        """Test get_protocol_version."""
        from remarkable_mcp.capabilities import get_protocol_version

        mock_ctx = Mock()
        mock_ctx.session = Mock()
        mock_ctx.session.client_params = Mock()
        mock_ctx.session.client_params.protocolVersion = "2024-11-05"

        result = get_protocol_version(mock_ctx)
        assert result == "2024-11-05"

    def test_get_protocol_version_without_context(self):
        """Test get_protocol_version returns None without valid context."""
        from remarkable_mcp.capabilities import get_protocol_version

        mock_ctx = Mock()
        mock_ctx.session = None

        result = get_protocol_version(mock_ctx)
        assert result is None

    def test_capability_imports_from_package(self):
        """Test that capability utilities can be imported from main package."""
        from remarkable_mcp import (
            client_supports_elicitation,
            client_supports_experimental,
            client_supports_roots,
            client_supports_sampling,
            get_client_capabilities,
            get_client_info,
            get_protocol_version,
        )

        # Verify all functions are callable
        assert callable(get_client_capabilities)
        assert callable(client_supports_sampling)
        assert callable(client_supports_elicitation)
        assert callable(client_supports_roots)
        assert callable(client_supports_experimental)
        assert callable(get_client_info)
        assert callable(get_protocol_version)


# =============================================================================
# Test Sampling OCR
# =============================================================================


class TestSamplingOCR:
    """Test sampling-based OCR functionality."""

    def test_get_ocr_backend_default(self):
        """Test default OCR backend is auto."""
        import os

        from remarkable_mcp.ocr import get_ocr_backend

        # Clear any env var
        env_backup = os.environ.get("REMARKABLE_OCR_BACKEND")
        if "REMARKABLE_OCR_BACKEND" in os.environ:
            del os.environ["REMARKABLE_OCR_BACKEND"]

        try:
            result = get_ocr_backend()
            assert result == "auto"
        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_OCR_BACKEND"] = env_backup

    def test_get_ocr_backend_sampling(self):
        """Test OCR backend can be set to sampling."""
        import os

        from remarkable_mcp.ocr import get_ocr_backend

        env_backup = os.environ.get("REMARKABLE_OCR_BACKEND")
        os.environ["REMARKABLE_OCR_BACKEND"] = "sampling"

        try:
            result = get_ocr_backend()
            assert result == "sampling"
        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_OCR_BACKEND"] = env_backup
            elif "REMARKABLE_OCR_BACKEND" in os.environ:
                del os.environ["REMARKABLE_OCR_BACKEND"]

    def test_should_use_sampling_ocr_false_when_not_configured(self):
        """Test should_use_sampling_ocr returns False when not configured."""
        import os

        from mcp.types import ClientCapabilities, SamplingCapability

        from remarkable_mcp.ocr import should_use_sampling_ocr

        env_backup = os.environ.get("REMARKABLE_OCR_BACKEND")
        if "REMARKABLE_OCR_BACKEND" in os.environ:
            del os.environ["REMARKABLE_OCR_BACKEND"]

        try:
            # Create mock context with sampling capability
            mock_caps = ClientCapabilities(sampling=SamplingCapability())
            mock_ctx = Mock()
            mock_ctx.session = Mock()
            mock_ctx.session.client_params = Mock()
            mock_ctx.session.client_params.capabilities = mock_caps

            # Should return False because backend is "auto", not "sampling"
            result = should_use_sampling_ocr(mock_ctx)
            assert result is False
        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_OCR_BACKEND"] = env_backup

    def test_should_use_sampling_ocr_true_when_configured(self):
        """Test should_use_sampling_ocr returns True when configured and client supports it."""
        import os

        from mcp.types import ClientCapabilities, SamplingCapability

        from remarkable_mcp.ocr import should_use_sampling_ocr

        env_backup = os.environ.get("REMARKABLE_OCR_BACKEND")
        os.environ["REMARKABLE_OCR_BACKEND"] = "sampling"

        try:
            # Create mock context with sampling capability
            mock_caps = ClientCapabilities(sampling=SamplingCapability())
            mock_ctx = Mock()
            mock_ctx.session = Mock()
            mock_ctx.session.client_params = Mock()
            mock_ctx.session.client_params.capabilities = mock_caps

            result = should_use_sampling_ocr(mock_ctx)
            assert result is True
        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_OCR_BACKEND"] = env_backup
            elif "REMARKABLE_OCR_BACKEND" in os.environ:
                del os.environ["REMARKABLE_OCR_BACKEND"]

    def test_should_use_sampling_ocr_false_when_client_doesnt_support(self):
        """Test should_use_sampling_ocr returns False when client doesn't support sampling."""
        import os

        from mcp.types import ClientCapabilities

        from remarkable_mcp.ocr import should_use_sampling_ocr

        env_backup = os.environ.get("REMARKABLE_OCR_BACKEND")
        os.environ["REMARKABLE_OCR_BACKEND"] = "sampling"

        try:
            # Create mock context WITHOUT sampling capability
            mock_caps = ClientCapabilities(sampling=None)
            mock_ctx = Mock()
            mock_ctx.session = Mock()
            mock_ctx.session.client_params = Mock()
            mock_ctx.session.client_params.capabilities = mock_caps

            result = should_use_sampling_ocr(mock_ctx)
            assert result is False
        finally:
            if env_backup is not None:
                os.environ["REMARKABLE_OCR_BACKEND"] = env_backup
            elif "REMARKABLE_OCR_BACKEND" in os.environ:
                del os.environ["REMARKABLE_OCR_BACKEND"]

    def test_ocr_system_prompt_structure(self):
        """Test the OCR system prompt is properly structured."""
        from remarkable_mcp.sampling import OCR_SYSTEM_PROMPT, OCR_USER_PROMPT

        # Check that system prompt contains key instructions
        assert "OCR" in OCR_SYSTEM_PROMPT
        assert "ONLY" in OCR_SYSTEM_PROMPT
        assert "[NO TEXT DETECTED]" in OCR_SYSTEM_PROMPT
        assert "reMarkable" in OCR_SYSTEM_PROMPT

        # Check user prompt is concise
        assert "text" in OCR_USER_PROMPT.lower()
        assert len(OCR_USER_PROMPT) < 200  # Should be short and focused

    @pytest.mark.asyncio
    async def test_ocr_via_sampling_returns_none_without_session(self):
        """Test ocr_via_sampling returns None when session is not available."""
        from remarkable_mcp.sampling import ocr_via_sampling

        mock_ctx = Mock()
        mock_ctx.session = None

        result = await ocr_via_sampling(mock_ctx, b"fake_png_data")
        assert result is None

    def test_sampling_imports_from_module(self):
        """Test that sampling utilities can be imported."""
        from remarkable_mcp.ocr import get_ocr_backend, should_use_sampling_ocr
        from remarkable_mcp.sampling import (
            OCR_SYSTEM_PROMPT,
            OCR_USER_PROMPT,
            ocr_pages_via_sampling,
            ocr_via_sampling,
        )

        # Verify all functions/constants are accessible
        assert callable(ocr_via_sampling)
        assert callable(ocr_pages_via_sampling)
        assert callable(get_ocr_backend)
        assert callable(should_use_sampling_ocr)
        assert isinstance(OCR_SYSTEM_PROMPT, str)
        assert isinstance(OCR_USER_PROMPT, str)


class TestOcrConfig:
    """Config + reachability for the unified OCR module."""

    def test_get_ocr_backend_default_and_env(self):
        import os

        from remarkable_mcp.ocr import get_ocr_backend

        os.environ.pop("REMARKABLE_OCR_BACKEND", None)
        try:
            assert get_ocr_backend() == "auto"
            os.environ["REMARKABLE_OCR_BACKEND"] = "OLLAMA"
            assert get_ocr_backend() == "ollama"
        finally:
            os.environ.pop("REMARKABLE_OCR_BACKEND", None)

    def test_ollama_config_defaults(self):
        import os

        from remarkable_mcp import ocr

        for k in (
            "REMARKABLE_OLLAMA_MODEL",
            "REMARKABLE_OLLAMA_HOST",
            "OLLAMA_HOST",
            "REMARKABLE_OLLAMA_TIMEOUT",
        ):
            os.environ.pop(k, None)
        assert ocr.get_ollama_model() == "gemma4:31b"
        assert ocr.get_ollama_host() == "http://localhost:11434"
        assert ocr.get_ollama_timeout() == 180

    def test_ollama_available_true_false_and_cache(self):
        from unittest.mock import MagicMock, patch

        from remarkable_mcp import ocr

        ocr._reset_ollama_cache()
        with patch("remarkable_mcp.ocr.requests.get") as g:
            g.return_value = MagicMock(status_code=200)
            assert ocr.ollama_available() is True
            assert ocr.ollama_available() is True  # cached, no second call
            assert g.call_count == 1
        ocr._reset_ollama_cache()
        with patch("remarkable_mcp.ocr.requests.get", side_effect=Exception("refused")):
            assert ocr.ollama_available() is False
        ocr._reset_ollama_cache()


class TestOllamaEngine:
    """The local Ollama OCR engine."""

    def test_ocr_png_ollama_success(self):
        from unittest.mock import MagicMock, patch

        from remarkable_mcp import ocr

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"response": "hello world\n"}
        with patch("remarkable_mcp.ocr.requests.post", return_value=resp) as p:
            out = ocr.ocr_png_ollama(b"PNGDATA")
        assert out == "hello world"
        body = p.call_args.kwargs["json"]
        assert body["model"] == "gemma4:31b"
        assert body["images"] and isinstance(body["images"][0], str)
        assert body["stream"] is False

    def test_ocr_png_ollama_no_text_sentinel(self):
        from unittest.mock import MagicMock, patch

        from remarkable_mcp import ocr

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"response": "[NO TEXT DETECTED]"}
        with patch("remarkable_mcp.ocr.requests.post", return_value=resp):
            assert ocr.ocr_png_ollama(b"x") is None

    def test_ocr_png_ollama_error_returns_none(self):
        from unittest.mock import patch

        from remarkable_mcp import ocr

        with patch("remarkable_mcp.ocr.requests.post", side_effect=Exception("refused")):
            assert ocr.ocr_png_ollama(b"x") is None


# =============================================================================
# Test Tag Support
# =============================================================================


class TestTagSupport:
    """Test tag-related functionality."""

    @pytest.mark.asyncio
    async def test_document_has_tags_field(self):
        """Test that Document dataclass includes tags field."""
        from remarkable_mcp.sync import Document

        doc = Document(
            id="test-id",
            hash="test-hash",
            name="Test Doc",
            doc_type="DocumentType",
            tags=["work", "important"],
        )
        assert hasattr(doc, "tags")
        assert doc.tags == ["work", "important"]

    @pytest.mark.asyncio
    async def test_document_tags_default_empty(self):
        """Test that Document tags default to empty list."""
        from remarkable_mcp.sync import Document

        doc = Document(
            id="test-id",
            hash="test-hash",
            name="Test Doc",
            doc_type="DocumentType",
        )
        assert hasattr(doc, "tags")
        assert doc.tags == []

    @pytest.mark.asyncio
    async def test_browse_includes_tags(self):
        """Test that remarkable_browse includes tags in response."""
        mock_client = Mock()
        mock_doc = Mock()
        mock_doc.VissibleName = "Tagged Doc"
        mock_doc.ID = "doc-1"
        mock_doc.Parent = ""
        mock_doc.is_folder = False
        mock_doc.ModifiedClient = None
        mock_doc.tags = ["work", "project"]

        mock_client.get_meta_items.return_value = [mock_doc]

        with patch("remarkable_mcp.tools.get_rmapi", return_value=mock_client):
            with patch("remarkable_mcp.tools._is_cloud_archived", return_value=False):
                result = await mcp.call_tool("remarkable_browse", {"path": "/"})
                data = json.loads(result[0][0].text)

                assert data["mode"] == "browse"
                assert len(data["documents"]) == 1
                assert data["documents"][0]["name"] == "Tagged Doc"
                assert "tags" in data["documents"][0]
                assert data["documents"][0]["tags"] == ["work", "project"]

    @pytest.mark.asyncio
    async def test_browse_filter_by_tags(self):
        """Test that remarkable_browse can filter documents by tags."""
        mock_client = Mock()

        mock_doc1 = Mock()
        mock_doc1.VissibleName = "Work Doc"
        mock_doc1.ID = "doc-1"
        mock_doc1.Parent = ""
        mock_doc1.is_folder = False
        mock_doc1.ModifiedClient = None
        mock_doc1.tags = ["work"]

        mock_doc2 = Mock()
        mock_doc2.VissibleName = "Personal Doc"
        mock_doc2.ID = "doc-2"
        mock_doc2.Parent = ""
        mock_doc2.is_folder = False
        mock_doc2.ModifiedClient = None
        mock_doc2.tags = ["personal"]

        mock_client.get_meta_items.return_value = [mock_doc1, mock_doc2]

        with patch("remarkable_mcp.tools.get_rmapi", return_value=mock_client):
            with patch("remarkable_mcp.tools._is_cloud_archived", return_value=False):
                result = await mcp.call_tool("remarkable_browse", {"path": "/", "tags": ["work"]})
                data = json.loads(result[0][0].text)

                assert data["mode"] == "browse"
                assert len(data["documents"]) == 1
                assert data["documents"][0]["name"] == "Work Doc"
                assert "filter_tags" in data
                assert data["filter_tags"] == ["work"]

    @pytest.mark.asyncio
    async def test_browse_search_mode_includes_tags(self):
        """Test that remarkable_browse in search mode includes tags in results."""
        mock_client = Mock()
        mock_doc = Mock()
        mock_doc.VissibleName = "Meeting Notes"
        mock_doc.ID = "doc-1"
        mock_doc.Parent = ""
        mock_doc.is_folder = False
        mock_doc.ModifiedClient = None
        mock_doc.tags = ["meeting", "important"]

        mock_client.get_meta_items.return_value = [mock_doc]

        with patch("remarkable_mcp.tools.get_rmapi", return_value=mock_client):
            with patch("remarkable_mcp.tools._is_cloud_archived", return_value=False):
                result = await mcp.call_tool("remarkable_browse", {"query": "meeting"})
                data = json.loads(result[0][0].text)

                assert data["mode"] == "search"
                assert len(data["results"]) == 1
                assert "tags" in data["results"][0]
                assert data["results"][0]["tags"] == ["meeting", "important"]

    @pytest.mark.asyncio
    async def test_browse_search_mode_filter_by_tags(self):
        """Test that remarkable_browse in search mode can filter by tags."""
        mock_client = Mock()

        mock_doc1 = Mock()
        mock_doc1.VissibleName = "Work Meeting"
        mock_doc1.ID = "doc-1"
        mock_doc1.Parent = ""
        mock_doc1.is_folder = False
        mock_doc1.ModifiedClient = None
        mock_doc1.tags = ["work", "meeting"]

        mock_doc2 = Mock()
        mock_doc2.VissibleName = "Personal Meeting"
        mock_doc2.ID = "doc-2"
        mock_doc2.Parent = ""
        mock_doc2.is_folder = False
        mock_doc2.ModifiedClient = None
        mock_doc2.tags = ["personal", "meeting"]

        mock_client.get_meta_items.return_value = [mock_doc1, mock_doc2]

        with patch("remarkable_mcp.tools.get_rmapi", return_value=mock_client):
            with patch("remarkable_mcp.tools._is_cloud_archived", return_value=False):
                result = await mcp.call_tool(
                    "remarkable_browse", {"query": "meeting", "tags": ["work"]}
                )
                data = json.loads(result[0][0].text)

                assert data["mode"] == "search"
                assert len(data["results"]) == 1
                assert data["results"][0]["name"] == "Work Meeting"
                assert "filter_tags" in data
                assert data["filter_tags"] == ["work"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Test USB Web Interface
# =============================================================================


class TestUSBWebInterface:
    """Test USB web interface client."""

    @patch("requests.request")
    def test_usb_web_check_connection(self, mock_request):
        """Test USB web interface connection check."""
        from remarkable_mcp.usb_web import USBWebClient

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_request.return_value = mock_response

        client = USBWebClient()
        assert client.check_connection() is True

        # Verify request was made
        mock_request.assert_called_once()

    @patch("requests.request")
    def test_usb_web_connection_error(self, mock_request):
        """Test USB web interface connection error."""
        from remarkable_mcp.usb_web import USBWebClient

        # Mock connection error
        mock_request.side_effect = Exception("Connection refused")

        client = USBWebClient()
        assert client.check_connection() is False

    @patch("requests.request")
    def test_usb_web_get_meta_items(self, mock_request):
        """Test fetching documents via USB web interface."""
        from remarkable_mcp.usb_web import USBWebClient

        # Mock successful response with documents
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"ID": "doc1", "VissibleName": "Test Doc", "Type": "DocumentType", "fileType": "pdf"},
            {"ID": "folder1", "VissibleName": "Test Folder", "Type": "CollectionType"},
        ]
        mock_request.return_value = mock_response

        client = USBWebClient()
        docs = client.get_meta_items()

        assert len(docs) >= 2
        assert any(d.name == "Test Doc" for d in docs)
        assert any(d.is_folder for d in docs)
        # fileType from API response is captured
        pdf_doc = next(d for d in docs if d.name == "Test Doc")
        assert pdf_doc.file_type == "pdf"
        assert client.get_file_type(pdf_doc) == "pdf"

    @patch("requests.request")
    def test_usb_web_get_meta_items_root_unreachable_raises(self, mock_request):
        """Root /documents/ failure must raise, not return an empty list silently."""
        import requests

        from remarkable_mcp.usb_web import USBWebClient

        mock_request.side_effect = requests.ConnectionError("Connection refused")

        client = USBWebClient()
        with pytest.raises(RuntimeError, match="Cannot connect"):
            client.get_meta_items()

    @patch("requests.request")
    def test_usb_web_download(self, mock_request):
        """Test downloading document via USB web interface."""
        from remarkable_mcp.usb_web import Document, USBWebClient

        # Mock successful download response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake zip content"
        mock_request.return_value = mock_response

        client = USBWebClient()
        doc = Document(id="doc1", hash="doc1", name="Test", doc_type="DocumentType")

        content = client.download(doc)
        assert content == b"fake zip content"

    @patch("remarkable_mcp.usb_web.create_usb_web_client")
    def test_get_rmapi_usb_web_mode(self, mock_create_client):
        """Test get_rmapi in USB web mode."""
        import os
        import sys

        # Set USB web mode before importing
        os.environ["REMARKABLE_USE_USB_WEB"] = "1"

        # Reload the module to pick up the new env var
        if "remarkable_mcp.api" in sys.modules:
            import importlib

            import remarkable_mcp.api

            importlib.reload(remarkable_mcp.api)
            from remarkable_mcp.api import get_rmapi
        else:
            from remarkable_mcp.api import get_rmapi

        # Mock USB web client
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        try:
            client = get_rmapi()
            assert client == mock_client
            mock_create_client.assert_called_once()
        finally:
            # Clean up
            if "REMARKABLE_USE_USB_WEB" in os.environ:
                del os.environ["REMARKABLE_USE_USB_WEB"]
            # Reload to reset
            if "remarkable_mcp.api" in sys.modules:
                import importlib

                import remarkable_mcp.api

                importlib.reload(remarkable_mcp.api)

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_status_usb_web_mode(self, mock_get_rmapi):
        """Test remarkable_status in USB web mode."""
        import os
        import sys

        # Set USB web mode before importing
        os.environ["REMARKABLE_USE_USB_WEB"] = "1"

        # Reload the modules to pick up the new env var
        if "remarkable_mcp.api" in sys.modules:
            import importlib

            import remarkable_mcp.api

            importlib.reload(remarkable_mcp.api)

        try:
            # Mock USB web client
            mock_client = Mock()
            mock_doc = Mock()
            mock_doc.is_folder = False
            mock_doc.VissibleName = "Test"
            mock_doc.ID = "doc1"
            mock_doc.Parent = ""
            mock_client.get_meta_items.return_value = [mock_doc]
            mock_get_rmapi.return_value = mock_client

            result = await mcp.call_tool("remarkable_status", {})
            data = json.loads(result[0][0].text)

            assert data["authenticated"] is True
            assert data["transport"] == "usb-web"
            assert "USB web interface" in data["connection"]
        finally:
            # Clean up
            if "REMARKABLE_USE_USB_WEB" in os.environ:
                del os.environ["REMARKABLE_USE_USB_WEB"]
            # Reload to reset
            if "remarkable_mcp.api" in sys.modules:
                import importlib

                import remarkable_mcp.api

                importlib.reload(remarkable_mcp.api)


# =============================================================================
# Test write tools (upload, create_folder, create_notebook, move, delete)
# =============================================================================


class TestUSBWebWriteOperations:
    """Test the USBWebClient write methods against a mocked HTTP layer."""

    @patch("requests.request")
    def test_upload_posts_multipart_with_file_field(self, mock_request):
        from remarkable_mcp.usb_web import USBWebClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_request.return_value = mock_response

        client = USBWebClient()
        client.upload(b"%PDF-1.4 hello", filename="test.pdf")

        # Verify the request was a POST to /upload with multipart 'file' field
        call = mock_request.call_args
        assert call.args[0] == "POST"
        assert call.args[1].endswith("/upload")
        files = call.kwargs["files"]
        assert "file" in files
        assert files["file"][0] == "test.pdf"
        assert files["file"][1] == b"%PDF-1.4 hello"
        assert files["file"][2] == "application/pdf"

    @patch("requests.request")
    def test_create_notebook_uploads_pdf_with_requested_page_count(self, mock_request):
        from remarkable_mcp.usb_web import USBWebClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Upload successful"}
        mock_request.return_value = mock_response

        client = USBWebClient()
        result = client.create_notebook("Notes", pages=3)

        assert result["name"] == "Notes"
        assert result["pages"] == 3
        # Verify a PDF was uploaded
        call = mock_request.call_args
        files = call.kwargs["files"]
        assert files["file"][0] == "Notes.pdf"
        assert files["file"][1].startswith(b"%PDF")
        # Confirm the PDF has 3 pages
        import fitz

        doc = fitz.open(stream=files["file"][1], filetype="pdf")
        assert doc.page_count == 3
        doc.close()


class TestSSHWriteOperations:
    """Test the SSHClient write methods with mocked SSH. All writes stream over
    _ssh_pipe (cat > tmp && mv) rather than base64 — the tablet has no base64
    binary, a bug the real-device smoke test caught."""

    @staticmethod
    def _json_writes(mock_pipe):
        """Decode JSON payloads streamed via _ssh_pipe, asserting none of the
        write commands depend on a `base64` binary (regression guard for the
        'base64: command not found' device failure)."""
        writes = []
        for call in mock_pipe.call_args_list:
            data, command = call.args[0], call.args[1]
            assert "base64" not in command, f"write must not use base64: {command}"
            try:
                writes.append(json.loads(data.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass  # non-JSON payload (e.g. a raw PDF) — skip
        return writes

    def test_delete_marks_metadata_and_restarts_xochitl(self):
        from remarkable_mcp.ssh import SSHClient

        client = SSHClient()
        # _ssh_command: cat metadata (read), then restart. The write goes via pipe.
        with patch.object(
            client,
            "_ssh_command",
            side_effect=[
                json.dumps({"deleted": False, "type": "DocumentType", "parent": ""}),
                "",
            ],
        ) as mock_cmd, patch.object(client, "_ssh_pipe") as mock_pipe:
            result = client.delete("doc-uuid-1")

        assert result["deleted"] is True
        assert result["transport"] == "ssh"
        writes = self._json_writes(mock_pipe)
        assert writes and writes[0]["deleted"] is True
        assert writes[0]["metadatamodified"] is True
        assert any(
            "systemctl restart xochitl" in c.args[0] for c in mock_cmd.call_args_list
        )

    def test_create_folder_writes_collectiontype_metadata(self):
        from remarkable_mcp.ssh import SSHClient

        client = SSHClient()
        with (
            patch.object(client, "_ssh_command", return_value="") as mock_cmd,
            patch.object(client, "_ssh_pipe") as mock_pipe,
        ):
            result = client.create_folder("Reading", parent_id="parent-uuid")

        assert result["name"] == "Reading"
        assert result["parent"] == "parent-uuid"
        assert result["transport"] == "ssh"
        writes = self._json_writes(mock_pipe)
        assert writes and writes[0]["type"] == "CollectionType"
        assert writes[0]["visibleName"] == "Reading"
        assert writes[0]["parent"] == "parent-uuid"
        assert any(
            "systemctl restart xochitl" in c.args[0] for c in mock_cmd.call_args_list
        )

    def test_move_updates_parent_and_restarts_xochitl(self):
        from remarkable_mcp.ssh import SSHClient

        client = SSHClient()
        with patch.object(
            client,
            "_ssh_command",
            side_effect=[
                json.dumps({"deleted": False, "type": "DocumentType", "parent": "old"}),
                "",
            ],
        ), patch.object(client, "_ssh_pipe") as mock_pipe:
            result = client.move("doc-uuid-2", "new-parent-uuid")

        assert result["parent"] == "new-parent-uuid"
        writes = self._json_writes(mock_pipe)
        assert writes and writes[0]["parent"] == "new-parent-uuid"

    def test_upload_writes_payload_metadata_content_and_restarts(self):
        from remarkable_mcp.ssh import SSHClient

        client = SSHClient()
        with (
            patch.object(client, "_ssh_command", return_value="") as mock_cmd,
            patch.object(client, "_ssh_pipe") as mock_pipe,
        ):
            result = client.upload(
                b"%PDF-1.4 data", filename="paper.pdf", parent_id="folder-uuid"
            )

        assert result["fileType"] == "pdf"
        assert result["parent"] == "folder-uuid"
        assert result["name"] == "paper"
        assert result["transport"] == "ssh"

        # First pipe call streams the PDF payload straight to {uuid}.pdf.
        first_data, first_cmd = (
            mock_pipe.call_args_list[0].args[0],
            mock_pipe.call_args_list[0].args[1],
        )
        assert first_data == b"%PDF-1.4 data"
        assert first_cmd.startswith("cat > ")
        assert f"{result['id']}.pdf" in first_cmd

        # .content (fileType pdf) and .metadata (DocumentType) stream as JSON —
        # _json_writes also asserts no command uses base64.
        writes = self._json_writes(mock_pipe)
        assert any(w.get("fileType") == "pdf" for w in writes)
        meta = [w for w in writes if w.get("type") == "DocumentType"]
        assert meta and meta[0]["parent"] == "folder-uuid"
        assert meta[0]["visibleName"] == "paper"
        assert any(
            "systemctl restart xochitl" in c.args[0] for c in mock_cmd.call_args_list
        )

    def test_create_notebook_generates_pdf_and_uploads_via_ssh(self):
        from remarkable_mcp.ssh import SSHClient

        client = SSHClient()
        with patch.object(client, "_ssh_command", return_value=""), patch.object(
            client, "_ssh_pipe"
        ) as mock_pipe:
            result = client.create_notebook("Notes", pages=2, parent_id="p1")

        assert result["name"] == "Notes"
        assert result["pages"] == 2
        assert result["parent"] == "p1"
        assert result["transport"] == "ssh"

        # First pipe call is the PDF payload.
        uploaded = mock_pipe.call_args_list[0].args[0]
        assert uploaded.startswith(b"%PDF")
        import fitz

        doc = fitz.open(stream=uploaded, filetype="pdf")
        assert doc.page_count == 2
        doc.close()

    def test_upload_rejects_unsupported_extension(self):
        from remarkable_mcp.ssh import SSHClient

        client = SSHClient()
        with pytest.raises(RuntimeError, match="supports .pdf and .epub"):
            client.upload(b"data", filename="note.rmdoc")


class TestWriteToolParity:
    """Lock the write-method set per backend so the two transports can't silently
    drift. Changing these sets is intentional — update the parity matrix in
    docs/tools.md to match."""

    def test_backend_write_method_sets(self):
        from remarkable_mcp.ssh import SSHClient
        from remarkable_mcp.usb_web import USBWebClient

        ssh_writes = {"upload", "create_notebook", "create_folder", "move", "delete"}
        usb_writes = {"upload", "create_notebook"}

        for m in ssh_writes:
            assert callable(getattr(SSHClient, m, None)), f"SSHClient missing {m}"
        for m in usb_writes:
            assert callable(getattr(USBWebClient, m, None)), f"USBWebClient missing {m}"

        # USB web genuinely lacks folder/move/delete endpoints — assert they stay
        # absent so we never ship a stub that pretends to work.
        for m in ("create_folder", "move", "delete"):
            assert getattr(USBWebClient, m, None) is None, (
                f"USBWebClient unexpectedly has '{m}'. If intentional, update the "
                "parity matrix in docs/tools.md and this test together."
            )


class TestWriteToolGuards:
    """Test that write tools refuse to run in the wrong transport mode."""

    @pytest.mark.asyncio
    async def test_upload_refuses_in_cloud_mode(self):
        import os
        import sys

        # Force cloud mode (the default when neither flag is set). upload works
        # in SSH or USB-web, so it should only refuse here in cloud mode.
        for var in ("REMARKABLE_USE_USB_WEB", "REMARKABLE_USE_SSH"):
            os.environ.pop(var, None)
        if "remarkable_mcp.api" in sys.modules:
            import importlib

            import remarkable_mcp.api

            importlib.reload(remarkable_mcp.api)

        result = await mcp.call_tool("remarkable_upload", {"file_path": "/tmp/x.pdf"})
        data = json.loads(result[0][0].text)
        assert "_error" in data
        assert "write transport" in str(data)
        assert "cloud mode" in str(data)

    @pytest.mark.asyncio
    async def test_delete_refuses_when_not_ssh(self):
        import os
        import sys

        os.environ["REMARKABLE_USE_USB_WEB"] = "1"
        os.environ.pop("REMARKABLE_USE_SSH", None)
        if "remarkable_mcp.api" in sys.modules:
            import importlib

            import remarkable_mcp.api

            importlib.reload(remarkable_mcp.api)

        try:
            result = await mcp.call_tool("remarkable_delete", {"document": "anything"})
            data = json.loads(result[0][0].text)
            assert "_error" in data
            assert "SSH" in str(data)
        finally:
            os.environ.pop("REMARKABLE_USE_USB_WEB", None)
            if "remarkable_mcp.api" in sys.modules:
                import importlib

                import remarkable_mcp.api

                importlib.reload(remarkable_mcp.api)


# =============================================================================
# Test SSH metadata framing (regression: trailing newline / listing parser)
# =============================================================================


class TestSSHMetadataFraming:
    """Regression tests for the SSH write/read framing bug.

    Metadata written by the SSH tools had no trailing newline, so the listing
    command's `cat "$f"` ran the next `===FILE===` marker onto the previous
    file's closing brace (`}===FILE===nextid`). The line-start parser then
    treated the merged marker as content, so json.loads hit "Extra data" and
    the item (and the next one in glob order) silently vanished from listings.
    """

    def test_write_remote_json_payload_ends_with_newline(self):
        """Writer must terminate metadata with a newline so files stay separable."""
        from remarkable_mcp.ssh import SSHClient

        client = SSHClient(password="x")
        captured = {}

        def fake_pipe(data, remote_command, timeout=120):
            captured["data"] = data

        with patch.object(client, "_ssh_pipe", side_effect=fake_pipe):
            client._write_remote_json("/x/y.metadata", {"a": 1})

        assert captured["data"].endswith(b"\n"), (
            "metadata payload must end with a newline so the listing parser can "
            "separate adjacent files"
        )

    def test_get_meta_items_parses_file_without_trailing_newline(self):
        """Listing must recover a file whose content abuts the next marker."""
        from remarkable_mcp.ssh import SSHClient

        client = SSHClient(password="x")
        md1 = json.dumps({"visibleName": "Alpha", "type": "CollectionType", "parent": ""})
        md2 = json.dumps({"visibleName": "Beta", "type": "CollectionType", "parent": ""})
        # md1 has NO trailing newline, so its closing brace abuts the next
        # marker -- exactly the corruption observed on-device.
        merged = f"===FILE===id-alpha\n{md1}===FILE===id-beta\n{md2}\n"

        with patch.object(client, "_ssh_command", return_value=merged):
            docs = client.get_meta_items()

        names = sorted(d.name for d in docs)
        assert names == ["Alpha", "Beta"], (
            f"both folders should parse despite the missing newline; got {names}"
        )

    def test_created_folder_is_not_cloud_archived(self):
        """A freshly created local folder must not read as cloud-archived.

        Document.is_cloud_archived is `not synced or parent == 'trash'`, and
        browse/read/recent skip cloud-archived items. Writing synced=False on a
        brand-new on-device folder therefore makes it invisible to those tools
        (delete/move look up by name and were unaffected, masking the bug).
        """
        from remarkable_mcp.ssh import Document, SSHClient

        client = SSHClient(password="x")
        captured = {}
        with patch.object(
            client, "_write_metadata", side_effect=lambda doc_id, md: captured.setdefault("md", md)
        ), patch.object(client, "_restart_xochitl"):
            client.create_folder("Test Folder")

        md = captured["md"]
        doc = Document(
            id="x",
            hash="x",
            name=md["visibleName"],
            doc_type=md["type"],
            parent=md.get("parent", ""),
            synced=md.get("synced", True),  # same default get_meta_items uses
        )
        assert not doc.is_cloud_archived, (
            "create_folder produced a cloud-archived folder; browse/read/recent "
            "will hide it. Newly created on-device items must not be synced=False."
        )
