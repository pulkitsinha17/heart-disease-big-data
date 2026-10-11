"""
HDFS Integration Module for Heart Disease Big Data Pipeline.

Provides a robust WebHDFS client that:
1. Connects to NameNode WebHDFS endpoint (default: http://localhost:9870/webhdfs/v1).
2. Intercepts HTTP 307 temporary redirects from NameNode to DataNodes and automatically
   rewrites internal Docker hostnames (e.g., 'datanode:9864') to host-accessible
   endpoints ('localhost:9864').
3. Supports creating directories, writing JSON Lines (.jsonl), appending records,
   reading files back, and verifying cluster health.
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import requests

DEFAULT_NAMENODE_URL: str = os.getenv("HDFS_NAMENODE_URL", "http://localhost:9870")
DEFAULT_HDFS_DIR: str = "/heart-disease/raw"
DEFAULT_HDFS_FILE: str = f"{DEFAULT_HDFS_DIR}/heart_disease_stream.jsonl"


class HDFSWriter:
    """
    Client wrapper for interacting with Hadoop HDFS via WebHDFS REST API.
    Designed to work seamlessly from a Windows host communicating with Docker containers.
    """

    def __init__(
        self,
        namenode_url: str = DEFAULT_NAMENODE_URL,
        datanode_host_map: Tuple[str, str] = ("datanode", "localhost"),
        timeout_seconds: int = 15,
    ):
        self.namenode_url = namenode_url.rstrip("/")
        self.base_url = f"{self.namenode_url}/webhdfs/v1"
        self.dn_old, self.dn_new = datanode_host_map
        self.timeout = timeout_seconds
        self.session = requests.Session()

    def _resolve_datanode_url(self, url: str) -> str:
        """
        Rewrite internal Docker DataNode hostname in redirect URL to host-accessible address.
        Example: http://datanode:9864/... -> http://localhost:9864/...
        """
        if self.dn_old and self.dn_new:
            return url.replace(f"://{self.dn_old}:", f"://{self.dn_new}:")
        return url

    def is_healthy(self) -> Tuple[bool, str]:
        """Verify WebHDFS service responsiveness."""
        try:
            url = f"{self.base_url}/?op=LISTSTATUS"
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return True, "NameNode WebHDFS is reachable and healthy."
            return False, f"NameNode returned unexpected status {resp.status_code}: {resp.text}"
        except Exception as e:
            return False, f"Failed to connect to NameNode at {self.namenode_url}: {e}"

    def exists(self, hdfs_path: str) -> bool:
        """Check whether an HDFS file or directory exists."""
        clean_path = hdfs_path.lstrip("/")
        url = f"{self.base_url}/{clean_path}?op=GETFILESTATUS"
        resp = self.session.get(url, timeout=self.timeout)
        return resp.status_code == 200

    def mkdirs(self, hdfs_dir: str) -> bool:
        """Ensure an HDFS directory path exists (creates parents as needed)."""
        clean_path = hdfs_dir.lstrip("/")
        if not clean_path:
            return True
        url = f"{self.base_url}/{clean_path}?op=MKDIRS"
        resp = self.session.put(url, timeout=self.timeout)
        if resp.status_code == 200:
            return resp.json().get("boolean", True)
        resp.raise_for_status()
        return False

    def write_jsonl(
        self,
        hdfs_path: str,
        records: List[Dict[str, Any]],
        overwrite: bool = True,
    ) -> int:
        """
        Write a batch of dictionary records to HDFS formatted as JSON Lines (.jsonl).
        Ensures parent directory exists and rewrites DataNode redirect.
        Returns the number of written records.
        """
        clean_path = hdfs_path.lstrip("/")
        parent_dir = clean_path.rsplit("/", 1)[0] if "/" in clean_path else ""
        if parent_dir:
            self.mkdirs(parent_dir)

        # Step 1: Request file creation from NameNode (expects 307 redirect)
        ow_str = "true" if overwrite else "false"
        create_url = f"{self.base_url}/{clean_path}?op=CREATE&overwrite={ow_str}"
        resp_nn = self.session.put(create_url, allow_redirects=False, timeout=self.timeout)

        if resp_nn.status_code != 307:
            raise RuntimeError(
                f"NameNode rejected CREATE for '{hdfs_path}' with status {resp_nn.status_code}: {resp_nn.text}"
            )

        # Step 2: Upload payload to DataNode via resolved redirect location
        redirect_location = resp_nn.headers.get("Location")
        if not redirect_location:
            raise RuntimeError("NameNode 307 response did not provide a Location header.")

        datanode_url = self._resolve_datanode_url(redirect_location)

        # Serialize records as JSON Lines; preserve nulls without NaN tokens
        jsonl_content = "\n".join([json.dumps(r, allow_nan=False) for r in records]) + "\n"

        resp_dn = self.session.put(
            datanode_url,
            data=jsonl_content.encode("utf-8"),
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout,
        )

        if resp_dn.status_code not in (200, 201):
            raise RuntimeError(
                f"DataNode write failed for '{hdfs_path}' with status {resp_dn.status_code}: {resp_dn.text}"
            )

        return len(records)

    def append_record_jsonl(self, hdfs_path: str, record: Dict[str, Any]) -> bool:
        """
        Append a single dictionary record to an existing HDFS file.
        If the file does not exist, it creates it.
        """
        clean_path = hdfs_path.lstrip("/")
        if not self.exists(clean_path):
            self.write_jsonl(clean_path, [record], overwrite=True)
            return True

        # Step 1: Request APPEND from NameNode
        append_url = f"{self.base_url}/{clean_path}?op=APPEND"
        resp_nn = self.session.post(append_url, allow_redirects=False, timeout=self.timeout)

        if resp_nn.status_code != 307:
            raise RuntimeError(
                f"NameNode rejected APPEND for '{hdfs_path}' with status {resp_nn.status_code}: {resp_nn.text}"
            )

        # Step 2: Upload to DataNode
        datanode_url = self._resolve_datanode_url(resp_nn.headers["Location"])
        line = json.dumps(record, allow_nan=False) + "\n"

        resp_dn = self.session.post(
            datanode_url,
            data=line.encode("utf-8"),
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout,
        )

        if resp_dn.status_code not in (200, 201):
            raise RuntimeError(
                f"DataNode append failed for '{hdfs_path}' with status {resp_dn.status_code}: {resp_dn.text}"
            )

        return True

    def read_jsonl(self, hdfs_path: str) -> List[Dict[str, Any]]:
        """
        Read a JSON Lines file from HDFS and parse every line into a dictionary.
        """
        clean_path = hdfs_path.lstrip("/")
        open_url = f"{self.base_url}/{clean_path}?op=OPEN"
        resp_nn = self.session.get(open_url, allow_redirects=False, timeout=self.timeout)

        if resp_nn.status_code != 307:
            raise RuntimeError(
                f"NameNode rejected OPEN for '{hdfs_path}' with status {resp_nn.status_code}: {resp_nn.text}"
            )

        datanode_url = self._resolve_datanode_url(resp_nn.headers["Location"])
        resp_dn = self.session.get(datanode_url, timeout=self.timeout)
        resp_dn.raise_for_status()

        lines = [l.strip() for l in resp_dn.text.strip().split("\n") if l.strip()]
        return [json.loads(l) for l in lines]

    def delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        """Delete an HDFS file or folder."""
        clean_path = hdfs_path.lstrip("/")
        rec_str = "true" if recursive else "false"
        url = f"{self.base_url}/{clean_path}?op=DELETE&recursive={rec_str}"
        resp = self.session.delete(url, timeout=self.timeout)
        if resp.status_code == 200:
            return resp.json().get("boolean", True)
        return False
