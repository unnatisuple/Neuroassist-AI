"""
NeuroAssist AI v2 — File Encryption Service (AES-256 / Fernet)
Ensures patient MRI files are encrypted at rest for compliance (HIPAA).
"""

from cryptography.fernet import Fernet
import os
from backend.config import settings
from loguru import logger

_fernet = None


def get_fernet() -> Fernet:
    """Get or initialize the Fernet cipher instance."""
    global _fernet
    if _fernet is not None:
        return _fernet

    key_str = getattr(settings, "file_encryption_key", "")

    if not key_str:
        # Generate a temporary key for local development
        logger.warning(
            "FILE_ENCRYPTION_KEY not configured in settings. "
            "Generating a temporary encryption key. Files will not be decryptable "
            "after server restarts."
        )
        key = Fernet.generate_key()
    else:
        try:
            # Ensure it is valid bytes key
            key = key_str.encode()
            # Test key validity
            Fernet(key)
        except Exception as e:
            logger.error(f"Invalid FILE_ENCRYPTION_KEY format: {e}. Generating fallback key.")
            key = Fernet.generate_key()

    _fernet = Fernet(key)
    return _fernet


def encrypt_data(data: bytes) -> bytes:
    """Encrypt binary data."""
    return get_fernet().encrypt(data)


def decrypt_data(data: bytes) -> bytes:
    """Decrypt binary data."""
    return get_fernet().decrypt(data)


from contextlib import contextmanager
import tempfile

@contextmanager
def decrypted_temp_file(encrypted_file_path: str):
    """
    Decrypts an encrypted file at rest to a temporary file,
    yields the temp file path, and deletes it afterward.
    """
    with open(encrypted_file_path, "rb") as f:
        encrypted_data = f.read()

    decrypted = decrypt_data(encrypted_data)
    ext = os.path.splitext(encrypted_file_path)[1]

    temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(temp_fd, "wb") as f_temp:
            f_temp.write(decrypted)
        yield temp_path
    finally:
        try:
            os.close(temp_fd)
        except OSError:
            pass
        try:
            os.remove(temp_path)
        except OSError:
            pass
