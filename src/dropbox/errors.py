"""Custom exceptions for the dropbox module."""

class DropboxError(Exception):
    """Base exception for all dropbox errors."""
    pass


class EmptyFileError(DropboxError):
    """Raised when an uploaded file has no content or no rows."""
    pass


class InvalidFileTypeError(DropboxError):
    """Raised when an uploaded file is not a supported format (.csv or .xlsx)."""
    pass


class FileSizeLimitError(DropboxError):
    """Raised when an uploaded file exceeds the 5 MB limit."""
    pass


class NoTextColumnError(DropboxError):
    """Raised when automatic text column detection fails."""
    pass


class ColumnNotFoundError(DropboxError):
    """Raised when a specified column override does not exist in the dataset."""
    pass
