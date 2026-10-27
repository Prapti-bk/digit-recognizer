"""Domain-specific exceptions so callers can handle failures precisely."""


class DigitRecognizerError(Exception):
    """Base class for all errors raised by this package."""


class BlankImageError(DigitRecognizerError, ValueError):
    """The supplied image contains no visible ink."""


class ModelNotFoundError(DigitRecognizerError, FileNotFoundError):
    """No trained model exists at the expected path."""
