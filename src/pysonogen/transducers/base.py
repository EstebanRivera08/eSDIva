from abc import ABC, abstractmethod

class BaseTransducer(ABC):
    """
    Abstract base class for all transducer types.
    Provides a common interface and repr for parameters.
    """
    @abstractmethod
    def __repr__(self):
        """Return a concise summary of constructor parameters."""
        pass
    