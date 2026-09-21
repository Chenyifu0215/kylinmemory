"""kylinmemory: standalone L0/L1/L2/L3 memory subsystem."""
__version__ = '0.2.0'


def __getattr__(name):
    if name == 'MemorySystem':
        from .runtime import MemorySystem
        return MemorySystem
    raise AttributeError(name)


__all__ = ['MemorySystem']
