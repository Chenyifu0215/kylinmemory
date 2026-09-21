"""Discoverable Hermes MemoryProvider registration."""
from kylinmemory.plugin import LayeredMemoryProvider


def register(ctx):
    ctx.register_memory_provider(LayeredMemoryProvider())
