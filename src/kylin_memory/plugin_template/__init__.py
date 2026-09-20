"""Discoverable Hermes MemoryProvider registration."""
from kylin_memory.plugin import LayeredMemoryProvider


def register(ctx):
    ctx.register_memory_provider(LayeredMemoryProvider())
