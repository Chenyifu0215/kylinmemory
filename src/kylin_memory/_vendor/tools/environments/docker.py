import logging
import os
import shutil
from typing import Optional
logger = logging.getLogger(__name__)
_DOCKER_SEARCH_PATHS = ['/usr/local/bin/docker', '/opt/homebrew/bin/docker', '/Applications/Docker.app/Contents/Resources/bin/docker']
_docker_executable: Optional[str] = None

def find_docker() -> Optional[str]:
    """Locate the docker (or podman) CLI binary.

    Resolution order:
    1. ``HERMES_DOCKER_BINARY`` env var — explicit override (e.g. ``/usr/bin/podman``)
    2. ``docker`` on PATH via ``shutil.which``
    3. ``podman`` on PATH via ``shutil.which``
    4. Well-known macOS Docker Desktop install locations

    Returns the absolute path, or ``None`` if neither runtime can be found.
    """
    global _docker_executable
    if _docker_executable is not None:
        return _docker_executable
    override = os.getenv('HERMES_DOCKER_BINARY')
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        _docker_executable = override
        logger.info('Using HERMES_DOCKER_BINARY override: %s', override)
        return override
    found = shutil.which('docker')
    if found:
        _docker_executable = found
        return found
    found = shutil.which('podman')
    if found:
        _docker_executable = found
        logger.info('Using podman as container runtime: %s', found)
        return found
    for path in _DOCKER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _docker_executable = path
            logger.info('Found docker at non-PATH location: %s', path)
            return path
    return None

