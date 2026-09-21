"""Command line and persistent JSON-lines service for benchmark adapters."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys


def emit(value):
    print(json.dumps(value, ensure_ascii=False, default=str), flush=True)


def install_plugin(destination):
    root = Path(destination).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('plugin_template')
    for name in ('__init__.py', 'plugin.yaml'):
        target = root / name
        data = (source / name).read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f'Refusing to overwrite existing plugin file: {target}')
    for name in ('__init__.py', 'plugin.yaml'):
        shutil.copyfile(source / name, root / name)
    return {'installed': str(root)}


def dispatch(system, request):
    method = request.get('method')
    params = request.get('params') or {}
    methods = {'observe', 'ingest', 'context', 'recall', 'scenes', 'read_scene', 'commit', 'consolidate', 'status', 'switch_session'}
    if method not in methods:
        raise ValueError(f'Unknown method: {method}')
    return getattr(system, method)(**params)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='kylinmemory', description='kylinmemory: standalone four-layer memory')
    parser.add_argument('--home', type=Path, help='isolated data/config directory')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--session', default='default')
    parser.add_argument('--user')
    parser.add_argument('--platform', default='cli')
    parser.add_argument('--team', default='hermes')
    parser.add_argument('--agent', default='default')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('serve', help='persistent JSON-lines requests on stdin/stdout')
    sub.add_parser('status')
    sub.add_parser('scenes')
    sub.add_parser('commit')
    sub.add_parser('consolidate')
    observe = sub.add_parser('observe')
    observe.add_argument('user_text')
    observe.add_argument('--assistant', default='')
    ingest = sub.add_parser('ingest')
    ingest.add_argument('file', type=Path, help='JSON message array, or {messages: [...]}')
    ingest.add_argument('--backfill', action='store_true')
    recall = sub.add_parser('recall')
    recall.add_argument('query')
    recall.add_argument('--limit', type=int, default=5)
    context = sub.add_parser('context')
    context.add_argument('query')
    read_scene = sub.add_parser('read-scene')
    read_scene.add_argument('filename')
    install = sub.add_parser('install-plugin')
    install.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    if args.command == 'install-plugin':
        emit(install_plugin(args.directory))
        return
    from .config import load_config
    from .runtime import MemorySystem
    config = load_config(args.config) if args.config else None
    system = MemorySystem(args.home, config=config, session_id=args.session, user_id=args.user,
                          platform=args.platform, team_id=args.team, agent_id=args.agent)
    try:
        if args.command == 'serve':
            for line in sys.stdin:
                if not line.strip():
                    continue
                request = None
                try:
                    request = json.loads(line)
                    result = dispatch(system, request)
                    emit({'id': request.get('id'), 'result': result})
                except Exception as exc:
                    from .redact import redact_sensitive_text
                    emit({'id': request.get('id') if isinstance(request, dict) else None,
                          'error': {'type': type(exc).__name__, 'message': redact_sensitive_text(str(exc), force=True)}})
        elif args.command == 'observe':
            emit(system.observe(args.user_text, args.assistant))
        elif args.command == 'ingest':
            payload = json.loads(args.file.read_text())
            emit(system.ingest(payload['messages'] if isinstance(payload, dict) else payload, backfill=args.backfill))
        elif args.command in {'recall', 'context'}:
            emit(system.recall(args.query, limit=args.limit) if args.command == 'recall' else system.context(args.query))
        elif args.command == 'read-scene':
            emit(system.read_scene(args.filename))
        else:
            emit(getattr(system, args.command)())
    finally:
        system.close()
