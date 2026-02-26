#!/usr/bin/env python3

import re
import subprocess
import git_filter_repo as fr

def fix_tabs(data: bytes) -> bytes:
    result = []
    for line in data.split(b'\n'):
        # Trouver la longueur du prefixe d'indentation (TABs + espaces)
        i = 0
        while i < len(line) and line[i:i+1] in (b'\t', b' '):
            i += 1

        prefix = line[:i]
        rest = line[i:]

        # Recalculer le prefixe en respectant les tabulations sur 8 colonnes
        col = 0
        new_prefix = b''
        for ch in prefix:
            if ch == ord('\t'):
                # Sauter a la prochaine colonne multiple de 8
                spaces = 8 - (col % 8)
                new_prefix += b' ' * spaces
                col += spaces
            else:  # espace
                new_prefix += b' '
                col += 1

        # Supprimer les espaces et tabs en fin de ligne
        rest = rest.rstrip(b' \t')

        result.append(new_prefix + rest)

    return b'\n'.join(result)

# Blobs deja traites : ancien_id -> nouvel_id (ou même id si inchange)
blob_cache: dict[bytes, bytes] = {}

def read_blob(blob_id: bytes) -> bytes:
    """Lit le contenu d'un blob Git via git cat-file."""
    result = subprocess.run(
        ['git', 'cat-file', 'blob', blob_id.decode()],
        capture_output=True
    )
    return result.stdout

def write_blob(data: bytes) -> bytes:
    """Ecrit un nouveau blob Git et retourne son SHA1."""
    result = subprocess.run(
        ['git', 'hash-object', '-w', '--stdin'],
        input=data,
        capture_output=True
    )
    return result.stdout.strip()

def commit_callback(commit, metadata):
    for change in commit.file_changes:
        if change.blob_id is None:
            continue
        if not (change.filename.endswith(b'.cpp') or
                change.filename.endswith(b'.notcpp') or
                change.filename.endswith(b'.noth') or
                change.filename.endswith(b'.h')):
            continue

        original_id = change.blob_id

        if original_id in blob_cache:
            change.blob_id = blob_cache[original_id]
            continue

        original_data = read_blob(original_id)
        new_data = fix_tabs(original_data)

        if new_data != original_data:
            new_id = write_blob(new_data)
            blob_cache[original_id] = new_id
            change.blob_id = new_id
        else:
            blob_cache[original_id] = original_id

args = fr.FilteringOptions.parse_args(['--force','--replace-refs','delete-no-add'])
fr.RepoFilter(args, commit_callback=commit_callback).run()

