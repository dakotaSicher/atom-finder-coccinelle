from collections import defaultdict
from pathlib import Path
from pygit2.enums import DeltaStatus, DiffOption



def get_file_content_at_commit(repo, commit, file_path):
    """
    Retrieve the content of a file at a specific commit in a Git repository.

    Args:
    repo_path (str): Path to the repository.
    commit_sha (str): The SHA of the commit.
    file_path (str): Path to the file within the repository.

    Returns:
    str: Content of the file at the specified commit, or an error message if the file does not exist.
    """
    try:
        # Retrieve the entry and blob from the commit's tree
        file_entry = commit.tree[file_path]
        blob = repo.get(file_entry.id)
        return blob.data.decode(
            "utf-8"
        )  # Assuming the file content is text and utf-8 encoded
    except KeyError:
        return "File not found in the specified commit."


def get_diff(repo, commit):
    parent = commit.parents[0]
    diff = repo.diff(parent, commit, context_lines=0, interhunk_lines=0)

    added_lines = defaultdict(list)
    removed_lines = defaultdict(list)
    for patch in diff:
        status = patch.delta.status
        if status != DeltaStatus.MODIFIED:
            continue
        file_name = patch.delta.new_file.path
        if Path(file_name).suffix not in (".c", ".h"):
            continue

        for hunk in patch.hunks:
            for line in hunk.lines:
                if line.old_lineno != -1:
                    removed_lines[file_name].append(line)
                if line.new_lineno != -1:
                    added_lines[file_name].append(line)
        """ for hunk in patch.hunks:
            for line in hunk.lines:
                removed_lines[file_name].append(line)
                added_lines[file_name].append(line)
                assert len(removed_lines[file_name]) == len(added_lines[file_name]) """

    return added_lines, removed_lines


def get_line_mappings(repo, commit):
    """
    Return a dict: { filename: [ (old_lineno or -1, new_lineno or -1), ... ] }
    Only processes files where the commit->parent diff status is MODIFIED
    and only for .c/.h files (change the suffix filter as needed).
    """
    # get the first parent (assumes commit has at least one parent)
    parent = commit.parents[0]
    # zero context to keep hunks tight (same as your original)
    diff = repo.diff(parent, commit, context_lines=1, interhunk_lines=0)

    mappings = defaultdict(list)

    for patch in diff:
        status = patch.delta.status
        # compare with pygit2 constant (some codebases use other enums; this is standard)
        if status != DeltaStatus.MODIFIED:
            continue

        file_name = patch.delta.new_file.path
        if Path(file_name).suffix not in (".c", ".h"):
            continue

        for hunk in patch.hunks:
            removed_buf = []
            added_buf = []

            def flush_buffers():
                # pair as many as possible -> treated as modified
                nonlocal removed_buf, added_buf
                n = min(len(removed_buf), len(added_buf))
                for i in range(n):
                    mappings[file_name].append((removed_buf[i].old_lineno, added_buf[i].new_lineno))
                # leftovers
                for r in removed_buf[n:]:
                    mappings[file_name].append((r.old_lineno, -1))
                for a in added_buf[n:]:
                    mappings[file_name].append((-1, a.new_lineno))
                removed_buf = []
                added_buf = []

            for line in hunk.lines:
                # line.origin sometimes comes as a byte/int; normalize to a char
                origin = line.origin
                #print(line.origin)
                if isinstance(origin, int):
                    origin = chr(origin)
                # ' ' is context, '-' is removal, '+' is addition
                if origin == ' ':
                    # context breaks runs — flush any pairing run
                    flush_buffers()
                    continue
                elif origin == '-':
                    removed_buf.append(line)
                elif origin == '+':
                    added_buf.append(line)
                else:
                    # unusual origin (e.g. '\0' or something else) — flush to be safe
                    flush_buffers()

            # at end of hunk flush any remaining buffers
            flush_buffers()

    return mappings


def get_block_mappings(repo, commit):
    """
    Return a dict: { filename: [ ([block removed lines],[block added lines]), ... ] }
    Only processes files where the commit->parent diff status is MODIFIED
    and only for .c/.h files (change the suffix filter as needed).
    """
    # get the first parent (assumes commit has at least one parent)
    parent = commit.parents[0]
    # zero context to keep hunks tight (same as your original)
    diff = repo.diff(parent, commit,  context_lines=1, interhunk_lines=0)
    
    mappings = defaultdict(list)

    for patch in diff:
        status = patch.delta.status
        # compare with pygit2 constant (some codebases use other enums; this is standard)
        if status != DeltaStatus.MODIFIED:
            continue

        file_name = patch.delta.new_file.path
        if Path(file_name).suffix not in (".c", ".h"):
            continue

        for hunk in patch.hunks:
            removed_buf = []
            added_buf = []

            def flush_buffers():
                nonlocal removed_buf, added_buf
                if removed_buf and added_buf:
                    mappings[file_name].append(
                        ([(r.old_lineno,r.content) for r in removed_buf],
                         [(a.new_lineno,a.content) for a in added_buf]))
                elif removed_buf:
                    mappings[file_name].append(([(r.old_lineno,r.content) for r in removed_buf],[]))
                elif added_buf:
                    mappings[file_name].append(([],[(a.new_lineno,a.content) for a in added_buf]))
                removed_buf = []
                added_buf = []

            for line in hunk.lines:
                origin = line.origin
                if isinstance(origin, int):
                    origin = chr(origin)
                # ' ' is context, '-' is removal, '+' is addition
                if origin == ' ':
                    # context breaks runs — flush any pairing run
                    flush_buffers()
                    continue
                elif origin == '-':
                    removed_buf.append(line)
                elif origin == '+':
                    added_buf.append(line)
                else:
                    flush_buffers()

            flush_buffers()

    return mappings


 

