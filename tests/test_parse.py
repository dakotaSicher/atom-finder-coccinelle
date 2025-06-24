import pygit2
import tempfile
import difflib
import shutil
import time
from pathlib import Path
from collections import defaultdict
from src.log import logger
from src.analysis.utils.git import get_diff, get_file_content_at_commit
from src.analysis.utils.parsing import save_headers_to_temp, parse_and_modify_functions, parse_and_reduce_code
from src.analysis.utils.parsing_with_cscope import parse_and_modify_with_cscope
from src.utils import empty_directory
#from tests.test_utils import save_and_diff_contents

def test_parsing_methods(repo, commit):
    logger.info(f"Testing parsing on commit: {commit.id}")

    _, removed_lines = get_diff(repo, commit)

    if removed_lines:
        loaded_headers = defaultdict(list)
        invalid_headers = defaultdict(list)
        parent = commit.parents[0]
    output = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name, removed in removed_lines.items():
            removed_line_numbers = [line.old_lineno for line in removed]
            headers_dir = Path(temp_dir, "headers")
            content = get_file_content_at_commit(repo, parent, file_name)
            save_headers_to_temp(
                commit=parent,
                output_dir=headers_dir,
                repo=repo,
                full_code=content,
                loaded_headers=loaded_headers,
                invalid_headers=invalid_headers,
            )
            logger.info(file_name)
            start = time.time()
            shorter_original, modified_lines_original = parse_and_modify_functions(
                content, removed_line_numbers, headers_dir, file_name
            )
            end = time.time()
            original_time = end - start
            logger.info(f"original parser: {original_time}")

            start = time.time
            shorter_agressive, modified_lines_agressive = parse_and_reduce_code(
                content, removed_line_numbers, headers_dir, file_name
            )
            end = time.time()
            agressive_time = end - start
            logger.info(f"agressive parser: {agressive_time}")

            start = time.time()
            shorter_cscope, modified_lines_cscope = parse_and_modify_with_cscope(
                content, removed_line_numbers, temp_dir, file_name 
            )
            end = time.time()
            cscope_time = end - start
            logger.info(f"cscope parser: {cscope_time}")
    
            safe_file_name = file_name.replace('.', '_')
            output_dir = Path(f"parse_{commit.id}_output") / safe_file_name
            output_dir.mkdir(parents=True, exist_ok=True)

            orig_path = output_dir / file_name
            orig_path.write_text(content)
            orig_shorter = output_dir / f"original_{file_name}"
            orig_shorter.write_text(shorter_original)
            aggr_path = output_dir / f"agressive_{file_name}"
            aggr_path.write_text(shorter_agressive)
            cscope_path = output_dir / f"cscope_{file_name}"
            cscope_path.write_text(shorter_cscope)

            def save_diff(a, b, a_name, b_name, diff_name):
                diff = difflib.unified_diff(
                    a.splitlines(keepends=True),
                    b.splitlines(keepends=True),
                    fromfile=a_name,
                    tofile=b_name
                )
                (output_dir / diff_name).write_text(''.join(diff))

            save_diff(content, shorter_original, file_name, f"original_{file_name}", f"original_{file_name}_diff")
            save_diff(content, shorter_agressive, file_name, f"agressive_{file_name}", f"agressive_{file_name}_diff")
            save_diff(content, shorter_cscope, file_name, f"cscope_{file_name}", f"cscope_{file_name}_diff")

            empty_directory(temp_dir, files_to_keep=[output_dir])

    shutil.rmtree(temp_dir)
