"""
Test script for comparing the new parsing method with the old
"""
import pygit2
import tempfile
import difflib
import shutil
import time
import os
from src import ROOT_DIR
from pathlib import Path
from collections import defaultdict
from src.log import logger
from src.analysis.utils.git import get_diff, get_file_content_at_commit
from src.analysis.utils.parsing import save_headers_to_temp, parse_and_modify_functions, parse_and_reduce_code,save_all_headers,save_header_directories
from src.analysis.utils.parsing_with_cscope import parse_and_modify_with_cscope
from src.utils import empty_directory

REPO_PATH = (ROOT_DIR.parent / "projects/linux").absolute()

def compare_parsing_methods(repo, commit):
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
            print(file_name)
            
            arch = None
            parts = file_name.split('/')
            if "arch" in parts:
                idx = parts.index("arch")
                if idx + 1 < len(parts):
                    arch = parts[idx + 1]  

            content = get_file_content_at_commit(repo, parent, file_name)

            include_paths = [str(Path(file_name).parent)]
            if(True):
                build_args = [ "-D__KERNEL__"]
            if arch is not None:
                include_paths += [f"arch/{arch}/include",f"arch/{arch}/include/uapi"]
                build_args += [f"-D__{arch}__"]
            include_paths += ["include","include/uapi"]

            """ save_headers_to_temp(
                commit=parent,
                output_dir=headers_dir,
                repo=repo,
                full_code=content,
                loaded_headers=loaded_headers,
                invalid_headers=invalid_headers,
                include_paths=include_paths,
            ) """

            save_header_directories(
                commit=parent,
                output_dir=headers_dir,
                repo=repo,
                include_paths=include_paths
                )
            
            base_name = Path(file_name).name 
            logger.info(base_name)
            file_dir_name = base_name.replace('.', '_')
            output_dir = Path(f"parsed_{commit.id}") / file_dir_name
            output_dir.mkdir(parents=True, exist_ok=True)
            input(...)

            """ start = time.time()
            shorter_original, modified_lines_original = parse_and_modify_functions(
                content, removed_line_numbers, headers_dir, file_name
            )
            stop = time.time()
            original_time = stop - start
            logger.info(f"original parser: {original_time}")
            orig_shorter = output_dir / f"original_{base_name}"
            orig_shorter.write_text(shorter_original) """

            start = time.time()
            shorter_agressive, modified_lines_agressive = parse_and_reduce_code(
                content, removed_line_numbers, headers_dir, include_paths, build_args, file_name,
            )
            stop = time.time()
            agressive_time = stop - start
            logger.info(f"agressive parser: {agressive_time}")
            aggr_path = output_dir / f"agressive_{base_name}"
            aggr_path.write_text(shorter_agressive)

            """ start = time.time()
            shorter_cscope, modified_lines_cscope = parse_and_modify_with_cscope(
                content, removed_line_numbers, temp_dir, file_name 
            )
            stop = time.time()
            cscope_time = stop - start
            logger.info(f"cscope parser: {cscope_time}")
            cscope_path = output_dir / f"cscope_{base_name}"
            cscope_path.write_text(shorter_cscope) """

            def save_diff(a, b, a_name, b_name, diff_name):
                diff = difflib.unified_diff(
                    a.splitlines(keepends=True),
                    b.splitlines(keepends=True),
                    fromfile=a_name,
                    tofile=b_name
                )
                (output_dir / diff_name).write_text(''.join(diff))

            #save_diff(content, shorter_original, base_name, f"original_{base_name}", f"original_{base_name}_diff")
            #save_diff(content, shorter_agressive, base_name, f"agressive_{base_name}", f"agressive_{base_name}_diff")
            #save_diff(content, shorter_cscope, base_name, f"cscope_{base_name}", f"cscope_{base_name}_diff")

            empty_directory(temp_dir, files_to_keep=[output_dir])


if __name__=="__main__":
    repo = pygit2.Repository(str(REPO_PATH))
    commit_sha = "59ba025948be2a92e8bc9ae1cbdaf197660bd508" #"20d72b00ca814d748f5663484e5c53bb2bf37a3a"
    commit = repo.get(commit_sha)
    compare_parsing_methods(repo, commit)