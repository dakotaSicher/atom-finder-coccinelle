from collections import defaultdict
import click
import csv
import json
import multiprocessing
import re
import tempfile
import pygit2
import time
from pathlib import Path
from timelength import TimeLength
from src import ROOT_DIR
from src.analysis.utils.parsing import run_coccinelle_for_file_at_commit
from src.analysis.utils.git import get_diff
from src.analysis.utils.utils import append_rows_to_csv, append_to_json, safely_load_json
from src.run_cocci import CocciPatch
from src.log import logger


PATCHES_TO_SKIP = [CocciPatch.OMITTED_CURLY_BRACES]
REPO_PATH = (ROOT_DIR.parent / "projects/linux").absolute()  # Path to the Linux kernel Git repository
COMMITS_FILE_PATH = Path("commits.json")  # Path to a JSON file containing commit hashes
RESULTS_DIR = Path("./results")
LAST_PROCESSED_DIR = Path("./last_processed")
ERRORS_FILE_DIR = Path("./extract")
NUMBER_OF_PROCESSES = 5


def find_removed_atoms(repo, commit):
    """
    Get removed lines (lines removed in a commit) by comparing the commit to its parent.
    """
    logger.info(f"Current commit: {commit.id}")
    atoms = []

    _, removed_lines = get_diff(repo, commit)

    if removed_lines:
        loaded_headers = defaultdict(list)
        invalid_headers = defaultdict(list)

    parent = commit.parents[0]
    output = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name, removed in removed_lines.items():
            removed_line_numbers = [line.old_lineno for line in removed]
            atoms = run_coccinelle_for_file_at_commit(
                repo,
                file_name,
                parent,
                removed_line_numbers,
                temp_dir,
                loaded_headers,
                invalid_headers,
                PATCHES_TO_SKIP,
            )

            for row in atoms:
                atom, path, start_line, start_col, end_line, end_col, code = row
                output_row = [atom, file_name, str(commit.id), start_line, start_col, code]
                output.append(output_row)

    return output


def iterate_commits_and_extract_removed_code(repo_path, stop_commit, commits_file_path ,history_length = None):
    """
    Iterate through commits, check commit message, and retrieve removed code if condition is met.
    Stop iteration when the stop_commit is reached.

    Args:
        repo_path (str): Path to the repository.
        stop_commit (str): SHA of the commit where the iteration should stop.
        sha_condition (str): Pattern to match 'Fixes: <sha>'.
    """
    repo = pygit2.Repository(repo_path)
    head = repo.head.target  # Get the HEAD commit

    # Regular expression to match 'Fixes: <SHA>'
    fixes_pattern = re.compile(r"Fixes:\s+([0-9a-fA-F]{6,12})", re.IGNORECASE)

    # Calculate cutoff timestamp if history_length is provided
    cutoff_timestamp = None
    if history_length is not None:
        tl = TimeLength(history_length)
        now = int(time.time())
        cutoff_timestamp = now - int(tl.to_minutes()*60)


    commit_fixes = []

    stop_iteration = False
    for commit in repo.walk(head, pygit2.GIT_SORT_TIME):
        commit_message = commit.message.strip()
        print(commit.id)

        # Filter by cutoff timestamp if set
        if cutoff_timestamp is not None and commit.commit_time < cutoff_timestamp:
            break

        # make sure the last pair is added
        if stop_iteration:
            break

        # Check the condition using the regex
        if fixes_pattern.search(commit_message):
            commit_fixes.append(str(commit.id))

        # Stop when the specific commit is reached
        if str(commit.id) == stop_commit:
            stop_iteration = True

    commits_file_path.write_text(json.dumps(commit_fixes))


def load_processed_data(processed_path):
    """Load processed data from a JSON file."""
    if processed_path.is_file():
        return json.loads(processed_path.read_text())
    return {"count": 0, "count_w_atoms": 0, "last_commit": None}


def get_removed_lines(repo_path, commits, output, processed_path, errors_path):
    """Process commits and extract lines removed in each commit."""
    repo = pygit2.Repository(str(repo_path))
    processed = load_processed_data(processed_path)
    count, count_w_atoms = processed["count"], processed["count_w_atoms"]
    first_commit = processed["last_commit"]
    found_first_commit = first_commit is None

    for commit_sha in commits:
        if first_commit and not found_first_commit:
            found_first_commit = commit_sha == first_commit
            continue
        if not found_first_commit:
            continue
        try:
            commit = repo.get(commit_sha)
            atoms = find_removed_atoms(repo, commit)
            count += 1
            if atoms:
                append_rows_to_csv(output, atoms)
                count_w_atoms += 1
            logger.info(f"Processed {count} commits, {count_w_atoms} with atoms.")
        except Exception as e:
            logger.error(e)
            append_to_json(errors_path, {"commit_sha": commit_sha, "error": str(e)})
            continue
        processed.update({"count": count, "count_w_atoms": count_w_atoms, "last_commit": str(commit.id)})
        processed_path.write_text(json.dumps(processed))


def execute(repo_path, commits, number_of_processes, results_dir, last_procesed_dir, errors_dir):
    """
    Main function to spawn the processes.
    """
    # Create a pool of worker processes
    chunks = chunkify(commits, number_of_processes)
    with multiprocessing.Pool(processes=number_of_processes) as pool:
        # Create a list of tuples, each containing arguments for task_function
        tasks = []
        for i in range(number_of_processes):
            tasks.append((repo_path, chunks[i], results_dir / f"atoms{i+1}.csv", last_procesed_dir / f"last_processed{i+1}.json",  errors_dir / f"errors{i+1}.json"))
        # Use starmap to pass multiple arguments to the task function
        pool.starmap(get_removed_lines, tasks)


def chunkify(lst, n):
    """
    Divide the input list into n chunks.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def combine_results(results_folder):

    combined_file_path = results_folder / "atoms.csv"

    with combined_file_path.open("w", newline="") as combined_file:
        writer = csv.writer(combined_file)

        for file_path in results_folder.iterdir():
            if file_path.name != combined_file_path.name:
                if file_path.is_file() and file_path.suffix == ".csv":
                    print(f"Adding {file_path.name}")
                    with file_path.open("r", newline="") as file:
                        reader = csv.reader(file)
                        for row in reader:
                            writer.writerow(row)


#if __name__ == "__main__":
@click.command()
@click.argument("linux_dir",type=Path)
@click.option("-o", "--output-dir", type=Path, default="./output")
@click.option("-t","--history_length",type=str, default=None)

def extract_linux_fixes(linux_dir,output_dir:Path,history_length):
    stop_commit = "c511851de162e8ec03d62e7d7feecbdf590d881d" # this is the commit when the fix: convention was introduced
    output_dir.mkdir(exist_ok=True)
    commits_file_path = output_dir / "commits.json"


    commits = safely_load_json(commits_file_path)
    if not commits or commits[-1] != stop_commit:
        iterate_commits_and_extract_removed_code(linux_dir, stop_commit, commits_file_path, history_length)

    last_processed = output_dir / LAST_PROCESSED_DIR
    last_processed.mkdir(exist_ok=True)
    results_dir = output_dir/RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    errors_dir = results_dir / ERRORS_FILE_DIR
    errors_dir.mkdir(exist_ok=True)

    commits = json.loads(commits_file_path.read_text())
    # commits = ["e589f9b7078e1c0191613cd736f598e81d2390de"]

"""    if len(commits) == 1 or NUMBER_OF_PROCESSES == 1:
        get_removed_lines(linux_dir, commits, results_dir / "atoms.csv", last_processed / "last_processed.json", errors_dir / "errors.json")
    else:
        execute(linux_dir, commits, NUMBER_OF_PROCESSES, results_dir, last_processed, errors_dir)

    combine_results(results_dir)"""

if __name__ == "__main__":
    extract_linux_fixes()