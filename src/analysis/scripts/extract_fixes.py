from collections import defaultdict
import csv
import sys
import cProfile
import queue
import json
import signal
import multiprocessing
import re
import tempfile
import pygit2
import time
from datetime import datetime
from pathlib import Path
from timelength import TimeLength
from src import ROOT_DIR
from src.analysis.utils.parsing import run_coccinelle_for_file_at_commit
from src.analysis.utils.git import get_diff
from src.analysis.utils.utils import append_rows_to_csv, append_to_json, safely_load_json
from src.run_cocci import CocciPatch
from src.log import logger

PATCHES_TO_SKIP = [CocciPatch.OMITTED_CURLY_BRACES]

def _timeout_handler(signum, frame):
    raise TimeoutError("find_removed_atoms timed out")

def find_removed_atoms(repo, commit, timeout, jobs):
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
    skipped = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name, removed in removed_lines.items():
            removed_line_numbers = [line.old_lineno for line in removed]

            atoms, skipped_patches = run_coccinelle_for_file_at_commit(
                repo,
                file_name,
                parent,
                removed_line_numbers,
                temp_dir,
                loaded_headers,
                invalid_headers,
                timeout,
                jobs,
                PATCHES_TO_SKIP,
            )

            for row in atoms:
                atom, path, start_line, start_col, end_line, end_col, code = row
                output_row = [atom, file_name, str(parent.id), start_line, start_col, code]
                output.append(output_row)
            if skipped_patches:
                row = [commit.id, file_name] + skipped_patches
                skipped.append(row)

    return output, skipped, len(removed_lines)


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
        #print(commit.id)

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
    return {"count": 0, "count_w_atoms": 0, "last_commit": None,"number_files": 0,}


'''
Single thread version 
'''
def get_removed_lines(repo_path, commits, output, processed_path, errors_path, skipped_path, timeout, jobs):
    """Process commits and extract lines removed in each commit."""
    repo = pygit2.Repository(str(repo_path))
    processed = load_processed_data(processed_path)
    count,number_files, count_w_atoms = processed["count"],processed["number_files"], processed["count_w_atoms"]

    processing_info = {
        "current_commit": None, 
        "start_time": None,
        "count": None,
        "number_files": None,
        "count_w_atoms": None,
        "last_commit": None,
    }

    for commit_sha in commits:
        processing_info["current_commit"] = commit_sha
        processing_info["start_time"] = datetime.now().strftime("%H:%M:%S")
        processed.update(processing_info)
        processed_path.write_text(json.dumps(processed))

        try:
            commit = repo.get(commit_sha)
            atoms,skipped,num_files = find_removed_atoms(repo, commit, timeout, jobs)
            number_files += num_files
            count += 1
            if atoms:
                append_rows_to_csv(output, atoms)
                count_w_atoms += 1
            #logger.info(f"{datetime.now().strftime("%H:%M:%S")}: Processed {count} commits, {count_w_atoms} with atoms.")
            if skipped:
                append_rows_to_csv(skipped_path, skipped)
            logger.info(f" Processed {count} commits, {count_w_atoms} with atoms.")
        except Exception as e:
            logger.error(e)
            append_to_json(errors_path, {"commit_sha": commit_sha, "error": str(e)})
            continue
            
        processing_info["count"] = count
        processing_info["number_files"] = number_files
        processing_info["count_w_atoms"] = count_w_atoms
        processing_info["last_commit"] = str(commit_sha)

        
'''
Mutliprocessing using static commit lists
'''
def execute(repo_path, commits, number_of_processes, results_dir, last_procesed_dir, errors_dir):
    """
    Main function to spawn the processes.
    """
    if not number_of_processes:
        number_of_processes = multiprocessing.cpu_count()
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


def _get_removed_lines_worker(repo_path, commit_queue, output, processed_path, errors_path, skipped_path, timeout, jobs, pid):
    """
    Worker function that pulls commits from the shared queue.
    """
    #sys.stdout = open(f"worker_{pid}.out", "w")
    #sys.stderr = open(f"worker_{pid}.err", "w")

    repo = pygit2.Repository(str(repo_path))
    processed = load_processed_data(processed_path)
    count, number_files, count_w_atoms = processed["count"],processed["number_files"], processed["count_w_atoms"]

    processing_info = {
        "current_commit": None, 
        "start_time": None,
        "count": count,
        "number_files": number_files,
        "count_w_atoms": count_w_atoms,
        "last_commit": None,
    }

    while True:
        commit_sha = commit_queue.get()
        if commit_sha is None:
            break

        processing_info["current_commit"] = commit_sha
        processing_info["start_time"] = datetime.now().strftime("%H:%M:%S")
        processed.update(processing_info)
        processed_path.write_text(json.dumps(processed))

        try:
            commit = repo.get(commit_sha)
            atoms,skipped,num_files = find_removed_atoms(repo, commit, timeout, jobs)
            number_files += num_files
            count += 1
            if atoms:
                append_rows_to_csv(output, atoms)
                count_w_atoms += 1
            if skipped:
                append_rows_to_csv(skipped_path, skipped)
            logger.info(f"PID {pid}: Processed {count} commits, {count_w_atoms} with atoms.")
        except Exception as e:
            logger.error(e)
            append_to_json(errors_path, {"commit_sha": commit_sha, "error": str(e)})
            continue
        
        processing_info["count"] = count
        processing_info["number_files"] = number_files
        processing_info["count_w_atoms"] = count_w_atoms
        processing_info["last_commit"] = str(commit.id)
        
    
def execute_queue(repo_path, commits, number_of_processes, results_dir, last_procesed_dir, errors_dir, timeout, jobs):
    '''
    Multiprocessing using shared queue of commits 

    Args:
        repo_path (str): Path to the repository.
        commits (list[str]): list of commits to be processed
        number_of_processes (int): number of worker processes to use
        *_dir (path): output directories 
        timeout (int): timeout in seconds for find_removed_atoms
    '''
    if not number_of_processes:
        number_of_processes = multiprocessing.cpu_count() // 2
    if not jobs:
        jobs = 1

    commit_queue = multiprocessing.Queue()

    for commit in commits:
        commit_queue.put(commit)
    for _ in range(number_of_processes):
        commit_queue.put(None)
    
    processes = []
    for i in range(number_of_processes):
        p = multiprocessing.Process(
            target=_get_removed_lines_worker,
            args=(
                repo_path,
                commit_queue,
                results_dir / f"atoms{i+1}.csv",
                last_procesed_dir / f"last_processed{i+1}.json",
                errors_dir / f"errors{i+1}.json",
                results_dir / f"skipped{i+1}.csv",
                timeout,
                jobs,
                i,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


'''
Combines multiprocessing results into single data set
'''
def combine_results(results_dir,last_processed_dir):
    """
    Combine all atoms*.csv files into atoms.csv and all long_commits*.csv files into long_commits.csv.
    """
    atoms_combined = results_dir / "atoms.csv"
    skipped_combined = results_dir / "skipped.csv"
    files_to_remove = []
    # Helper to combine files matching a pattern
    def combine_pattern(pattern, output_file):
        lines = 0
        with output_file.open("w", newline="") as combined_file:
            writer = csv.writer(combined_file)
            for file_path in results_dir.glob(pattern):
                if file_path.name == output_file.name:
                    continue
                if file_path.is_file():
                    print(f"Adding {file_path.name} to {output_file.name}")
                    with file_path.open("r", newline="") as file:
                        reader = csv.reader(file)
                        for row in reader:
                            writer.writerow(row)
                            lines += 1
                files_to_remove.append(file_path)
        return lines

    num_atoms = combine_pattern("atoms*.csv", atoms_combined)
    num_skipped = combine_pattern("skipped*.csv", skipped_combined)

    total_count = total_count_w_atoms = total_files = 0
    total_processed = results_dir / "stats.json"
    for file_path in last_processed_dir.glob("last_processed*.json"):
        if file_path.name == total_processed.name:
            continue
        if file_path.is_file():
            print(f"Adding {file_path.name} to {total_processed.name}")
            processed_data = load_processed_data(file_path)
            total_count += processed_data["count"]
            total_count_w_atoms += processed_data["count_w_atoms"]
            total_files += processed_data["number_files"]
            files_to_remove.append(file_path)
    total_processed.write_text(json.dumps(
        {"commits_processed": total_count, 
         "commits_with_atoms": total_count_w_atoms,
         "number_atoms_found": num_atoms, 
         "files_processed": total_files,
         "files_skipped_by_timeout": num_skipped,
         "percent_files_timedout": num_skipped/total_files,
         }))
    
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")


def resumeRun(last_processed_dir, commits):
    """
    reads the 'last_processed' files if they exist
    gets the commits that were being processes at interupt
    counts past the number of commits already processed while saving those that were in processing
    """
    total_count = 0
    resume_commits = []
    last_found = False
    for file_path in last_processed_dir.glob("last_processed*.json"):
        if file_path.is_file():
            processed_data = load_processed_data(file_path)
            total_count += processed_data["count"]
            resume_commits.append(processed_data["current_commit"]) 
            last_found = True
    if last_found:
        logger.info(f"Resuming previously interupted run. {total_count} commits already processed. Beginning with commits that were interupted.")
    print(resume_commits)
    unprocesed_commits = resume_commits + commits[total_count+len(resume_commits):]  #if no last_processed files this will be all of commits(total_count==0 and len(resume_commits)==0)
    return unprocesed_commits



PATCHES_TO_SKIP = [CocciPatch.OMITTED_CURLY_BRACES]
REPO_PATH = (ROOT_DIR.parent / "projects/linux").absolute()  # Path to the Linux kernel Git repository
COMMITS_FILE_PATH = Path("commits.json")  # Path to a JSON file containing commit hashes
RESULTS_DIR = Path("./results")
LAST_PROCESSED_DIR = Path("./last_processed")
ERRORS_FILE_DIR = Path("./extract")
NUMBER_OF_PROCESSES = 1

#profiling batch processing of commits
def profile_linux_fixes(linux_dir = REPO_PATH,output_dir = Path("./output"),history_length = None, cpus = None, timeout = None, jobs = 4):

    stop_commit = "c511851de162e8ec03d62e7d7feecbdf590d881d" # this is the commit when the fix: convention was introduced
    output_dir.mkdir(exist_ok=True)
    commits_file_path = output_dir / "commits.json"

    commits = safely_load_json(commits_file_path)
    if not commits or (commits[-1] != stop_commit and history_length == None):
        iterate_commits_and_extract_removed_code(linux_dir, stop_commit, commits_file_path, history_length)

    last_processed = output_dir / LAST_PROCESSED_DIR
    last_processed.mkdir(exist_ok=True)
    results_dir = output_dir/RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    errors_dir = results_dir / ERRORS_FILE_DIR
    errors_dir.mkdir(exist_ok=True)

    commits = json.loads(commits_file_path.read_text())
    commits = resumeRun(last_processed, commits)

    if len(commits) == 1 or cpus == 1:
        get_removed_lines(linux_dir, 
                          commits, 
                          results_dir / "atoms.csv", 
                          last_processed / "last_processed.json", 
                          errors_dir / "errors.json",
                          results_dir / "skipped.csv",
                          timeout,
                          jobs
                          )
    else:
        #execute(linux_dir, commits, cpus, results_dir, last_processed, errors_dir)
        execute_queue(linux_dir, commits, cpus, results_dir, last_processed, errors_dir, timeout, jobs)
        combine_results(results_dir,last_processed)

if __name__ == "__main__":
    profile_linux_fixes(Path("../projects/linux/"), Path("./output_all"), None, 8, 60, 1)
    #cProfile.run('profile_linux_fixes(Path("../projects/linux/"), Path("./output_test_1job_8proc"), "3 month", 8, 60, 1)',"profile_linux_1job_8proc" ) # pretty close
    #cProfile.run('profile_linux_fixes(Path("../projects/linux/"), Path("./output_test_2job_4proc"), "3 month", 4, 60, 2)',"profile_linux_2job_4proc" )
    #cProfile.run('profile_linux_fixes(Path("../projects/linux/"), Path("./output_test_4job_2proc"), "3 month", 2, 60, 4)',"profile_linux_4job_2proc" )
    #cProfile.run('profile_linux_fixes(Path("../projects/linux/"), Path("./output_test_2job_8proc"), "3 month", 8, 60, 2)',"profile_linux_2job_8proc" ) # my optimal
    #cProfile.run('profile_linux_fixes(Path("../projects/linux/"), Path("./output_test_4job_4proc"), "3 month", 4, 60, 4)',"profile_linux_4job_4proc" )