import csv
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

from src import COCCI_DIR
from src.log import logging
from src.exceptions import RunCoccinelleError
from src.utils import empty_directory, run, check_cocci_version


class CocciPatch(Enum):
    ASSIGNMENT_AS_VALUE = "assignment_as_value.cocci"
    CHANGE_OF_LITERAL_ENCODING = "change_of_literal_encoding.cocci"
    COMMA_OPERATOR = "comma_operator.cocci"
    CONDITIONAL_OPERATOR = "conditional_operator.cocci"
    IMPLICIT_PREDICATE = "implicit_predicate.cocci"
    LOGIC_AS_CONTROLFLOW = "logic_as_controlflow.cocci"
    MACRO_OPERATOR_PRECEDENCE = "macro_operator_precedence.cocci"
    OMITTED_CURLY_BRACES = "omitted_curly_braces.cocci"
    OPERATOR_PRECEDENCE = "operator_precedence.cocci"
    POST_INCDEC = "post_incdec.cocci"
    PRE_INCDEC = "pre_incdec.cocci"
    REPURPOSED_VARIABLE = "repurposed_variable.cocci"
    REVERSED_SUBSCRIPT = "reversed_subscripts.cocci"
    TYPE_CONVERSION = "type_conversion.cocci"

    def get_full_path(self, cocci_dir):
        """ Return the full path of the cocci file. """
        return cocci_dir / self.value

    @staticmethod
    def from_string(value):
        patch_mapping = {patch.name.lower(): patch for patch in CocciPatch}
        # Lookup the enum member from the mapping
        enum_member = patch_mapping[value.lower()]
        return enum_member


# in some cases, subexpressions should be counted as separate atoms, in others, it seems unnecessary
remove_subexpressions_patches = (CocciPatch.ASSIGNMENT_AS_VALUE, CocciPatch.COMMA_OPERATOR, CocciPatch.TYPE_CONVERSION)
include_headers_patches = (CocciPatch.COMMA_OPERATOR, CocciPatch.MACRO_OPERATOR_PRECEDENCE)


def _check_if_already_added(start_line, start_col, end_line, end_col, processed):
    if not len(processed) or start_line not in processed:
        new_range = {'start_line': int(start_line), 'start_col': int(start_col), 'end_line': int(end_line), 'end_col': int(end_col)}
        processed[start_line] = [new_range]
        return False
    already_added = any(line["start_line"] == int(start_line) and line["end_line"] == int(end_line) and
        line["start_col"] == int(start_col) and line["end_col"] == int(end_col)
        for  line in processed[start_line])
    return already_added


def _check_if_subexpression(start_line, start_col, end_line, end_col, processed):
    new_range = {'start_line': int(start_line), 'start_col': int(start_col), 'end_line': int(end_line), 'end_col': int(end_col)}
    if start_line in processed:
        subset = any(_is_subset(new_range, existing) for existing in processed[start_line])
        if not subset:
            processed[start_line].append(new_range)
            processed[start_line] = [existing for existing in processed[start_line] if not _is_subset(existing, new_range)]
            return False
        else:
            return True
    else:
        processed[start_line] = [new_range]
        return False

def _is_subset(current, previous):
    # Check if the current range is entirely within the previous range
    if (current['start_line'] > previous['start_line'] or
        (current['start_line'] == previous['start_line'] and current['start_col'] >= previous['start_col'])) and \
       (current['end_line'] < previous['end_line'] or
        (current['end_line'] == previous['end_line'] and current['end_col'] <= previous['end_col'])):
        return True
    return False


def run_cocci(cocci_patch_path, c_input_path, output_file=None, opts=None):
    # keep all paths for this file to avoid additional imports from pathlib
    logging.debug(f"Running patch: {cocci_patch_path} against {c_input_path}")
    try:
        opts = opts or []
        if cocci_patch_path in [patch.value for patch in include_headers_patches]:
            opts.append("--include-headers")

        cmd = ["spatch","--jobs 4", "--sp-file", str(cocci_patch_path), str(c_input_path)] + opts
        if output_file is not None:
            output_file.touch()
            cmd.append(">>")
            cmd.append(str(output_file))
        run(cmd)

    except subprocess.CalledProcessError as e:
        raise RunCoccinelleError(f"An error occurred while running patch {cocci_patch_path}: {e}")


def read_csv_generator(file_path):
    with open(file_path, 'r', newline='', encoding="utf8") as file:
        reader = csv.reader(file)
        for row in reader:
            yield row


def postprocess_and_generate_output(file_path: Path,  patch: CocciPatch, remove_end_line_and_col=True):
    seen = set() 
    filtered_data = [] 
    removed_lines_count = 0
    processed = {}
    logging.debug("Posptocessing: removing duplicate lines")

    previous_debug_row = None
    for row in read_csv_generator(file_path):
        key = tuple(row[1:-1])
        if row[0].startswith("Rule"):
            previous_debug_row = row
            continue

        if len(row) < 7:
            filtered_data.append(row)
            continue
        start_line = row[2]
        start_col = row[3]
        end_line = row[4]
        end_col = row[5]
        if len(row) > 7:
            code = ",".join(row[6:])
            row = row[:6]
            row.append(code)
            
        if key not in seen and (
            patch not in remove_subexpressions_patches and not _check_if_already_added(
                start_line, start_col, end_line, end_col, processed)) or \
                not _check_if_subexpression(start_line, start_col, end_line, end_col, processed):
            if previous_debug_row is not None:
                filtered_data.append(previous_debug_row)
            seen.add(key)

            if remove_end_line_and_col:
                row = row[:4] + row[6:]
            filtered_data.append(row)

        else:
            removed_lines_count += 1
            previous_debug_row = None

    logging.debug(f"Removed {removed_lines_count} lines")
    return filtered_data


def run_patches_and_generate_output(
        input_path: Path, 
        output_path: Optional[Path] = None, 
        temp_dir: Optional[Path] = None, 
        split_output = True, 
        patch: Optional[CocciPatch] = None,  
        patches_to_skip: Optional[list] = None, 
        remove_end_line_and_col: Optional[bool]=True, 
        cocci_dir:Optional[Path]=COCCI_DIR
):
    logging.debug("Running patches")
    patches_to_skip = patches_to_skip or []
    if patch is None:
        # run all patche, except for patches to skip
        patches_to_run = [cocci_patch for cocci_patch in CocciPatch if cocci_patch not in patches_to_skip]
    else:
        patches_to_run = [patch]

    delete_temp = temp_dir is None
    all_atoms = []
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    def _write_lines_to_file(file_path, data_list):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            for row in data_list:
                writer.writerow(row)

    for patch_to_run in patches_to_run:
        full_patch_path = patch_to_run.get_full_path(cocci_dir)
        temp_output_file = Path(temp_dir, f"{full_patch_path.stem}.csv")
        if not full_patch_path.is_file():
            continue
        try:
            run_cocci(full_patch_path, input_path, output_file=temp_output_file)
        except RunCoccinelleError as e:
            # log the error and continue
            logging.error(str(e))
        
        atoms = postprocess_and_generate_output(temp_output_file, patch, remove_end_line_and_col)
        if split_output:
            output_file = output_path / f"{full_patch_path.stem}.csv"
            _write_lines_to_file(output_file, atoms)
        else:
            all_atoms.extend(atoms)
    
    if not split_output:
        _write_lines_to_file(output_path, all_atoms)

    if delete_temp:
        shutil.rmtree(temp_dir)
    else:
       empty_directory(temp_dir, files_to_keep=[output_path])
