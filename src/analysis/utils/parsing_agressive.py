import csv
import pygit2
import os
from pathlib import Path

from clang.cindex import Index, CursorKind, TokenKind, Config
from src.analysis.utils.git import get_file_content_at_commit
from src.run_cocci import run_patches_and_generate_output
from src.log import logger


Config.set_library_file("/usr/lib/llvm-14/lib/libclang-14.so.1")

def save_header_directories(output_dir, commit, repo, include_paths):
    """
    Save entire include directories (e.g., 'include', 'arch/x/include') 
    from a specific commit in a Git repo to a temporary output directory.
    
    :param repo: pygit2 Repository object.
    :param commit: pygit2 Commit object.
    :param include_paths: list of include paths (relative to root of repo).
    :param output_dir: Path to output directory where headers will be saved.
    """
    tree = commit.tree

    for include_path in include_paths:
        if 'generated' in include_path:
            continue
        try:
            sub_tree = _get_tree_at_path(repo, tree, include_path)
        except KeyError:
            print(f"[WARN] '{include_path}' not found in commit tree.")
            continue

        _save_tree_to_dir(repo, sub_tree, Path(output_dir) / include_path)

def _get_tree_at_path(repo, tree, path):
    """Walks the tree to the subdirectory specified by `path`."""
    for part in path.strip("/").split("/"):
        tree_entry = tree[part]
        tree = repo.get(tree_entry.id)
    return tree

def _save_tree_to_dir(repo, tree, target_dir, relative_path=""):
    """
    Recursively saves a Git tree to a directory.
    
    :param repo: pygit2 Repo object
    :param tree: pygit2 Tree object
    :param target_dir: Path where to save the contents
    :param relative_path: Used internally to preserve subdirectories
    """
    for entry in tree:
        full_path = Path(target_dir) / relative_path / entry.name
        if entry.type == pygit2.GIT_OBJECT_TREE:
            _save_tree_to_dir(repo, repo.get(entry.id), target_dir, Path(relative_path) / entry.name)
        elif entry.type == pygit2.GIT_OBJECT_BLOB:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            blob = repo.get(entry.id)
            full_path.write_bytes(blob.data)

def is_complex_structure(cursor):
    return cursor.kind in [
        CursorKind.IF_STMT,
        CursorKind.FOR_STMT,
        CursorKind.WHILE_STMT,
        CursorKind.DO_STMT,
        CursorKind.SWITCH_STMT,
        CursorKind.STRUCT_DECL,
    ]

def node_is_declaration(cursor):
    return cursor.kind in [
        CursorKind.STRUCT_DECL,
        CursorKind.UNION_DECL,
        CursorKind.ENUM_DECL,
        CursorKind.TYPEDEF_DECL,
        CursorKind.MACRO_DEFINITION,
        CursorKind.VAR_DECL,
        CursorKind.PARM_DECL,
        CursorKind.FIELD_DECL,
    ]

def node_is_control_stmt(cursor):
    return cursor.kind in [
        CursorKind.BREAK_STMT,
        CursorKind.RETURN_STMT,
        CursorKind.CONTINUE_STMT,
        CursorKind.CASE_STMT,
        CursorKind.DEFAULT_STMT,
        CursorKind.GOTO_STMT,
        CursorKind.INDIRECT_GOTO_STMT,
        CursorKind.LABEL_STMT,
    ]

##############################################
#Ast Print functions for Debug
def print_ast(cursor, indent=0):
    print('  ' * indent + f"{cursor.kind} {cursor.spelling} (line {cursor.location.line})")
    for child in cursor.get_children():
        print_ast(child, indent + 1)

def print_ast_for_modified(cursor, modified_lines, indent=0):
    element_start = cursor.extent.start.line
    element_end = cursor.extent.end.line
    if any(line in modified_lines for line in range(element_start, element_end + 1)):
        print('  ' * indent + f"{cursor.kind} {cursor.spelling} (line {cursor.location.line})")
    for child in cursor.get_children():
        print_ast_for_modified(child, modified_lines, indent + 1)
############################################
def parse_file(code, include_dir, include_paths, build_args, file_name):
    """
    -I./arch/powerpc/include 
    -I./arch/powerpc/include/generated 
    -I./include 
    -I./include 
    -I./arch/powerpc/include/uapi 
    -I./arch/powerpc/include/generated/uapi 
    -I./include/uapi 
    -I./include/generated/uapi 
    -include ./include/linux/compiler-version.h -include ./include/linux/kconfig.h -include ./include/linux/compiler_types.h"""
    index = Index.create()
    linux_dir = "/home/atoms/projects/linux"
    other_includes = []#[f"-include {include_dir}/include/linux/compiler-version.h", f"-include {include_dir}/include/linux/kconfig.h", f"-include {include_dir}/include/linux/compiler_types.h"]
    include_args = []
    for include_path in include_paths:
        if 'generated' in include_path:
            include_args += [f"-I{linux_dir}/{include_path}"]
        else:
            include_args += [f"-I{include_dir}/{include_path}"]
    #include_args = [f"-I{include_dir}/{ipath}" for ipath in include_paths]
    tu = index.parse(
        file_name,
        args=[
            "-x", "c",
            "-std=c11",
            "-nostdinc",
            "-fsyntax-only",
            "-ferror-limit=0"
        ] + build_args + include_args + other_includes,
        unsaved_files=[(file_name, code)],
    )
    ##########################################
    #debug 
    for d in tu.diagnostics:
        print(d)
    ##########################################
    input(...)
    return tu

def get_references_on_lines(cursor, target_lines):
    """
    Finds all referenced variables and called functions on the modified lines.
    """
    var_refs = []
    funcs_called = []
    for token in cursor.get_tokens():
        if token.kind is not TokenKind.IDENTIFIER:
            continue
        if token.cursor.referenced and (token.location.line in target_lines):
            ref_kind = token.cursor.referenced.kind
            if ref_kind in [CursorKind.VAR_DECL, CursorKind.PARM_DECL]:
                if token.cursor.referenced not in var_refs:
                    #print(token.spelling, token.location.line)
                    var_refs.append(token.cursor.referenced)
            """ elif ref_kind == CursorKind.FUNCTION_DECL:
                if token.cursor.referenced not in funcs_called:
                    funcs_called.append(token.cursor.referenced) """
    return var_refs, funcs_called

def node_contains_var_reference(cursor, referenced_set):
    ''' 
    Checks tokens at cursor location to see if node contains 
    references to vairables in the referenced set.
    '''
    for token in cursor.get_tokens():
        if token.cursor.referenced and (token.cursor.referenced in referenced_set):
            return True
    return False


def node_is_referenced_func(cursor, func_set):
    '''
    Determines if the AST node is function contained in set
    '''
    return cursor.kind == CursorKind.FUNCTION_DECL and cursor in func_set

def node_is_atomic(cursor):
    return cursor.kind in [
        CursorKind.BINARY_OPERATOR,
        CursorKind.UNARY_OPERATOR,
        CursorKind.DECL_STMT,
        CursorKind.INIT_LIST_EXPR,
        CursorKind.ARRAY_SUBSCRIPT_EXPR,
        CursorKind.MEMBER_REF_EXPR,
        CursorKind.CALL_EXPR,
        CursorKind.CONDITIONAL_OPERATOR,
        CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
    ]

def strip_inner(cursor, code_chars, var_refs, removed_line_numbers):
    for child in cursor.get_children():
        #do not strip down "atomic" statements or control statements
        if node_is_atomic(child) or node_is_control_stmt(child):
            continue
        #do not strip parts of complex structrues that are not the compound statements
        #ie don't strip if, loop or switch conditionals
        #parent = child.semantic_parent
        if is_complex_structure(cursor) and child.kind != CursorKind.COMPOUND_STMT:
            continue

        #continue inside if still contains variable references or else remove node from lines
        if node_contains_var_reference(child, var_refs):
            strip_inner(child,code_chars,var_refs,removed_line_numbers)
        else:
            #add check to not delete modified lines if the ast in incomplete
            start_line = child.extent.start.line
            end_line = child.extent.end.line
            element_lines = [line for line in range(start_line, end_line + 1)]
            if any(line in removed_line_numbers for line in element_lines):
                continue

            start_offset = child.extent.start.offset
            end_offset = child.extent.end.offset
            if child.kind == CursorKind.COMPOUND_STMT:
                start_offset+=1
                end_offset-=1
            # include trailing semicolon if present
            if end_offset < len(code_chars) and code_chars[end_offset] == ';':
                end_offset += 1
            for i in range(start_offset, end_offset):
                if code_chars[i] not in ('\n'): 
                    code_chars[i] = ' '



#should be called on tu.cursor(global cursor)
def strip_unrelated_code(cursor, code_chars, removed_line_numbers, file_name):
    for child in cursor.get_children():
        if not child.location.file or file_name not in child.location.file.name:
            continue
        is_function = child.kind == CursorKind.FUNCTION_DECL
        if is_function:
            element_start = child.extent.start.line
            element_end = child.extent.end.line
            element_lines = [line for line in range(element_start, element_end + 1)]

            any_contained = any(line in removed_line_numbers for line in element_lines)
            all_contained = all(line in removed_line_numbers for line in element_lines)

            for c in child.get_children():
                if c.kind == CursorKind.COMPOUND_STMT:
                    if not any_contained or all_contained:
                        body_start = c.extent.start.offset
                        body_end = c.extent.end.offset
                        for i in range(body_start+1, body_end-1):
                            if code_chars[i] not in ('\n'):
                                code_chars[i] = ' '
                        break
                    if all_contained:       
                        for line in element_lines:
                            removed_line_numbers.remove(line)
                    else:
                        #print_ast(c)
                        #input(...)
                        var_refs,_ = get_references_on_lines(c,removed_line_numbers)
                        strip_inner(c,code_chars,var_refs,removed_line_numbers)


def _strip_unrelated_code(cursor, lines, var_refs,funcs_called,removed_line_numbers, file_name):
    """
    Removes Functions that don't contain any modified lines
    Or functions that are entirely removed/modified by the commit
    For functions that contain modified lines, further strip off lines of code not likely to be involved in atoms
    """
    for child in cursor.get_children():
        if not child.location.file or file_name not in child.location.file.name:
            continue
        parent = child.semantic_parent
        if parent and (parent.kind == CursorKind.FUNCTION_DECL or is_complex_structure(parent)) and child.kind != CursorKind.COMPOUND_STMT:
            continue
        if node_is_control_stmt(child):
            continue
        #not sure this is neccessary
        """ if node_is_referenced_func(child, funcs_called):
            continue """

        is_function = child.kind == CursorKind.FUNCTION_DECL
        is_complex = is_complex_structure(child)
        continue_inner_search = True
        if is_function or is_complex:
            element_start = child.extent.start.line
            element_end = child.extent.end.line
            element_lines = [line for line in range(element_start, element_end + 1)]

            any_contained = any(
                line in removed_line_numbers for line in element_lines
            )
            all_contained = all(
                line in removed_line_numbers for line in element_lines
            )

            # if all contained, the whole function was removed
            if is_function:
                if not any_contained or all_contained:
                    # Find the compound statement that is the body of the function
                    for c in child.get_children():
                        if c.kind == CursorKind.COMPOUND_STMT:
                            # Calculate the start and end offsets for the body
                            body_start_line = c.extent.start.line
                            body_end_line = c.extent.end.line - 2
                            # Store the offsets and the count of newlines to preserve formatting
                            lines[body_start_line : body_end_line + 1] = [
                                ""
                                for _ in range(body_end_line - body_start_line + 1)
                            ]
                            break
                else:
                    #lines_to_print = [line in removed_line_numbers for line in element_lines]
                    #print_ast_for_modified(child, removed_line_numbers)
                    input(...)

            if all_contained and len(element_lines) > 2:
                for line in element_lines:
                    removed_line_numbers.remove(line)

            if all_contained or not any_contained:
                continue_inner_search = False

        elif not node_contains_var_reference(child, var_refs):
            continue_inner_search = False        
            start_line = child.extent.start.line
            end_line = child.extent.end.line
            for i in range(start_line-1, end_line):
                lines[i] = ""

        if continue_inner_search and not node_is_atomic(child):
            strip_unrelated_code(child, lines, var_refs,funcs_called,removed_line_numbers, file_name)


def parse_and_reduce_code(code, removed_line_numbers, include_dir, include_paths, build_args, file_name):
    '''
    Uses Cindex AST to strip off code that is not relevant to any modified line.
        - Keeps any functions that are called on the modified lines
        - Keeps any AST node that contains a reference to a variable in the modified lines
    '''
    tu = parse_file(code, include_dir, include_paths, build_args, file_name)
    code_chars = list(code)
    modified_line_numbers = list(set(removed_line_numbers))
    strip_unrelated_code(tu.cursor, code_chars, modified_line_numbers, file_name)
    cleaned_code = "".join(code_chars)
    return cleaned_code, modified_line_numbers


def run_coccinelle_for_file_at_commit_aggressive(
    repo,
    file_name,
    commit,
    modified_line_numbers,
    temp_dir,
    loaded_headers,
    invalid_headers,
    patches_to_skip=None,
    save_headers=True,
):
    atoms = []
    headers_dir = Path(temp_dir, "headers")
    arch = None
    parts = file_name.split('/')
    if "arch" in parts:
        idx = parts.index("arch")
        if idx + 1 < len(parts):
            arch = parts[idx + 1]  

    content = get_file_content_at_commit(repo, commit, file_name)


    include_paths = [str(Path(file_name).parent)]
    build_args = []
    if(True):
        build_args = [ "-D__KERNEL__"]
    if arch is not None:
        build_args += [f"-D__{arch}__"]
        include_paths += [f"arch/{arch}/include",f"arch/{arch}/include/generated"]
        include_paths += ["include"]
        include_paths += [f"arch/{arch}/include/uapi",f"arch/{arch}/include/generated/uapi"]
        include_paths += ["include/uapi","include/generated/uapi"]
    else:
        include_paths += ["include"]
        include_paths += ["include/uapi","include/generated/uapi"]
    
    """ include_paths = [str(Path(file_name).parent)]
    build_args = []
    if(True):
        build_args = [ "-D__KERNEL__"]
    if arch is not None:
        build_args += [f"-D__{arch}__"]
        include_paths += [f"arch/{arch}/include"]
        include_paths += ["include"]
        include_paths += [f"arch/{arch}/include/uapi"]
        include_paths += ["include/uapi"]
    else:
        include_paths += ["include"]
        include_paths += ["include/uapi"] """
    
    save_header_directories(
        commit=commit,
        output_dir=headers_dir,
        repo=repo,
        include_paths=include_paths
        )
    

 
    base_name = Path(file_name).name 
    logger.info(base_name)
    """file_dir_name = base_name.replace('.', '_')
    output_dir = Path(f"parsed_{commit.id}") / file_dir_name
    output_dir.mkdir(parents=True, exist_ok=True) """

    shorter_content, modified_lines = parse_and_reduce_code(
                content, modified_line_numbers, headers_dir, include_paths, build_args, file_name,
            )


    input = Path(temp_dir, "input", file_name)
    input.parent.mkdir(parents=True, exist_ok=True)
    input.write_text(shorter_content)

    output = Path(temp_dir, "output.csv")
    input_dir = Path(temp_dir, "input")

    run_patches_and_generate_output(
        input_dir, output, temp_dir, False, None, patches_to_skip, False
    )

    with open(output, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 1:
                continue
            atom, path, start_line, start_col, end_line, end_col, code = row
            file_name = path.split(f"{input_dir}/")[1]
            if int(start_line) in modified_lines:
                row[1] = file_name
                atoms.append(row)

    return atoms

