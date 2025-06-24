import csv
from pathlib import Path
import re
import pygit2
from clang.cindex import Index, CursorKind, TokenKind, Config
from src.analysis.utils.git import get_file_content_at_commit
from src.run_cocci import run_patches_and_generate_output


Config.set_library_file("/usr/lib/llvm-14/lib/libclang-14.so.1")


def is_complex_structure(cursor):
    return cursor.kind in [
        CursorKind.IF_STMT,
        CursorKind.FOR_STMT,
        CursorKind.WHILE_STMT,
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
        CursorKind.IF_STMT,
        CursorKind.FOR_STMT,
        CursorKind.WHILE_STMT,
        CursorKind.DO_STMT,
        CursorKind.SWITCH_STMT,
        CursorKind.CASE_STMT,
        CursorKind.DEFAULT_STMT,
    ]

def node_goto_label(cursor):
    return cursor.kind in [
        CursorKind.GOTO_STMT,
        CursorKind.INDIRECT_GOTO_STMT,
        CursorKind.LABEL_STMT,
    ]

def save_headers_to_temp(
    full_code, output_dir, repo, commit, loaded_headers, invalid_headers
):
    _extract_headers(
        output_dir, full_code, repo, commit, loaded_headers, invalid_headers
    )


def save_all_headers(output_dir, commit, repo):
    # if it's better to save all headers, this can be used
    # doesn't seem to make a difference
    subfolder_tree = commit.tree
    subfolder_tree = repo.get(subfolder_tree["include"].id)
    _save_all_headers(output_dir, subfolder_tree, repo, path_prefix="include")


def _save_all_headers(output_dir, tree, repo, path_prefix):
    for entry in tree:
        entry_path = f"{path_prefix}/{entry.name}".strip("/")
        if entry.type == pygit2.GIT_OBJ_TREE:
            sub_tree = repo.get(entry.id)
            _save_all_headers(output_dir, sub_tree, repo, entry_path)
        elif entry.type == pygit2.GIT_OBJ_BLOB:
            header_name = f"{path_prefix.split('include/')[1]}/{entry.name}"
            path = Path(output_dir / header_name)
            file_content = entry.read_raw().decode()
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(file_content)


def _extract_headers(output_dir, code, repo, commit, processed, invalid):
    """
    Recursively extract all unique header file names from the C code.

    :param code: C code from which to extract header files.
    :param base_path: The base directory where header files are searched (as a Path object).
    :param processed: A set to keep track of processed header files to avoid cyclic includes.
    :return: A set of all header files included in the code, directly or indirectly.
    """
    header_pattern = re.compile(r"#include\s+<([^>]+)>")
    headers = set(header_pattern.findall(code))
    all_headers = set(headers)

    for header in headers:
        if header in processed[commit] or header in invalid[commit]:
            continue
        if header not in processed[commit]:
            processed[commit].append(header)
            header_name = str(Path("include", header))
            try:
                file_content = get_file_content_at_commit(repo, commit, header_name)
                path = Path(output_dir / header)
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_text(file_content)
            except Exception as e:
                print(f"Cannot load {header} at {commit} due to {e}")
                invalid[commit].append(header)
                continue
            included_headers = _extract_headers(
                output_dir, file_content, repo, commit, processed, invalid
            )
            all_headers.update(included_headers)
    return all_headers


def _normalize_code(text):
    """
    Normalize code by removing extra spaces around punctuation and making it lowercase.
    This function also standardizes common variations in array declarations.
    """
    # text = text.replace('\t', ' ').replace('\n', ' ')
    # text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    # text = re.sub(r'\s*\[\s*', '[', text)  # Remove spaces around [
    # text = re.sub(r'\s*\]\s*', ']', text)  # Remove spaces around ]
    # text = re.sub(r'\s*\(\s*', '(', text)  # remove spaces around parentheses
    # text = re.sub(r'\s*\)\s*', ')', text)  # remove spaces around parentheses
    # text = re.sub(r'\s*\)\s*', '*', text)  # remove spaces around *
    return text.replace(" ", "")


def contains_expression(node, expression, line_number=None):
    """
    Check if the normalized node text contains the normalized expression.
    """
    if line_number:
        if line_number < node.extent.start.line or line_number > node.extent.end.line:
            return False
    node_text = " ".join([token.spelling for token in node.get_tokens()])
    if expression.endswith(";"):
        expression = expression[:-1]
    normalized_node_text = _normalize_code(node_text)
    normalized_expression = _normalize_code(expression)
    return normalized_expression in normalized_node_text


def find_smallest_containing_node(
    node, expression, line_number, ancestors, best_match=None
):
    """
    Recursively find the smallest node that contains the given expression.
    """
    expression = expression.strip()
    if contains_expression(node, expression, line_number):
        ancestors.append(node)
        best_match = node
        # print("--------------------")
        # node_text = " ".join([token.spelling for token in node.get_tokens()])
        # print(node_text)
        # print("**************************88")
        children = [child for child in node.get_children()]
        for child in children:
            # node_text = " ".join([token.spelling for token in child.get_tokens()])
            # print(node_text)
            # print("***************8")
            inner_best_match = find_smallest_containing_node(
                child, expression, line_number, ancestors, best_match
            )
            if inner_best_match is not None:
                best_match = inner_best_match
                break
    return best_match


def get_code_from_extent(code, extent):
    lines = code.splitlines()
    start = extent.start
    end = extent.end

    if start.line == end.line:
        return lines[start.line - 1][start.column - 1 : end.column - 1]

    code_lines = []
    code_lines.append(lines[start.line - 1][start.column - 1 :])
    for line in range(start.line, end.line - 1):
        code_lines.append(lines[line])
    try:
        code_lines.append(lines[end.line - 1][: end.column - 1])
    except:
        pass
    return code_lines


def get_function_or_statement_context(root_node, full_code, source_code, line_number):

    # _run_diagnostics(tu, file_path)

    ancestors = []
    try:
        node = find_smallest_containing_node(
            root_node, source_code, line_number, ancestors
        )
    except UnicodeDecodeError as e:
        print(f"Could not parse file")
        return None, None

    # if node is not None:
    #     ancestors.reverse()
    #     # Ensure we capture broader context by moving up the AST if needed
    #     for parent_node in ancestors:
    #         if parent_node.kind in (clang.cindex.CursorKind.FUNCTION_DECL,
    #                                    clang.cindex.CursorKind.CXX_METHOD,
    #                                    clang.cindex.CursorKind.STRUCT_DECL,
    #                                    clang.cindex.CursorKind.CLASS_DECL):
    #             node = parent_node
    #             break
    if node is not None and node != root_node:
        return node, get_code_from_extent(full_code, node.extent)
    return None, None

def parse_file(code, include_dir, file_name):
    index = Index.create()
    tu = index.parse(
        file_name,
        args=["-x", "c", "-std=c11", "-nostdinc", "-w", "-fsyntax-only", f"-I{include_dir}"],
        unsaved_files=[(file_name, code)],
    )
    return tu

def get_references_on_lines(tu, target_lines):
    """
    Finds all referenced variables and called functions on the modified lines.
    """
    var_refs = []
    funcs_called = []
    for token in tu.get_tokens(extent=tu.cursor.extent):
        if token.cursor.referenced and token.location.line in target_lines:
            ref_kind = token.cursor.referenced.kind
            if ref_kind in (CursorKind.VAR_DECL, CursorKind.PARM_DECL):
                if token.cursor.referenced not in var_refs:
                    var_refs.append(token.cursor.referenced)
            elif ref_kind == CursorKind.FUNCTION_DECL:
                if token.cursor.referenced not in funcs_called:
                    funcs_called.append(token.cursor.referenced)
    return var_refs, funcs_called

def node_contains_var_reference(cursor, referenced_set):
    ''' 
    Checks tokens at cursor location to see if node contains 
    references to vairables in the referenced set.
    '''
    for token in cursor.get_tokens():
        if token.cursor.referenced and token.cursor.referenced in referenced_set:
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

def strip_unrelated_code(cursor, lines, var_refs,funcs_called,removed_line_numbers, file_name):
    """
    Removes Functions that don't contain any modified lines
    Or functions that are entirely removed/modified by the commit
    Then strips the remaining functions
    """
    for child in cursor.get_children():
        if not child.location.file or file_name not in child.location.file.name:
            continue

        start_line = child.extent.start.line
        end_line = child.extent.end.line

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

            if all_contained and len(element_lines) > 2:
                for line in element_lines:
                    removed_line_numbers.remove(line)

            if all_contained or not any_contained:
                continue_inner_search = False

        else:
            if not node_contains_var_reference(child, var_refs):
                continue_inner_search = False
                for i in range(start_line-1, end_line):
                    lines[i] = ""

        if continue_inner_search and not node_is_atomic(child):
            strip_unrelated_code(child, lines, var_refs,funcs_called,removed_line_numbers, file_name)


def parse_and_reduce_code(code, removed_line_numbers, include_dir, file_name):
    '''
    Uses Cindex AST to strip off code that is not relevant to any modified line.
        - Keeps any functions that are called on the modified lines
        - Keeps any AST node that contains a reference to a variable in the modified lines
    '''
    tu = parse_file(code, include_dir, file_name)
    lines = code.splitlines()
    modified_line_numbers = list(set(removed_line_numbers))
    var_refs,funcs_called = get_references_on_lines(tu, modified_line_numbers)
    strip_unrelated_code(tu.cursor, lines, var_refs, funcs_called,modified_line_numbers, file_name)
    cleaned_code = "\n".join(lines)
    return cleaned_code, modified_line_numbers


