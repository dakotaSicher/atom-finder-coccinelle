from clang.cindex import Index, CursorKind, TokenKind, Config
from src.analysis.utils.git import get_file_content_at_commit
from src.run_cocci import run_patches_and_generate_output


Config.set_library_file("/usr/lib/llvm-14/lib/libclang-14.so.1")


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

def node_goto_label(cursor):
    return cursor.kind in [
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
    index = Index.create()
    include_args = [f"-I{include_dir}/{ipath}" for ipath in include_paths]
    tu = index.parse(
        file_name,
        args=["-x", "c", "-std=c11", "-w", "-nostdinc", "-fsyntax-only","-ferror-limit=0"] + include_args + build_args,
        unsaved_files=[(file_name, code)],
    )
    ##########################################
    #debug 
    """ for d in tu.diagnostics:
        print(d)
    for token in tu.get_tokens(extent=tu.cursor.extent):
        if 972 <= token.location.line <= 996:
            print(f"{token.spelling} - {token.kind} - {token.cursor.kind} - {token.cursor.referenced}") """
    ##########################################
    return tu

def get_references_on_lines(tu, target_lines):
    """
    Finds all referenced variables and called functions on the modified lines.
    """
    var_refs = []
    funcs_called = []
    for token in tu.get_tokens(extent=tu.cursor.extent):
        if token.kind is not TokenKind.IDENTIFIER:
            continue
        if token.cursor.referenced and (token.location.line in target_lines):
            ref_kind = token.cursor.referenced.kind
            if ref_kind in [CursorKind.VAR_DECL, CursorKind.PARM_DECL]:
                if token.cursor.referenced not in var_refs:
                    print(token.spelling, token.location.line)
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

def strip_unrelated_code(cursor, lines, var_refs,funcs_called,removed_line_numbers, file_name):
    """
    Removes Functions that don't contain any modified lines
    Or functions that are entirely removed/modified by the commit
    Then strips the remaining functions
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
    lines = code.splitlines()
    modified_line_numbers = list(set(removed_line_numbers))
    var_refs,funcs_called = get_references_on_lines(tu, modified_line_numbers)
    strip_unrelated_code(tu.cursor, lines, var_refs, funcs_called,modified_line_numbers, file_name)
    cleaned_code = "\n".join(lines)
    return cleaned_code, modified_line_numbers


