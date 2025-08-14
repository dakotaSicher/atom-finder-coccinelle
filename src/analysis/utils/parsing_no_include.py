import csv
import pygit2
from pathlib import Path
import shutil
import re
import time
from collections import defaultdict

from clang.cindex import Index, CursorKind, TokenKind, Config, SourceLocation, SourceRange
from src.analysis.utils.git import get_file_content_at_commit
from src.run_cocci import run_patches_and_generate_output
from src.log import logger


Config.set_library_file("/usr/lib/llvm-14/lib/libclang-14.so.1")

RESERVED = {
    "auto","break","case","char","const","continue","default","do",
    "double","else","enum","extern","float","for","goto","if","int",
    "long","register","return","short","signed","sizeof","static",
    "struct","switch","typedef","union","unsigned","void",
    "volatile","while"
}

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
def replace_case_values(code):
    case_pattern = re.compile(r'(case)([^:]*)(:)', flags=re.MULTILINE)
    counter = 0

    def replacer(match):
        nonlocal counter
        prefix, middle, suffix = match.groups()
        middle_str = middle

        # Calculate replacement keeping length same as original
        replacement_value = ' '+str(counter)
        counter += 1

        replacement = replacement_value.ljust(len(middle_str)) if len(middle_str) > len(replacement_value) else replacement_value[:len(middle_str)]

        return f'{prefix}{replacement}{suffix}'

    return case_pattern.sub(replacer, code)

def remove_function_like_macros(code):
    pattern = re.compile(
        r'(?:#define)\s*'                # match '#define' and whitespace
        r'([_a-zA-Z][_a-zA-Z0-9]*)'       # macro name
        r'\s*\(([^\)]*)\)'            # arguments in parentheses
        r'(.*?)'                          # capture the body (lazy)
        r'(?:(?<!\\)\n|$)',            # stop at newline not ending in '\\'
        re.DOTALL
    )

    def replacer(match):
        start = match.start()
        end = match.end()
        return ' ' * (end - start -1) + '\n'  # replace with same number of spaces to preserve offsets

    # now iteratively find each multiline macro
    pos = 0
    result = []
    while pos < len(code):
        match = pattern.search(code, pos)
        if not match:
            result.append(code[pos:])
            break
        result.append(code[pos:match.start()])

        # Handle line continuations
        macro_text = code[match.start():]
        lines = macro_text.splitlines(keepends=True)
        total = ''
        for i, line in enumerate(lines):
            total += line
            if not line.rstrip().endswith('\\'):
                break
        result.append(' ' * len(total))
        pos = match.start() + len(total)

    return ''.join(result)

def errors_to_dummy_header(tu,dummy_header):
    identifiers = set()
    types = set()
    structs = set()
    found_types = False
    define_const = 0
    for d in tu.diagnostics:
        if "use of undeclared identifier" in d.spelling:
            name = d.spelling.split("'")[1]
            if name == 'NULL':
                continue
            if name not in identifiers:
                if name.isupper():
                    dummy_header += f"#define {name} {define_const}\n"
                    define_const+=1
                else:
                    dummy_header += f"int {name};\n"
                    #print(f"non-macro unknown identifier found: {name} line {d.location.line}")
                identifiers.add(name)
        if "unknown type name" in d.spelling:
            name = d.spelling.split("'")[1]
            print(d)
            if name not in types:
                dummy_header += f"typedef int {name};\n"
                types.add(name)

        #doesn't seem like these affect node formation

        if "variable has incomplete type" in d.spelling:
            pass
        if "incomplete definition of type" in d.spelling:
            name = d.spelling.split("'")[1]
            if name not in structs:
                dummy_header += f"{name} {{}};\n"
                structs.add(name)

    if(len(types)):
        logger.info(f"Found more undefined types: {len(types)}")
        found_types =True
    return dummy_header,found_types

def pre_build_dummy_header(code, local_headers):
    #straight forward - looks for declaration statements using patern of  "[;|{|}|const|extern|for(] ident \** ident [;|=]"
    types_from_decls = re.findall(r"(?:[;{}]\s*|(?:for\s*\()|(?:const)|(?:extern))\s*([_a-zA-Z][_a-zA-Z0-9]*)(?:\s*\*+\s*|\s+)[_a-zA-Z][_a-zA-Z0-9]*\s*(?=[\[;=])", code)
    #probably a good assumption to assume identifiers that end in '_t' are types
    types_from_underscore_t = re.findall(r"[_a-zA-Z][_a-zA-Z0-9]*(?:_t)",code)

    #captures the possible type name from cast expression, inner '*'s and outer '*'s as separate groups to separate out the ambibiguous expression
    types_from_casts = re.findall(r"[^_a-zA-Z0-9\s]\s*\(\s*(?:const)?\s*(?:(?:unsigned)|(?:signed)|(?:struct))?\s*([_a-zA-Z][_a-zA-Z0-9]*)\s*(\**)\s*\)\s*(\**)\s*\s*[_a-zA-Z][_a-zA-Z0-9]*\s*",code)
    types_ambiguous_cast = []
    types_definite_cast =[]
    for t,p1,p2 in types_from_casts:
        if p2 == '*' and p1 == '': #cast expression that are like (ident)*ident are ambiguous and we cant tell if the first ident is a variable or type with a declaration or typedef for the name
            types_ambiguous_cast.append(t) #maybe add some guess to try and resolve the ambiguous ones. Best i can come up with is if it end in '_t' or does it end in a number that is power of 2
        else:
            types_definite_cast.append(t)
            

    param_lists = re.findall(r"[_a-zA-Z][_a-zA-Z0-9]*\s*\(\s*((?:const)?\s*(?:(?:unsigned)|(?:signed)|(?:struct))?\s*[_a-zA-Z][_a-zA-Z0-9]*(?:\s*\*+\s*|\s+)[_a-zA-Z][_a-zA-Z0-9]*(?:\,\s*(?:const)?\s*(?:(?:unsigned)|(?:signed)|(?:struct))?\s*[_a-zA-Z][_a-zA-Z0-9]*(?:\s*\*+\s*|\s+)[_a-zA-Z][_a-zA-Z0-9]*\s*)*)\s*\)\s*[\{;]",code)
    types_from_params = []
    for pl in param_lists:
        param_types = re.findall(r"(?:const)?\s*(?:(?:unsigned)|(?:signed))?\s*([_a-zA-Z][_a-zA-Z0-9]*)(?:\s*\*+\s*|\s+)[_a-zA-Z][_a-zA-Z0-9]*",pl)
        types_from_params += param_types
    
    types_all = set(types_from_decls + types_from_underscore_t + types_definite_cast + types_from_params)
    types_all = types_all - RESERVED

    for _, header_content in local_headers:
        typedefs = re.findall(r"(?:typedef)\s*([_a-zA-Z][_a-zA-Z0-9]*)\s*([_a-zA-Z][_a-zA-Z0-9]*);",header_content)
        for t1,t2 in typedefs:
            if t1 not in RESERVED:
                types_all.add(t1) #need ensure it is defined in dummy
            if t2 in types_all:
                types_all.remove(t2) #this typedef already defines it, don't need to add to dummy


    dummy_header = ""
    for t in types_all:
        dummy_header += f"typedef int {t};\n"
    #print(dummy_header)
    return dummy_header

def parse_file(code, file_name, use_dummy_header, local_headers):
    # Use a virtual absolute path for the dummy header
    dummy_header ="# define fallthrough  __attribute__((__fallthrough__))\n"
    if(use_dummy_header):
        dummy_header += pre_build_dummy_header(code, local_headers)
    dummy_path = str(Path("/virtual/dummy.h"))
    index = Index.create()
    tu = index.parse(
        file_name,
        args=[
            "-x", "c",
            "-std=c11",
            "-nostdinc",
            "-fsyntax-only",
            "-ferror-limit=0",
            "-include", dummy_path,
        ] 
        ,
        unsaved_files=[
            (file_name, code),
            (dummy_path, dummy_header)
        ] + local_headers
    )

    dummy_header,_ = errors_to_dummy_header(tu,dummy_header)
    input(...)
    print(dummy_header)
    input(...)
    tu = index.parse(
        file_name,
        args=[
            "-x", "c",
            "-std=c11",
            "-nostdinc",
            "-fsyntax-only",
            "-ferror-limit=0",
            "-include", dummy_path
        ]
        ,
        unsaved_files=[
            (file_name, code),
            (dummy_path,dummy_header)
        ] + local_headers
    )
    
    for d in tu.diagnostics:
        """ if "implicit declaration of function" in d.spelling:
            continue """
        print(d) 
    return tu

def get_references_on_lines(cursor, target_lines):
    """
    Finds all referenced variables and called functions on the modified lines.
    """
    var_refs = []
    for token in cursor.get_tokens():
        if token.kind is not TokenKind.IDENTIFIER:
            continue
        if token.cursor.referenced and (token.location.line in target_lines):
            ref_kind = token.cursor.referenced.kind
            if ref_kind in [CursorKind.VAR_DECL, CursorKind.PARM_DECL]:
                if token.cursor.referenced not in var_refs:
                    print(token.spelling, token.location.line)
                    var_refs.append(token.cursor.referenced)
    return var_refs

def node_contains_var_reference(cursor, referenced_set):
    ''' 
    Checks tokens at cursor location to see if node contains 
    references to vairables in the referenced set.
    '''
    for token in cursor.get_tokens():
        if token.cursor.referenced and (token.cursor.referenced in referenced_set):
            return True
    return False

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
                        var_refs = get_references_on_lines(c,removed_line_numbers)
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

def parse_and_reduce_code(code, removed_line_numbers, file_name, use_dummy_header = False, local_headers=None):
    '''
    Uses Cindex AST to strip off code that is not relevant to any modified line.
        - Keeps any functions that are called on the modified lines
        - Keeps any AST node that contains a reference to a variable in the modified lines
    '''
    tu = parse_file(code, file_name, use_dummy_header, local_headers)
    print_ast(tu.cursor)
    #return None, None
    code_chars = list(code)
    modified_line_numbers = list(set(removed_line_numbers))
    #strip_unrelated_code(tu.cursor, code_chars, modified_line_numbers, file_name)
    cleaned_code = "".join(code_chars)
    return cleaned_code, modified_line_numbers


def run_coccinelle_for_file_at_commit_no_inlcude(
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
    parent_dir = Path(file_name).parent
    logger.info(file_name)
    #content = get_file_content_at_commit(repo, commit, file_name)
    content ="""static int invoke_bpf_prog(u32 *image, u32 *ro_image, struct codegen_context *ctx,
			   struct bpf_tramp_link *l, int regs_off, int retval_off,
			   int run_ctx_off, bool save_ret)
{
	struct bpf_prog *p = l->link.prog;
	ppc_inst_t branch_insn;
	u32 jmp_idx;
	int ret = 0;


	/* Save cookie */
	if (IS_ENABLED(CONFIG_PPC64)) {
		PPC_LI64(_R3, l->cookie);
		EMIT(PPC_RAW_STD(_R3, _R1, run_ctx_off + offsetof(struct bpf_tramp_run_ctx,
				 bpf_cookie)));
	} else {
		PPC_LI32(_R3, l->cookie >> 32);
		PPC_LI32(_R4, l->cookie);
		EMIT(PPC_RAW_STW(_R3, _R1,
				 run_ctx_off + offsetof(struct bpf_tramp_run_ctx, bpf_cookie)));
		EMIT(PPC_RAW_STW(_R4, _R1,
				 run_ctx_off + offsetof(struct bpf_tramp_run_ctx, bpf_cookie) + 4));
	}

	/* __bpf_prog_enter(p, &bpf_tramp_run_ctx) */
	PPC_LI_ADDR(_R3, p);
	EMIT(PPC_RAW_MR(_R25, _R3));
	EMIT(PPC_RAW_ADDI(_R4, _R1, run_ctx_off));
	ret = bpf_jit_emit_func_call_rel(image, ro_image, ctx,
					 (unsigned long)bpf_trampoline_enter(p));
	if (ret)
		return ret;

	/* Remember prog start time returned by __bpf_prog_enter */
	EMIT(PPC_RAW_MR(_R26, _R3));

	/*
	 * if (__bpf_prog_enter(p) == 0)
	 *	goto skip_exec_of_prog;
	 *
	 * Emit a nop to be later patched with conditional branch, once offset is known
	 */
	EMIT(PPC_RAW_CMPLI(_R3, 0));
	jmp_idx = ctx->idx;
	EMIT(PPC_RAW_NOP());

	/* p->bpf_func(ctx) */
	EMIT(PPC_RAW_ADDI(_R3, _R1, regs_off));
	if (!p->jited)
		PPC_LI_ADDR(_R4, (unsigned long)p->insnsi);
	if (!create_branch(&branch_insn, (u32 *)&ro_image[ctx->idx], (unsigned long)p->bpf_func,
			   BRANCH_SET_LINK)) {
		if (image)
			image[ctx->idx] = ppc_inst_val(branch_insn);
		ctx->idx++;
	} else {
		EMIT(PPC_RAW_LL(_R12, _R25, offsetof(struct bpf_prog, bpf_func)));
		EMIT(PPC_RAW_MTCTR(_R12));
		EMIT(PPC_RAW_BCTRL());
	}

	if (save_ret)
		EMIT(PPC_RAW_STL(_R3, _R1, retval_off));

	/* Fix up branch */
	if (image) {
		if (create_cond_branch(&branch_insn, &image[jmp_idx],
				       (unsigned long)&image[ctx->idx], COND_EQ << 16))
			return -EINVAL;
		image[jmp_idx] = ppc_inst_val(branch_insn);
	}

	/* __bpf_prog_exit(p, start_time, &bpf_tramp_run_ctx) */
	EMIT(PPC_RAW_MR(_R3, _R25));
	EMIT(PPC_RAW_MR(_R4, _R26));
	EMIT(PPC_RAW_ADDI(_R5, _R1, run_ctx_off));
	ret = bpf_jit_emit_func_call_rel(image, ro_image, ctx,
					 (unsigned long)bpf_trampoline_exit(p));

	return ret;
}"""

    #need to move this up the call chain as an arg to give options to retrieve headers from git tree or to make and use a dummy header
    #maybe put in a config
    using_repo_header = False
    using_stdinc = False
    use_dummy_header = True

    if(use_dummy_header):
        #remove includes to prevent clang fatal errors
        content = re.sub(r'#\s*include\s*<[^>]+>', lambda m: ' ' * len(m.group(0)), content, flags=re.MULTILINE)

        #replace all case statements with incremental integers to prevent any case statement conflicts
        content = replace_case_values(content)

        #extract local headers and save content to memory for use by clang
        local_headers = re.findall(r'#\s*include\s*"([^"]+)"', content, flags=re.MULTILINE)
        local_headers_and_content = []
        for h in local_headers:
            header_content = get_file_content_at_commit(repo, commit, parent_dir / h)
            header_content = re.sub(r'#\s*include\s*<[^>]+>', lambda m: ' ' * len(m.group(0)), header_content, flags=re.MULTILINE)
            header_content = remove_function_like_macros(header_content)
            header_content = ""
            local_headers_and_content.append((parent_dir/h,header_content))
    else:
        pass #need to put back retrieving headers from the git tree

    shorter_content, modified_lines = parse_and_reduce_code(
                content, modified_line_numbers, file_name, use_dummy_header, local_headers_and_content
            )

    """ input = Path(temp_dir, "input", file_name)
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
                atoms.append(row) """

    return atoms

if __name__ == "__main__":
    code = '''
    //start of code
    extern u21 myGlobal;
    #define AGE1 1
    #define AGE2 2
    #define AGE3 3
    
    int func(s8 hello, s16 world)
    {
        struct abcd                                                                                                                                                                                                                                                                                                                                                                                                                                                ;
        const u4 myconst = 10;
        int a = 1;
        int j;
        u32 arr[5];
        u64 b;
        u128 e = 1;
        a = (u32) b;
        u8* c;
        u16** d;
        a = (s16*)c;
        a = (s32**)d;
        
        a =(s8*)func2();
        a = func3 (b) * a;
        
        char x = PAGE; 
        x = MACRO(MAC(a));
        a = func1(y->z);
        printk(age);

        if(a==b){
            a++;
        }
        for(s64 i =0; i < 10; i++) do_stuff();

        switch(age){
            case AGE1:
                x++; 
                a.u=1;
                break;
            case AGE2:
                x++;
                break;
            case AGE3:
                break;
        }
    }
    //end
    '''

    content = re.sub(r'#\s*include\s*<[^>]+>', lambda m: ' ' * len(m.group(0)), code, flags=re.MULTILINE)

    content = replace_case_values(content)
    print(content)

    #extract local headers and save content to memory for use by clang
    local_headers_and_content = []

    start = time.time()
    tu = parse_file(code,'test.c',True,[])
    stop = time.time()
    print(stop - start)

    def walk(cursor, depth=0):
        print("  " * depth, cursor.kind, cursor.spelling, cursor.referenced)
        for child in cursor.get_children():
            walk(child, depth+1)

    def tokens(cursor):
        for token in tu.get_tokens(extent=tu.cursor.extent):
            print(f"{token.spelling} - {token.kind} - {token.cursor.kind} - {token.cursor.referenced}")
    #input(...)
    #walk(tu.cursor)
    


'''
	int arch_bpf_trampoline_size(const struct btf_func_model *m, u32 flags,
			     struct bpf_tramp_links *tlinks, void *func_addr)
    {
        struct bpf_tramp_image im;
        void *image;
        int ret;

        image = bpf_jit_alloc_exec(PAGE_SIZE);
        if (!image)
            return -ENOMEM;
        ret = __arch_prepare_bpf_trampoline(&im, image, image + PAGE_SIZE, image,
                            m, flags, tlinks, func_addr);
        bpf_jit_free_exec(image);
        return ret;
    }
    '''

'''
    int func(){
        struct abcd *y;
        int a = 1;
        char x = PAGE; 
        x = MACRO(MAC(a));
        a = func1(y->z);
        printk(age);
    }
    '''