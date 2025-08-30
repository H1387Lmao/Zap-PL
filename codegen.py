from parser import *
from _errors import *
import sys, os, subprocess

class Type:
    def __init__(self, base, pointer_depth=0, array_dimensions=None):
        self.base = base
        self.pointer_depth = pointer_depth
        self.array_dimensions = array_dimensions or []
        
    def element_type(self):
        """Get the type of array elements"""
        if self.is_array():
            return Type(self.base, self.pointer_depth, self.array_dimensions[1:])
        return self
        
    def total_elements(self):
        """Get total number of elements in the array"""
        if not self.is_array():
            return 1
        total = 1
        for dim in self.array_dimensions:
            total *= dim
        return total

    def is_pointer(self):
        return self.pointer_depth > 0

    def is_array(self):
        return len(self.array_dimensions) > 0

    def size(self):
        # Base size
        base_size = self.base_size()

        # If it's a pointer, return 8 (64-bit)
        if self.is_pointer():
            return 8

        # If it's an array, calculate its total size based on the dimensions
        if self.is_array():
            size = base_size
            for dim in self.array_dimensions:
                size *= dim
            return size
        return base_size

    def base_size(self):
        return {
            'char': 1,
            'int8': 1,
            'int16': 2,
            'int32': 4,
            'int': 8,
            'char*': 8,  # Pointer size
        }.get(self.base, 8)  # Default to 8 if not found

    def deref(self):
        if self.pointer_depth == 0:
            raise Exception(f"Cannot dereference non-pointer type {self}")
        return Type(self.base, self.pointer_depth - 1, self.array_dimensions)

    def __str__(self):
        # If it's an array, print its dimensions
        if self.is_array():
            dims = "][".join(str(d) for d in self.array_dimensions)
            return f"{self.base}[{dims}]"
        return self.base + "_pointer" * self.pointer_depth



def parse_type(type_str):
    parts = type_str.split('_pointer')
    base = parts[0]
    ptr_depth = len(parts) - 1
    return Type(base, ptr_depth)

class Label:
    def __init__(self, name):
        self.name = name
        self.assemblies = []

    def generate(self):
        s = f"\n{self.name}:\n"
        for asm in self.assemblies:
            s += f"{asm}\n"
        return s+"\n\n"

    def add(self, instructions):
        if isinstance(instructions, list):
            self.assemblies.extend(instructions)
        else:
            self.assemblies.append(instructions)

class Variable:
    def __init__(self, type: Type, name, offset):
        self.type = type
        self.name = name
        self.offset = offset
class CodeGen:
    label_count = 0
    def __init__(self, target='elf'):
        self.target = target
        self.entry_point = "_main" if target=="elf" else "main"
        self.code = f"global {self.entry_point}\nsection .text\n"

        self.externs = []
        self.data_section = []
        self.string_counter = 0

        self.constants = []
        self.labels = {}
        self.current_label = None

        # Stack of variable scopes: each is dict var_name -> stack offset
        self.scopes = []
        self.stack_offset = 0  # Total stack size used (positive number)
        self.current_scope_offset = 0  # Track offset within current scope
        
    def new_label(self, prefix="label"):
        self.label_count += 1
        return f"{prefix}_{self.label_count}"
        
    def enter_scope(self):
        # Push a new scope dict
        self.scopes.append({})
        self.current_scope_offset = 0

    def exit_scope(self):
        # Pop scope and free its stack space
        if not self.scopes:
            ZapError("No scope to exit")

        scope = self.scopes.pop()
        # Reset current scope offset
        self.current_scope_offset = 0

    def get_variable(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        ZapError(f"Undefined variable: {name}")

    def allocate_var(self, name, var_type_str, array_size=None):
        if not self.scopes:
            ZapError("No active scope for variable allocation")

        current_scope = self.scopes[-1]
        if name in current_scope:
            ZapError(f"Variable already defined in this scope: {name}")

        var_type = parse_type(var_type_str)
        if array_size:
            var_type.array_dimensions = array_size  # Set array size
        size = var_type.size()

        # Align to 8 bytes
        size = (size + 7) & ~7
        offset = self.current_scope_offset
        self.current_scope_offset += size
        self.stack_offset = max(self.stack_offset, self.current_scope_offset)
        
        current_scope[name] = Variable(var_type, name, offset)

    def get_var_offset(self, name):
        # Search scopes from innermost to outermost
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        ZapError(f"Undefined variable: {name}")

    def create_stack(self, size):
        if size > 0:
            # Align stack to 16 bytes
            aligned_size = (size + 15) & ~15
            self.current_label.add([
                "push rbp",
                "mov rbp, rsp",
                f"sub rsp, {aligned_size}"
            ])

    def end_stack(self, size):
        if size > 0:
            self.current_label.add([
                "mov rsp, rbp",
                "pop rbp"
            ])
        self.current_label.add(["ret"])

    def use_external(self, ext):
        if ext not in self.externs:
            self.externs.append(ext)

    def generate(self, ast):
        if isinstance(ast, int):
            return [f"mov rax, {ast}"]

        elif isinstance(ast, str):
            var = self.get_variable(ast)
            return [f"mov rax, [rbp-{var.offset+8}]"]  # +8 for pushed RBP

        elif ast.type == "StringLiteral":
            label = f"__str{self.string_counter}"
            self.string_counter += 1
            escaped = ast.value.encode('unicode_escape').decode('ascii')
            self.data_section.append(f'{label}: db "{escaped}", 0')
            return [f"lea rax, [rel {label}]"]

        elif ast.type == "array_access":
            var = self.get_variable(ast.name)
            code = []
            
            # Get base address of array
            code.append(f"lea rax, [rbp-{var.offset+8}]")
            
            # Calculate offset for each index
            total_offset = 0
            for i, index_expr in enumerate(ast.indexes):
                index_code = self.generate(index_expr)
                code.extend(index_code)
                
                # Calculate element size for this dimension
                element_size = var.type.base_size()
                for dim in var.type.array_dimensions[i+1:]:
                    element_size *= dim
                
                code.append(f"mov rbx, {element_size}")
                code.append("imul rax, rbx")
                
                if i == 0:
                    code.append("mov rsi, rax")  # Save the offset
                else:
                    code.append("add rsi, rax")  # Accumulate offsets
            
            # Load from calculated address
            if i == 0:
                code.append("mov rsi, rax")
            
            if var.type.base == 'char':
                code.append("movzx eax, byte [rsi]")
            else:
                code.append(f"mov eax, [rsi]")
                
            return code

        elif ast.type == "array_assign":
            var = self.get_variable(ast.name)
            code = []
            
            # Get base address of array
            code.append(f"lea rax, [rbp-{var.offset+8}]")
            
            # Calculate offset
            total_offset = 0
            for i, index_expr in enumerate(ast.indexes):
                index_code = self.generate(index_expr)
                code.extend(index_code)
                
                element_size = var.type.base_size()
                for dim in var.type.array_dimensions[i+1:]:
                    element_size *= dim
                
                code.append(f"mov rbx, {element_size}")
                code.append("imul rax, rbx")
                
                if i == 0:
                    code.append("mov rsi, rax")
                else:
                    code.append("add rsi, rax")
            
            # Save address
            code.append("push rsi")
            
            # Generate value to store
            value_code = self.generate(ast.value)
            code.extend(value_code)
            
            # Store value at calculated address
            code.append("pop rdi")
            if var.type.base == 'char':
                code.append("mov [rdi], al")
            else:
                code.append("mov [rdi], eax")
                
            return code

        elif ast.type == "assign_existing":
            var = self.get_variable(ast.name)
            code = self.generate(ast.value)
            code.append(f"mov [rbp-{var.offset+8}], rax")
            return code
            
        elif ast.type == "if":
            cond = ast.cond
            body = ast.body

            # Generate condition code for left and right
            left_code = self.generate(cond.l)
            right_code = self.generate(cond.r)

            end_label = self.new_label("ifend")

            code = []
            # Evaluate left side into eax/rax
            code += left_code
            code.append("push rax")   # Save left on stack

            # Evaluate right side into eax/rax
            code += right_code
            code.append("mov ebx, eax")  # Move right side to ebx

            code.append("pop rax")  # Restore left side in rax

            # Compare left (rax/eax) and right (ebx/ebx)
            code.append("cmp eax, ebx")

            # Depending on operator, jump if false
            if cond.o == "==":
                code.append(f"jne {end_label}")
            elif cond.o == "!=":
                code.append(f"je {end_label}")
            elif cond.o == "<":
                code.append(f"jge {end_label}")
            elif cond.o == "<=":
                code.append(f"jg {end_label}")
            elif cond.o == ">":
                code.append(f"jle {end_label}")
            elif cond.o == ">=":
                code.append(f"jl {end_label}")
            else:
                ZapError(f"Unknown comparison operator: {cond.o}")

            # Generate body code
            for stmt in body:
                code += self.generate(stmt)

            # End label
            code.append(f"\n{end_label}:")

            return code

        elif ast.type == "prog":
            self.current_label = Label(self.entry_point)
            self.labels[self.entry_point] = self.current_label

            self.enter_scope()  # enter global scope

            body_instructions = []
            for stmt in ast.body:
                instr = self.generate(stmt)
                if instr:
                    body_instructions.extend(instr)

            stack_size = self.stack_offset
            self.create_stack(stack_size)
            self.current_label.add(body_instructions)
            self.end_stack(stack_size)

            self.exit_scope()  # exit global scope

            for label in self.labels.values():
                self.code += label.generate()

            for extern in self.externs:
              self.code = f"extern {extern}\n"+self.code

            if self.data_section:
                self.code += "section .data\n"+"\n".join(self.data_section)+'\n'

            return self.code

        elif ast.type == "assign":
            name = ast.name
            var_type_str = ast.v_type
            array_sizes = ast.array_sizes
            
            # Handle array initialization
            if array_sizes and ast.value != 0:
                # For arrays, we need to initialize each element
                var_type = parse_type(var_type_str)
                if array_sizes:
                    var_type.array_dimensions = array_sizes
                
                self.allocate_var(name, var_type_str, array_sizes)
                var = self.get_variable(name)
                
                code = []
                if isinstance(ast.value, int) and ast.value == 0:
                    # Initialize array to zeros
                    total_size = var.type.size()
                    code.append(f"lea rdi, [rbp-{var.offset+8}]")
                    code.append(f"mov rcx, {total_size}")
                    code.append("xor al, al")
                    code.append("rep stosb")
                else:
                    # Single value initialization (first element only)
                    val_code = self.generate(ast.value)
                    code.extend(val_code)
                    code.append(f"mov [rbp-{var.offset+8}], rax")
                
                return code
            else:
                # Regular variable assignment
                self.allocate_var(name, var_type_str, array_sizes)
                var = self.get_variable(name)
                val_code = self.generate(ast.value)
                return val_code + [f"mov [rbp-{var.offset+8}], rax"]

        elif ast.type == "binop":
            left = self.generate(ast.l)
            right = self.generate(ast.r)
            op = ast.o

            code = []
            code += left
            code.append("push rax")
            code += right
            code.append("mov rbx, rax")
            code.append("pop rax")

            if op == "+":
                code.append("add rax, rbx")
            elif op == "-":
                code.append("sub rax, rbx")
            elif op == "*":
                code.append("imul rax, rbx")
            elif op == "/":
                code.append("xor rdx, rdx")
                code.append("idiv rbx")

            return code

        elif ast.type == "func_call":
            if ast.name == "print":
                arg = self.generate(ast.args[0])
                if self.target == "elf":
                    return arg + [
                        "mov rdi, rax",
                        "call print_str"
                    ]
                else:
                    self.use_external("puts")
                    # For Windows, we need to handle different data types
                    code = arg
                    # Check if it's a char (single character)
                    if isinstance(ast.args[0], int) or (hasattr(ast.args[0], 'type') and ast.args[0].type != 'StringLiteral'):
                        # Convert number to string for printing
                        code.extend([
                            "sub rsp, 32",  # Shadow space for Windows
                            "mov rcx, rax",
                            "call puts",
                            "add rsp, 32"
                        ])
                    else:
                        # String literal
                        code.extend([
                            "sub rsp, 32",  # Shadow space
                            "mov rcx, rax",
                            "call puts",
                            "add rsp, 32"
                        ])
                    return code
            elif ast.name == "exit":
                if ast.args[0] == 0:
                    if self.target == "elf":
                        return ["mov rax, 60", "mov rdi, 0", "syscall"]
                    else:
                        self.use_external("ExitProcess")
                        return ["sub rsp, 32", "mov rcx, 0", "call ExitProcess", "add rsp, 32"]
                else:
                    v = self.generate(ast.args[0])
                    if self.target == "elf":
                        return v + ["mov rdi, rax", "mov rax, 60", "syscall"]
                    else:
                        self.use_external("ExitProcess")
                        return v + ["sub rsp, 32", "mov rcx, rax", "call ExitProcess", "add rsp, 32"]
        else:
            ZapError(f"Unknown AST node type: {ast.type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description="Your compiler frontend",
      usage="%(prog)s [options] input_files"
    )

    parser.add_argument(
      'input_file',
      help='input source file'
    )

    parser.add_argument(
        '-f', '--format',
        choices=['elf', 'win'],
        default='elf',
        help='target output format (default: elf)'
    )

    parser.add_argument(
      '-o', '--output',
      help='output source file'
    )

    parser.add_argument(
      '-S', '--Asm',
      action='store_true',
      help='output only assembly file'
    )

    parser.add_argument(
        '--keep-asm',
        action='store_true',
        help='keep intermediate assembly files'
    )
    parser.add_argument(
        '--keep-o',
        action='store_true',
        help='keep intermediate object files'
    )

    args = parser.parse_args()

    if not args.input_file:
      print("fatal: no input file")
      sys.exit(1)
    output = args.output if args.output is not None else args.input_file
    # ASSEMBLY
    __base = ".".join(output.split(".")[:-1])
    __asm_o = __base+'.asm'
    __obj_o = __base+'.o'
    __exe_o = __base+'.exe' if args.format == "win" else __base
    
    r = open(args.input_file)
    
    f = open(__asm_o, "w")
    AST = Parser(r.read()).p_prog()
    print(AST)
    cg = CodeGen(target=args.format)
    asm = cg.generate(AST)

    f.write(asm)
    f.close()

    r.close()

    if not args.Asm:
        subprocess.run(["nasm", f"-f{args.format}64", __asm_o, "-o", __obj_o])
        subprocess.run(["gcc", "-mconsole",__obj_o, "-o", __exe_o])

        if not args.keep_asm:
            os.remove(__asm_o)
        if not args.keep_o:
            os.remove(__obj_o)
