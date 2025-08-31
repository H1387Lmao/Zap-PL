from parser import *
from _errors import *
import sys, os, subprocess, argparse

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

    def is_numeric(self):
        return self.base in ['int', 'int8', 'int16', 'int32', 'char'] and not self.is_pointer()

    def is_char_array(self):
        return self.base == 'char' and self.is_array()

    def is_string_like(self):
        """Check if type can be treated as a string (char* or char[])"""
        return (self.base == 'char' and self.pointer_depth == 1) or self.is_char_array()

    def can_assign_to(self, other_type):
        """Check if this type can be assigned to another type"""
        # Same type
        if self.base == other_type.base and self.pointer_depth == other_type.pointer_depth:
            return True
        
        # Numeric types can be assigned to each other (with potential warnings)
        if self.is_numeric() and other_type.is_numeric():
            return True
            
        # String literal can be assigned to char array
        if self.base == "string_literal" and other_type.is_char_array():
            return True
            
        # Array name can decay to pointer
        if (self.is_array() and other_type.is_pointer() and 
            self.base == other_type.base and self.pointer_depth == other_type.pointer_depth - 1):
            return True
            
        return False

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

    def add_pointer(self):
        """Add one level of pointer indirection"""
        return Type(self.base, self.pointer_depth + 1, self.array_dimensions)

    def __str__(self):
        # If it's an array, print its dimensions
        if self.is_array():
            dims = "][".join(str(d) for d in self.array_dimensions)
            return f"{self.base}[{dims}]" + "*" * self.pointer_depth
        return self.base + "*" * self.pointer_depth

def parse_type(type_str):
    parts = type_str.split('_pointer')
    base = parts[0]
    ptr_depth = len(parts) - 1
    return Type(base, ptr_depth)

class TypeChecker:
    """Handles type checking and compatibility"""
    
    @staticmethod
    def check_binary_op(left_type, right_type, operator):
        """Check if binary operation is valid between two types"""
        if operator in ['+', '-', '*', '/', '%']:
            if not (left_type.is_numeric() and right_type.is_numeric()):
                ZapError(f"Arithmetic operator '{operator}' requires numeric operands, got {left_type} and {right_type}")
            return left_type  # Result type is left operand type
            
        elif operator in ['==', '!=', '<', '>', '<=', '>=']:
            if left_type.is_numeric() and right_type.is_numeric():
                return Type('int')  # Comparison result is int
            elif left_type.is_pointer() and right_type.is_pointer():
                if left_type.base == right_type.base:
                    return Type('int')
                else:
                    ZapError(f"Cannot compare pointers of different types: {left_type} and {right_type}")
            else:
                ZapError(f"Cannot compare {left_type} and {right_type}")
                
        else:
            ZapError(f"Unknown binary operator: {operator}")
    
    @staticmethod
    def check_assignment(target_type, source_type):
        """Check if assignment is valid"""
        if not source_type.can_assign_to(target_type):
            ZapError(f"Cannot assign {source_type} to {target_type}")
    
    @staticmethod
    def check_array_access(array_type, index_type):
        """Check if array access is valid"""
        if not array_type.is_array() and not array_type.is_pointer():
            ZapError(f"Cannot index non-array/non-pointer type: {array_type}")
        
        if not index_type.is_numeric():
            ZapError(f"Array index must be numeric, got {index_type}")
            
        # Return element type
        if array_type.is_array():
            return array_type.element_type()
        else:  # pointer
            return array_type.deref()
    
    @staticmethod
    def check_dereference(pointer_type):
        """Check if dereferencing is valid"""
        if not pointer_type.is_pointer():
            ZapError(f"Cannot dereference non-pointer type: {pointer_type}")
        return pointer_type.deref()

class Label:
    def __init__(self, name):
        self.name = name
        self.assemblies = []

    def generate(self):
        s = f"\n{self.name}:\n"
        for asm in self.assemblies:
            s += f"    {asm}\n"
        return s+"\n"

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
        self.entry_point = "_start" if target=="elf" else "main"
        self.code = f"global {self.entry_point}\nsection .text\n"

        self.externs = []
        self.data_section = []
        self.string_counter = 0

        self.constants = []
        self.labels = {}
        self.current_label = None

        # Stack of variable scopes: each is dict var_name -> Variable object
        self.scopes = []
        self.stack_offset = 0  # Total stack size used (positive number)
        
        # Type checking
        self.type_checker = TypeChecker()
        
    def get_expression_type(self, expr):
        """Get the type of an expression"""
        if isinstance(expr, int):
            # Check if it's a character literal (ASCII value)
            if 0 <= expr <= 127:
                return Type('char')  # Could be char, but default to int for safety
            return Type('int')
        elif isinstance(expr, str):
            # Variable reference
            var = self.get_variable(expr)
            # Arrays decay to pointers when used as expressions
            if var.type.is_array():
                return Type(var.type.base, var.type.pointer_depth + 1)
            return var.type
        elif hasattr(expr, 'type'):
            if expr.type == "StringLiteral":
                return Type('char', 1)  # char*
            elif expr.type == "array_access":
                var = self.get_variable(expr.name)
                index_type = self.get_expression_type(expr.indexes[0])  # Simplified for first index
                return self.type_checker.check_array_access(var.type, index_type)
            elif expr.type == "binop":
                left_type = self.get_expression_type(expr.l)
                right_type = self.get_expression_type(expr.r)
                return self.type_checker.check_binary_op(left_type, right_type, expr.o)
            elif expr.type == "dereference":
                pointer_type = self.get_expression_type(expr.expr)
                return self.type_checker.check_dereference(pointer_type)
            elif expr.type == "address_of":
                expr_type = self.get_expression_type(expr.expr)
                return expr_type.add_pointer()
        
        return Type('int')  # Default fallback
        
    def new_label(self, prefix="label"):
        CodeGen.label_count += 1
        return f"{prefix}_{CodeGen.label_count}"
        
    def enter_scope(self):
        # Push a new scope dict and reset stack offset for this scope
        self.scopes.append({})

    def exit_scope(self):
        # Pop scope - variables will be cleaned up by stack unwinding
        if not self.scopes:
            ZapError("No scope to exit")
        self.scopes.pop()

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
            # Convert AST expressions to integers if needed
            dimensions = []
            for dim in array_size:
                if isinstance(dim, int):
                    dimensions.append(dim)
                elif hasattr(dim, 'type') and dim.type == 'Number':
                    dimensions.append(dim.value)
                elif dim == 0:  # Handle empty array brackets []
                    dimensions.append(0)  # Will be set later based on initializer
                else:
                    # For now, assume it's a literal integer
                    dimensions.append(dim)
            var_type.array_dimensions = dimensions
        
        size = var_type.size()
        # For arrays with unknown size (0), we'll update the size later
        if size == 0:
            size = 8  # Temporary size, will be updated
            
        # Align to 8 bytes
        size = (size + 7) & ~7
        
        self.stack_offset += size
        current_scope[name] = Variable(var_type, name, self.stack_offset)

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
            # If it's an array, load the address, not the value
            if var.type.is_array():
                return [f"lea rax, [rbp-{var.offset}]"]
            else:
                return [f"mov rax, [rbp-{var.offset}]"]

        elif ast.type == "StringLiteral":
            label = f"__str{self.string_counter}"
            self.string_counter += 1
            escaped = ast.value.encode('unicode_escape').decode('ascii')
            self.data_section.append(f'{label}: db "{escaped}", 0')
            return [f"lea rax, [rel {label}]"]

        elif ast.type == "dereference":
            # *expr - dereference a pointer
            pointer_code = self.generate(ast.expr)
            pointer_type = self.get_expression_type(ast.expr)
            
            # Type check
            if not pointer_type.is_pointer():
                ZapError(f"Cannot dereference non-pointer type: {pointer_type}")
            
            result_type = pointer_type.deref()
            code = pointer_code
            
            # Load value from the address in rax
            if result_type.base == 'char':
                code.append("movzx rax, byte [rax]")
            elif result_type.base == 'int16':
                code.append("movzx rax, word [rax]")
            elif result_type.base == 'int32':
                code.append("mov eax, [rax]")
            else:  # int, int8, or pointer
                code.append("mov rax, [rax]")
            
            return code

        elif ast.type == "address_of":
            # &expr - get address of a variable or array element
            if isinstance(ast.expr, str):
                # &variable
                var = self.get_variable(ast.expr)
                return [f"lea rax, [rbp-{var.offset}]"]
            elif hasattr(ast.expr, 'type') and ast.expr.type == "array_access":
                # &array[index] - get address of array element
                var = self.get_variable(ast.expr.name)
                code = []
                
                # Get base address of array
                code.append(f"lea rbx, [rbp-{var.offset}]")
                
                # Calculate offset
                code.append("xor rsi, rsi")
                for i, index_expr in enumerate(ast.expr.indexes):
                    index_code = self.generate(index_expr)
                    code.extend(index_code)
                    
                    element_size = var.type.base_size()
                    for dim in var.type.array_dimensions[i+1:]:
                        element_size *= dim
                    
                    if element_size != 1:
                        code.append(f"mov rcx, {element_size}")
                        code.append("imul rax, rcx")
                    
                    code.append("add rsi, rax")
                
                # Return address
                code.append("add rbx, rsi")
                code.append("mov rax, rbx")
                return code
            else:
                ZapError("Cannot take address of this expression")

        elif ast.type == "array_access":
            var = self.get_variable(ast.name)
            
            # Type check
            for index_expr in ast.indexes:
                index_type = self.get_expression_type(index_expr)
                if not index_type.is_numeric():
                    ZapError(f"Array index must be numeric, got {index_type}")
            
            code = []
            
            # Get base address of array
            code.append(f"lea rbx, [rbp-{var.offset}]")
            
            # Calculate offset for each index
            code.append("xor rsi, rsi")
            
            for i, index_expr in enumerate(ast.indexes):
                # Generate code to evaluate index
                index_code = self.generate(index_expr)
                code.extend(index_code)
                
                # Calculate element size for this dimension
                element_size = var.type.base_size()
                for dim in var.type.array_dimensions[i+1:]:
                    element_size *= dim
                
                if element_size != 1:
                    code.append(f"mov rcx, {element_size}")
                    code.append("imul rax, rcx")
                
                code.append("add rsi, rax")
            
            # Load from calculated address
            code.append("add rbx, rsi")
            if var.type.base == 'char':
                code.append("movzx rax, byte [rbx]")
            else:
                code.append("mov rax, [rbx]")
                
            return code

        elif ast.type == "array_assign":
            var = self.get_variable(ast.name)
            
            # Type check
            for index_expr in ast.indexes:
                index_type = self.get_expression_type(index_expr)
                if not index_type.is_numeric():
                    ZapError(f"Array index must be numeric, got {index_type}")
            
            # Type check assignment
            element_type = var.type.element_type()
            value_type = self.get_expression_type(ast.value)
            self.type_checker.check_assignment(element_type, value_type)
            
            code = []
            
            # Get base address of array
            code.append(f"lea rbx, [rbp-{var.offset}]")
            
            # Calculate offset
            code.append("xor rsi, rsi")
            for i, index_expr in enumerate(ast.indexes):
                index_code = self.generate(index_expr)
                code.extend(index_code)
                
                element_size = var.type.base_size()
                for dim in var.type.array_dimensions[i+1:]:
                    element_size *= dim
                
                if element_size != 1:
                    code.append(f"mov rcx, {element_size}")
                    code.append("imul rax, rcx")
                
                code.append("add rsi, rax")
            
            # Save address
            code.append("add rbx, rsi")
            code.append("push rbx")
            
            # Generate value to store
            value_code = self.generate(ast.value)
            code.extend(value_code)
            
            # Store value at calculated address
            code.append("pop rbx")
            if var.type.base == 'char':
                code.append("mov [rbx], al")
            elif var.type.base == 'int16':
                code.append("mov [rbx], ax")
            elif var.type.base == 'int32':
                code.append("mov [rbx], eax")
            else:  # int, pointers
                code.append("mov [rbx], rax")
                
            return code

        elif ast.type == "assign_existing":
            var = self.get_variable(ast.name)
            
            # Type check assignment
            value_type = self.get_expression_type(ast.value)
            self.type_checker.check_assignment(var.type, value_type)
            
            code = self.generate(ast.value)
            
            if var.type.base == 'char' and not var.type.is_pointer():
                code.append(f"mov [rbp-{var.offset}], al")
            elif var.type.base == 'int16':
                code.append(f"mov [rbp-{var.offset}], ax")
            elif var.type.base == 'int32':
                code.append(f"mov [rbp-{var.offset}], eax")
            else:  # int, pointers
                code.append(f"mov [rbp-{var.offset}], rax")
            
            return code

        elif ast.type == "deref_assign":
            # *target = value - assign through pointer
            target_type = self.get_expression_type(ast.target)
            if not target_type.is_pointer():
                ZapError(f"Cannot dereference non-pointer type: {target_type}")
            
            deref_type = target_type.deref()
            value_type = self.get_expression_type(ast.value)
            self.type_checker.check_assignment(deref_type, value_type)
            
            code = []
            
            # Generate code to get the pointer address
            target_code = self.generate(ast.target)
            code.extend(target_code)
            code.append("push rax")  # Save pointer address
            
            # Generate code to get the value
            value_code = self.generate(ast.value)
            code.extend(value_code)
            
            # Store value through pointer
            code.append("pop rbx")  # Get pointer address back
            if deref_type.base == 'char':
                code.append("mov [rbx], al")
            elif deref_type.base == 'int16':
                code.append("mov [rbx], ax")
            elif deref_type.base == 'int32':
                code.append("mov [rbx], eax")
            else:  # int, pointers
                code.append("mov [rbx], rax")
            
            return code
            
        elif ast.type == "if":
            cond = ast.cond
            body = ast.body

            # Generate condition code for left and right
            left_code = self.generate(cond.l)
            right_code = self.generate(cond.r)

            end_label = self.new_label("ifend")

            code = []
            # Evaluate left side into rax
            code += left_code
            code.append("push rax")   # Save left on stack

            # Evaluate right side into rax
            code += right_code
            code.append("mov rbx, rax")  # Move right side to rbx

            code.append("pop rax")  # Restore left side in rax

            # Compare left (rax) and right (rbx)
            code.append("cmp rax, rbx")

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
            code.append(f"{end_label}:")

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
            
            # Add exit syscall for ELF format
            if self.target == "elf":
                self.current_label.add([
                    "mov rax, 60",  # sys_exit
                    "mov rdi, 0",   # exit status
                    "syscall"
                ])
            else:
                self.end_stack(stack_size)

            self.exit_scope()  # exit global scope

            # Generate all labels
            for label in self.labels.values():
                self.code += label.generate()

            # Add external declarations
            for extern in self.externs:
                self.code = f"extern {extern}\n" + self.code

            # Add data section
            if self.data_section:
                self.code += "section .data\n" + "\n".join(self.data_section) + '\n'

            return self.code

        elif ast.type == "assign":
            name = ast.name
            var_type_str = ast.v_type
            array_sizes = ast.array_sizes
            
            code = []
            
            # Special handling for string literals assigned to char arrays
            if (var_type_str == "char" and array_sizes and 
                hasattr(ast.value, 'type') and ast.value.type == "StringLiteral"):
                
                # Calculate actual array size from string length
                string_len = len(ast.value.value) + 1  # +1 for null terminator
                if array_sizes[0] == 0:  # If size not specified, use string length
                    array_sizes[0] = string_len
                
                # Allocate the variable
                self.allocate_var(name, var_type_str, array_sizes)
                var = self.get_variable(name)
                
                # Create string literal in data section
                label = f"__str{self.string_counter}"
                self.string_counter += 1
                escaped = ast.value.value.encode('unicode_escape').decode('ascii')
                self.data_section.append(f'{label}: db "{escaped}", 0')
                
                # Copy string data to array
                code.extend([
                    f"lea rsi, [rel {label}]",      # Source: string literal
                    f"lea rdi, [rbp-{var.offset}]", # Dest: array on stack
                    f"mov rcx, {string_len}",       # Count: string length + null
                    "rep movsb"                     # Copy bytes
                ])
                
            else:
                # Allocate the variable first
                self.allocate_var(name, var_type_str, array_sizes)
                var = self.get_variable(name)
                
                # Type check assignment if there's a value
                if ast.value != 0:
                    value_type = self.get_expression_type(ast.value)
                    self.type_checker.check_assignment(var.type, value_type)
                
                # Handle other initialization
                if array_sizes and ast.value != 0:
                    if isinstance(ast.value, int) and ast.value == 0:
                        # Initialize array to zeros
                        total_size = var.type.size()
                        code.append(f"lea rdi, [rbp-{var.offset}]")
                        code.append(f"mov rcx, {total_size}")
                        code.append("xor al, al")
                        code.append("rep stosb")
                    else:
                        # Single value initialization (first element only)
                        val_code = self.generate(ast.value)
                        code.extend(val_code)
                        if var.type.base == 'char':
                            code.append(f"mov [rbp-{var.offset}], al")
                        elif var.type.base == 'int16':
                            code.append(f"mov [rbp-{var.offset}], ax")
                        elif var.type.base == 'int32':
                            code.append(f"mov [rbp-{var.offset}], eax")
                        else:
                            code.append(f"mov [rbp-{var.offset}], rax")
                elif ast.value != 0:
                    # Regular variable assignment
                    val_code = self.generate(ast.value)
                    code.extend(val_code)
                    if var.type.base == 'char' and not var.type.is_pointer():
                        code.append(f"mov [rbp-{var.offset}], al")
                    elif var.type.base == 'int16':
                        code.append(f"mov [rbp-{var.offset}], ax")
                    elif var.type.base == 'int32':
                        code.append(f"mov [rbp-{var.offset}], eax")
                    else:
                        code.append(f"mov [rbp-{var.offset}], rax")
                # If value is 0, we don't need to initialize (stack is already zeroed)
            
            return code

        elif ast.type == "func_call":
            if ast.name == "print":
                # Type check print argument
                arg_type = self.get_expression_type(ast.args[0])
                if not arg_type.is_string_like():
                    print(f"Warning: print() expects string-like type, got {arg_type}")
                
                arg_code = self.generate(ast.args[0])
                if self.target == "elf":
                    # For ELF, use write syscall with proper string length calculation
                    strlen_label = self.new_label("strlen")
                    strlen_end = self.new_label("strlen_end")
                    print_label = self.new_label("print_str")
                    
                    # Add strlen function if not already added
                    if "print_str_func" not in self.labels:
                        strlen_func = Label(print_label)
                        strlen_func.add([
                            "mov rsi, rax",      # String address
                            "xor rdx, rdx",      # Length counter
                            f"{strlen_label}:",
                            f"mov al, [rsi + rdx]", # Load character
                            "test al, al",       # Check for null terminator
                            f"jz {strlen_end}",  # Jump if null
                            "inc rdx",           # Increment length
                            f"jmp {strlen_label}", # Continue loop
                            f"{strlen_end}:",
                            "mov rax, 1",        # sys_write
                            "mov rdi, 1",        # stdout
                            "syscall",           # Write string
                            "ret"
                        ])
                        self.labels["print_str_func"] = strlen_func
                    
                    return arg_code + [f"call {print_label}"]
                else:
                    self.use_external("puts")
                    code = arg_code
                    code.extend([
                        "sub rsp, 32",  # Shadow space for Windows
                        "mov rcx, rax",
                        "call puts",
                        "add rsp, 32"
                    ])
                    return code
                    
            elif ast.name == "exit":
                if ast.args and len(ast.args) > 0:
                    # Type check exit code
                    exit_type = self.get_expression_type(ast.args[0])
                    if not exit_type.is_numeric():
                        ZapError(f"exit() requires numeric argument, got {exit_type}")
                    
                    v = self.generate(ast.args[0])
                    if self.target == "elf":
                        return v + ["mov rdi, rax", "mov rax, 60", "syscall"]
                    else:
                        self.use_external("ExitProcess")
                        return v + ["sub rsp, 32", "mov rcx, rax", "call ExitProcess", "add rsp, 32"]
                else:
                    if self.target == "elf":
                        return ["mov rax, 60", "mov rdi, 0", "syscall"]
                    else:
                        self.use_external("ExitProcess")
                        return ["sub rsp, 32", "mov rcx, 0", "call ExitProcess", "add rsp, 32"]
        else:
            ZapError(f"Unknown AST node type: {ast.type}")
            # Special handling for string literals assigned to char arrays
            if (var_type_str == "char" and array_sizes and 
                hasattr(ast.value, 'type') and ast.value.type == "StringLiteral"):
                
                # Calculate actual array size from string length
                string_len = len(ast.value.value) + 1  # +1 for null terminator
                if array_sizes[0] == 0:  # If size not specified, use string length
                    array_sizes[0] = string_len
                
                # Allocate the variable
                self.allocate_var(name, var_type_str, array_sizes)
                var = self.get_variable(name)
                
                # Create string literal in data section
                label = f"__str{self.string_counter}"
                self.string_counter += 1
                escaped = ast.value.value.encode('unicode_escape').decode('ascii')
                self.data_section.append(f'{label}: db "{escaped}", 0')
                
                # Copy string data to array
                code.extend([
                    f"lea rsi, [rel {label}]",      # Source: string literal
                    f"lea rdi, [rbp-{var.offset}]", # Dest: array on stack
                    f"mov rcx, {string_len}",       # Count: string length + null
                    "rep movsb"                     # Copy bytes
                ])
                
            else:
                # Allocate the variable first
                self.allocate_var(name, var_type_str, array_sizes)
                var = self.get_variable(name)
                
                # Handle other array initialization
                if array_sizes and ast.value != 0:
                    if isinstance(ast.value, int) and ast.value == 0:
                        # Initialize array to zeros
                        total_size = var.type.size()
                        code.append(f"lea rdi, [rbp-{var.offset}]")
                        code.append(f"mov rcx, {total_size}")
                        code.append("xor al, al")
                        code.append("rep stosb")
                    else:
                        # Single value initialization (first element only)
                        val_code = self.generate(ast.value)
                        code.extend(val_code)
                        code.append(f"mov [rbp-{var.offset}], rax")
                elif ast.value != 0:
                    # Regular variable assignment
                    val_code = self.generate(ast.value)
                    code.extend(val_code)
                    code.append(f"mov [rbp-{var.offset}], rax")
                # If value is 0, we don't need to initialize (stack is already zeroed)
            
            return code

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
    
    with open(args.input_file) as r:
        source = r.read()
    
    AST = Parser(source).p_prog()
    print(AST)
    cg = CodeGen(target=args.format)
    asm = cg.generate(AST)

    with open(__asm_o, "w") as f:
        f.write(asm)

    if not args.Asm:
        subprocess.run(["nasm", f"-f{args.format}64", __asm_o, "-o", __obj_o])
        if args.format == "elf":
            subprocess.run(["ld", __obj_o, "-o", __exe_o])
        else:
            subprocess.run(["gcc", "-mconsole", __obj_o, "-o", __exe_o])

        if not args.keep_asm:
            os.remove(__asm_o)
        if not args.keep_o:
            os.remove(__obj_o)
