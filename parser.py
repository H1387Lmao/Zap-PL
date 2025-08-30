from lexer import tokenize
from _errors import *
import json

class AST:
    def __init__(self, type, **kwargs):
        self.type = type
        [setattr(self, a, b) for a, b in kwargs.items()]
        self.kw = kwargs

    def to_dict(self):
        def serialize(value):
            if isinstance(value, AST):
                return value.to_dict()
            elif isinstance(value, list):
                return [serialize(v) for v in value]
            else:
                return value

        return {self.type: {k: serialize(v) for k, v in self.kw.items()}}

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)


class Parser:
    def __init__(self, source):
        self.tokens = tokenize(source)
        print(self.tokens)
        self.pos = 0

    def peek(self, offset=0):
        if self.pos + offset >= len(self.tokens):
            return None
        return self.tokens[self.pos + offset]

    def consume(self):
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def expect(self, value):
        tok = self.consume()
        if value.startswith("!"):
          if tok is None or tok.type != value[1:]:
            ZapError(f"Expected '{value[1:]}', got '{tok.type if tok else None}'", getattr(tok, 'position', None))
        elif tok is None or tok.value != value:
            ZapError(f"Expected '{value}', got '{tok.value if tok else None}'", getattr(tok, 'position', None))
        return tok

    def p_binop(self, func, ops):
        left = func()
        while self.peek() and self.peek().value in ops:
            op = self.consume().value
            right = func()
            left = AST('binop', l=left, o=op, r=right)
        return left

    def p_factor(self):
        tok = self.peek()
        if tok is None:
            ZapError("Unexpected end of input in factor")

        if tok.type in ('Number', 'Identifier', 'Character'):
            self.consume()
            if tok.type == 'Identifier' and self.peek() and self.peek().value == '[':
                indexes = []
                while self.peek() and self.peek().value == '[':
                    self.consume()  # consume '['
                    index_expr = self.p_expr()
                    indexes.append(index_expr)
                    self.expect(']')
                
                return AST('array_access', name=tok.value, indexes=indexes)
            return tok.value
        elif tok.value == '(':
            self.consume()
            expr = self.p_expr()
            self.expect(')')
            return expr
        elif tok.type == "StringLiteral":
            self.consume()
            return AST("StringLiteral", value=tok.value)
        else:
            ZapError(f"Expected value, got '{tok.value}'", tok.position)

    def p_term(self):
        return self.p_binop(self.p_factor, ['*', '/'])

    def p_expr(self):
        return self.p_binop(self.p_term, ['+', '-'])

    def p_cond(self):
        l = self.p_expr()
        condition = self.expect("!Comparison")
        r = self.p_expr()
        return AST("cond", l=l, o=condition.value, r=r)

    def p_stmt(self):
        if self.peek() and self.peek().value in ('int8', 'int16', 'int32', 'int', 'char'):
            type = self.consume().value
            while self.peek().value == "*": #POINTER FOUND
                type+="_pointer"
                self.consume()
            name_tok = self.consume()
            suffixes = []
            while self.peek().value == "[":
                self.consume()
                if self.peek().value == "]":
                    suffixes.append(0)
                    self.expect("]")
                    continue
                suffixes.append(self.p_expr())
                self.expect("]")
            if self.peek().value != "=":
                return AST("assign", v_type=type, name=name_tok.value, array_sizes=suffixes, value=0)
            self.consume()
            expr = self.p_expr()
            return AST("assign", v_type=type, name=name_tok.value, array_sizes=suffixes, value=expr)
        elif self.peek() and self.peek().value == "if":
            self.consume()
            self.expect("(")
            cond = self.p_cond()
            self.expect(")")
            self.expect("{")
            body = self.p_list(self.p_stmt, None, "}")
            return AST("if", cond=cond, body=body)
        elif self.peek() and self.peek().type == "Identifier":
            if self.peek(1) and self.peek(1).value == "(":
                name = self.consume().value
                self.consume()  # consume '('
                args = self.p_list(self.p_expr, ",", ")")
                return AST("func_call", name=name, args=args)
            elif self.peek(1) and self.peek(1).value == "[":
                name = self.consume().value
                indexes = []
                while self.peek() and self.peek().value == '[':
                    self.consume()  # consume '['
                    index_expr = self.p_expr()
                    indexes.append(index_expr)
                    self.expect(']')
                            
                self.expect('=')
                value = self.p_expr()
                return AST("array_assign", name=name, indexes=indexes, value=value)
            else:
                name = self.consume().value
                self.expect('=')
                value = self.p_expr()
                return AST("assign_existing", name=name, value=value)


        # fallback: just an expression
        return self.p_expr()

    def p_list(self, func, sep, end_value):
        items = []
        if self.peek() and self.peek().value == end_value:
            self.consume()
            return items

        while True:
            items.append(func())
            if self.peek() and self.peek().value == end_value:
                self.consume()
                break
            if sep:
                sep_tok = self.consume()
                if not sep_tok or sep_tok.value != sep:
                    ZapError(f"Expected separator '{sep}'", sep_tok.position)
        return items

    def p_stmts(self):
        return self.p_list(self.p_stmt, None, "EOF")

    def p_prog(self):
        return AST("prog", body=self.p_stmts())


if __name__ == "__main__":
    p = Parser(input(">> "))
    print(p.p_prog())

