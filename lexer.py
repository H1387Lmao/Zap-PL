import re
from collections import namedtuple
from _errors import *

Keywords = ("int", "char", "if", "int8", "int16", "int32")

TokenRegexes = [
    ("Identifier", r"[A-Za-z_]+[A-Za-z0-9]*"),
    ("Number", r"\d+(\.\d+)?"), #"
    ("Character", r"'.'"),
    ("StringLiteral", r"\".*\""),
    ("Ignore", r"[ \t\n]+"),
    ("Comparison", r"==|!="),
    ("Assignment", r"\+|\-|\*|/|\="),
    ("Symbol", r"\(|\)|\{|\}|\[|\]|\,|\.|\:|\&|\||\^|\~|\<|\>"),
    ("Unknown", '.')
]

pattern = "|".join([f"(?P<{type}>{regex})" for type, regex in TokenRegexes])
Token = namedtuple('Token', ['type','value', 'position'])
Position = namedtuple('Position', ['start', 'end'])

def tokenize(text):
	tokens = []
	for match in re.finditer(pattern, text):
		for name in match.groupdict():
			if match.group(name):
				start = match.start(name)
				end = match.end(name)-1
				pos = Position(start, end)
				value = match.group(name)
				if name == "Unknown":
					ZapError(f"Unused character: {value}")
				if name == "Character":
					tokens.append(Token(name, ord(value[1]), pos))
					continue
				if name == "StringLiteral":
					tokens.append(Token(name, value[1:-1], pos))
					continue
				if name == "Ignore":
					continue
				if name == "Identifier":
					if value in Keywords:
						tokens.append(Token(value, value, pos))
						continue
				if name == "Number":
					if '.' in value:
						value = float(value)
					else:
						value = int(value)
				tokens.append(Token(name, value, pos))
	tokens.append(Token("EOF", "EOF", Position(len(text), len(text))))
	return tokens

if __name__ == "__main__":
	for token in tokenize(input(">> ")):
		print(token)
