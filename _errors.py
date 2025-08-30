import sys
import argparse

class ZapError:
	def __init__(self, mes, pos=None):
		self.message = mes
		self.position = pos
		self._error_out()
	def _error_out(self):
		print(self.message, end=' ')
		if self.position:
			print("At position:",self.position.start)
		print()
		return sys.exit()

