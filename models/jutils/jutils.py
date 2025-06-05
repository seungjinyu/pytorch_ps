import sys

class DualLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # for compatibility with Python’s output buffering
        self.terminal.flush()
        self.log.flush()