class TextDocument():
    
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'w') as f:
            pass
    def add_line(self, string):
        with open(self.filename, 'a') as f:
            f.write(string + "\n")
            