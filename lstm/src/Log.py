import sys
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

'''
#test
placename="place"
sys.stdout = Logger(placename+'.txt')
#type = sys.getfilesystemencoding()
#print(os.path.dirname(__file__))
print("test")
'''