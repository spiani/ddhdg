from os import path

# The directory where the python script is saved
SCRIPT_DIR = path.dirname(path.realpath(__file__))

# The directory of the git repo of ddhdg
REPO_DIR = path.realpath(path.join(SCRIPT_DIR, '..'))

# The directory of the compiled executable
EXECUTABLE_DIR = path.join(REPO_DIR, 'build')

# The complete path of the executable
EXECUTABLE = path.join(EXECUTABLE_DIR, 'MAIN')
