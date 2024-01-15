class ParseResult(object):
    n_task = 0
    task_difficult = []
    task_timing = []
    n_developers = 0
    developers_cof = []

    def __init__(self, n_task, task_difficult, task_timing, n_developers, developers_cof):
        self.n_task = n_task
        self.task_difficult = task_difficult
        self.task_timing = task_timing
        self.n_developers = n_developers
        self.developers_cof = developers_cof


def parse_from_txt(filename):
    n_task = 0
    task_difficult = []
    task_timing = []
    n_developers = 0
    developers_cof = []
    
    with open(filename, 'r') as file:
        n_task = int(file.readline())
        task_difficult = [int(item) for item in file.readline().split(' ')]
        task_timing = [float(item) for item in file.readline().split(' ')]
        n_developers = int(file.readline())
        
        for i in range(n_developers):
            developer_cof = [float(item) for item in file.readline().split(' ')]
            developers_cof.append(developer_cof)
            
    result = ParseResult(n_task, task_difficult, task_timing, n_developers, developers_cof)
    
    return result
