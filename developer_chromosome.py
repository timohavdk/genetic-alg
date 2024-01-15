class DeveloperChromosome(object):
    number=0
    productivity=[]
    timing_for_task=0
    task_number=0
    
    def __init__(self, number, productivity, timing_for_task, task):
        self.number = number
        self.productivity = productivity
        self.timing_for_task = timing_for_task
        self.task = task