import os
from time import gmtime, strftime

DEBUG=0
INFO=1
WARN=2
ERROR=3

LEVEL = ERROR

_idx2str = ['D', 'I', 'W', 'E']

get_logger = lambda x:Logger(x)

class Logger():
    def __init__(self, name='') -> None:
        self.name = name
        if self.name != '':
            self.name = '[' + self.name + ']'
        self.path = None

        self.debug = self._generate_print_func(DEBUG)
        self.info = self._generate_print_func(INFO)
        self.warn = self._generate_print_func(WARN)
        self.error = self._generate_print_func(ERROR)
    
    def to_json(self):
        return {
            'path': self.path,
            'name': self.name
        }

    @classmethod
    def from_json(cls, js):
        a = cls(js['name'])
        a.set_path(js['path'])
        return a

    def set_path(self, path):
        self.path = path

    def _generate_print_func(self, level=DEBUG):
        def prin(*args, end='\n'):
            strs = ' '.join([str(a) for a in args])
            str_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            if level >= LEVEL:
                print('[' + _idx2str[level] + '][' + str_time + ']' + self.name, strs, end=end)
            if self.path is not None:
                open(self.path, 'a').write(
                    '[' + _idx2str[level] + '][' + str_time + ']' + self.name + strs + end
                )
        return prin
