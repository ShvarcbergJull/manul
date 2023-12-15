from copy import deepcopy

class SingletonClass(type):
    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]
    

class GeneticOperator:
    """
    Abstract class (need to implement method 'apply').
    Operator is applied to some object and change its properties.
    Work inplace (object is being changed by it applying).
    """

    def __init__(self, params: dict = None) -> None:
        if params is None:
            params = {}
        self.params = params

    def _check_params(self, *keys) -> None:
        params_keys = self.params.keys()
        for key in keys:
            assert key in params_keys, "Key {} must be in {}.   params".format(key, type(self).__name__)

    def apply(self, target, *args, **kwargs) -> None:
        raise NotImplementedError("Genetic Operator must doing something with target")
    

class OperatorMap(metaclass=SingletonClass):

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        assert isinstance(value, GeneticOperator), 'Attribute must be "GeneticOperator" object'
        self.__dict__[key] = value


class GeneticOperatorIndivid(GeneticOperator):
    """
    Genetic Operator influencing object Individ.
    Change Individ, doesn't create a new one.
    """
    def __init__(self, params: dict = None):
        super().__init__(params=params)

    def apply_to(self, individ, *args, **kwargs) -> None:
        """Использует метод apply, не переопределять в наследниках."""
        if kwargs:
            tmp_params = self.params.copy()
            for key, value in kwargs.items():
                if key in self.params.keys():
                    self.params[key] = kwargs[key]
        else:
            tmp_params = self.params

        ret = self.apply(individ, *args, **kwargs)
        self.params = tmp_params
        return ret

class GeneticOperatorPopulation(GeneticOperator):  
    """
    Genetic Operator influencing list of Individs in Population.
    May be parallelized.
    May change Individs in population and create new ones. Return new list of Individs.
    """
    def __init__(self, params: dict = None):
        super().__init__(params=params)

    def apply_to(self, population, *args, **kwargs):
        return self.apply(population, *args, **kwargs)

def _methods_decorator(method):
    def wrapper(*args, **kwargs):
        self = args[0]
        # self.change_all_fixes(False)
        return method(*args, **kwargs)
    return wrapper

def apply_decorator(method):
    def wrapper(*args, **kwargs):
        self = args[0]
        try:
            individ = kwargs['individ']
        except KeyError:
            individ = args[1]

        ret = method(*args, **kwargs)
        return ret
    return wrapper