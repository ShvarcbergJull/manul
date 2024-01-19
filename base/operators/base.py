from copy import deepcopy
from datetime import datetime
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

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

class ProgramRun(metaclass=SingletonClass):
    name_of_dir = f"info_log/{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"

    def __init__(self, name_directory=None) -> None:
        # create directory
        if name_directory is None:
            os.makedirs(self.name_of_dir)
        else:
            self.name_of_dir = name_directory

    def save_confusion_matrix(self, name: str, data, data2 = None):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        target_true = data[0]
        target_predict = data[1]
        cm = confusion_matrix(target_true.reshape(-1), target_predict.reshape(-1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        

        if data2 is not None:
            f, axes = plt.subplots(1, 2, sharey='row')
            target_true = data2[0]
            target_predict = data2[1]
            cm2 = confusion_matrix(target_true.reshape(-1), target_predict.reshape(-1))
            disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
            disp.plot(ax=axes[0])
            disp.im_.colorbar.remove()
            disp2.plot(ax=axes[1])
            disp2.im_.colorbar.remove()

            f.colorbar(disp.im_, ax=axes)
        else:
            disp.plot()

        plt.savefig(f"{self.name_of_dir}/{name}")
        plt.close()

    def save_plot(self, name, data):
        plt.plot(data)
        plt.savefig(f"{self.name_of_dir}/{name}")
        plt.close()

    def save_plots(self, name, data, labels=None):
        for i, data_i in enumerate(data):
            label_data = i
            if labels is not None:
                label_data = labels[i]
            plt.plot(data_i, label=label_data)
        plt.savefig(f"{self.name_of_dir}/{name}")
        plt.close()

    def save_graph(self, data):
        graph_data = []
        for i, edges in enumerate(data):
            graph_data.append(list(edges.values()))
        with open(f"{self.name_of_dir}/graph.txt", "w") as fl:
            fl.write(str(graph_data))

    def save_boxplot(self, name, data):
        data = np.array(data)
        jg = data[:, 0, :]
        eag = data[:, 1, :]

        box_data = np.concatenate((jg, eag), axis=1)
        plt.boxplot(box_data, labels=["1 класс,\nbase", "2 класс,\nbase", "1 класс,\nman", "2 класс,\nman"])
        plt.savefig(f"{self.name_of_dir}/{name}")
        plt.close()

    def save_model(self, name, model):
        torch.save(model.state_dict(), f"{self.name_of_dir}/{name}.pt")
        
    def load_model(self, name):
        the_model = torch.load(name)
        return the_model


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