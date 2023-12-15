import numpy as np
from abc import ABC, abstractmethod
import networkx as nx

from base.operators.base import OperatorMap

class ComplexStructure:
    """
    """

    def __init__(self, structure):
        if structure is None:
            structure = []
        self._structure = structure

    @property
    def structure(self) -> list:
        return self._structure
    
    @structure.setter
    def structure(self, structure):
        assert type(structure) == list, "structure must be a list"
        self._structure = structure

    def get_substructure(self, idx: int):
        return self.structure[idx]

    def set_substructure(self, substructure, idx: int) -> None:
        self.structure[idx] = substructure

    def add_substructure(self, substructure, idx: int = -1) -> None:
        if idx is None or idx == -1 or idx == len(self.structure):
            try:
                self.structure.extend(substructure)
            except TypeError:
                self.structure.append(substructure)
        else:
            tmp_structure = self.structure[:idx]
            try:
                tmp_structure.extend(substructure)
            except TypeError:
                tmp_structure.append(substructure)
            tmp_structure.extend(self.structure[idx:])
            self.structure = tmp_structure

    def del_substructure(self, substructure):
        #TODO обработать исключения и работу с массивами
        self.structure.remove(substructure)   
    

    def apply_operator(self, name: str, *args, **kwargs):
        """
        Apply an operator with the given name.

        Parameters
        ----------
        name: str
            Name of the operator in genetic_operators dict.

        args
        kwargs

        Returns
        -------
        None
        """
        operators = OperatorMap()
        try:
            operator = operators[name]
        except KeyError:
            raise KeyError("Operator with name '{}' is not implemented in"
                           " object {}".format(name, operators))
        except TypeError:
            raise TypeError("Argument 'operators' cannot be '{}'".format(type(operators)))
        return operator.apply_to(self, *args, **kwargs)


class Individ(ComplexStructure):
    """
    """

    def __init__(self, fitness: float = 0):
        self._fitness = fitness
        self.elitism = False
        self.selected = False
        self.graph = nx.Graph()

    @property
    def fitness(self):
        return self._fitness
    
    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness

    def __eq__(self, __value: object) -> bool:
        pass

    @abstractmethod
    def _count_fitness_(self):
        pass

    @abstractmethod
    def mutation(self):
        pass

    @abstractmethod
    def cross_over(self, _other):
        pass


class Population(ComplexStructure):
    """
    """

    def __init__(self, structure):
        super().__init__(structure=structure)
        self.individs = []

    def evolutionary(self):
        raise NotImplementedError("Define evolution by 'Population.evolutionary_step()'")