from abc import ABC, abstractmethod


class Detector(ABC):
    name: str
    fmin: float
    fmax: float
    armlength: float
    res: float

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def satellite_positions(self):
        pass

    @abstractmethod
    def detector_arms(self):
        pass
