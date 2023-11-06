import numpy as np
import scipy as sp
import pandas as pd
import dataclasses as dataclass
import nltk

@dataclass
class Term:
    term_index = 0
    document_index = 0
    term_appearance = 0

class VectorSpaceModel:

    def __init__(self) -> None:
        pass