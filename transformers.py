from typing import Optional, Tuple
import pandas
import numpy
from sklearn.base import BaseEstimator, TransformerMixin


class ValuesToFloat(BaseEstimator, TransformerMixin):
    def fit(self, X: pandas.DataFrame, y: Optional[pandas.Series]=None):
        return self
    
    def transform(self, X: pandas.DataFrame, y: Optional[pandas.Series]=None) -> Tuple[pandas.DataFrame, Optional[pandas.Series]]:
        return X.astype(numpy.float64)