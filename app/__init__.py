

#ML Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
import re
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, Imputer
from sklearn.externals import joblib