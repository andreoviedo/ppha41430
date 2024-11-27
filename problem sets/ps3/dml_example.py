######## FILE FROM THE DML BOOK #########

# Import relevant packages
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.simplefilter('ignore')

np.random.seed(1234)


file = "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/gun_clean.csv"
data = pd.read_csv(file)
data.shape

# County FE
county_vars = data.filter(like='X_J')

# Time variables and population weights
# Pull out time variables
time_vars = data.filter(like='X_T')

# Use these to construct population weights
popweights = time_vars.sum(axis=1)

# Unweighted time variables
time_vars = time_vars.div(popweights, axis=0)

# For any columns with only zeros, drop them
time_vars = time_vars.loc[:, (time_vars != 0).any(axis=0)]

# Create time index
time_ind = (time_vars * np.arange(1, 21)).sum(axis=1)


# Function to find variable names
def varlist(df=None, type=["numeric", "factor", "character"], pattern="", exclude=None):
    vars = []
    if any(t in type for t in ["numeric", "factor", "character"]):
        if "numeric" in type:
            vars += df.select_dtypes(include=["number"]).columns.tolist()
        if "factor" in type:
            vars += df.select_dtypes(include=["category"]).columns.tolist()
        if "character" in type:
            vars += df.select_dtypes(include=["object"]).columns.tolist()

    if exclude:
        vars = [var for var in vars if var not in exclude]

    if pattern:
        vars = [var for var in vars if re.search(pattern, var)]

    return vars


# Census control variables
census = []
census_var = ["^AGE", "^BN", "^BP", "^BZ", "^ED", "^EL", "^HI", "^HS",
              "^INC", "^LF", "^LN", "^PI", "^PO", "^PP", "^PV", "^SPR", "^VS"]

for pattern in census_var:
    census += varlist(df=data, pattern=pattern)

# Other control variables
X1 = ["logrobr", "logburg", "burg_missing", "robrate_missing"]
X2 = ["newblack", "newfhh", "newmove", "newdens", "newmal"]

# "Treatment" variable
d = "logfssl"

# Outcome variable
y = "logghomr"

# New DataFrame for time index
usedata = pd.DataFrame({'time.ind': time_ind})
usedata['weights'] = popweights

varlist_all = [y, d] + X1 + X2 + census
for var in varlist_all:
    usedata[var] = data[var]

# Construct initial conditions
varlist0 = [y] + X1 + X2 + census
for var in varlist0:
    usedata[f'{var}0'] = np.kron(usedata.loc[usedata['time.ind'] == 1, var], np.ones(20))

# County means
county_vars = pd.DataFrame(county_vars)
for var in varlist_all:
    usedata[f'{var}J'] = county_vars.dot(np.linalg.pinv(county_vars).dot(usedata[var]))

# Time means
time_vars = pd.DataFrame(time_vars)
for var in varlist_all:
    usedata[f'{var}T'] = time_vars.dot(np.linalg.pinv(time_vars).dot(usedata[var]))