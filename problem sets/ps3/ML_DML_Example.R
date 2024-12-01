library(keras3)
library(tensorflow)
#library(xtable)
library(dplyr)
#library(sandwich)
library(magrittr)

file <- "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/gun_clean.csv"
data <- read.csv(file)
dim(data)

# Note: These data are population weighted. Specifically,
# looking at the JBES replication files, they seem to be multiplied
# by sqrt((1/T sum_t population_{j,t})/100000). To get the
# unweighted variables need to divide by this number - which we can
# get from the time effects. We are mostly just going to use the weighted
# variables as inputs - except for time and county. We'll take
# cross-sectional and time series means of these weighted variables
# as well. Note that there is nothing wrong with this, but it does not
# reproduce a weighted regression in a setting where covariates may
# enter nonlinearly and flexibly.

## County FE
county_vars <- select(data, starts_with("X_J"))

## Time variables and population weights
# Pull out time variables
time_vars <- select(data, starts_with("X_T"))

# Use these to construct population weights
pop_weights <- rowSums(time_vars)

# Unweighted time variables
time_vars <- time_vars / pop_weights

# For any columns with only zero (like the first one), just drop
time_vars <- time_vars[, colSums(time_vars != 0) > 0]

# Create time index
time_ind <- rowSums(time_vars * (seq(1:20)))


###### Create new data frame with variables we'll use

# Function to find variable names
var_list <- function(df = NULL, type = c("numeric", "factor", "character"), pattern = "", exclude = NULL) {
  vars <- character(0)
  if (any(type %in% "numeric")) {
    vars <- c(vars, names(df)[sapply(df, is.numeric)])
  }
  if (any(type %in% "factor")) {
    vars <- c(vars, names(df)[sapply(df, is.factor)])
  }
  if (any(type %in% "character")) {
    vars <- c(vars, names(df)[sapply(df, is.character)])
  }
  vars[(!vars %in% exclude) & grepl(vars, pattern = pattern)]
}

# census control variables
census <- NULL
census_var <- c("^AGE", "^BN", "^BP", "^BZ", "^ED", "^EL", "^HI", "^HS", "^INC", "^LF", "^LN",
                "^PI", "^PO", "^PP", "^PV", "^SPR", "^VS")

for (i in seq_along(census_var)) {
  census <- append(census, var_list(data, pattern = census_var[i]))
}

# other control variables
X1 <- c("logrobr", "logburg", "burg_missing", "robrate_missing")
X2 <- c("newblack", "newfhh", "newmove", "newdens", "newmal")

# "treatment" variable
d <- "logfssl"

# outcome variable
y <- "logghomr"

# new data frame for time index
usedata <- as.data.frame(time_ind)
colnames(usedata) <- "time_ind"
usedata[, "weights"] <- pop_weights

var_list <- c(y, d, X1, X2, census)
for (i in seq_along(var_list)) {
  usedata[, var_list[i]] <- data[, var_list[i]]
}

####################### Construct county specific means,
# time specific means, initial conditions

# Initial conditions
var_list0 <- c(y, X1, X2, census)
for (i in seq_along(var_list0)) {
  usedata[, paste(var_list0[i], "0", sep = "")] <- kronecker(
    usedata[time_ind == 1, var_list0[i]],
    rep(1, 20)
  )
}

# County means
var_list_j <- c(X1, X2, census)
county_vars <- as.matrix(county_vars)
for (i in seq_along(var_list_j)) {
  usedata[, paste(var_list_j[i], "J", sep = "")] <-
    county_vars %*% qr.solve(county_vars, as.matrix(usedata[, var_list_j[i]]))
}

# Time means
time_vars <- as.matrix(time_vars)
for (i in seq_along(var_list_j)) {
  usedata[, paste(var_list_j[i], "T", sep = "")] <-
    time_vars %*% qr.solve(time_vars, as.matrix(usedata[, var_list_j[i]]))
}