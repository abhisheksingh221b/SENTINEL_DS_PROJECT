"""
make_results.py  -  STEP 1: do all the heavy work once and save the answers as small CSV files.
Run it with:   python make_results.py
The Streamlit app (app.py) then only READS these files, so it opens instantly.
This is the same analysis as the research notebook, written as one script.
"""

import warnings                                           # lets us hide harmless warnings
import numpy as np                                        # maths on numbers
import pandas as pd                                       # tables
from scipy import stats                                   # statistical tests
from sklearn.dummy import DummyRegressor                  # lazy baseline model
from sklearn.linear_model import LinearRegression         # straight-line model
from sklearn.tree import DecisionTreeRegressor            # one decision tree
from sklearn.ensemble import RandomForestRegressor        # many trees voting
from sklearn.cluster import KMeans, AgglomerativeClustering   # two grouping methods
from sklearn.preprocessing import StandardScaler          # puts numbers on one scale
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score, adjusted_rand_score   # scoring tools
from statsmodels.tsa.arima.model import ARIMA             # ARIMA forecasting model

warnings.filterwarnings("ignore")                         # hide harmless warnings

# ---------------------------------------------------------------- settings
DATA_FILE = "data/acled_weekly_aggregated.csv"            # the ACLED file (already in the data folder)
COUNTRIES = ["Ukraine", "Palestine", "Sudan", "Myanmar", "Syria", "Democratic Republic of Congo", "Yemen"]   # our 7 conflicts
TYPE_COLS = {"Battles": "battles", "Explosions/Remote violence": "explosions", "Violence against civilians": "civilians",
             "Protests": "protests", "Riots": "riots", "Strategic developments": "strategic"}   # long names -> short names
COUNT_COLS = ["battles", "explosions", "civilians", "protests", "riots", "strategic"]           # the 6 short names

# ---------------------------------------------------------------- 1. load and clean
print("1. Loading and cleaning the data")                 # progress message
weekly = pd.read_csv(DATA_FILE)                           # read the file
weekly = weekly[weekly["country"].isin(COUNTRIES)]        # keep our 7 countries
weekly["week"] = pd.to_datetime(weekly["week"])           # text dates -> real dates
exposure_text = weekly["population_exposure"].astype(str) # treat exposure as text first
exposure_text = exposure_text.str.replace(",", "")        # remove commas
weekly["population_exposure"] = exposure_text.astype(float)   # back to numbers
weekly["month"] = weekly["week"].dt.to_period("M").dt.to_timestamp()   # which month each week belongs to
weekly = weekly[weekly["week"] >= "2020-01-01"]           # drop one stray week from 2019
newest_month = weekly["month"].max()                      # the newest month in the file
weekly = weekly[weekly["month"] < newest_month]           # drop it: it is unfinished
last_month = weekly["month"].max()                        # the last complete month

# ---------------------------------------------------------------- 2. monthly table
print("2. Building the monthly table")                    # progress message
weekly["type_col"] = weekly["event_type"].map(TYPE_COLS)  # short event-type names
counts = weekly.pivot_table(index=["country", "admin1", "month"], columns="type_col", values="events",
                            aggfunc="sum", fill_value=0)  # events per region per month per type
for col in COUNT_COLS:                                    # make sure all 6 columns exist
    if col not in counts.columns:
        counts[col] = 0
counts = counts[COUNT_COLS]                               # fixed column order
group = weekly.groupby(["country", "admin1", "month"])    # group by region and month
extra = pd.DataFrame()                                    # empty table to fill
extra["fatalities"] = group["fatalities"].sum()           # deaths
extra["exposure"] = group["population_exposure"].sum()    # people exposed
extra["n_subtypes"] = group["sub_event_type"].nunique()   # variety of events
monthly = counts.join(extra).reset_index()                # glue together

all_months = pd.date_range(weekly["month"].min(), weekly["month"].max(), freq="MS")   # every month
pairs = monthly[["country", "admin1"]].drop_duplicates()  # list of regions
grid = pairs.merge(pd.DataFrame({"month": all_months}), how="cross")   # every region x every month
monthly = grid.merge(monthly, on=["country", "admin1", "month"], how="left")   # add real numbers
monthly = monthly.fillna(0)                               # months with no data -> 0
monthly["total_events"] = monthly[COUNT_COLS].sum(axis=1) # all events
monthly["civilian_share"] = 0.0                           # start at 0
monthly["unrest_share"] = 0.0                             # start at 0
has_events = monthly["total_events"] > 0                  # months with events
monthly.loc[has_events, "civilian_share"] = monthly["civilians"] / monthly["total_events"]   # civilian share
monthly.loc[has_events, "unrest_share"] = (monthly["protests"] + monthly["riots"]) / monthly["total_events"]   # protest/riot share
monthly = monthly.sort_values(["country", "admin1", "month"]).reset_index(drop=True)   # sort

monthly_small = monthly[["country", "admin1", "month", "total_events", "fatalities"] + COUNT_COLS]   # columns the app needs for charts
monthly_small.to_csv("data/monthly_region.csv", index=False)   # save for the app

# ---------------------------------------------------------------- 3. severity models
print("3. Severity: comparing models")                    # progress message
SPLIT_DATE = pd.Timestamp("2025-06-01")                   # train before, test after
FEATURES = COUNT_COLS + ["civilian_share", "n_subtypes", "log_exposure"]   # model inputs
active = monthly[monthly["total_events"] > 0].copy()      # months with events
active["log_exposure"] = np.log1p(active["exposure"])     # log of exposure
active["target"] = np.log1p(active["fatalities"])         # log(deaths + 1)
train = active[active["month"] <= SPLIT_DATE]             # training months
test = active[active["month"] > SPLIT_DATE]               # test months

models = {"Baseline (always the average)": DummyRegressor(),
          "Linear Regression": LinearRegression(),
          "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
          "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42)}
names_list = []                                           # model names
r2_list = []                                              # R2 scores
error_list = []                                           # average errors
for name in models:                                       # each model in turn
    model = models[name]                                  # pick it
    model.fit(train[FEATURES], train["target"])           # learn
    guess = model.predict(test[FEATURES])                 # estimate test months
    names_list.append(name)
    r2_list.append(round(r2_score(test["target"], guess), 3))
    error_list.append(round(mean_absolute_error(test["target"], guess), 3))
severity_table = pd.DataFrame({"model": names_list, "R2": r2_list, "average_error": error_list})
severity_table.to_csv("data/severity_models.csv", index=False)   # save

best_name = severity_table.set_index("model")["R2"].idxmax()     # best model
final_model = models[best_name]                           # use it
final_model.fit(active[FEATURES], active["target"])       # retrain on all months
reference = np.sort(final_model.predict(active[FEATURES]))   # sorted estimates for the 0-100 score
importance = pd.DataFrame({"feature": FEATURES, "importance": models["Random Forest"].feature_importances_})
importance.to_csv("data/feature_importance.csv", index=False)    # save
print("   best model:", best_name)

# ---------------------------------------------------------------- 4. risk groups
print("4. Risk groups: comparing K-means and hierarchical")   # progress message
start_of_window = last_month - pd.DateOffset(months=11)   # first of the last 12 months
recent = monthly[monthly["month"] >= start_of_window]     # last 12 months
sum_columns = COUNT_COLS + ["fatalities", "total_events"] # columns to add up
sums = recent.groupby(["country", "admin1"])[sum_columns].sum()   # per region
sums = sums[sums["total_events"] > 0]                     # drop silent regions
feat = pd.DataFrame(index=sums.index)                     # one row per region
feat["avg_events"] = sums["total_events"] / 12
feat["deaths_per_event"] = sums["fatalities"] / sums["total_events"]
feat["civilian_share"] = sums["civilians"] / sums["total_events"]
feat["unrest_share"] = (sums["protests"] + sums["riots"]) / sums["total_events"]
feat["avg_deaths"] = sums["fatalities"] / 12
X = feat[["avg_events", "deaths_per_event", "civilian_share", "unrest_share"]].copy()
X["avg_events"] = np.log1p(X["avg_events"])
X["deaths_per_event"] = np.log1p(X["deaths_per_event"])
X_scaled = StandardScaler().fit_transform(X)              # same scale for all

kmeans_labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)   # method 1
hier_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_scaled)   # method 2
cluster_compare = pd.DataFrame({
    "method": ["K-means", "Hierarchical (Ward)"],
    "silhouette": [round(silhouette_score(X_scaled, kmeans_labels), 3), round(silhouette_score(X_scaled, hier_labels), 3)]})
cluster_compare["agreement_between_methods"] = round(adjusted_rand_score(kmeans_labels, hier_labels), 3)
cluster_compare.to_csv("data/cluster_compare.csv", index=False)   # save

feat["cluster"] = kmeans_labels                           # use K-means groups
avg_deaths_by_cluster = feat.groupby("cluster")["avg_deaths"].mean()   # deaths per group
order = list(avg_deaths_by_cluster.sort_values().index)   # fewest to most
names = {}                                                # group number -> name
names[order[0]] = "Low activity"
names[order[1]] = "Moderate"
names[order[2]] = "High intensity"
feat["risk_group"] = feat["cluster"].map(names)           # name for each region
profile_columns = ["avg_events", "deaths_per_event", "civilian_share", "unrest_share"]
profile = feat.groupby("risk_group")[profile_columns].mean().round(2)   # group profiles
profile["regions"] = feat["risk_group"].value_counts()    # group sizes
profile.reset_index().to_csv("data/cluster_profile.csv", index=False)   # save

# ---------------------------------------------------------------- 5. forecast methods
print("5. Forecast: comparing SMA, EMA and ARIMA")        # progress message

def forecast_sma(values):                                 # simple moving average
    return float(np.mean(values[-3:]))                    # average of the last 3 months

def forecast_ema(values):                                 # exponential moving average
    alpha = 0.5                                           # weight of the newest month
    level = values[0]                                     # start at the first month
    for value in values[1:]:                              # go through the rest
        level = alpha * value + (1 - alpha) * level       # mix new with old
    return float(level)

def forecast_arima(values):                               # ARIMA
    try:
        model = ARIMA(values, order=(1, 1, 1))            # simple ARIMA setting
        fitted = model.fit()                              # learn
        result = float(fitted.forecast(1)[0])             # forecast 1 month
        if result < 0:                                    # no negative events
            result = 0.0
        return result
    except Exception:                                     # if ARIMA fails
        return forecast_sma(values)                       # use the moving average

METHODS = {"SMA": forecast_sma, "EMA": forecast_ema, "ARIMA": forecast_arima}

country_month = monthly.groupby(["country", "month"])[["total_events", "fatalities"]].sum().reset_index()   # per country per month

def get_series(country):                                  # monthly event counts of one country
    rows = country_month[country_month["country"] == country].sort_values("month")
    return rows["total_events"].values

error_total = {"SMA": 0.0, "EMA": 0.0, "ARIMA": 0.0}      # total mistakes
actual_total = {"SMA": 0.0, "EMA": 0.0, "ARIMA": 0.0}     # total real events
for country in COUNTRIES:                                 # each country
    series = get_series(country)
    for method_name in METHODS:                           # each method
        for i in range(6, 0, -1):                         # last 6 months one by one
            guess = METHODS[method_name](series[:-i])     # forecast using only earlier months
            truth = series[-i]                            # what really happened
            error_total[method_name] = error_total[method_name] + abs(guess - truth)
            actual_total[method_name] = actual_total[method_name] + truth
method_list = []                                          # method names
error_percent = []                                        # error as % of events
for method_name in METHODS:
    method_list.append(method_name)
    error_percent.append(round(error_total[method_name] / actual_total[method_name] * 100, 1))
backtest = pd.DataFrame({"method": method_list, "error_percent": error_percent})
backtest.to_csv("data/forecast_backtest.csv", index=False)   # save
best_method_name = backtest.loc[backtest["error_percent"].idxmin(), "method"]   # winner
print("   best method:", best_method_name)

def forecast_row(values):                                 # forecast numbers for one series
    guess = METHODS[best_method_name](values)             # forecast with the winner
    spread = np.std(values[-12:])                         # how much it jumps around
    baseline = float(np.mean(values[-9:-3]))              # average of the 6 months before the latest 3
    if baseline >= 3:
        ratio = guess / baseline                          # forecast vs earlier level
    else:
        ratio = 1.0                                       # tiny numbers: stable
    if ratio > 1.10:
        direction = "Rising"
    elif ratio < 0.90:
        direction = "Falling"
    else:
        direction = "Stable"
    low = guess - 1.96 * spread                           # rough lower end
    if low < 0:
        low = 0
    high = guess + 1.96 * spread                          # rough upper end
    return {"last_month": float(values[-1]), "earlier_avg": round(baseline, 1), "forecast": round(guess, 1),
            "low": round(low, 1), "high": round(high, 1), "direction": direction}

def trend_test(values):                                   # is the 12-month trend real?
    if values[-12:].std() > 0:                            # numbers must vary
        rho, p = stats.spearmanr(np.arange(12), values[-12:])
    else:
        rho, p = 0, 1
    if p < 0.05:
        verdict = "Yes"
    else:
        verdict = "No"
    return round(rho, 2), round(p, 3), verdict

# ---------------------------------------------------------------- 6. statistical tests (per country)
print("6. Statistical tests")                             # progress message
test_rows = []                                            # one row per country
for country in COUNTRIES:
    series = get_series(country)
    stat, shapiro_p = stats.shapiro(series[-24:])         # Shapiro-Wilk on the last 24 months
    rho, trend_p, trend_real = trend_test(series)         # Spearman trend
    before = series[-12:-6]                               # the 6 months before
    recent6 = series[-6:]                                 # the latest 6 months
    stat, mw_p = stats.mannwhitneyu(recent6, before, alternative="two-sided")   # Mann-Whitney
    test_rows.append({"country": country, "shapiro_p": round(shapiro_p, 3), "trend_rho": rho, "trend_p": trend_p,
                      "trend_is_real": trend_real, "avg_before": round(before.mean()), "avg_recent": round(recent6.mean()),
                      "level_change_p": round(mw_p, 3)})
stat_tests = pd.DataFrame(test_rows)
stat_tests.to_csv("data/stat_tests.csv", index=False)     # save

country_month["deaths_per_event"] = 0.0                   # deaths per event
busy = country_month["total_events"] > 0
country_month.loc[busy, "deaths_per_event"] = country_month["fatalities"] / country_month["total_events"]
lists = []
for country in COUNTRIES:
    lists.append(country_month[country_month["country"] == country]["deaths_per_event"].values)
h, kruskal_p = stats.kruskal(lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], lists[6])
table = monthly.groupby("country")[COUNT_COLS].sum()
chi2, chi_p, dof, expected = stats.chi2_contingency(table)
overall = pd.DataFrame({"test": ["Kruskal-Wallis: do countries differ in deaths per event?",
                                 "Chi-square: is the event mix linked to the country?"],
                        "p_value": [kruskal_p, chi_p],
                        "answer": ["Yes, they differ" if kruskal_p < 0.05 else "No clear difference",
                                   "Yes, the mix differs" if chi_p < 0.05 else "No clear difference"]})
overall.to_csv("data/overall_tests.csv", index=False)     # save

# ---------------------------------------------------------------- 7. results per region and per country
print("7. Putting it together (this can take a minute)")  # progress message
three_back = last_month - pd.DateOffset(months=2)         # first of the last 3 months
last3 = monthly[monthly["month"] >= three_back].copy()    # last 3 months
last3["log_exposure"] = np.log1p(last3["exposure"])
last3["estimate"] = final_model.predict(last3[FEATURES])
last3.loc[last3["total_events"] == 0, "estimate"] = 0     # no events = lowest
avg_estimate = last3.groupby(["country", "admin1"])["estimate"].mean()   # average of 3 months

rows = []
for (country, region), g in monthly.groupby(["country", "admin1"]):   # every region
    values = g.sort_values("month")["total_events"].values
    estimate = avg_estimate[(country, region)]            # its average estimate
    below = np.sum(reference < estimate)                  # how many past values are lower
    severity = round(below / len(reference) * 100)        # 0-100 score
    if values[-12:].sum() == 0:                           # silent for a year
        fc = {"last_month": 0.0, "earlier_avg": 0.0, "forecast": 0.0, "low": 0.0, "high": 0.0, "direction": "Stable"}
    else:
        fc = forecast_row(values)
    rho, trend_p, trend_real = trend_test(values)
    if (country, region) in feat.index:
        risk_group = feat.loc[(country, region), "risk_group"]
    else:
        risk_group = "Inactive"
    row = {"country": country, "admin1": region, "risk_group": risk_group, "severity": severity}
    row.update(fc)                                        # add the forecast numbers
    row["trend_p"] = trend_p
    row["trend_is_real"] = trend_real
    rows.append(row)
results = pd.DataFrame(rows)
results.to_csv("data/region_results.csv", index=False)    # save

country_rows = []
for country in COUNTRIES:
    series = get_series(country)
    row = {"country": country}
    row.update(forecast_row(series))
    rho, trend_p, trend_real = trend_test(series)
    row["trend_p"] = trend_p
    row["trend_is_real"] = trend_real
    country_rows.append(row)
country_results = pd.DataFrame(country_rows)
for_deaths = monthly[monthly["month"] >= start_of_window].groupby("country")[["fatalities", "total_events"]].sum()   # last 12 months
country_results["deaths_per_event"] = country_results["country"].map(for_deaths["fatalities"] / for_deaths["total_events"]).round(2)
high_share = []                                           # % of regions that are high intensity
avg_sev = []                                              # average severity
for country in COUNTRIES:
    region_rows = results[results["country"] == country]
    high_share.append(round((region_rows["risk_group"] == "High intensity").mean() * 100))
    avg_sev.append(round(region_rows["severity"].mean()))
country_results["pct_regions_high_intensity"] = high_share
country_results["avg_region_severity"] = avg_sev
country_results.to_csv("data/country_results.csv", index=False)   # save

summary = pd.DataFrame({"item": ["last_complete_month", "best_severity_model", "best_forecast_method"],
                        "value": [last_month.strftime("%Y-%m-%d"), best_name, best_method_name]})
summary.to_csv("data/summary.csv", index=False)           # save
print("Done. Files saved in the data folder.")
