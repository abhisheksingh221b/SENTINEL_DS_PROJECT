"""
app.py  -  the SENTINEL web app (Streamlit)
Run it with:   streamlit run app.py

IMPORTANT: this app does not train anything. It only READS the small CSV files that
make_results.py created in the data folder. That is why it opens instantly.
The research itself (models, tests, comparisons) is in the notebook.
"""

import os                                                 # os: read environment settings (the API key)
import json                                               # json: read and write JSON text
import pandas as pd                                       # pandas: tables
import plotly.graph_objects as go                         # plotly: charts
import streamlit as st                                    # streamlit: turns this script into a web page

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="SENTINEL", layout="wide")  # browser-tab title and wide layout

# ---------------------------------------------------------------- load the data once
# Streamlit re-runs this whole file from the top on EVERY click.
# @st.cache_data stores the result so the files are read only once.
@st.cache_data
def load_file(name):                                      # read one CSV file from the data folder
    return pd.read_csv("data/" + name)                    # return it as a table

region_results = load_file("region_results.csv")          # one row per region
country_results = load_file("country_results.csv")        # one row per country
monthly = load_file("monthly_region.csv")                 # monthly counts per region
monthly["month"] = pd.to_datetime(monthly["month"])       # text dates -> real dates
summary = load_file("summary.csv").set_index("item")["value"]   # a few facts such as the last month in the data
last_month = pd.to_datetime(summary["last_complete_month"])     # the last month with complete data
next_month = last_month + pd.DateOffset(months=1)         # the month we forecast

# ---------------------------------------------------------------- the LLM helper functions
MODEL = "claude-haiku-4-5-20251001"                       # a fast, cheap Claude model

def get_api_key():                                        # find the API key (never written in the code)
    key = os.environ.get("ANTHROPIC_API_KEY")             # 1st place: environment setting
    if key:                                               # if found...
        return key                                        # ...use it
    try:                                                  # 2nd place: Streamlit Secrets (set on the website)
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:                                     # no secret set
        return None                                       # no key

def ask_claude(system_prompt, user_text, max_tokens):     # send one message to Claude, return its reply text
    import anthropic                                      # the official Claude library
    client = anthropic.Anthropic(api_key=get_api_key())   # connect using the key
    reply = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system_prompt,
                                   messages=[{"role": "user", "content": user_text}])   # send the message
    return reply.content[0].text                          # Claude's answer as text

READ_PROMPT = '''You read a short news report about armed conflict and return ONLY a JSON object with these fields:
"event_type": one of ["Battles", "Explosions/Remote violence", "Violence against civilians", "Protests", "Riots", "Strategic developments"],
"location": place named or "unknown", "deaths_mentioned": a number or null, "tone": "calm", "tense" or "violent",
"summary": one plain-English sentence, "confidence": a number from 0 to 1.
Use only what the text says. Do not guess.'''           # instructions for reading a report

EXPLAIN_PROMPT = '''You explain conflict-monitoring results to a non-technical NGO coordinator.
Write 3 to 4 short sentences in plain English.
Use ONLY the numbers given. Do not add facts, causes or names that are not in the data.
Say clearly if the trend might just be random noise.
End with one cautious sentence: this is decision support and local knowledge matters.'''     # instructions for explaining a result

def read_report(text):                                    # news text in -> dictionary of details out
    raw = ask_claude(READ_PROMPT, text, 400)              # ask Claude
    start = raw.find("{")                                 # where the JSON starts
    end = raw.rfind("}") + 1                              # where the JSON ends
    return json.loads(raw[start:end])                     # cut out the JSON and read it

def explain_row(row):                                     # one results row in -> plain-English note out
    facts = json.dumps(row.to_dict(), default=str)        # the numbers as text
    return ask_claude(EXPLAIN_PROMPT, "Region data: " + facts, 300)   # ask Claude

# ---------------------------------------------------------------- sidebar: pick a conflict and a region
st.sidebar.title("SENTINEL")                              # sidebar title
st.sidebar.caption("Signal Extraction & Explainable Intelligence for Conflict Monitoring")   # sidebar subtitle
country_list = sorted(country_results["country"])         # all 7 countries, A to Z
country = st.sidebar.selectbox("1. Pick a conflict", country_list)   # dropdown 1

in_country = region_results[region_results["country"] == country]    # regions of that country
in_country = in_country.sort_values("severity", ascending=False)     # most severe first
region_options = ["(whole country)"] + list(in_country["admin1"])    # dropdown choices
region = st.sidebar.selectbox("2. Pick a region (optional)", region_options)   # dropdown 2

st.sidebar.info("Fixed snapshot: ACLED data up to " + last_month.strftime("%B %Y") +
                ". Forecast is for " + next_month.strftime("%B %Y") + ".")   # a note about the data
st.sidebar.caption("Decision support only. It does not replace human judgment.")   # a reminder

# ---------------------------------------------------------------- work out the numbers for the choice
if region == "(whole country)":                           # the user wants the whole country
    row = country_results[country_results["country"] == country].iloc[0]   # the country's row
    history = monthly[monthly["country"] == country].groupby("month")["total_events"].sum().reset_index()   # events per month
    title = country
else:                                                     # the user picked one region
    row = in_country[in_country["admin1"] == region].iloc[0]               # the region's row
    history = monthly[(monthly["country"] == country) & (monthly["admin1"] == region)]   # its monthly rows
    title = region + ", " + country

# ---------------------------------------------------------------- tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Results", "Compare conflicts", "Model comparison",
                                              "Statistical tests", "Paste a report", "How it works"])

# ================================================================ TAB 1: results for the choice
with tab1:
    st.header(title)                                      # big title

    col1, col2, col3, col4 = st.columns(4)                # four boxes side by side
    if region == "(whole country)":                       # whole-country view
        col1.metric("Regions that are high intensity", str(row["pct_regions_high_intensity"]) + "%")
        col2.metric("Average regional severity (0-100)", row["avg_region_severity"])
    else:                                                 # one-region view
        col1.metric("Risk group", row["risk_group"])
        col2.metric("Severity score (0-100)", row["severity"])
    change = row["forecast"] - row["last_month"]          # forecast minus last month
    col3.metric("Forecast for " + next_month.strftime("%b %Y") + " (events)", round(row["forecast"]),
                delta=str(round(change)) + " vs last month", delta_color="off")
    col4.metric("Direction", row["direction"], help="Forecast compared with the average of the 6 months before the latest 3.")

    if row["trend_is_real"] == "Yes":                     # the statistical check
        st.success("Statistical check: the 12-month trend is REAL (p = " + str(row["trend_p"]) + "), not just noise.")
    else:
        st.warning("Statistical check: the 12-month trend is NOT CLEAR (p = " + str(row["trend_p"]) + "). It could be random ups and downs.")

    recent_history = history.sort_values("month").tail(36)     # the last 36 months
    figure = go.Figure()                                       # an empty chart
    figure.add_trace(go.Scatter(x=recent_history["month"], y=recent_history["total_events"],
                                mode="lines+markers", name="Recorded events"))   # the real history
    figure.add_trace(go.Scatter(x=[next_month], y=[row["forecast"]], mode="markers", name="Forecast",
                                marker=dict(size=12, color="crimson")))          # the forecast point
    figure.update_layout(title="Events per month (last 36 months) and next-month forecast", height=380)
    st.plotly_chart(figure, width="stretch")                   # show the chart
    st.caption("Rough 95% range for the forecast: " + str(row["low"]) + " to " + str(row["high"]) + " events.")

    st.subheader("Plain-English explanation")             # a smaller heading
    if st.button("Explain this result"):                  # a button
        if get_api_key():                                 # only if a key exists
            try:                                          # the call may fail (bad key, no internet)
                st.write(explain_row(row))                # show Claude's explanation
            except Exception as error:                    # if it failed...
                st.error("The LLM call failed: " + type(error).__name__ + ": " + str(error)[:150])   # ...say why
        else:
            st.info("No API key found. Add ANTHROPIC_API_KEY in Streamlit Secrets to use this button.")

    if region == "(whole country)":                       # extra table for the country view
        st.subheader("Most severe regions in this country")
        table = in_country[["admin1", "severity", "risk_group", "last_month", "forecast", "direction"]].head(10)
        st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("What kinds of events happened (last 12 months)?")
    type_columns = ["battles", "explosions", "civilians", "protests", "riots", "strategic"]   # the 6 types
    if region == "(whole country)":
        pick = monthly[monthly["country"] == country]     # all rows of the country
    else:
        pick = monthly[(monthly["country"] == country) & (monthly["admin1"] == region)]   # rows of the region
    pick = pick.sort_values("month").groupby("month")[type_columns].sum().tail(12)   # the last 12 months
    st.bar_chart(pick.sum())                              # add up the 12 months and draw bars

# ================================================================ TAB 2: compare the 7 conflicts
with tab2:
    st.header("How do the 7 conflicts compare?")
    table = country_results[["country", "last_month", "forecast", "direction", "trend_is_real",
                             "deaths_per_event", "pct_regions_high_intensity", "avg_region_severity"]]
    table = table.rename(columns={"country": "Conflict", "last_month": "Events last month", "forecast": "Forecast",
                                  "direction": "Direction", "trend_is_real": "Trend real?",
                                  "deaths_per_event": "Deaths per event (12 months)",
                                  "pct_regions_high_intensity": "% regions high intensity",
                                  "avg_region_severity": "Average regional severity"})   # friendlier column names
    st.dataframe(table, hide_index=True, width="stretch")
    bars = go.Figure(go.Bar(x=country_results["country"], y=country_results["deaths_per_event"]))   # bar chart
    bars.update_layout(title="Deaths per event (last 12 months): how deadly is a typical event?", height=350)
    st.plotly_chart(bars, width="stretch")
    st.caption("A conflict with many events is not always the deadliest per event: compare Ukraine and Sudan.")

# ================================================================ TAB 3: model comparison
with tab3:
    st.header("Which model is best? (tested on months the models never saw)")

    st.subheader("1. Severity: four models")
    st.dataframe(load_file("severity_models.csv"), hide_index=True, width="stretch")
    st.caption("R2: 1 is perfect, 0 is no better than guessing the average. Average error is in log units (lower is better).")
    importance = load_file("feature_importance.csv").set_index("feature")   # what drives the estimate
    st.bar_chart(importance)                              # draw it
    st.caption("This chart is the 'explainable' part: it shows which event types push the severity estimate up.")

    st.subheader("2. Risk groups: two methods")
    st.dataframe(load_file("cluster_compare.csv"), hide_index=True, width="stretch")
    st.caption("Silhouette: higher means better separated groups. Below 0.5 means the groups overlap, so read them as rough categories.")
    st.dataframe(load_file("cluster_profile.csv"), hide_index=True, width="stretch")

    st.subheader("3. Forecast: three methods")
    st.dataframe(load_file("forecast_backtest.csv"), hide_index=True, width="stretch")
    st.caption("Error as a % of events over a 6-month backtest (lower is better). The methods are close, so the winner is only slightly better.")

# ================================================================ TAB 4: statistical tests
with tab4:
    st.header("Is it signal or noise?")
    st.write("A trend only counts as real when its p-value is below 0.05.")
    st.dataframe(load_file("stat_tests.csv"), hide_index=True, width="stretch")
    st.caption("shapiro_p: is the data bell-shaped? trend_p: Spearman trend over 12 months. level_change_p: last 6 months vs the 6 before (Mann-Whitney).")
    st.dataframe(load_file("overall_tests.csv"), hide_index=True, width="stretch")

# ================================================================ TAB 5: paste a news report
with tab5:
    st.header("Paste a news report")
    st.write("The LLM reads the text and pulls out structured details. It does NOT change the forecast.")
    if not get_api_key():                                 # no key yet
        st.info("No API key found. Add ANTHROPIC_API_KEY in Streamlit Secrets to use this tab.")

    examples = {"(write my own)": "",
                "Example 1 (made up)": "Armed men attacked a village near the border on Tuesday, killing at least eight civilians, local officials said.",
                "Example 2 (made up)": "Thousands of people marched through the capital on Saturday to demand lower fuel prices. The demonstration ended peacefully."}
    choice = st.selectbox("Try an example", list(examples.keys()))   # dropdown of examples
    text = st.text_area("News text", value=examples[choice], height=160)   # a big text box

    if st.button("Extract details"):                      # a button
        if text.strip() == "":                            # nothing typed
            st.warning("Please paste some text first.")
        elif not get_api_key():
            st.warning("No API key found.")
        else:
            try:
                result = read_report(text)                # ask Claude
                a, b, c = st.columns(3)                   # three boxes
                a.metric("Event type", result.get("event_type", "?"))
                b.metric("Tone", result.get("tone", "?"))
                c.metric("Confidence", result.get("confidence", "?"))
                st.write("Summary: " + str(result.get("summary", "")))
                st.write("Location: " + str(result.get("location", "unknown")))
                st.write("Deaths mentioned: " + str(result.get("deaths_mentioned")))
                if result.get("confidence", 1) < 0.6:     # not sure
                    st.warning("Low confidence: please check this one by hand.")
            except Exception as error:
                st.error("The LLM call failed: " + type(error).__name__ + ": " + str(error)[:150])

# ================================================================ TAB 6: how it works
with tab6:
    st.header("How SENTINEL works")
    st.markdown("""
1. **Data:** ACLED weekly conflict counts per region for 7 conflicts (a fixed snapshot).
2. **Severity (regression):** a model estimates how severe a region-month is from its mix of events. Four models were compared.
3. **Risk groups (clustering):** regions that behave alike are grouped. Two methods were compared.
4. **Forecast (time series):** next month's event count. SMA, EMA and ARIMA were compared with a backtest.
5. **Statistical checks:** tests show whether a trend is real or just noise.
6. **LLM:** reads pasted text and explains results in plain English. It never makes the forecast.
""")
    st.subheader("Limits")
    st.markdown("""
- The data is weekly counts per region, not individual events, so there are no event descriptions or actors.
- It is a fixed snapshot, not live data.
- It shows patterns, not causes, and does not predict specific attacks.
- The LLM can misread text and cannot check whether a report is true.
- Decision support only. People make the decisions.
""")
    st.caption("Best severity model: " + summary["best_severity_model"] + " | Best forecast method: " + summary["best_forecast_method"])
