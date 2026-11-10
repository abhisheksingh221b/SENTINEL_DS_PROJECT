# SENTINEL app

Files
- `app.py`                     the Streamlit web app (reads the CSV files in `data/`)
- `make_results.py`            does all the analysis once and writes the CSV files in `data/`
- `data/`                      the ACLED file and all the result files
- `requirements.txt`           what Streamlit Cloud installs to run the app
- `requirements_make_results.txt`  extra packages only needed to re-run `make_results.py`

## Run on your computer
    pip install -r requirements.txt
    streamlit run app.py

(Only if you want to recompute the results: `pip install -r requirements_make_results.txt` then `python make_results.py`.)

## Deploy on Streamlit Community Cloud
1. On GitHub, create a repository and upload EVERYTHING in this folder: `app.py`, `requirements.txt`, and the whole `data` folder
   (drag the folder itself into GitHub, not just the files inside it).
2. Go to share.streamlit.io, sign in with GitHub, click "Create app", choose the repository, the main file `app.py`.
3. Before or after deploying: open the app's Settings -> Secrets and paste this line (with your own key between the quotes):
       ANTHROPIC_API_KEY = "paste-your-key-here"
   Never write the key in any file. Without a key the app still works; only the LLM buttons are disabled.
4. Click Deploy.

## Five Streamlit ideas used in app.py
1. The script re-runs from the top on every click.
2. `@st.cache_data` stores results so files are not re-read on every click.
3. `st.secrets` holds the API key (set on the website).
4. `st.tabs` and `st.columns` control the layout.
5. `st.sidebar` holds the pickers; widgets like `st.selectbox` return what the user chose.
