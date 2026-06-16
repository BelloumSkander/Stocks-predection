# Stocks Market Analyses

This repo contains a Streamlit dashboard for financial market analysis, technical indicators, forecasting, sentiment analysis, and portfolio comparison built with Yahoo Finance data.

## Deployment

This project is ready for deployment to Streamlit Community Cloud.

### Recommended deployment settings
- Repository root file: `streamlit_app.py`
- Requirements file: `requirements.txt`
- Working directory: repository root

### Deploy steps
1. Push this repo to GitHub.
2. Open Streamlit Community Cloud and add a new app.
3. Select the repo and branch.
4. Set the main file to `streamlit_app.py` if needed.
5. Deploy.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

- `streamlit_app.py` – Streamlit Cloud entry point.
- `stocks_predection.py` – Main app logic and layout.
- `styles.html` – Custom CSS for the app.
- `.streamlit/config.toml` – Streamlit server configuration.
- `.streamlit/theme.toml` – Theme settings.
- `requirements.txt` – Python dependencies.

## Notes

- The app uses `nltk` and downloads the `vader_lexicon` and `punkt` corpora at startup.
- `TA-Lib` may require a prebuilt wheel on Streamlit Cloud. If installation fails, try installing a compatible binary wheel or use a local development environment.
- The app relies on Yahoo Finance and public web scraping; data endpoints may change over time.
