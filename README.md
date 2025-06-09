
MarketTrackr
===============

MarketTrackr is your personal, AI-powered stock insight tool designed to help traders and investors visualize portfolio performance and get meaningful statistics that guide next-day decisions. Whether you're a beginner or a seasoned trader, MarketTrackr provides an edge by analyzing key metrics and making data intuitive.

Why It’s Cool
----------------
- Insight-Driven Analysis: Not just raw data—MarketTrackr provides meaningful statistics for daily performance reviews.
- AI-Enhanced Logic: Lays the groundwork for integrating future predictive models to make smarter trades.
- Portfolio Focused: Easily upload your own portfolio (portfolio.json) and get insights tailored to your assets.
- Beginner Friendly, Pro Capable: Designed for clarity and impact without sacrificing flexibility.

What Sets It Apart
---------------------
Unlike basic stock trackers, MarketTrackr focuses on daily strategy insights and actionable summaries, allowing you to analyze outcomes and make data-informed decisions before the next market day. It’s your post-trade analysis assistant.

Tech Stack
-------------
- Python — Core logic and backend
- JSON — Used for portfolio data
- Matplotlib / Plotly — For data visualization (if applicable in future versions)
- FastAPI / Flask (future integration planned) — For turning this into a web app/API

# How to Run MarketTrackr Locally
----------------------------------

> Follow these steps to set up and run the project on your computer.

## Prerequisites:
- Python 3.8 or higher
- pip package manager
- Git (optional)
- A code editor (e.g., VS Code)

## Step-by-Step:
### 1. Clone the Repo or Download ZIP
   git clone https://github.com/your-username/MarketTrackr.git
   cd MarketTrackr

### 2. Create & Activate Virtual Environment 
  ``` bash
   python -m venv venv 
   source venv/bin/activate   (On Windows: venv\Scripts\activate)
```
### 3. Install Dependencies
``` bash
   pip install -r requirements.txt
```
### 5. Run the Application
```bash
   streamlit run app.py

   #if port 5000 is in use, run with a custom port:
   streamlit run app.py --server.port 8502
```

### 6. View Output
   The app will analyze your portfolio and output meaningful insights in the terminal.

Future Plans
---------------
- Add options trading analysis
- Integrate Alpaca/Yahoo Finance for real-time data
- Web dashboard via FastAPI + React
- Export reports as PDFs
