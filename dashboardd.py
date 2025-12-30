import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from duckduckgo_search import DDGS
import wikipedia
import time
import re

# ==========================================
# 1. ADVANCED PSYCHOMETRIC AI ENGINE
# ==========================================
@st.cache_resource
def load_psychometric_model():
    """
    Trains a model that correlates behavioral traits with fraud probability.
    """
    np.random.seed(101)
    n = 10000
    
    # --- INPUT VARIABLES ---
    debt_ratio = np.random.uniform(0.1, 5.0, n)
    margins = np.random.uniform(-20, 40, n)
    insider_activity = np.random.uniform(-100, 100, n) 
    board_control = np.random.uniform(0, 1, n)
    auditor_tenure = np.random.randint(1, 20, n) 
    narcissism = np.random.randint(1, 10, n) 
    risk_appetite = np.random.randint(1, 10, n)
    transparency = np.random.randint(1, 10, n) 
    market_pressure = np.random.choice([0, 1], size=n) 
    
    # --- FRAUD LOGIC ---
    risk_score = (
        (narcissism * 4.5) + (risk_appetite * 3.0) + (market_pressure * 15.0) + 
        (debt_ratio * 4.0) - (margins * 0.2) - (board_control * 20.0) - 
        (transparency * 8.0) + (auditor_tenure * 0.5)
    )
    risk_score += np.where(insider_activity < -50, 25, 0) # Panic selling penalty
    risk_score += np.random.normal(0, 4, n)
    
    threshold = np.percentile(risk_score, 88)
    is_fraud = (risk_score > threshold).astype(int)
    
    df = pd.DataFrame({
        'debt': debt_ratio, 'margins': margins, 'insider': insider_activity,
        'control': board_control, 'audit_tenure': auditor_tenure,
        'narcissism': narcissism, 'risk_app': risk_appetite, 'transparency': transparency,
        'market': market_pressure, 'is_fraud': is_fraud
    })
    
    model = RandomForestClassifier(n_estimators=300, max_depth=18, random_state=101)
    model.fit(df.drop('is_fraud', axis=1), df['is_fraud'])
    return model

model = load_psychometric_model()

# ==========================================
# 2. LIVE INTELLIGENCE AGENT
# ==========================================
def fetch_intelligence(name, use_live):
    data = {
        "found": False, "bio": "Simulation Mode - No Bio", "url": "#", 
        "headlines": [], "sentiment": 0.0, "red_flags": []
    }
    
    if not use_live:
        return data

    try:
        page = wikipedia.page(name, auto_suggest=False)
        data['bio'] = page.summary[:800]
        data['url'] = page.url
        data['found'] = True
    except:
        pass

    try:
        results = DDGS().text(f"{name} CEO investigation fraud lawsuit scandal", max_results=10)
        data['headlines'] = [r['title'] for r in results]
    except:
        pass

    if data['headlines']:
        pols = [TextBlob(h).sentiment.polarity for h in data['headlines']]
        data['sentiment'] = np.mean(pols)
        
        toxic_words = ['fraud', 'probe', 'subpoena', 'embezzlement', 'misconduct', 'allegation', 'guilty', 'scandal', 'bribe', 'laundering', 'ponzi', 'indicted']
        for h in data['headlines']:
            for w in toxic_words:
                if w in h.lower():
                    # High visibility formatting for red flags
                    formatted_flag = f"⚠️ **{w.upper()}** detected in: *'{h[:50]}...'* "
                    data['red_flags'].append(formatted_flag)
    
    data['red_flags'] = list(set(data['red_flags']))
    return data

# ==========================================
# 3. DASHBOARD UI (HIGH VISIBILITY)
