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
    Uses the 'Fraud Pentagon' framework: Pressure, Opportunity, Rationalization, Capability, Arrogance.
    """
    np.random.seed(101)
    n = 10000
    
    # --- INPUT VARIABLES (The Toggles) ---
    # Financial
    debt_ratio = np.random.uniform(0.1, 5.0, n)
    margins = np.random.uniform(-20, 40, n)
    insider_activity = np.random.uniform(-100, 100, n) # Neg = Sell, Pos = Buy
    
    # Governance
    board_control = np.random.uniform(0, 1, n)
    auditor_tenure = np.random.randint(1, 20, n) # Long tenure = cozy relationship
    
    # Behavioral (Psychometrics)
    narcissism = np.random.randint(1, 10, n) # Derived from social media + speeches
    risk_appetite = np.random.randint(1, 10, n)
    transparency = np.random.randint(1, 10, n) # 1=Secretive, 10=Open Book
    
    # External
    market_pressure = np.random.choice([0, 1], size=n) # 1 = Bear Market/Recession
    
    # --- FRAUD LOGIC ---
    # The "Dark Triad" Logic: High Narcissism + Low Control + High Pressure = FRAUD
    risk_score = (
        (narcissism * 4.5) + 
        (risk_appetite * 3.0) + 
        (market_pressure * 15.0) + 
        (debt_ratio * 4.0) - 
        (margins * 0.2) - 
        (board_control * 20.0) - 
        (transparency * 8.0) + 
        (auditor_tenure * 0.5)
    )
    
    # Insider Selling Panic Multiplier
    risk_score += np.where(insider_activity < -50, 25, 0)
    
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

    # 1. Wikipedia
    try:
        page = wikipedia.page(name, auto_suggest=False)
        data['bio'] = page.summary[:800]
        data['url'] = page.url
        data['found'] = True
    except:
        pass

    # 2. DuckDuckGo News
    try:
        # Search for high-signal keywords
        results = DDGS().text(f"{name} CEO investigation fraud lawsuit scandal", max_results=10)
        data['headlines'] = [r['title'] for r in results]
    except:
        pass

    # 3. Sentiment & Red Flag Scanner
    if data['headlines']:
        pols = [TextBlob(h).sentiment.polarity for h in data['headlines']]
        data['sentiment'] = np.mean(pols)
        
        # Toxic Keyword Search
        toxic_words = ['fraud', 'probe', 'subpoena', 'embezzlement', 'misconduct', 'allegation', 'guilty', 'scandal', 'bribe', 'laundering', 'ponzi']
        for h in data['headlines']:
            for w in toxic_words:
                if w in h.lower():
                    data['red_flags'].append(f"⚠️ {w.upper()} detected in: '{h[:40]}...'")
    
    data['red_flags'] = list(set(data['red_flags'])) # Remove dupes
    return data

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.set_page_config(page_title="Risk Command", page_icon="☢️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-font { font-size:30px !important; font-weight: bold; }
    .risk-critical { color: #FF4B4B; border: 2px solid #FF4B4B; padding: 10px; border-radius: 5px; }
    .risk-safe { color: #00CC96; border: 2px solid #00CC96; padding: 10px; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 5px 5px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: MISSION CONTROL ---
with st.sidebar:
    st.header("🎛️ Mission Control")
    
    # MASTER SWITCH
    use_live = st.toggle("Activate Live Web Scraper", value=True)
    name = st.text_input("Target Name", "Sam Bankman-Fried")
    
    with st.expander("1. Financial Toggles", expanded=True):
        debt = st.slider("Debt Ratio (D/E)", 0.0, 10.0, 3.5, help="High debt = Pressure to cook books")
        margins = st.slider("Profit Margins (%)", -20, 50, -5, help="Negative margins = Pressure to lie")
        insider = st.slider("Insider Net Buying (%)", -100, 100, -80, help="-100 = Dumping stock, +100 = Buying")

    with st.expander("2. Governance Toggles", expanded=False):
        control = st.slider("Board Independence", 0.0, 1.0, 0.1, help="0 = CEO is Dictator, 1 = Board has power")
        audit_yrs = st.number_input("Auditor Tenure (Yrs)", 1, 50, 15, help=">10 years implies cozy relationship")

    with st.expander("3. Psychometric Toggles", expanded=False):
        narcissism = st.slider("Public Narcissism Score", 1, 10, 9, help="1=Humble, 10=God Complex")
        risk_app = st.slider("Risk Appetite", 1, 10, 10, help="Gambler mentality")
        transparency = st.slider("Transparency", 1, 10, 2, help="Willingness to share data")
        
    with st.expander("4. Macro Environment", expanded=False):
        market = st.radio("Market Condition", ["Bull Market (Easy Money)", "Bear Market (High Pressure)"])
        mkt_val = 1 if market == "Bear Market (High Pressure)" else 0

    btn = st.button("🚀 EXECUTE RISK ANALYSIS", type="primary", use_container_width=True)

# --- MAIN DASHBOARD ---
if btn:
    # 1. PROCESS DATA
    with st.spinner("Processing Psychometric & Financial Vectors..."):
        # Fetch Live
        intel = fetch_intelligence(name, use_live)
        
        # Predict Risk
        input_df = pd.DataFrame({
            'debt': [debt], 'margins': [margins], 'insider': [insider],
            'control': [control], 'audit_tenure': [audit_yrs],
            'narcissism': [narcissism], 'risk_app': [risk_app], 
            'transparency': [transparency], 'market': [mkt_val]
        })
        prob = model.predict_proba(input_df)[0][1] * 100
        
        # Determine Psychometric Archetype
        if narcissism > 7 and risk_app > 8:
            archetype = "The Icarus (High Ego + High Risk)"
        elif control < 0.3 and transparency < 3:
            archetype = "The Shadow Monarch (Secretive + Dictator)"
        elif debt > 5.0 and insider < -50:
            archetype = "The Desperate Debtor"
        else:
            archetype = "The Corporate Steward"
            
        time.sleep(0.5) # UX Pacing

    # 2. HEADER METRICS
    st.title(f"Risk Dossier: {name}")
    st.markdown(f"**Archetype:** `{archetype}` | **Live Status:** {'ONLINE' if intel['found'] else 'OFFLINE/SIMULATED'}")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Fraud Probability", f"{prob:.1f}%", delta="CRITICAL" if prob > 75 else "Stable", delta_color="inverse")
    col_m2.metric("Psychometric Risk", f"{narcissism}/10", "Narcissism Score")
    col_m3.metric("Financial Pressure", "EXTREME" if debt > 4 and margins < 0 else "Low", "Debt + Margins")
    col_m4.metric("Media Sentiment", f"{intel['sentiment']:.2f}", "Live Polarity")

    # 3. ADVANCED VISUALIZATION TABS
    tab1, tab2, tab3, tab4 = st.tabs(["🌪️ Risk Sunburst", "🧠 Psychometric Radar", "📰 Live Intel Feed", "📝 Audit Report"])

    with tab1:
        # SUNBURST CHART (Hierarchical Risk)
        c_sun, c_explain = st.columns([2, 1])
        with c_sun:
            # Structuring data for Sunburst
            labels = ["Total Risk", "Behavioral", "Financial", "Governance", "Ego", "Risk Taking", "Debt", "Margins", "Board Control"]
            parents = ["", "Total Risk", "Total Risk", "Total Risk", "Behavioral", "Behavioral", "Financial", "Financial", "Governance"]
            values = [prob, (narcissism+risk_app)*5, (debt+abs(margins/10))*10, (1-control)*20, narcissism*10, risk_app*10, debt*10, abs(margins), (1-control)*20]
            
            fig_sun = go.Figure(go.Sunburst(
                labels=labels, parents=parents, values=values,
                branchvalues="total",
                marker=dict(colors=values, colorscale='RdBu_r')
            ))
            fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=400)
            st.plotly_chart(fig_sun, use_container_width=True)
        
        with c_explain:
            st.info("**How to read:** The inner circle is the Total Risk. The outer rings show which specific category contributes most to that risk. Red areas are the most dangerous.")

    with tab2:
        # RADAR CHART (Psychometric)
        categories = ['Narcissism', 'Transparency (Inv)', 'Risk Appetite', 'Market Pressure', 'Board Power (Inv)']
        r_vals = [narcissism, 10-transparency, risk_app, mkt_val*10, (1-control)*10]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals, theta=categories, fill='toself', name='Subject Profile',
            line_color='#FF0000' if prob > 50 else '#00CC96'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), title="The Fraud Pentagon")
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        # LIVE FEED WITH HIGHLIGHTS
        col_news, col_kw = st.columns([2, 1])
        with col_news:
            if intel['red_flags']:
                st.error(f"🚨 **{len(intel['red_flags'])} CRITICAL RED FLAGS DETECTED**")
                for flag in intel['red_flags']:
                    st.write(flag)
                st.divider()
            
            if intel['headlines']:
                st.subheader("Latest Wire News")
                for h in intel['headlines']:
                    st.markdown(f"> {h}")
            else:
                st.warning("No live news found. System using manual inputs only.")
                
        with col_kw:
            st.subheader("Bio Summary")
            st.caption(intel['bio'])
            st.markdown(f"[Source]({intel['url']})")

    with tab4:
        # AUTO-GENERATED NARRATIVE
        st.subheader("Official Risk Determination")
        risk_text = f"""
        **AUDIT REPORT: {name.upper()}**
        
        **1. EXECUTIVE SUMMARY**
        The system has flagged this individual as **{archetype.upper()}**. 
        The calculated fraud probability is **{prob:.2f}%**.
        
        **2. KEY RISK DRIVERS**
        - **Psychometric:** A Narcissism score of {narcissism}/10 combined with {risk_app}/10 Risk Appetite suggests a leader who may rationalize unethical behavior to maintain their self-image.
        - **Financial:** With a Debt Ratio of {debt} and Profit Margins of {margins}%, there is significant 'Pressure' (Step 1 of Fraud Triangle) to manipulate earnings.
        - **Governance:** Board independence is at {control*100:.0f}%. {'This low level of oversight creates the Opportunity for unchecked misconduct.' if control < 0.3 else 'This suggests reasonable oversight.'}
        
        **3. LIVE SIGNALS**
        Recent media sentiment is {intel['sentiment']:.2f}. { 'Toxic keywords detected in news stream.' if intel['red_flags'] else 'No immediate red flags in recent headlines.'}
        """
        st.text_area("Copy Report", risk_text, height=350)

else:
    st.info("👈 Open the Sidebar to configure your 'Mission Control' toggles before launching.")
