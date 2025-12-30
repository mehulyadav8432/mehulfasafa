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
import random
from collections import Counter

# ==========================================
# 1. PSYCHOMETRIC & RISK ENGINE
# ==========================================
@st.cache_resource
def load_titan_model():
    """
    Trains the advanced risk model using the 'Fraud Diamond' theory.
    """
    np.random.seed(42)
    n = 10000
    
    # --- INPUT VARIABLES ---
    debt_ratio = np.random.uniform(0.1, 8.0, n)
    insider_sell = np.random.uniform(0, 100, n)
    margins = np.random.uniform(-30, 50, n)
    
    narcissism = np.random.randint(1, 10, n)
    machiavellianism = np.random.randint(1, 10, n) # Manipulative behavior
    psychopathy = np.random.randint(1, 10, n) # Lack of empathy
    
    control = np.random.uniform(0, 1, n)
    culture = np.random.uniform(1, 5, n)
    
    # --- LOGIC: The Dark Triad Correlator ---
    # High Dark Triad + High Pressure (Debt) + Opportunity (Control) = FRAUD
    dark_triad_score = (narcissism + machiavellianism + psychopathy) / 3
    
    risk_score = (
        (dark_triad_score * 5.0) + 
        (debt_ratio * 3.0) + 
        (insider_sell * 0.1) - 
        (margins * 0.2) - 
        (control * 15.0) - 
        (culture * 5.0)
    )
    
    risk_score += np.random.normal(0, 5, n)
    threshold = np.percentile(risk_score, 90)
    is_fraud = (risk_score > threshold).astype(int)
    
    df = pd.DataFrame({
        'debt': debt_ratio, 'insider': insider_sell, 'margins': margins,
        'narcissism': narcissism, 'mach': machiavellianism, 'psych': psychopathy,
        'control': control, 'culture': culture, 'is_fraud': is_fraud
    })
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(df.drop('is_fraud', axis=1), df['is_fraud'])
    return model

model = load_titan_model()

# ==========================================
# 2. INTELLIGENCE GATHERING (FAIL-SAFE)
# ==========================================
def fetch_live_intelligence(name, use_live):
    """
    Tries to fetch real data. If it fails, generates realistic MOCK data
    so the dashboard always works.
    """
    data = {
        "status": "OFFLINE", "bio": "Bio unavailable.", "url": "#",
        "sentiment_score": 0.0, "headlines": [], "keywords": {},
        "sentiment_history": []
    }
    
    if use_live:
        try:
            # 1. Wiki Bio
            page = wikipedia.page(name, auto_suggest=False)
            data['bio'] = page.summary[:600] + "..."
            data['url'] = page.url
            
            # 2. News Search
            results = DDGS().text(f"{name} CEO controversy fraud business news", max_results=8)
            data['headlines'] = [r['title'] for r in results]
            data['status'] = "LIVE"
            
        except Exception as e:
            data['status'] = "ERROR (Using Simulation)"
            data['headlines'] = [
                f"Analyst report raises questions about {name}'s accounting practices.",
                f"{name} denies allegations of misconduct in recent press conference.",
                f"Stock drops 5% amidst rumors of regulatory probe into {name}.",
                f"Board of Directors expresses full confidence in {name}.",
                f"Leaked memos suggest internal culture clash under {name}."
            ]

    # 3. Sentiment & Keywords (Works on both Live and Mock data)
    if data['headlines']:
        # Current Sentiment
        pols = [TextBlob(h).sentiment.polarity for h in data['headlines']]
        data['sentiment_score'] = np.mean(pols) if pols else 0.0
        
        # Keyword Extraction
        all_text = " ".join(data['headlines']).lower()
        words = re.findall(r'\w+', all_text)
        stops = ['the', 'to', 'of', 'in', 'and', 'for', 'with', 'on', 'at', 'from', 'ceo', 'news']
        filtered = [w for w in words if w not in stops and len(w) > 4]
        data['keywords'] = dict(Counter(filtered).most_common(7))
        
        # Simulated History (Trend Line)
        # We generate a trend that ends at the current calculated sentiment
        trend = np.linspace(data['sentiment_score'] - 0.5, data['sentiment_score'], 30) 
        trend += np.random.normal(0, 0.2, 30) # Add noise
        data['sentiment_history'] = trend

    return data

def determine_archetype(narc, mach, psych):
    score = narc + mach + psych
    if score > 24: return "The Dark Triad Leader"
    if narc > 8 and mach < 5: return "The Grandiose Narcissist"
    if mach > 8: return "The Machiavellian Strategist"
    if psych > 7: return "The Corporate Ruthless"
    return "The Balanced Executive"

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.set_page_config(page_title="ExecuWatch Titan", page_icon="🛡️", layout="wide")

# NEON CSS
st.markdown("""
<style>
    .metric-card { background-color: #111; border: 1px solid #333; padding: 20px; border-radius: 10px; }
    .status-live { color: #00FF00; font-weight: bold; border: 1px solid #00FF00; padding: 5px; border-radius: 5px;}
    .status-sim { color: #FFA500; font-weight: bold; border: 1px solid #FFA500; padding: 5px; border-radius: 5px;}
    .big-alert { font-size: 24px; color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Titan Control")
    
    use_live = st.toggle("Live Web Uplink", value=True)
    name = st.text_input("Target Subject", "Elon Musk")
    
    st.subheader("1. The Dark Triad (Psychometrics)")
    narc = st.slider("Narcissism", 1, 10, 8, help="Ego, self-importance")
    mach = st.slider("Machiavellianism", 1, 10, 7, help="Manipulation, strategy")
    psych = st.slider("Psychopathy", 1, 10, 4, help="Lack of empathy, risk-taking")
    
    st.subheader("2. Financial Stressors")
    debt = st.slider("Debt Ratio", 0.0, 10.0, 2.5)
    insider = st.slider("Insider Selling %", 0, 100, 15)
    margins = st.slider("Profit Margins", -50, 50, 10)
    
    st.subheader("3. Governance")
    control = st.slider("Board Control", 0.0, 1.0, 0.3)
    culture = st.slider("Culture Score", 1.0, 5.0, 3.0)
    
    btn = st.button("RUN TITAN ANALYSIS", type="primary", use_container_width=True)

# --- MAIN PANEL ---
if btn:
    # 1. EXECUTE ANALYSIS
    with st.spinner("Initializing Titan Risk Engine..."):
        intel = fetch_live_intelligence(name, use_live)
        
        # AI Prediction
        input_data = pd.DataFrame({
            'debt': [debt], 'insider': [insider], 'margins': [margins],
            'narcissism': [narc], 'mach': [mach], 'psych': [psych],
            'control': [control], 'culture': [culture]
        })
        prob = model.predict_proba(input_data)[0][1] * 100
        archetype = determine_archetype(narc, mach, psych)
        
        time.sleep(1)

    # 2. HEADER
    st.title(f"Risk Dossier: {name}")
    
    stat_color = "status-live" if intel['status'] == "LIVE" else "status-sim"
    st.markdown(f"Status: <span class='{stat_color}'>{intel['status']}</span> | Archetype: **{archetype}**", unsafe_allow_html=True)

    # 3. METRICS ROW
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Probability", f"{prob:.1f}%", delta="CRITICAL" if prob > 70 else "Stable", delta_color="inverse")
    c2.metric("Dark Triad Score", f"{(narc+mach+psych)/3:.1f}/10", "Psychometric Avg")
    c3.metric("Live Sentiment", f"{intel['sentiment_score']:.2f}", "News Polarity")
    c4.metric("Financial Stress", "High" if debt > 4 else "Low", "Debt Load")

    # 4. TABS & GRAPHS
    tab1, tab2, tab3 = st.tabs(["📉 Visual Analytics", "📰 Live Intelligence", "📝 Risk Report"])

    with tab1:
        g1, g2 = st.columns(2)
        
        with g1:
            st.subheader("Psychometric Radar")
            # Radar Chart
            cats = ['Narcissism', 'Machiavellianism', 'Psychopathy', 'Aggression (Margin)', 'Power (Control)']
            vals = [narc, mach, psych, abs(margins/5), (1-control)*10]
            
            fig = go.Figure(go.Scatterpolar(
                r=vals, theta=cats, fill='toself', 
                line_color='#FF4B4B' if prob > 50 else '#00CC96'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), title="The Dark Triad Profile")
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            st.subheader("Sentiment Volatility (30 Days)")
            # Trend Line
            days = list(range(1, 31))
            fig_line = px.line(x=days, y=intel['sentiment_history'], 
                               labels={'x': 'Days Ago', 'y': 'Sentiment'}, 
                               title="Reputation Trend (Simulated History)")
            st.plotly_chart(fig_line, use_container_width=True)
            
        # Keyword Heatmap
        st.subheader("Keyword Risk Heatmap")
        if intel['keywords']:
            kw_df = pd.DataFrame(list(intel['keywords'].items()), columns=['Term', 'Count'])
            fig_bar = px.bar(kw_df, x='Count', y='Term', orientation='h', color='Count', color_continuous_scale='Magma')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No sufficient text data for heatmap.")

    with tab2:
        col_news, col_bio = st.columns([2, 1])
        
        with col_news:
            st.subheader("News Wire Feed")
            if intel['headlines']:
                for i, h in enumerate(intel['headlines']):
                    # Color code sentiment
                    sent = TextBlob(h).sentiment.polarity
                    icon = "🔴" if sent < -0.1 else "🟢" if sent > 0.1 else "⚪"
                    st.markdown(f"{icon} **{h}**")
            else:
                st.warning("Feed Offline.")

        with col_bio:
            st.subheader("Subject Bio")
            st.info(intel['bio'])
            st.markdown(f"[Read Full Profile]({intel['url']})")

    with tab3:
        st.subheader("Automated Risk Audit")
        report = f"""
        **TITAN RISK AUDIT: {name.upper()}**
        
        **1. PSYCHOMETRIC EVALUATION**
        Subject displays characteristics of **{archetype}**.
        - Narcissism: {narc}/10 (High need for validation)
        - Machiavellianism: {mach}/10 (Manipulative tendencies)
        
        **2. FRAUD TRIANGLE ANALYSIS**
        - **Pressure:** Debt levels are at {debt}x, creating significant strain.
        - **Opportunity:** Board control is {(1-control)*100:.0f}% consolidated in the CEO's hands.
        - **Rationalization:** High 'Dark Triad' scores often correlate with the ability to justify misconduct.
        
        **3. CONCLUSION**
        Probability of material financial misstatement is estimated at **{prob:.2f}%**.
        """
        st.text_area("Official Record", report, height=300)

else:
    st.info("👈 Use the Titan Control Panel to begin analysis.")
