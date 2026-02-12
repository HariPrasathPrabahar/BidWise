import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BidWise | IIM Amritsar",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for UI
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .title-text {color: #1e3a8a; font-family: 'Helvetica', sans-serif; font-weight: 800; font-size: 3rem;}
    .subtitle-text {color: #64748b; font-size: 1.2rem;}
    .stButton>button {background-color: #2563eb; color: white; border-radius: 8px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_resource
def load_data():
    # Load Network
    try:
        with open('skill_network.pkl', 'rb') as f:
            G = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading Network: {e}")
        G = nx.Graph()

    # Load Role Mappings
    try:
        with open('onet_map.pkl', 'rb') as f:
            roles_map = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading ONET Map: {e}")
        roles_map = {}

    # Load Electives (FIXED FOR EXCEL)
    try:
        # Reads .xlsx file now
        df_electives = pd.read_excel('Elective List.xlsx')
        
        # Filter for Electives only if the column exists
        if 'Category' in df_electives.columns:
            df_biddable = df_electives[df_electives['Category'] == 'Elective'].copy()
        else:
            df_biddable = df_electives.copy()
    except Exception as e:
        st.error(f"Error loading Elective List: {e}")
        df_biddable = pd.DataFrame(columns=['Course Name', 'Department/Area', 'Brief Description for NLP Mapping'])
        
    return G, roles_map, df_biddable

G_final, onet_target_roles, df_electives = load_data()

# --- 2. NLP ENGINE SETUP ---
@st.cache_resource
def setup_nlp_engine(df):
    if df.empty: return None, None
    
    # Fill NA to prevent errors
    df['Weighted_Text'] = ((df['Course Name'].fillna('') + " ") * 3 + 
                           df['Brief Description for NLP Mapping'].fillna('')).str.lower()
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(df['Weighted_Text'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = setup_nlp_engine(df_electives)

# --- 3. HELPER FUNCTIONS ---

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
        return text
    except:
        return ""

def extract_skills_robust(text, network_skills):
    if not text: return []
    text = re.sub(r'\s+', ' ', text).lower()
    found = set()
    
    # 1. Exact Phrase Match
    for skill in network_skills:
        pattern = r'\b' + re.escape(skill.lower()).replace(r'\ ', r'\s+') + r'\b'
        if re.search(pattern, text):
            found.add(skill)
            
    return list(found)

def get_recommendations(needed_skills):
    if vectorizer is None: return pd.DataFrame()
    
    recs = []
    # Domain Mapping for Boosting
    domain_map = {
        'Analytics': ['data', 'analytics', 'sql', 'python', 'mining'],
        'Finance': ['financial', 'valuation', 'accounting', 'investment'],
        'Marketing': ['marketing', 'brand', 'sales', 'consumer'],
        'Strategy': ['strategy', 'consulting', 'innovation']
    }
    
    for skill in needed_skills:
        skill_vec = vectorizer.transform([skill.lower()])
        cosine_sim = cosine_similarity(skill_vec, tfidf_matrix).flatten()
        
        # Domain Boost Logic
        intended_domain = next((d for d, k in domain_map.items() if any(x in skill.lower() for x in k)), None)
        if intended_domain:
            for i, area in enumerate(df_electives['Department/Area']):
                if intended_domain in str(area):
                    cosine_sim[i] *= 1.5
        
        top_idx = cosine_sim.argsort()[-1]
        score = cosine_sim[top_idx]
        
        if score > 0.15: # Threshold
            course = df_electives.iloc[top_idx]
            recs.append({
                "Skill Gap": skill.title(),
                "Recommended Elective": course['Course Name'],
                "Area": course['Department/Area']
            })
    
    if recs:
        return pd.DataFrame(recs).drop_duplicates(subset=['Recommended Elective'])
    return pd.DataFrame()

# --- 4. UI LAYOUT ---

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.write("🎓")
with col2:
    st.markdown('<h1 class="title-text">BidWise</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">IIM Amritsar Elective Bidding Intelligence System</p>', unsafe_allow_html=True)

st.divider()

# Tabs
tab1, tab2 = st.tabs(["🚀 Role Explorer", "📄 Resume Gap Analyzer"])

# --- TAB 1: ROLE EXPLORER ---
with tab1:
    st.header("Find Electives for a Target Role")
    
    role_options = sorted([r.replace('_', ' ').title() for r in onet_target_roles.keys()])
    selected_role = st.selectbox("Select a Career Goal:", [""] + role_options)
    
    if selected_role:
        role_key = selected_role.lower().replace(' ', '_')
        target_skills = onet_target_roles.get(role_key, [])
        
        st.info(f"Market requires {len(target_skills)} key skills for this role.")
        
        if st.button("Show Bidding Strategy"):
            df_recs = get_recommendations(target_skills)
            
            if not df_recs.empty:
                st.success("✅ Optimal Bidding Bundle Generated")
                st.dataframe(df_recs, use_container_width=True, hide_index=True)
            else:
                st.warning("No specific electives matched. Focus on core curriculum.")

# --- TAB 2: RESUME GAP ANALYZER ---
with tab2:
    st.header("Upload CV for Personalized Analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    with col_b:
        target_role_cv = st.selectbox("Select Target Goal:", role_options, key='cv_role')
        
    if uploaded_file and target_role_cv:
        if st.button("Analyze Gap"):
            with st.spinner("Analyzing..."):
                # 1. Parse CV
                text = extract_text_from_pdf(uploaded_file)
                current_skills = extract_skills_robust(text, list(G_final.nodes()))
                
                st.write(f"**Detected Skills:** {', '.join(list(current_skills)[:10])}...")
                
                # 2. Identify Gaps
                role_key = target_role_cv.lower().replace(' ', '_')
                needed_skills = onet_target_roles.get(role_key, [])
                
                # Simple set difference
                gap_skills = [s for s in needed_skills if s not in current_skills]
                
                if gap_skills:
                    st.write(f"**Missing Skills:** {len(gap_skills)}")
                    df_gap = get_recommendations(gap_skills)
                    
                    if not df_gap.empty:
                        st.subheader("🌉 Recommended Electives to Bridge the Gap")
                        st.dataframe(df_gap, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Gaps detected, but no specific electives found to cover them.")
                else:
                    st.success("🎉 No Skill Gaps! You are fully prepared for this role.")

# Footer
st.markdown("---")
st.markdown("built by P HariPrasath")
