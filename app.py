import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import graphviz 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BidWise | IIM Amritsar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR "ATTRACTIVE" UI ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #eef2f6);
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e3a8a;
        font-weight: 800;
        font-size: 4rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0px;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.5rem;
        font-weight: 300;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* Card Styling for Results */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #2563eb;
    }
    
    /* Table Styling */
    .dataframe {
        font-size: 14px !important;
        font-family: 'Arial', sans-serif !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 30px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: scale(1.05);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_resource
def load_data():
    try:
        with open('skill_network.pkl', 'rb') as f: G = pickle.load(f)
    except: G = nx.Graph()

    try:
        with open('onet_map.pkl', 'rb') as f: roles_map = pickle.load(f)
    except: roles_map = {}

    try:
        df = pd.read_excel('Elective List.xlsx')
        if 'Category' in df.columns:
            df = df[df['Category'] == 'Elective'].copy()
    except:
        df = pd.DataFrame(columns=['Course Name', 'Department/Area', 'Brief Description for NLP Mapping'])
        
    return G, roles_map, df

G_final, onet_target_roles, df_electives = load_data()

# --- 2. NLP ENGINE ---
@st.cache_resource
def setup_nlp_engine(df):
    if df.empty: return None, None
    df['Weighted_Text'] = ((df['Course Name'].fillna('') + " ") * 3 + 
                           df['Brief Description for NLP Mapping'].fillna('')).str.lower()
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(df['Weighted_Text'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = setup_nlp_engine(df_electives)

# --- 3. HELPER FUNCTIONS ---

def create_skill_flowchart(current, gaps, target_role):
    """Generates a DOT graph for GraphViz"""
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', bgcolor='transparent')
    
    # Nodes
    graph.node('Start', 'You\n(Current Profile)', shape='circle', style='filled', fillcolor='#dbeafe', color='#2563eb')
    graph.node('Target', target_role.title().replace('_', ' '), shape='doublecircle', style='filled', fillcolor='#1e40af', fontcolor='white')
    
    # Current Skills Cluster
    with graph.subgraph(name='cluster_current') as c:
        c.attr(style='dashed', color='#94a3b8', label='Existing Assets')
        for i, s in enumerate(current[:4]): # Show top 4 to avoid clutter
            c.node(f'c_{i}', s.title(), shape='box', style='filled', fillcolor='#dcfce7', color='#16a34a')
            graph.edge('Start', f'c_{i}', color='#cbd5e1')

    # Bridge/Gap Skills Cluster
    with graph.subgraph(name='cluster_gaps') as c:
        c.attr(style='solid', color='#f59e0b', label='The Bidding Bridge (Needed)')
        for i, s in enumerate(gaps[:5]): # Show top 5 gaps
            c.node(f'g_{i}', s.title(), shape='box', style='filled', fillcolor='#fef3c7', color='#d97706')
            # Edges from Current (Logic: Network Path) -> Gap -> Target
            # For visual simplicity, we just flow left to right
            graph.edge(f'g_{i}', 'Target', color='#2563eb', penwidth='2.0')

    # Invisible edges to force layout if needed
    if current and gaps:
        graph.edge(f'c_0', f'g_0', style='invis')
        
    return graph

def get_recommendations_expanded(needed_skills, top_n=15):
    if vectorizer is None: return pd.DataFrame()
    recs = []
    
    # Mapping for Domain Boost
    domain_map = {
        'Analytics': ['data', 'analytics', 'sql', 'python'],
        'Finance': ['finance', 'valuation', 'investment', 'risk'],
        'Marketing': ['marketing', 'brand', 'sales', 'consumer'],
        'Operations': ['supply chain', 'operations', 'logistics', 'project'],
        'Strategy': ['strategy', 'consulting', 'management', 'business']
    }

    for skill in needed_skills:
        skill_vec = vectorizer.transform([skill.lower()])
        cosine_sim = cosine_similarity(skill_vec, tfidf_matrix).flatten()
        
        # Domain Boost
        intended_domain = next((d for d, k in domain_map.items() if any(x in skill.lower() for x in k)), None)
        if intended_domain:
            for i, area in enumerate(df_electives['Department/Area']):
                if intended_domain in str(area): cosine_sim[i] *= 1.5

        # Get Top 3 matches per skill to fill the list
        top_indices = cosine_sim.argsort()[-3:][::-1]
        
        for idx in top_indices:
            score = cosine_sim[idx]
            if score > 0.12: # Slightly lower threshold to ensure we get enough electives
                course = df_electives.iloc[idx]
                recs.append({
                    "Skill to Bridge": skill.title(),
                    "Elective": course['Course Name'],
                    "Department": course['Department/Area'],
                    "Description": course.get('Brief Description for NLP Mapping', 'No description available')[:100] + "...",
                    "Relevance": score
                })

    if not recs: return pd.DataFrame()
    
    # Dedup and Sort by Relevance
    df_recs = pd.DataFrame(recs)
    df_recs = df_recs.sort_values('Relevance', ascending=False).drop_duplicates(subset=['Elective'])
    return df_recs.head(top_n)

# --- 4. MAIN UI LAYOUT ---

# -- Header Section --
col1, col2 = st.columns([0.8, 5])
with col1:
    # You can host a logo on imgur or github and link it here
    st.markdown("## 🏛️") 
with col2:
    st.markdown('<h1 class="main-header">BidWise</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Strategic Elective Bidding for IIM Amritsar | Powered by Network Science</p>', unsafe_allow_html=True)

st.markdown("---")

# -- Navigation --
selected_tab = st.radio("", ["🎯 Role Explorer", "🌉 Resume Gap Analyzer"], horizontal=True, label_visibility="collapsed")

if selected_tab == "🎯 Role Explorer":
    st.markdown("### 🔎 Search Your Dream Career")
    
    # 1. Search Bar with many roles
    role_options = sorted([r.replace('_', ' ').title() for r in onet_target_roles.keys()])
    selected_role = st.selectbox("Type to search (e.g., 'Product Manager', 'Brand Manager')", [""] + role_options)
    
    if selected_role:
        role_key = selected_role.lower().replace(' ', '_')
        target_skills = onet_target_roles.get(role_key, [])
        
        # 2. Skill Visualization (Flowchart)
        st.markdown("#### 🧬 Skill DNA for this Role")
        # For Role explorer, we assume no current skills, so just show the Target Skills
        graph = create_skill_flowchart([], target_skills, selected_role)
        st.graphviz_chart(graph)
        
        if st.button("Generate Bidding Strategy"):
            with st.spinner("Calculating Optimal Elective Bundle..."):
                df_bundle = get_recommendations_expanded(target_skills, top_n=15)
                
                if not df_bundle.empty:
                    st.success(f"✅ Generated {len(df_bundle)} Electives for {selected_role}")
                    
                    # 3. Enhanced Table Display
                    st.dataframe(
                        df_bundle[['Elective', 'Department', 'Description', 'Skill to Bridge']],
                        column_config={
                            "Elective": st.column_config.TextColumn("Elective Name", width="medium"),
                            "Department": st.column_config.TextColumn("Area", width="small"),
                            "Description": st.column_config.TextColumn("Course Snapshot", width="large"),
                            "Skill to Bridge": st.column_config.TextColumn("Why take this?", width="medium"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.warning("Could not map specific electives. Please check Core requirements.")

elif selected_tab == "🌉 Resume Gap Analyzer":
    st.markdown("### 📄 Personalized Gap Analysis")
    
    c1, c2 = st.columns(2)
    with c1:
        uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")
    with c2:
        role_options = sorted([r.replace('_', ' ').title() for r in onet_target_roles.keys()])
        target_role_cv = st.selectbox("Target Goal:", [""] + role_options)

    if uploaded_file and target_role_cv:
        if st.button("Analyze My Profile"):
            
            # Extract
            def extract_pdf(f):
                try:
                    r = PyPDF2.PdfReader(f)
                    return " ".join([p.extract_text() for p in r.pages])
                except: return ""
            
            text = extract_pdf(uploaded_file)
            
            # Simple Extraction for Demo (You can paste your 'extract_skills_robust' logic here)
            # Using basic regex matching against G_final nodes
            current_skills = []
            clean_text = re.sub(r'\s+', ' ', text).lower()
            for node in G_final.nodes():
                if f" {node.lower()} " in f" {clean_text} ":
                    current_skills.append(node)
            
            # Gap Logic
            role_key = target_role_cv.lower().replace(' ', '_')
            needed = onet_target_roles.get(role_key, [])
            gaps = [s for s in needed if s not in current_skills]
            
            # Visuals
            st.markdown("### 🛤️ Your Career Path Visualization")
            graph = create_skill_flowchart(current_skills, gaps, target_role_cv)
            st.graphviz_chart(graph)
            
            # Recommendations
            st.markdown("### 🎒 Your Personalized Bidding List")
            if gaps:
                df_gap_recs = get_recommendations_expanded(gaps, top_n=12)
                st.dataframe(
                    df_gap_recs[['Elective', 'Department', 'Description', 'Skill to Bridge']],
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("You are fully matched! Consider advanced electives for specialization.")
