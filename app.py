import os
import json
import hashlib
import streamlit as st
import numpy as np
from typing import TypedDict, List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
)
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from bson import json_util

# Load environment variables
load_dotenv()

# MongoDB setup
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['vendor_matching_db']  # Database name
vendors_collection = db['vendors']  # Collection for vendor data
ps_collection = db['problem_statements']  # Collection for problem statements
vendor_capabilities_collection = db['vendor_capabilities']  # Collection for vendor capabilities
ps_analysis_collection = db['ps_analysis']  # Collection for problem statement analysis
vendor_embeddings_collection = db['vendor_embeddings']  # Collection for vendor embeddings
ps_embeddings_collection = db['ps_embeddings']  # Collection for problem statement embeddings

# Verify MongoDB connection
try:
    client.admin.command('ping')
    st.sidebar.success("✅ Connected to MongoDB!")
except Exception as e:
    st.sidebar.error(f"❌ MongoDB connection failed: {str(e)}")

# Initialize sentence transformer model
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sentence_transformer()

# LLM setup
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Graph State
class GraphState(TypedDict):
    problem_statement: str
    vendors: List[Dict[str, str]]
    ps_analysis: Dict[str, Any]
    ps_embedding: np.ndarray
    vendor_capabilities: List[Dict[str, Any]]
    vendor_embeddings: List[np.ndarray]

# Helper functions
def load_document(file_path: str, file_bytes: bytes) -> str:
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".pptx") or file_path.endswith(".ppt"):
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    docs = loader.load()
    text = "\n".join([doc.page_content for doc in docs])
    os.remove(file_path)
    return text

def get_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def load_cached_analysis(collection, content_hash: str) -> Dict[str, Any]:
    doc = collection.find_one({"content_hash": content_hash})
    return doc.get("data") if doc else None

def save_analysis(collection, content_hash: str, data: Dict[str, Any]):
    collection.update_one(
        {"content_hash": content_hash},
        {"$set": {"content_hash": content_hash, "data": data}},
        upsert=True
    )

def load_embedding(collection, content_hash: str) -> np.ndarray:
    doc = collection.find_one({"content_hash": content_hash})
    if doc and "embedding" in doc:
        return np.array(doc["embedding"])
    return None

def save_embedding(collection, content_hash: str, embedding: np.ndarray):
    collection.update_one(
        {"content_hash": content_hash},
        {"$set": {"content_hash": content_hash, "embedding": embedding.tolist()}},
        upsert=True
    )

# Vendor onboarding: Extract capabilities and embedding
def process_vendor_profile(vendor_name: str, vendor_text: str) -> Dict[str, Any]:
    vendor_hash = get_content_hash(f"{vendor_name}:{vendor_text}")

    # Check MongoDB for capabilities
    capabilities = load_cached_analysis(vendor_capabilities_collection, vendor_hash)
    if not capabilities:
        prompt = PromptTemplate.from_template("""
        From this vendor profile, extract:
        1. Key technical domains (e.g., NLP, CV, ML)
        2. Tools and frameworks used
        3. Core capabilities (e.g., scalability, real-time processing)
        4. Industry experience
        5. Team size and project scale

        Vendor Profile: {vendor_text}

        Provide structured output in JSON format.
        """)
        chain = prompt | llm | JsonOutputParser()
        capabilities = chain.invoke({"vendor_text": vendor_text})
        capabilities["name"] = vendor_name
        save_analysis(vendor_capabilities_collection, vendor_hash, capabilities)

    # Check MongoDB for embedding
    embedding = load_embedding(vendor_embeddings_collection, vendor_hash)
    if embedding is None:
        text_representation = create_text_representation(capabilities)
        embedding = model.encode([text_representation])[0]
        save_embedding(vendor_embeddings_collection, vendor_hash, embedding)

    return capabilities, embedding

# Problem statement analysis: Extract requirements and embedding
def process_problem_statement(problem_statement: str) -> Dict[str, Any]:
    ps_hash = get_content_hash(problem_statement)

    # Check MongoDB for analysis
    analysis = load_cached_analysis(ps_analysis_collection, ps_hash)
    if not analysis:
        prompt = PromptTemplate.from_template("""
        Analyze this problem statement and extract:
        1. Primary technical domains (e.g., NLP, CV, ML)
        2. Required tools or frameworks
        3. Key technical requirements (e.g., real-time, accuracy)
        4. Deployment constraints (e.g., cloud, edge)
        5. Project complexity (e.g., research, production)

        Problem Statement: {problem_statement}

        Provide structured analysis in JSON format.
        """)
        chain = prompt | llm | JsonOutputParser()
        analysis = chain.invoke({"problem_statement": problem_statement})
        save_analysis(ps_analysis_collection, ps_hash, analysis)

    # Check MongoDB for embedding
    embedding = load_embedding(ps_embeddings_collection, ps_hash)
    if embedding is None:
        text_representation = create_text_representation(analysis)
        embedding = model.encode([text_representation])[0]
        save_embedding(ps_embeddings_collection, ps_hash, embedding)

    return analysis, embedding

# Semantic similarity and shortlisting
def shortlist_vendors(ps_embedding: np.ndarray, vendor_embeddings: List[np.ndarray], vendor_capabilities: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
    similarities = cosine_similarity([ps_embedding], np.array(vendor_embeddings))[0]
    similarity_results = [
        {
            "name": cap["name"],
            "semantic_similarity_score": float(similarities[i]),
            "similarity_percentage": float(similarities[i]) * 100,
            "vendor_capabilities": cap
        }
        for i, cap in enumerate(vendor_capabilities)
    ]
    similarity_results.sort(key=lambda x: x["semantic_similarity_score"], reverse=True)
    return similarity_results[:top_k]

# Batched LLM evaluation
def evaluate_shortlist(ps_analysis: Dict[str, Any], shortlist: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
    prompt = PromptTemplate.from_template("""
    Evaluate the following vendors against the problem statement requirements:
    Problem Requirements: {ps_analysis}
    Vendors: {vendor_batch}

    For each vendor, provide:
    1. Transferability score (1-100) assessing how well their capabilities match the problem
    2. Detailed justification for the score

    Return structured JSON output:
    [
        {{
            "name": str,
            "transferability_score": float,
            "justification": str
        }}
    ]
    """)
    chain = prompt | llm | JsonOutputParser()

    results = []
    for i in range(0, len(shortlist), batch_size):
        batch = shortlist[i:i + batch_size]
        batch_capabilities = [v["vendor_capabilities"] for v in batch]
        try:
            batch_results = chain.invoke({
                "ps_analysis": json.dumps(ps_analysis),
                "vendor_batch": json.dumps(batch_capabilities)
            })
            for j, result in enumerate(batch_results):
                results.append({
                    **batch[j],
                    "transferability_score": result["transferability_score"],
                    "justification": result["justification"]
                })
        except Exception as e:
            for vendor in batch:
                results.append({
                    **vendor,
                    "transferability_score": vendor["similarity_percentage"],
                    "justification": f"Fallback to semantic similarity due to error: {str(e)}"
                })
    results.sort(key=lambda x: x["transferability_score"], reverse=True)
    return results

# Text representation for embeddings
def create_text_representation(data: Dict[str, Any]) -> str:
    text_parts = []
    for key, value in data.items():
        if key != "name":
            if isinstance(value, (dict, list)):
                text_parts.append(f"{key}: {json.dumps(value)}")
            else:
                text_parts.append(f"{key}: {value}")
    return " ".join(text_parts)

# Streamlit UI
st.title("🎯 Optimized Vendor Matching System")

page = st.sidebar.selectbox("Select Page", ["📊 Dashboard", "🏢 Vendor Submission", "📝 PS Submission", "🔍 Vendor Matching"])

if page == "📊 Dashboard":
    st.header("System Overview")
    vendors = list(vendors_collection.find())
    ps_list = list(ps_collection.find())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Vendors", len(vendors))
    with col2:
        st.metric("Total Problem Statements", len(ps_list))
    with col3:
        cached_analyses = vendor_capabilities_collection.count_documents({}) + ps_analysis_collection.count_documents({})
        st.metric("Cached Analyses", cached_analyses)
    if vendors:
        st.subheader("Recent Vendors")
        for vendor in vendors[-3:]:
            st.write(f"• {vendor['name']}")
    if ps_list:
        st.subheader("Recent Problem Statements")
        for ps in ps_list[-3:]:
            st.write(f"• {ps['title']}")

elif page == "🏢 Vendor Submission":
    st.header("Submit Vendor Document")
    vendor_name = st.text_input("Vendor Name")
    uploaded_file = st.file_uploader("Upload Vendor Document (PDF, PPTX, DOCX)", type=["pdf", "pptx", "ppt", "docx"])

    if st.button("Submit Vendor") and uploaded_file and vendor_name:
        try:
            file_path = f"temp_{uploaded_file.name}"
            text = load_document(file_path, uploaded_file.getvalue())
            vendor_data = {"name": vendor_name, "text": text}
            vendors_collection.update_one(
                {"name": vendor_name},
                {"$set": vendor_data},
                upsert=True
            )
            with st.spinner("Processing vendor profile..."):
                capabilities, embedding = process_vendor_profile(vendor_name, text)
                st.success(f"✅ Vendor '{vendor_name}' onboarded and cached!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif page == "📝 PS Submission":
    st.header("Submit Problem Statement (PS)")
    title = st.text_input("Project Title")
    description = st.text_area("Project Description", height=150)
    outcomes = st.text_area("Expected Outcomes", height=100)

    if st.button("Submit PS") and title and description and outcomes:
        ps_id = hashlib.md5(title.encode()).hexdigest()[:8]
        problem_statement = f"Title: {title}\nDescription: {description}\nOutcomes: {outcomes}"
        ps_data = {
            "id": ps_id,
            "title": title,
            "description": description,
            "outcomes": outcomes,
            "full_statement": problem_statement
        }
        ps_collection.update_one(
            {"id": ps_id},
            {"$set": ps_data},
            upsert=True
        )
        with st.spinner("Processing problem statement..."):
            analysis, embedding = process_problem_statement(problem_statement)
            st.success(f"✅ PS '{title}' (ID: {ps_id}) processed and cached!")

elif page == "🔍 Vendor Matching":
    st.header("Match Vendors to Problem Statement")
    ps_list = list(ps_collection.find())
    if not ps_list:
        st.warning("⚠️ No PS submitted yet. Please submit one first.")
    else:
        ps_options = {f"{ps['title']} (ID: {ps['id']})": ps for ps in ps_list}
        selected_option = st.selectbox("Select Problem Statement", list(ps_options.keys()))
        top_k = st.slider("Number of vendors to shortlist", 5, 50, 20)
        batch_size = st.slider("Batch size for LLM evaluation", 1, 10, 5)

        if st.button("🔍 Match Vendors"):
            selected_ps = ps_options[selected_option]
            problem_statement = selected_ps["full_statement"]
            vendors = list(vendors_collection.find())
            if not vendors:
                st.warning("⚠️ No vendors submitted yet. Please submit vendors first.")
            else:
                with st.spinner("🔄 Running vendor matching..."):
                    try:
                        # Process problem statement
                        ps_analysis, ps_embedding = process_problem_statement(problem_statement)

                        # Process vendors and collect embeddings
                        vendor_capabilities = []
                        vendor_embeddings = []
                        for vendor in vendors:
                            cap, emb = process_vendor_profile(vendor["name"], vendor["text"])
                            vendor_capabilities.append(cap)
                            vendor_embeddings.append(emb)

                        # Shortlist vendors using semantic similarity
                        shortlist = shortlist_vendors(ps_embedding, vendor_embeddings, vendor_capabilities, top_k=top_k)

                        # Evaluate shortlist with batched LLM calls
                        final_results = evaluate_shortlist(ps_analysis, shortlist, batch_size=batch_size)

                        # Display results
                        st.subheader("🎯 Vendor Matching Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Vendors Analyzed", len(vendors))
                        with col2:
                            st.metric("Shortlisted Vendors", len(final_results))
                        with col3:
                            if final_results:
                                st.metric("Top Score", f"{final_results[0]['transferability_score']:.1f}%")

                        for i, result in enumerate(final_results, 1):
                            with st.expander(f"#{i} {result['name']} - Score: {result['transferability_score']:.1f}%"):
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.write("**Semantic Similarity**")
                                    st.progress(result["semantic_similarity_score"])
                                    st.write(f"Similarity: {result['similarity_percentage']:.1f}%")
                                with col2:
                                    st.write("**Transferability**")
                                    st.write(f"Score: {result['transferability_score']:.1f}%")
                                    st.write("**Justification**")
                                    st.write(result["justification"])

                        # Export results
                        st.subheader("📤 Export Results")
                        results_json = json.dumps({
                            "problem_statement": selected_ps,
                            "results": final_results
                        }, indent=2, default=json_util.default)
                        st.download_button(
                            label="Download Results as JSON",
                            data=results_json,
                            file_name=f"matching_results_{selected_ps['id']}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"❌ Error during matching: {str(e)}")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info("""
**System Features:**
- 🔄 One-time vendor onboarding with caching
- 🧠 Local embeddings for scalable matching
- 📊 Batched LLM evaluation for efficiency
- 💾 Persistent storage in MongoDB Atlas
- 🚀 Optimized for thousands of vendors
""")

if st.sidebar.button("🗑️ Clear All Cache"):
    try:
        vendor_capabilities_collection.delete_many({})
        ps_analysis_collection.delete_many({})
        vendor_embeddings_collection.delete_many({})
        ps_embeddings_collection.delete_many({})
        st.sidebar.success("Cache cleared!")
    except Exception as e:
        st.sidebar.error(f"❌ Error clearing cache: {str(e)}")