import streamlit as st
import weaviate
import uuid
from twelvelabs import TwelveLabs
import time
import os
from PIL import Image
import io


from dotenv import load_dotenv
import os

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
TWELVELABS_API_KEY = os.getenv("TWELVELABS_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "VideoEmbeddings") 

UPLOAD_DIR = "temp_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

schema = {
    "class": COLLECTION_NAME,
    "vectorizer": "none",
    "properties": [
        {"name": "video_url", "dataType": ["string"]},
        {"name": "title", "dataType": ["string"]},
        {"name": "description", "dataType": ["string"]},
        {"name": "start_time", "dataType": ["number"]},
        {"name": "end_time", "dataType": ["number"]}
    ]
}

def save_uploaded_file(uploaded_file):

    try:
        if uploaded_file is None:
            return None
        
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def init_weaviate():

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    
    if not client.schema.exists(COLLECTION_NAME):
        client.schema.create_class(schema)
        st.success(f"Created new class: {COLLECTION_NAME}")
    else:
        st.success(f"Using existing class: {COLLECTION_NAME}")
    
    return client

def add_video_page():
    """Page for adding new videos"""
    st.header("Add New Video")
    
    with st.form("video_input"):
        video_url = st.text_input("Video URL")
        title = st.text_input("Title (optional)")
        description = st.text_input("Description (optional)")
        
        submitted = st.form_submit_button("Process Video")
        
        if submitted and video_url:
            with st.spinner('Processing video...'):
                embeddings, error = generate_video_embedding(video_url, title, description)
                
                if error:
                    st.error(f"Error processing video: {error}")
                elif embeddings:
                    st.success(f"Generated {len(embeddings)} embeddings")
                    
                    if store_embeddings(client, embeddings):
                        st.success("Successfully stored embeddings in Weaviate")
                        
                        with st.expander("View sample embeddings"):
                            for i, emb in enumerate(embeddings[:2]):
                                st.write(f"\nEmbedding {i+1}:")
                                st.write(f"Time range: {emb['start_offset_sec']} - {emb['end_offset_sec']} seconds")
                                st.write(f"Embedding vector (first 5 values): {emb['embedding'][:5]}")
                    else:
                        st.error("Failed to store embeddings in Weaviate")
def generate_video_embedding(video_url, title="", description=""):

    try:
        st.write(f"Processing video: {video_url}")
        
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        
        task = twelvelabs_client.embed.task.create(
            model_name="Marengo-retrieval-2.7",
            video_url=video_url
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def on_task_update(task):
            status_text.write(f"Status: {task.status}")
            if task.status == "completed":
                progress_bar.progress(100)
            elif task.status == "processing":
                progress_bar.progress(50)
        
        task.wait_for_done(
            sleep_interval=2,
            callback=on_task_update
        )
        
        task_result = twelvelabs_client.embed.task.retrieve(task.id)
        
        embeddings = []
        for segment in task_result.video_embedding.segments:
            embeddings.append({
                'embedding': segment.embeddings_float,
                'start_offset_sec': segment.start_offset_sec,
                'end_offset_sec': segment.end_offset_sec,
                'video_url': video_url,
                'title': title,
                'description': description
            })
        
        return embeddings, None
    except Exception as e:
        return None, str(e)


def store_embeddings(client, embeddings):
 
    try:
        batch = client.batch.configure(batch_size=100)
        with batch:
            for emb in embeddings:
                properties = {
                    "video_url": emb['video_url'],
                    "title": emb['title'],
                    "description": emb['description'],
                    "start_time": emb['start_offset_sec'],
                    "end_time": emb['end_offset_sec']
                }
                
                client.batch.add_data_object(
                    data_object=properties,
                    class_name=COLLECTION_NAME,
                    vector=emb['embedding'],
                    uuid=str(uuid.uuid4())
                )
        return True
    except Exception as e:
        st.error(f"Error storing embeddings: {str(e)}")
        return False

client = init_weaviate()

st.title("Video Embedding and Search")

page = st.sidebar.selectbox("Choose a page", ["Add Video", "Search Videos"])

if page == "Add Video":
    add_video_page()
else:
    pass