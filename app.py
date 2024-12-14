import streamlit as st
import weaviate
import uuid
from twelvelabs import TwelveLabs
from openai import OpenAI
import time
import base64
import json
import os
from dotenv import load_dotenv


load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
TWELVELABS_API_KEY = os.getenv("TWELVELABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "VideoEmbeddings")



@st.cache_resource
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

def generate_image_embedding(image_file):

    try:
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        result = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            image_file=image_file.getvalue()
        )
        
        if result.image_embedding and result.image_embedding.segments:
            return result.image_embedding.segments[0].embeddings_float, None
        return None, "No embedding generated"
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

def search_similar_videos(client, image_embedding, top_k=5):
  
    try:
        result = (
            client.query
            .get(COLLECTION_NAME, ["video_url", "title", "description", "start_time", "end_time"])
            .with_near_vector({
                "vector": image_embedding,
                "certainty": 0.7
            })
            .with_additional(['certainty'])
            .with_limit(top_k)
            .do()
        )
        return result['data']['Get'][COLLECTION_NAME], None
    except Exception as e:
        return None, str(e)

def create_timestamped_video_url(video_url, start_time, end_time):

    base_url = video_url.split('#')[0]
    
    if 'youtube.com' in base_url or 'youtu.be' in base_url:
        return f"{base_url}?start={int(start_time)}&end={int(end_time)}"
    else:
        return f"{base_url}#t={int(start_time)},{int(end_time)}"

def render_video_result(result):
   
    confidence = result.get('_additional', {}).get('certainty', 0)
    
    st.markdown(f"""
    ### {result['title'] or 'Untitled'}
    **Similarity Score:** {confidence * 100:.2f}%
    
    {result['description'] if result['description'] else ''}
    
    **Timestamp:** {result['start_time']:.2f}s - {result['end_time']:.2f}s
    """)
    
    video_url = create_timestamped_video_url(
        result['video_url'], 
        result['start_time'], 
        result['end_time']
    )
    st.video(video_url)

# def get_chat_response(openai_client, question, context=""):
 
#     try:
#         if not openai_client:
#             return {
#                 "response": "OpenAI client is not properly initialized. Please check your API key.",
#                 "metadata": None
#             }
        
#         messages = [
#             {
#                 "role": "system",
#                 "content": """You are a helpful video search assistant. Help users find and understand video content.
#                 When discussing videos, focus on relevant details and timestamps."""
#             },
#             {
#                 "role": "user",
#                 "content": f"Question: {question}\nContext: {context}"
#             }
#         ]
        
#         try:
#             chat_response = openai_client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
            
#             if not chat_response or not chat_response.choices:
#                 raise Exception("Empty response from OpenAI")
            
#             return {
#                 "response": chat_response.choices[0].message.content,
#                 "metadata": {
#                     "model": "gpt-3.5-turbo",
#                     "timestamp": time.time(),
#                     "status": "success"
#                 }
#             }
            
#         except Exception as api_error:
#             st.error(f"OpenAI API Error: {str(api_error)}")
#             return {
#                 "response": "I apologize, but I'm having trouble generating a response right now. Please try again later.",
#                 "metadata": {
#                     "error": str(api_error),
#                     "timestamp": time.time(),
#                     "status": "error"
#                 }
#             }
            
#     except Exception as e:
#         st.error(f"Unexpected error in chat response: {str(e)}")
#         return {
#             "response": "An unexpected error occurred. Please try again.",
#             "metadata": {
#                 "error": str(e),
#                 "timestamp": time.time(),
#                 "status": "error"
#             }
#         }

def init_openai():
    try:
        client = OpenAI()
        client.models.list()  # Test the connection
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        return None

@st.cache_resource
def init_weaviate():
 
    try:
   
        auth_config = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY)
        
   
        client = weaviate.WeaviateClient(
            connection=weaviate.connect.Connection.from_url(
                url=WEAVIATE_URL,
                auth_client_secret=auth_config
            )
        )
        
     
        if not client.schema.exists(COLLECTION_NAME):
            client.schema.create_class(schema)
            st.success(f"Created new class: {COLLECTION_NAME}")
        else:
            st.success(f"Using existing class: {COLLECTION_NAME}")
        
        return client
    except Exception as e:
        st.error(f"Error initializing Weaviate client: {str(e)}")
        return None

def search_page(client):

    st.header("Search Videos Using Image")
    
    if client is None:
        st.error("Database connection not available. Please check your configuration.")
        return
    
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results", 1, 10, 5)
        with col2:
            min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.7)
        
        if st.button("Search"):
            with st.spinner('Processing...'):
                image_embedding, error = generate_image_embedding(uploaded_file)
                
                if error:
                    st.error(f"Error: {error}")
                elif image_embedding:
                    results, search_error = search_similar_videos(client, image_embedding, top_k)
                    
                    if search_error:
                        st.error(f"Search error: {search_error}")
                    elif results:
                        filtered_results = [
                            r for r in results 
                            if r.get('_additional', {}).get('certainty', 0) >= min_confidence
                        ]
                        
                        if filtered_results:
                            st.success(f"Found {len(filtered_results)} matches")
                            for result in filtered_results:
                                with st.expander(
                                    f"Match: {result['title'] or 'Untitled'} " 
                                    f"({result.get('_additional', {}).get('certainty', 0) * 100:.1f}%)"
                                ):
                                    render_video_result(result)
                        else:
                            st.warning(f"No results met the confidence threshold of {min_confidence:.1%}")
                    else:
                        st.warning("No matches found")

def add_video_page(client):
   
    st.header("Add New Video")
    
    if client is None:
        st.error("Database connection not available. Please check your configuration.")
        return
    
    with st.form("video_input"):
        video_url = st.text_input("Video URL")
        title = st.text_input("Title (optional)")
        description = st.text_area("Description (optional)")
        
        submitted = st.form_submit_button("Process Video")
        
        if submitted and video_url:
            with st.spinner('Processing video...'):
                embeddings, error = generate_video_embedding(video_url, title, description)
                
                if error:
                    st.error(f"Error: {error}")
                elif embeddings:
                    st.success(f"Generated {len(embeddings)} embeddings")
                    
                    if store_embeddings(client, embeddings):
                        st.success("Successfully stored embeddings")
                        
                        with st.expander("View embedding details"):
                            st.json(json.dumps({
                                'total_segments': len(embeddings),
                                'sample_segment': {
                                    'start_time': embeddings[0]['start_offset_sec'],
                                    'end_time': embeddings[0]['end_offset_sec'],
                                    'embedding_size': len(embeddings[0]['embedding'])
                                }
                            }, indent=2))
                    else:
                        st.error("Failed to store embeddings")

def chat_page(openai_client):
   
    st.header("Chat About Videos")
    
    if openai_client is None:
        st.error("Chat functionality not available. Please check your OpenAI API configuration.")
        return
    

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"]["response"])
                if message["content"]["metadata"]:
                    with st.expander("Response Details"):
                        st.json(message["content"]["metadata"])
            else:
                st.markdown(message["content"])


    if prompt := st.chat_input("Ask about videos..."):
      
        st.session_state.messages.append({"role": "user", "content": prompt})
        
    
        with st.chat_message("assistant"):
            response_data = get_chat_response(openai_client, prompt)
            st.session_state.messages.append({"role": "assistant", "content": response_data})
            st.markdown(response_data["response"])

def main():
    st.title("Video Search & Chat Assistant")
    

    weaviate_client = init_weaviate()
    openai_client = init_openai()
    
    page = st.sidebar.radio(
        "Choose a feature", 
        ["Add Video", "Image Search", "Chat"]
    )
    
    if page == "Add Video":
        add_video_page(weaviate_client)
    elif page == "Image Search":
        search_page(weaviate_client)
    else:
        chat_page(openai_client)
    
    with st.sidebar:
        st.markdown("""
        ### Features

        """)

if __name__ == "__main__":
    main()
