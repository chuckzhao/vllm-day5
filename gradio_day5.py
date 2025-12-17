# gradio_day5.py
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.vllm import Vllm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# Global index
index = None
llm = None
embed_model = None

def initialize_models():
    """Initialize models once"""
    global llm, embed_model
    if llm is None:
        embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
        llm = Vllm(
            model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            tensor_parallel_size=1,
            max_new_tokens=256,
            vllm_kwargs={        
                "gpu_memory_utilization": 0.9, 
                "max_model_len": 4096, 
                "enforce_eager": True,            
            },
        )

def upload_files(files):
    global index
    initialize_models() 
    os.makedirs("uploaded", exist_ok=True)
    for file in files:
        path = f"uploaded/{os.path.basename(file.name)}"
        os.system(f"cp '{file.name}' '{path}'")
    documents = SimpleDirectoryReader("uploaded").load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    return "Files uploaded and indexed! Ready to chat."

def respond(message, history):
    global index
    
    if index is None:
        return "Please upload files first!"
    
    query_engine = index.as_query_engine(llm=llm, embed_model=embed_model, streaming=True)
    response = query_engine.query(message)
    
    bot_message = ""
    for token in response.response_gen:
        bot_message += token
        yield bot_message
        
if __name__ == '__main__':
    with gr.Blocks(title="Day 5: Local 70B RAG Chat") as demo:
        gr.Markdown("# Day 5/30 — Local 70B RAG Chat with Gradio")
        gr.Markdown("Upload PDFs/text → Ask questions → 70B streams answers (100% local)")
        
        with gr.Row():
            file_upload = gr.File(label="Upload PDFs/Text Files", file_count="multiple")
            upload_btn = gr.Button("Index Files")
        status = gr.Textbox(label="Status", value="No files uploaded yet")
        upload_btn.click(upload_files, inputs=file_upload, outputs=status)
        
        gr.ChatInterface(
            respond,
            chatbot=gr.Chatbot(height=600),
            textbox=gr.Textbox(label="Ask a question about your documents"),
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
