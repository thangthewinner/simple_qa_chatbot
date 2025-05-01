import gradio as gr
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


## LLM
def get_llm():
    model_id = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype="auto")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=250,
        temperature=0.5,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    return HuggingFacePipeline(pipeline=pipe)

## Document Loader 
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document

## Text Splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len 
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Vector Database 
def vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## Retriever 
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever

## QA chain
def retriever_qa(file_path, query):
    llm = get_llm()
    retriever_obj = retriever(file_path)
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever_obj,
                                     return_source_documents=False)
    response = qa.invoke(query)
    return response['result']


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here ...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_application.launch()