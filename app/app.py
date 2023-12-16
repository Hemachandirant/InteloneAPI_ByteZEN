import streamlit as st
import os
import torch
import transformers
import time
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from datetime import datetime
import boto3
import numpy as np
import replicate
import base64
import textwrap
from botocore.exceptions import NoCredentialsError
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from streamlit.logger import get_logger
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import intel_extension_for_pytorch as ipex
from optimum.intel.generation.modeling import TSModelForCausalLM
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain import PromptTemplate, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from persist import persist, load_widget_state
import utils as utl


logger = get_logger(__name__)
parser = argparse.ArgumentParser()

st.set_page_config(layout="wide", page_title='BZ Support',page_icon=':zap:')
st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()

parser.add_argument("--auth_token",
                    help='HuggingFace authentification token for getting LLaMa2',
                    required=True)

parser.add_argument("--model_id",
                    type=str,
                    choices=["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"],
                    default="meta-llama/Llama-2-7b-chat-hf",
                    help="Hugging Face model id")

parser.add_argument("--window_len",
                    type=int,
                    help='Chat memory window length',
                    default=5)

parser.add_argument("--dtype",
                    type=str,
                    choices=["float32", "bfloat16"],
                    default="float32",
                    help="bfloat16, float32")

parser.add_argument("--device",
                    type=str,
                    choices=["cpu"],
                    default="cpu",
                    help="cpu")

parser.add_argument("--max_new_tokens",
                    type=int,
                    default=32,
                    help="Max tokens for warmup")

parser.add_argument("--prompt",
                    type=str,
                    default="solve the issue",
                    help="Text prompt for warmup")

parser.add_argument("--num_warmup",
                    type=int, 
                    default=15,
                    help="Number of warmup iterations")

parser.add_argument("--alpha",
                    default="auto",
                    help="Smooth quant parameter")

parser.add_argument("--output_dir",
                    default="./models",
                    help="Output directory for quantized model")

parser.add_argument("--ipex",
                    action="store_true",
                    help="Whether to use IPEX")

parser.add_argument("--jit",
                    action="store_true",
                    help="Whether to enable graph mode with IPEX")

parser.add_argument("--sq",
                    action="store_true",
                    help="Enable inference with smooth quantization")

parser.add_argument("--int4",
                    action="store_true",
                    help="Enable 4 bits quantization with bigdl-llm")


args = parser.parse_args()


if args.ipex:
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass
    
if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)
    
if args.int4:
    from bigdl.llm.transformers import AutoModelForCausalLM

    
# Check if amp is enabled
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)


def main(stop_keyword="restart", exit_keyword="exit"):
    
    st.markdown("<h1 style='text-align: centre;'> IT Customer Support Intel® oneAPI.</h1>", unsafe_allow_html=True)
    option = st.text_input(label = "Ask Something")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='position: absolute; z-index: -1; top: 0; left: 0; width: 100%; height: 100%;'>
            <img src='AI.jpg' style='object-fit: cover; width: 100%; height: 100%; opacity: 0.3;'/>
        </div>
    """, unsafe_allow_html=True)
    st.image("imagelog.jpeg",width=850)
    copyright_text = """
    <div style="text-align: center; padding: 10px; background-color: CCD1D1; border-radius: 5px;">
        <p style="margin: 0; font-size: 18px; font-weight: bold; color: #333;">🚀 ByteZEN © . All rights reserved. 🚀</p>
    </div>
"""
    st.markdown(copyright_text, unsafe_allow_html=True)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today ?"}]
    memory.clear()
    
def get_conversation(llm, window_len=args.window_len):
    # Define memory
    memory = ConversationBufferWindowMemory(k=window_len)
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory
    )
    history = ""
    
    conversation.prompt.template = """You are an IT support resolver. Provide helpful and informative responses to user queries related to IT issues in briefly. If you don't know the answer, indicate that you're not sure. Remember, your role is to assist with IT support and give answers briefly. Current conversation:\nIT Support Resolver: How can I help you with IT support today ? \n{history}\nUser: {input}\nIT Support Resolver:"""

    return conversation, memory

@st.cache_resource()
def LLMPipeline(temperature, 
                top_p,
                top_k,
                max_length,
                hf_auth,
                repetition_penalty=1.1,
                model_id=args.model_id):
    
    # Initialize tokenizer & model
    tokenizer = LlamaTokenizer.from_pretrained(model_id, token=hf_auth)
    #(PATH, local_files_only=True)
    #tokenizer = LlamaTokenizer.from_pretrained('./models/fp32',local_files_only=True,repo_type='other')
    model = LlamaForCausalLM.from_pretrained(model_id,
                                             torch_dtype=amp_dtype,
                                              torchscript=True if args.sq or args.jit else False,
                                              token=hf_auth)
    #/home/ubuntu/CustomerSupport_OpensourceLLM/models/fp32
    #model = LlamaForCausalLM.from_pretrained('/home/ubuntu/CustomerSupport_OpensourceLLM/models/fp32',config='models/fp32/config.json', local_files_only=True)
    # model_path = "/home/ubuntu/CustomerSupport_OpensourceLLM/models/fp32"  
    # tokenizer = LlamaTokenizer.from_pretrained(model_path)  
    # model = LlamaForCausalLM.from_pretrained(model_path) 
    model = model.to(memory_format=torch.channels_last)

    model.eval()
    
    # Model params
    num_att_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    
    # Apply IPEX llm branch optimizations
    if args.ipex:
        model = ipex._optimize_transformers(model, dtype=amp_dtype, inplace=True)
    
    # Smooth quantization option
    if args.sq:
        model = TSModelForCausalLM.from_pretrained(args.output_dir, file_name="best_model.pt")
    
    # 4bits quantization with bigdl
    if args.int4:
        # model.save_pretrained("models/fp32")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", load_in_4bit=True)

    
    # IPEX Graph mode
    if args.jit and args.ipex:
        dummy_input = tokenizer(args.prompt, return_tensors="pt")
        input_ids = dummy_input['input_ids']
        attention_mask = torch.ones(1, input_ids.shape[-1] + 1)
        attention_mask[:, 0] = 0
        past_key_values = tuple(
            [
                (
                    torch.ones(size=[1, num_att_heads, 1, head_dim]),
                    torch.ones(size=[1, num_att_heads, 1, head_dim]),
                )
                for _ in range(num_layers)
            ]
        )
        example_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        with torch.no_grad(), torch.autocast(
            device_type=args.device,
            enabled=amp_enabled,
            dtype=amp_dtype if amp_enabled else None,
        ):
            trace_model = torch.jit.trace(model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False)
            trace_model = torch.jit.freeze(trace_model)
            # Use TSModelForCausalLM wrapper since traced models don't have a generate method
            model = TSModelForCausalLM(trace_model, model.config)

    
    # Define HF pipeline
    generate_text = pipeline(model=model,
                             tokenizer=tokenizer,
                             return_full_text=True,
                             task='text-generation',
                             temperature=temperature,
                             top_p=top_p,
                             top_k=top_k,                         
                             max_new_tokens=max_length,
                             repetition_penalty=repetition_penalty)
    
    llm = HuggingFacePipeline(pipeline=generate_text)
    
    # Create langchain conversation
    conversation, memory = get_conversation(llm)
    
    return conversation, memory


def Chat_support():
    st.title("🤖 Intelligent Chat Support")
    st.write("Effortlessly connect with our AI chatbot for swift and expert IT support through text, avoiding the need for human intervention.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.first_response_captured = False
    st.sidebar.selectbox("Choose Your Preferred Language", st.session_state["languages"], key=persist("language_name"))


    temperature = 0.3
    top_p = 0.9
    top_k = 20
    max_length = 512

    # Load conversation
    conversation, memory = LLMPipeline(temperature, top_p, top_k, max_length, args.auth_token)


    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today ?"}]

    # Display chatbot messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="☘️" if message["role"] == "assistant" else "🧑‍💻"):
            st.write(message["content"])

    #Button to clear chatbot memory
    st.sidebar.write('\n')
    st.sidebar.write('\n')
    _, middle, _ = st.sidebar.columns([.16, 2.5, .1])
    with middle:
        clear_button = st.button(':arrows_counterclockwise: Clear Chatbot Memory', on_click=clear_chat_history)

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(prompt)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="☘️"):
            placeholder = st.empty()
            placeholder.markdown("▌")
            conversation.prompt.template = "You are an IT support resolver. Provide helpful and informative responses to user queries related to IT issues in briefly. If you don't know the answer, indicate that you're not sure. Remember, your role is to assist with IT support and provide answers briefly. Current conversation:\nIT Support Resolver: How can I help you with IT support today ? \n{history}\nUser: {input}\nIT Support Resolver:"
            #conversation.prompt.template = "You are an IT support resolver. Provide helpful and informative responses to user queries related to IT issues in briefly. If you don't know the answer, indicate that you're not sure. Remember, your role is to assist with IT support and provide answers briefly."

            start_time = datetime.now()  # Capture start time 
            response = conversation.predict(input=prompt)
            print('Original reponse from llm: ',response)
            end_time = datetime.now()  # Capture end time  
            # Assume `response` is the text response you got
            tokenized_response = tokenizer.tokenize(response)

            # Number of tokens in the response
            num_tokens = len(tokenized_response)
            print("Number of tokens in the response:", num_tokens)

          
            time_taken = end_time - start_time  # Calculate time taken  
            print(f"Time taken for inference: {time_taken}")  
            full_response = ""
            for item in response:
                full_response += item
                placeholder.markdown(full_response + "▌")
                #time.sleep(0.04)

            # Extract the specific portion only for the first response
            if not st.session_state.get("first_response_captured", False):
                extracted_response = full_response.split('\n\n')[0]  # Extract up to the first double line break
                placeholder.markdown(extracted_response)
                message = {"role": "assistant", "content": extracted_response}
                st.session_state.first_response_captured = True
            else:
                placeholder.markdown(full_response)
                message = {"role": "assistant", "content": full_response}
            
            print(message)
            st.session_state.messages.append(message)




if "page" not in st.session_state:
    # Initialize session state.
    st.session_state.update({
        # Default page.
        "page": "Home",

        "list": [],

        # Languages which you prefer
        "languages": ["English", "French", "Hindi", "Tamil"],
    })


def Speech_support():
    pass

def virtual_ai():
    st.subheader("Check out our [Virtual AI](https://e432-146-152-233-34.ngrok-free.app/)")
    st.markdown("<br>", unsafe_allow_html=True)

def image_support():
    st.title("📷 Image Analysis Assistance")
    st.write("Capture the challenge through an image upload, and let our AI-driven Image Analysis Assistance rapidly deliver detailed steps and troubleshooting guidance. Harness the efficiency of visuals for a seamless IT problem-solving experience.")
    

   
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    #   st.write("File uploaded")


    if uploaded_file is not None:
        
        st.image(uploaded_file, caption="", use_column_width=True,width=100)

        file_name = uploaded_file.name
        print(file_name)

        # Replace these with your actual AWS S3 credentials and bucket information
        AWS_ACCESS_KEY = 'AKIA4SHWXMHTSGYKIRE7'
        AWS_SECRET_KEY = 'j/a9x2Z++Iue2i0U0mNE0xbWXYdrfFttKsV7E0/n'
        BUCKET_NAME = 'errorimagesintel'

        # Replace this with the desired name of the file on S3
        s3_file_name = file_name

        # Set your AWS credentials
        boto3.setup_default_session(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )

        # Upload the image to S3 and get the S3 URI
        s3 = boto3.client('s3')
        try:
            s3.upload_file(file_name, BUCKET_NAME, s3_file_name)
            print(f"Upload Successful: {s3_file_name}")
            #s3_uri = f"s3://{BUCKET_NAME}/{s3_file_name}"

            #st.write(f"S3 URI: {s3_uri}")

            # Use the S3 URI in replicate.run
            output = replicate.run(
                "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
                input={
                    "image": f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_file_name}",
                    
                    "prompt": "Given the uploaded image that illustrates a particular issue, instruct the Language Model to provide detailed and step-by-step resolution steps to address and fix the problem depicted in the image. The response should be clear, concise, and include any necessary actions, configurations, or troubleshooting steps required for a successful resolution.The solution given by you should consist of interpretation of the user's error in 2 lines and then provide the solution for the error in steps.",
                    "max_tokens": 1024,
                    "temperature": 0.2
                }
            )
            progress_bar = st.progress(0)
            status_text = st.empty()
        # Display the uploaded image
            for i in range(100):
                progress_bar.progress(i + 1)

                new_rows = np.random.randn(10, 2)

                # Update status text.
                status_text.text(
                    'Computing the issue: %s' % new_rows[-1, 1])

                time.sleep(0.1)

                status_text.text('Done!')
            st.header("Resolution steps:")
            output_list = list(output)
            # Initialize an empty string to store the concatenated output
            complete_output = ''

# Iterate through each word in the output_list
            for word in output_list:
                # Print the word
                print(word)
                
                # Concatenate the word to the complete_output string
                complete_output += word + ''

# Now 'complete_output' contains the entire text
            print("Complete Output:", complete_output)

            # Display the output in the Streamlit app
            #st.header("Resolution steps:")
            st.text_area("", complete_output, height=400)
                #print(item)

        except FileNotFoundError:
            st.error(f"The file {file_name} was not found.")
        except NoCredentialsError:
            st.error("Credentials not available.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


#model and tokenizer loading
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


@st.cache_resource
def llm_pipeline():
    base_model.to(device='cpu') 
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample=True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    print(generated_text)
    answer = generated_text['result']
    print("Answer from process answer:",answer)
    return answer

def ragsupport():
    st.title("RAG : A CPU Approach")
    question = st.text_area("Enter your Question",height=50)
    if st.button("Ask"):
        answer = process_answer(question)
        st.text_area("Answer", answer)

def automation_support():
    pass

page_names_to_funcs = {
    "Home": main,
    "💬CHAT ASSIST": Chat_support,
    "🧠VIRTUAL AI":virtual_ai,
    "📷VISION GUIDANCE": image_support,
    "📚RAG": ragsupport,
    "🤖AUTOMATION_SUPPORT": automation_support,

}

demo_name = st.sidebar.selectbox("SUPPORT MODE", page_names_to_funcs.keys())

page_names_to_funcs[demo_name]()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
    st.image("logoside.jpeg")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Follow us")
    #st.write("[![Star](https://img.shields.io/github/stars/Hemachandirant/MetaHuman.svg?logo=github&style=social)](https://github.com/Hemachandirant/MetaHuman)")
    st.write("[![Follow on Twitter](https://img.shields.io/twitter/follow/bytezen?style=social)](https://twitter.com/hemac140)")
    st.write("[![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=social&logo=linkedin)](https://www.linkedin.com/in/hemachandiran-t-081836171/)")
  
    temperature = 0.3
    top_p = 0.9
    top_k = 20
    max_length = 512

    # Load conversation
    conversation, memory = LLMPipeline(temperature, top_p, top_k, max_length, args.auth_token)

    
    
    
    
