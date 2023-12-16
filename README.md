# ByteZen Customer Support Prototype

## Team Name
ByteZen

## Problem Statement
Customer Support

## Team Leader Email
[hemachandiran.t88@wipro.com](mailto:hemachandiran.t88@wipro.com)

## Brief of the Prototype

### Description
ByteZen is working on a groundbreaking solution for Customer IT Technical Support using a Large Language Model. The application leverages Intel's analytical toolkit and Cloud CPU to design and fine-tune AI models capable of understanding complex technical issues and providing prompt solutions. The toolkit includes the Intel Distribution of Modin for efficient data preprocessing, Intel Xeon CPUs coupled with Intel Extension for Transformers, and the Neural Compressor for model quantization.

- This project is a Streamlit chatbot with Langchain deploying a **LLM Model** on **Intel® Server and Client CPUs**and allows users to optimize the model using  **Intel® Extension for PyTorch (IPEX) in bfloat16 with graph mode** or **smooth quantization** (A new quantization technique specifically designed for LLMs: [ArXiv link](https://arxiv.org/pdf/2211.10438.pdf)), or **4-bit quantization**. The user can expect **up to 4.3x speed-up** compared to stock PyTorch in default mode.

- **IMPORTANT:** The CPU needs to support bfloat16 ops in order to be able to use such optimization. On top of software optimizations, I also introduced some hardware optimizations like non-uniform memory access (NUMA). User needs to **ask for access to LLaMA2** models by following this [link](https://huggingface.co/meta-llama#:~:text=Welcome%20to%20the%20official%20Hugging,processed%20within%201%2D2%20days). When getting approval from Meta, you can generate an authentification token from your HuggingFace account, and use it to load the model.

### Goal
The goal of ByteZen is to revolutionize technical support by building an AI-powered application that offers round-the-clock assistance. The application aims to provide instant solutions to customer problems, reducing downtime and enhancing the overall user experience.

## Prototype Description

The prototype utilizes Huggingface Transformers with large language models, including Mistral-7B and LLava-13B. Technologies such as Intel Extension for transformers, Intel Analytical Toolkit, Intel Neural Compressor, Intel Distribution for Python, streamlit, Langchain, node.js (Avatar Application), Azure Speech Service, and Ngrok are employed to achieve the project goals.

### Architecture
![WhatsApp Image 2023-12-16 at 6 43 31 AM](https://github.com/Hemachandirant/InteloneAPI_ByteZEN/assets/83321708/acfee12e-68f1-4144-952b-e670430cf8fc)


### Core Components of oneAPI AI Toolkit & IDC Used in the Project
![Core Components](https://github.com/Hemachandirant/Intel_Hackathon_Customer_Support-oneAPI/assets/83321708/dc0a4bb6-856b-4e65-bf4f-1930dc734f1f)

## Medium Article
[Revolutionizing Tech Support with Intel AI Toolkits and OneAPI](https://medium.com/@rshivanipriya/revolutionizing-tech-support-with-intel-ai-toolkits-and-oneapi-4cf7027909af)

## Benchmarking

# Pandas Vs Modin
![WhatsApp Image 2023-12-12 at 8 17 21 PM](https://github.com/Hemachandirant/InteloneAPI_ByteZEN/assets/83321708/6afe9bf2-a332-4866-afb2-5595ce3c4eab)

# No optimization vs IPEX vs BIGDL
![WhatsApp Image 2023-12-16 at 1 03 10 AM](https://github.com/Hemachandirant/InteloneAPI_ByteZEN/assets/83321708/042f084e-5982-40f4-86b8-6b7abea92979)

### Models
- Huggingface Transformers [https://huggingface.co/shivani05/Mistral-Finetuned-CPU/tree/main]
  - LLMs: Mistral-7B, LaMini-T5-738M

### Technologies Used
1. Intel Extension for transformers
2. Intel Analytical Toolkit
3. Intel Neural Compressor
4. Intel Distribution for Python
5. Streamlit
6. Langchain
7. Node.js (Avatar Application)
8. Azure Speech Service
9. Ngrok

### Xenon CPU Utilization during model training:

https://github.com/Hemachandirant/InteloneAPI_ByteZEN/assets/83321708/f3ad700f-d63f-4aef-a819-9e1d0d62a4df


### Training loss and saving the model:

![BeFunky-collage](https://github.com/Hemachandirant/Intel_Hackathon_Customer_Support-oneAPI/assets/83321708/ef4653da-1ffe-43d6-ba56-15fd14b4684c)

## :monocle_face: Description


## :scroll: Getting started

1. Start by cloning the repository:  
```bash
[git clone https://github.com/Hemachandirant/CustomerSupport_OpensourceLLM.git](https://github.com/Hemachandirant/InteloneAPI_ByteZEN.git)
cd llama2-chatbot-cpu
```
2. Create a Python 3.9 conda environment:
```bash
conda create -y -n myenv python=3.9
```
3. Activate the environment:  
```bash
conda activate myenv
```
4. Install requirements for NUMA:  
```bash
conda install -y gperftools -c conda-forge
conda install -y intel-openmp
sudo apt install numactl
```
5. Install the app requirements:  
```bash
pip install -r requirements.txt
```

## :rocket: Start the app

- Default mode (no optimizations):
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token>
```

- IPEX in graph mode with FP32:
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token> --ipex --jit
```

- IPEX in graph mode with bfloat16:
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token> --dtype=bfloat16 --ipex --jit
```

- Smooth quantization:
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token> --sq
```

- 4-bit quantization:
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token> --int4
```


## :computer: Chatbot UI
![Screenshot (363)](https://github.com/Hemachandirant/InteloneAPI_ByteZEN/assets/83321708/1156afde-30cf-4e3f-a587-6bec5f4d550d)

    


## :mailbox_closed: Contact
For any information, feedback or questions, please [contact me][hemac140@gmail.com]




[hemac-email]: mailto:hemac140@gmail.com


