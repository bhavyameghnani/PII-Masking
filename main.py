import re
import json
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from presidio_analyzer import Pattern, PatternRecognizer
from faker import Faker
from presidio_anonymizer.entities import OperatorConfig

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Reference code imports
import boto3
import os
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat

# Load environment variables from .env file
load_dotenv()
AWS_REGION = os.getenv("AWS_REGIONS")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Initialize AWS clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# Sample document content to be anonymized
document_content = """Date: October 19, 2021
 Witness: John Doe
 Subject: Testimony Regarding the Loss of Wallet

 Testimony Content:

 Hello Officer,

 My name is John Doe and on October 19, 2021, my wallet was stolen in the vicinity of Kilmarnock during a bike trip. This wallet contains some very important things to me.

 Firstly, the wallet contains my credit card with number 4111 1111 1111 1111, which is registered under my name and linked to my bank account, PL61109010140000071219812874.

 Additionally, the wallet had a driver's license - DL No: 999000680 issued to my name. It also houses my Social Security Number, 602-76-4532.

 What's more, I had my polish identity card there, with the number ABC123456.

 I would like this data to be secured and protected in all possible ways. I believe It was stolen at 9:30 AM.

 In case any information arises regarding my wallet, please reach out to me on my phone number, 999-888-7777, or through my personal email, johndoe@example.com.

 Please consider this information to be highly confidential and respect my privacy.

 The bank has been informed about the stolen credit card and necessary actions have been taken from their end. They will be reachable at their official email, support@bankname.com.
 My representative there is Victoria Cherry (her business phone: 987-654-3210).

 Thank you for your assistance,

 John Doe"""

# Define patterns for Polish ID and time
polish_id_pattern = Pattern(
    name="polish_id_pattern",
    regex="[A-Z]{3}\\d{6}",
    score=1,
)
time_pattern = Pattern(
    name="time_pattern",
    regex="(1[0-2]|0?[1-9]):[0-5][0-9] (AM|PM)",
    score=1,
)

# Define the recognizers with the patterns
polish_id_recognizer = PatternRecognizer(
    supported_entity="POLISH_ID", patterns=[polish_id_pattern]
)
time_recognizer = PatternRecognizer(supported_entity="TIME", patterns=[time_pattern])

# Initialize Faker for custom fake data generation
fake = Faker()

# Custom function to generate fake Polish ID
def fake_polish_id(_=None):
    return fake.bothify(text="???######").upper()

# Test the fake Polish ID function
fake_polish_id()

# Custom function to generate fake time
def fake_time(_=None):
    return fake.time(pattern="%I:%M %p")

# Test the fake time function
fake_time()

# Define custom operators for the anonymizer
new_operators = {
    "POLISH_ID": OperatorConfig("custom", {"lambda": fake_polish_id}),
    "TIME": OperatorConfig("custom", {"lambda": fake_time}),
}

# Initialize the anonymizer again with a seed for reproducibility
anonymizer = PresidioReversibleAnonymizer(
    faker_seed=42,
)

# Add the custom recognizers and operators again
anonymizer.add_recognizer(polish_id_recognizer)
anonymizer.add_recognizer(time_recognizer)
anonymizer.add_operators(new_operators)

# Anonymize the document before indexing
anonymized_content = anonymizer.anonymize(document_content)

# Extract the anonymization map to store in JSON
anonymization_map = anonymizer.deanonymizer_mapping

# Save the anonymization map to a JSON file
with open('anonymization_map.json', 'w') as f:
    json.dump(anonymization_map, f, indent=4)

# Split the anonymized content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(anonymized_content)

# Convert chunks to Document objects
documents = [Document(page_content=chunk) for chunk in chunks]

# Index the chunks using Bedrock embeddings
docsearch = FAISS.from_documents(documents, bedrock_embeddings)
retriever = docsearch.as_retriever()

# Create an anonymizer chain with prompt template and Bedrock model
template = """Answer the question based only on the following context:
{context}

Question: {anonymized_question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = BedrockChat(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", client=bedrock_client)

# Define parallel input processing for the chain
_inputs = RunnableParallel(
    question=RunnablePassthrough(),
    anonymized_question=RunnableLambda(anonymizer.anonymize),
)

# Create the anonymizer chain
anonymizer_chain = (
    _inputs
    | {
        "context": itemgetter("anonymized_question") | retriever,
        "anonymized_question": itemgetter("anonymized_question"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Invoke the chain with a sample question
anonymizer_chain.invoke(
    "Where did the theft of the wallet occur, at what time, and who was it stolen from?"
)

# Add deanonymization step to the chain
chain_with_deanonymization = anonymizer_chain | RunnableLambda(anonymizer.deanonymize)

# Invoke the chain with deanonymization and print the results
print(
    chain_with_deanonymization.invoke(
        "Where did the theft of the wallet occur, at what time, and who was it stolen from?"
    )
)

print(
    chain_with_deanonymization.invoke("What was the content of the wallet in detail?")
)

print(chain_with_deanonymization.invoke("Whose phone number is it: 999-888-7777?"))
