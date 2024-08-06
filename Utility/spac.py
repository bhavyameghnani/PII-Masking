import json
import spacy
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult
from faker import Faker
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

# Initialize Faker
fake = Faker()

# Initialize SpaCy
nlp = spacy.load("en_core_web_lg")

# Define custom patterns for legal terms
legal_patterns = [
    Pattern(name="party_name_pattern", regex=r"\b(Nomura)\b", score=0.5),
    Pattern(name="contract_terms_pattern", regex=r"\b(Confidentiality Agreement|Non-Disclosure Agreement|NDA)\b", score=0.5),
    # Add more patterns as needed
]

# Create custom recognizers for legal terms
legal_recognizers = [PatternRecognizer(supported_entity="LEGAL_TERM", patterns=[pattern]) for pattern in legal_patterns]

# Initialize the Presidio analyzer and add the custom recognizers
analyzer = AnalyzerEngine()
for recognizer in legal_recognizers:
    analyzer.registry.add_recognizer(recognizer)

# Initialize the Presidio anonymizer
anonymizer_engine = AnonymizerEngine()

# Initialize the PresidioReversibleAnonymizer from langchain_experimental
anonymizer = PresidioReversibleAnonymizer(
    add_default_faker_operators=False,
)

# Sample text containing legal terms
text = ("My name is John Doe, I am from Microsoft. As per our Confidentiality Agreement, "
        "I cannot disclose the Bank Account Number: 1234567890 of our client.")

# Analyze the text using SpaCy to identify entities
doc = nlp(text)
spacy_results = []
for ent in doc.ents:
    if ent.label_ in ["PERSON", "ORG", "GPE"]:
        spacy_results.append(RecognizerResult(entity_type=ent.label_, start=ent.start_char, end=ent.end_char, score=0.85))

# Analyze the text using Presidio to identify PII entities including the custom legal terms
presidio_results = analyzer.analyze(
    text=text, 
    entities=[
        "PERSON", 
        "PHONE_NUMBER", 
        "EMAIL_ADDRESS", 
        "ORGANIZATION", 
        "LOCATION", 
        "CREDIT_CARD", 
        "DATE_TIME", 
        "NRP", 
        "IP_ADDRESS", 
        "IBAN_CODE", 
        "US_DRIVER_LICENSE", 
        "URL", 
        "AWS_ACCESS_KEY", 
        "IPV4", 
        "IPV6",
        "LEGAL_TERM"
    ], 
    language="en"
)

# Combine the results from SpaCy and Presidio
results = presidio_results + spacy_results

# Create a mapping for PII to fake data
pii_to_fake = {}

# Configure the anonymizer to use Faker for generating fake data
anonymizer_config = {
    "default": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.text()}
    ),
    "PERSON": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.name()}
    ),
    "PHONE_NUMBER": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.phone_number()}
    ),
    "EMAIL_ADDRESS": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.email()}
    ),
    "ORGANIZATION": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.company()}
    ),
    "LOCATION": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.address()}
    ),
    "CREDIT_CARD": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.credit_card_number()}
    ),
    "DATE_TIME": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.date_time().isoformat()}
    ),
    "NRP": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.ssn()}
    ),
    "IP_ADDRESS": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.ipv4()}
    ),
    "IBAN_CODE": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.iban()}
    ),
    "US_DRIVER_LICENSE": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.license_plate()}
    ),
    "URL": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.url()}
    ),
    "AWS_ACCESS_KEY": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.uuid4()}
    ),
    "IPV4": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.ipv4()}
    ),
    "IPV6": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.ipv6()}
    ),
    "LEGAL_TERM": OperatorConfig(
        operator_name="custom",
        params={"function": lambda x: fake.bs()}
    )
}

# Function to generate fake data and keep the mapping
def custom_anonymize(entity_text, entity_type):
    if entity_type not in pii_to_fake:
        if entity_type in anonymizer_config:
            fake_data = anonymizer_config[entity_type].params['function'](entity_text)
            pii_to_fake[entity_text] = fake_data
        else:
            fake_data = anonymizer_config["default"].params['function'](entity_text)
            pii_to_fake[entity_text] = fake_data
    return pii_to_fake[entity_text]

# Perform the anonymization using PresidioReversibleAnonymizer
anonymized_text = anonymizer.anonymize(text)

# Update the anonymized text with fake data
for entity in results:
    original_text = text[entity.start:entity.end]
    anonymized_version = custom_anonymize(original_text, entity.entity_type)
    # Replace the original text with the anonymized version
    anonymized_text = anonymized_text.replace(original_text, anonymized_version, 1)

# Prepare the JSON output
output = {
    "original_text": text,
    "anonymized_text": anonymized_text,
    "anonymized_entities": [
        {
            "entity_type": entity.entity_type,
            "start": entity.start,
            "end": entity.end,
            "text": text[entity.start:entity.end],
            "anonymized_text": custom_anonymize(text[entity.start:entity.end], entity.entity_type)
        } for entity in results
    ]
}

# Save the output to a JSON file
with open("anonymized_data.json", "w") as json_file:
    json.dump(output, json_file, indent=4)

# Print the results
print("Original Text:", text)
print("Anonymized Text:", anonymized_text)
print("Anonymized Entities:", output["anonymized_entities"])
