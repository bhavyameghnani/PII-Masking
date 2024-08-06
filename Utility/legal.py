import json
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

# Define custom patterns for legal terms
legal_patterns = [
    Pattern(name="party_name_pattern", regex=r"\b(Nomura)\b", score=0.5),
    Pattern(name="contract_terms_pattern", regex=r"\b(Confidentiality Agreement|Non-Disclosure Agreement|NDA)\b", score=0.5),
    Pattern(name="financial_info_pattern", regex=r"\b(Bank Account Number: \d{10,12})\b", score=0.5),
    # Add more patterns as needed
]

# Create custom recognizers for legal terms
legal_recognizers = [PatternRecognizer(supported_entity="LEGAL_TERM", patterns=[pattern]) for pattern in legal_patterns]

# Initialize the Presidio analyzer and add the custom recognizers
analyzer = AnalyzerEngine()
for recognizer in legal_recognizers:
    analyzer.registry.add_recognizer(recognizer)

# Initialize the Presidio anonymizer
anonymizer = AnonymizerEngine()

# Sample text containing legal terms
text = ("My name is John Doe, I am from Nomura. As per our Confidentiality Agreement, "
        "I cannot disclose the Bank Account Number: 1234567890 of our client.")

# Analyze the text to identify PII entities including the custom legal terms
results = analyzer.analyze(
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

# Create a list of AnonymizerResult objects
anonymizer_results = [
    RecognizerResult(
        start=entity.start,
        end=entity.end,
        entity_type=entity.entity_type,
        score=entity.score
    ) for entity in results
]

# Configure the anonymizer to use a reversible anonymization method for each entity
anonymizer_config = {
    "default": OperatorConfig(
        operator_name="hash",
        params={"salt": "mysalt"}
    ),
    "PERSON": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{PERSON}"}
    ),
    "PHONE_NUMBER": OperatorConfig(
        operator_name="mask",
        params={
            "masking_char": "*",
            "chars_to_mask": 12,
            "from_end": True
        }
    ),
    "EMAIL_ADDRESS": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{EMAIL}"}
    ),
    "ORGANIZATION": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{ORGANIZATION}"}
    ),
    "LOCATION": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{LOCATION}"}
    ),
    "CREDIT_CARD": OperatorConfig(
        operator_name="mask",
        params={
            "masking_char": "*",
            "chars_to_mask": 16,
            "from_end": True
        }
    ),
    "DATE_TIME": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{DATE}"}
    ),
    "NRP": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{NRP}"}
    ),
    "IP_ADDRESS": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{IP}"}
    ),
    "IBAN_CODE": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{IBAN}"}
    ),
    "US_DRIVER_LICENSE": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{DRIVER_LICENSE}"}
    ),
    "URL": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{URL}"}
    ),
    "AWS_ACCESS_KEY": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{AWS_KEY}"}
    ),
    "IPV4": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{IPV4}"}
    ),
    "IPV6": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{IPV6}"}
    ),
    "LEGAL_TERM": OperatorConfig(
        operator_name="replace",
        params={"new_value": "{LEGAL_TERM}"}
    )
}

# Perform the anonymization
anonymized_text = anonymizer.anonymize(
    text=text, 
    analyzer_results=anonymizer_results, 
    operators=anonymizer_config
)

# Prepare the JSON output
output = {
    "original_text": text,
    "anonymized_text": anonymized_text.text,
    "anonymized_entities": [
        {
            "entity_type": entity.entity_type,
            "start": entity.start,
            "end": entity.end,
            "text": text[entity.start:entity.end],
            "anonymized_text": anonymized_text.text[entity.start:entity.end]
        } for entity in anonymizer_results
    ]
}

# Save the output to a JSON file
with open("anonymized_data.json", "w") as json_file:
    json.dump(output, json_file, indent=4)

# Print the results
print("Original Text:", text)
print("Anonymized Text:", anonymized_text.text)
print("Anonymized Entities:", output["anonymized_entities"])
