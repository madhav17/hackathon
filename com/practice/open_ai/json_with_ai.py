# pip install langchain-openai langchain langchain-openai openai python-dotenv langchain-community

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import json
import random

from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key is missing. Please set it in the .env file.")

os.environ["OPENAI_API_KEY"] = api_key

# Define the prompt template
prompt_template = """
You are a dataset schema generator. Based on the user's prompt, return a structured JSON schema specifying the dataset structure and metadata. 
Do NOT generate sample data points.

User Prompt: {user_prompt}

Expected Output:
The JSON schema should include:
- "row_count": Number of rows to generate (integer)
- "fields": A list of fields, each containing:
  - "name": The field name (string)
  - "type": The data type (string, integer, decimal, etc.)
  - "constraints" (if applicable): Additional constraints or rules for the field, such as "greater than 18" or "set of values".

Example Output:
{{
  "row_count": 10,
  "fields": [
    {{
      "name": "name",
      "type": "String"
    }},
    {{
      "name": "age",
      "type": "Integer",
      "constraints": ">18"
    }},
    {{
      "name": "salary",
      "type": "Decimal",
      "constraints": "2"
    }},
    {{
      "name": "department",
      "type": "String",
      "constraints": "SET[HR,IT,FINANCE]"
    }}
  ]
}}

Expectations: Json sould also included the any additional fileds not mentioned in given sample example but may occure in user prompt.
"""

# Create a LangChain prompt template
prompt = PromptTemplate(input_variables=["user_prompt"], template=prompt_template)

# Initialize the GPT-4o LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Define the LangChain LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Function to process user prompt and generate dataset schema
def generate_dataset_schema(user_prompt):
    response = chain.run(user_prompt=user_prompt)
    try:
        # Parse the JSON response if possible
        schema = json.loads(response)
        return schema
    except json.JSONDecodeError:
        print("Error: Could not parse the schema.")
        return response

# Test Example
if __name__ == "__main__":
    example_prompt = """
    Generate a schema for a dataset with 10 rows. Include:
    - Name (string)
    - Salary (decimal, rounded to 2 places)
    - Department (string from HR, IT, FINANCE).
    - City (String Metro cities of India )
    """
    
    # Generate schema using GPT-4
    schema = generate_dataset_schema(example_prompt)
    print("Generated Schema:")
    print(json.dumps(schema, indent=2))