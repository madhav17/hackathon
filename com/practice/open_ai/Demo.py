from openai import OpenAI

class Demo:

    def __init__(self):
        print("Intialize")
        self.client = OpenAI(
            api_key=""
        )

        self.command : str = '''
            You are an expert programming assistant.

Task:
Convert plain/simple English conditions into a structured JSON Abstract Syntax Tree (AST).
This AST must capture the exact logical structure of the condition without ambiguity.

Schema Rules:
- Each node has:
  - "op": one of ["AND","OR","NOT","CMP"]
  - "field": string (only for "CMP", otherwise null)
  - "cmp": one of ["eq","neq","gt","gte","lt","lte","in","between","contains","starts_with","ends_with", null]
  - "value": string | number | boolean | array | null
  - "children": array of child nodes (for AND/OR/NOT)

Output:
- Return ONLY valid JSON strictly following the schema.
- Do not add extra explanation.

Example Input:
status is not shipped and (amount greater than 500 or country is US)

Example Output (JSON AST):
{
  "op": "AND",
  "children": [
    {
      "op": "CMP",
      "field": "status",
      "cmp": "neq",
      "value": "shipped"
    },
    {
      "op": "OR",
      "children": [
        {
          "op": "CMP",
          "field": "amount",
          "cmp": "gt",
          "value": 500
        },
        {
          "op": "CMP",
          "field": "country",
          "cmp": "eq",
          "value": "US"
        }
      ]
    }
  ]
}
'''

        # self.english_instruction : str =  "[Filter all those records where status is not shipped and (amount greater than 500 or country is US)]"
        self.english_instruction : str =  "[Filter all those records where status is not shipped and amount greater than 500 or country is US]"


    def execute_api(self):

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            # You can also try "gpt-4o" or "gpt-4.1" (Codex is deprecated but GPT models now handle code better)
            messages=[
                {
                    "role": "system",
                    "content": f"{self.command}"
                 },
                {
                    "role": "user",
                    "content": f"Now convert the following English condition into JSON AST: \n{self.english_instruction}"
                }
            ]
        )

        print(response.choices[0].message.content)


if __name__ == '__main__':
    ob : Demo = Demo()
    ob.execute_api()
