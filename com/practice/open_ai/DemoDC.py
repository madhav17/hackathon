import json

from openai import OpenAI

class DemoGE:

    def __init__(self):
        print("Intialize")
        self.client = OpenAI(
            # api_key=""
        )

        self.command : str = '''
           You are an expert data-cleaning assistant.

Goal:
Convert the following plain-English requirements into a single JSON object that:
1) Encodes all FILTER logic as a JSON Abstract Syntax Tree (AST).
2) Specifies a cleaning plan compatible with pandas and scikit-learn.
3) Flags any missing/ambiguous columns for clarification.

Conventions:
- Assume the dataset is a pandas DataFrame named `df`.
- Do NOT invent columns. If the input mentions a column not in the provided list (if any), add it to "clarifications".
- Normalize comparison phrases (e.g., “greater than” → gt, “less than or equal to” → lte, “one of / in” → in, ranges → between).
- Return ONLY valid JSON (UTF-8, no comments, no trailing commas). No extra prose.

Inputs:
- English requirements: """[INSERT_ENGLISH_REQUIREMENTS_HERE]"""
- (Optional) Known columns & dtypes (may be empty): [INSERT_COLUMNS_JSON_OR_EMPTY]

### Output JSON Schema (must follow exactly)

{
  "filter_ast": {
    "op": "AND" | "OR" | "NOT" | "CMP",
    "field": string | null,
    "cmp": "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "in" | "between" | "contains" | "starts_with" | "ends_with" | null,
    "value": string | number | boolean | array | { "min": number | string, "max": number | string } | null,
    "children": [ <filter_ast>, ... ]
  },

  "targets": {
    "numeric": [string, ...],
    "categorical": [string, ...],
    "datetime": [string, ...],
    "text": [string, ...]
  },

  "cleaning_plan": {
    "pandas": {
      "steps": [
        { "coerce_numeric": { "columns": [string,...], "errors": "coerce" } },
        { "parse_datetime": { "columns": [string,...], "infer": true, "errors": "coerce", "dayfirst": false } },
        { "strip_whitespace": { "columns": [string,...] } },
        { "lowercase_text": { "columns": [string,...] } },
        { "standardize_categories": { "column": string, "mapping": { "raw_value": "standard_value", ... }, "case_insensitive": true } },
        { "clip_values": { "columns": [string,...], "min": number | null, "max": number | null } },
        { "winsorize_iqr": { "columns": [string,...], "iqr_multiplier": 1.5 } },
        { "fillna_numeric": { "columns": [string,...], "strategy": "median" | "mean" | "constant", "value": number | null } },
        { "fillna_categorical": { "columns": [string,...], "strategy": "most_frequent" | "constant", "value": string | null } },
        { "drop_duplicates": { "subset": [string,...] | null, "keep": "first" | "last" } },
        { "drop_invalid": { "rules": [ { "column": string, "cmp": "gt" | "gte" | "lt" | "lte" | "eq" | "neq" | "in" | "between", "value": number | string | array | { "min": number | string, "max": number | string } } ] } }
      ]
    },
    "scikit_learn": {
      "preprocess": {
        "numeric": {
          "imputer": { "strategy": "median" | "mean" | "constant", "fill_value": number | null },
          "scaler": "standard" | "minmax" | null
        },
        "categorical": {
          "imputer": { "strategy": "most_frequent" | "constant", "fill_value": string | null },
          "one_hot_encode": true | false,
          "handle_unknown": "ignore" | "error"
        }
      },
      "reports": {
        "missing_rate": true,
        "variance_threshold": { "threshold": 0.0 },
        "category_cardinality": true
      }
    }
  },

  "notes": [
    // implementation hints for the engineer, e.g.:
    // "Apply filter_ast first to create df_filtered; run cleaning_plan on df_filtered."
  ],

  "clarifications": [
    // questions for the user if columns/intent are ambiguous; [] if none
  ]
}

### Filter AST rules
- Use "CMP" for leaf comparisons (field vs value). For ranges, use cmp="between" with value={"min":X,"max":Y}.
- Logical nodes ("AND","OR","NOT") use "children".
- Values should be correctly typed (numbers as numbers, booleans as booleans).
- Map synonyms: not/in/one of/contains/starts with/ends with as per schema.



Expected JSON shape (illustrative):
{
  "filter_ast": {
    "op": "CMP",
    "field": "country",
    "cmp": "in",
    "value": ["US","IN"],
    "children": []
  },
  "targets": {
    "numeric": ["amount","salary"],
    "categorical": ["department","country"],
    "datetime": [],
    "text": ["department"]
  },
  "cleaning_plan": {
    "pandas": {
      "steps": [
        { "coerce_numeric": { "columns": ["amount","salary"], "errors": "coerce" } },
        { "clip_values": { "columns": ["amount"], "min": null, "max": 100000 } },
        { "fillna_numeric": { "columns": ["salary"], "strategy": "median", "value": null } },
        { "strip_whitespace": { "columns": ["department"] } },
        { "lowercase_text": { "columns": ["department"] } },
        { "standardize_categories": { "column": "department", "mapping": { "it":"IT","hr":"HR","fin":"FINANCE" }, "case_insensitive": true } },
        { "drop_duplicates": { "subset": null, "keep": "first" } }
      ]
    },
    "scikit_learn": {
      "preprocess": {
        "numeric": { "imputer": { "strategy": "median", "fill_value": null }, "scaler": "standard" },
        "categorical": { "imputer": { "strategy": "most_frequent", "fill_value": null }, "one_hot_encode": true, "handle_unknown": "ignore" }
      },
      "reports": { "missing_rate": true, "variance_threshold": { "threshold": 0.0 }, "category_cardinality": true }
    }
  },
  "notes": [
    "Apply filter_ast to create df_filtered; run cleaning_plan on df_filtered.",
    "Consider winsorize_iqr for heavy-tailed amount distributions instead of hard clipping."
  ],
  "clarifications": []
}

Now produce ONLY the JSON for the given inputs.
'''

        # self.english_instruction : str =  "[Filter all those records where status is not shipped and (amount greater than 500 or country is US)]"
        self.english_instruction : str =  "[Filter all those records where country in US or IN; coerce amount and salary to numeric; cap amount at 100000; fill missing salary with median; trim and lowercase department; one-hot encode department; fill missing age with constant 18; standard-scale numeric features.]"


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

        res = response.choices[0].message.content
        print(json.dumps(res,indent=2))
        print(res)


if __name__ == '__main__':
    ob : DemoGE = DemoGE()
    ob.execute_api()
