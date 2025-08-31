import json
from openai import OpenAI


class DemoGE:

    def __init__(self):
        print("Intialize")
        self.client = OpenAI(
            # api_key=""
        )

        self.command : str = '''
             You are an expert data-profiling assistant.

Goal:
Convert the following plain-English requirements into a single JSON object that:
1) Encodes all FILTER logic as a JSON Abstract Syntax Tree (AST).
2) Specifies a profiling plan compatible with pandas, scikit-learn, and ydata-profiling.
3) Flags any missing/ambiguous columns for clarification.

Conventions:
- Assume the dataset is a pandas DataFrame named df.
- Do NOT invent columns. If the input mentions a column not in the provided list (if any), record it under "clarifications".
- Normalize comparison phrases (e.g., “greater than” → gt, “less than or equal to” → lte, “one of / in” → in, ranges → between).
- Return ONLY valid JSON (UTF-8, no comments, no trailing commas). No extra prose.

### Output JSON Schema (must follow exactly)

{
  "filter_ast": {
    "op": "AND" | "OR" | "NOT" | "CMP",
    "field": string | null,
    "cmp": "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "in" | "between" | "contains" | "starts_with" | "ends_with" | null,
    "value": string | number | boolean | array | { "min": number | string, "max": number | string } | null,
    "children": [ <filter_ast>, ... ]  // only for AND/OR/NOT
  },

  "targets": {
    "numeric": [string, ...],        // columns to treat as numeric (e.g., ["amount","salary"])
    "categorical": [string, ...]     // columns to treat as categorical (e.g., ["department","country"])
  },

  "profiling_plan": {
    "pandas": {
      "steps": [
        "info",                                   // df.info()
        "describe_all",                           // df.describe(include="all", datetime_is_numeric=true)
        "missingness_summary",                    // counts + percentages
        { "value_counts": { "columns": [string,...], "dropna": false } },
        { "outliers": { "columns": [string,...] } }, // Q1,Q3,IQR, upper fence, count
        { "correlations": { "numeric_only": true, "round": 3 } }
      ]
    },
    "scikit_learn": {
      "checks": {
        "identify_types": true,                               // infer numeric/categorical via dtypes
        "missing_rate": true,                                 // per-column %
        "imputers": {
          "numeric_strategy": "median",
          "categorical_strategy": "most_frequent",
          "report_statistics": true                           // print medians/modes learned
        },
        "variance_threshold": { "threshold": 1e-8 },          // report near-constant features
        "standard_scaler_report": true                        // fit on imputed numerics, print mean/std
      }
    },
    "ydata_profiling": {
      "enabled": true,
      "title": "Profiling Report",
      "explorative": true,
      "output_file": "profile.html"
    }
  },

  "notes": [
    // free-form implementation hints for the engineer, e.g.:
    // "Coerce 'amount' and 'salary' using pd.to_numeric(errors='coerce') before stats/outliers."
  ],

  "clarifications": [
    // questions for the user if columns/intent are ambiguous; [] if none
  ]
}

### Filter AST rules
- Use "CMP" for leaf comparisons (field vs value). For ranges, use cmp="between" with value={"min":X,"max":Y}.
- Logical nodes ("AND","OR","NOT") use "children".
- Map synonyms: not/in/one of/contains/starts with/ends with as per schema.
- Values should be typed whenever clear (numbers as numbers, booleans as booleans).


Inputs:
- (Optional) Known columns & dtypes (may be empty): [INSERT_COLUMNS_JSON_OR_EMPTY]
'''

        # self.english_instruction : str =  "[Filter all those records where status is not shipped and (amount greater than 500 or country is US)]"
        self.english_instruction : str =  "[Profile amount and salary columns, find missing value in amount and salary columns, flag outliers in amount, show value counts for department, compute correlations among numeric columns]"


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
        print(json.dumps(res, indent=2))
        print(res)


if __name__ == '__main__':
    ob : DemoGE = DemoGE()
    ob.execute_api()
