import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# Request model
class ConvertRequest(BaseModel):
    from_language: str
    to_language: str
    code: str


@app.get("/")
def home():
    return {
        "message": "Programming Language Converter API is running!"
    }


@app.post("/convert")
def convert_code(request: ConvertRequest):

    # ==================================================
    # STEP 1: VALIDATE SOURCE LANGUAGE
    # ==================================================

    validation_prompt = f"""
You are a programming language code validator.

The user selected:

Source language: {request.from_language}

Here is the user's code:

{request.code}

Your job is ONLY to determine whether the code appears to belong
to the selected programming language.

IMPORTANT RULES:

1. Determine the language based on syntax, keywords, operators,
   libraries, function declarations, variables, and other clues.

2. Small syntax errors MUST NOT cause the code to be rejected.

   Examples:
   - Missing brackets
   - Missing parentheses
   - Missing colon
   - Missing semicolon
   - Indentation mistakes
   - Misspelled keywords caused by a small typo

   These should still be considered code in that language.

3. Logical errors MUST NOT cause the code to be rejected.

4. Incomplete code MUST NOT automatically be rejected if it clearly
   belongs to the selected language.

5. Do NOT fix the code.

6. Do NOT translate the code.

7. Do NOT explain the code.

8. If the code clearly belongs to another programming language,
   return INVALID.

9. If the code reasonably appears to belong to the selected language,
   return VALID.

10. If the input is clearly not programming code at all,
    return INVALID.

11. Be tolerant of short code snippets if they contain recognizable
    syntax of the selected language.

Return ONLY one of these formats:

VALID

or

INVALID: <short reason>
"""

    validation_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert programming language "
                    "validator."
                )
            },
            {
                "role": "user",
                "content": validation_prompt
            }
        ],
        temperature=0
    )

    validation_result = (
        validation_response.choices[0].message.content.strip()
    )

    # ==================================================
    # STEP 2: HANDLE LANGUAGE MISMATCH
    # ==================================================

    if validation_result.startswith("INVALID"):

        reason = validation_result.replace(
            "INVALID:",
            ""
        ).strip()

        return {
            "error": "language_mismatch",
            "message": (
                f"The entered code does not appear to be "
                f"{request.from_language} code. {reason}"
            )
        }

    # ==================================================
    # STEP 3: CONVERT CODE
    # ==================================================

    prompt = f"""
You are a programming language translator and syntax corrector.

Source language: {request.from_language}
Target language: {request.to_language}

Your task depends on whether the source and target languages
are the same.

RULES:

1. If the source language and target language are DIFFERENT:

   - Translate the code from the source language to the target language.
   - Preserve the original logic and behavior exactly.
   - Preserve functions, variables, conditions, loops, comments,
     input, output, and program flow.
   - Do NOT fix logical errors in the original code.
   - Do NOT add new features.
   - Do NOT remove functionality.
   - Do NOT optimize or improve the code.
   - Do NOT rename variables or functions unless required by the
     target language syntax.
   - Only make changes necessary to express the same code in the
     target language.

2. If the source language and target language are THE SAME:

   - Do NOT translate or rewrite the code.
   - Only fix syntax errors that prevent the code from being valid
     in that language.
   - Do NOT fix logical errors.
   - Do NOT improve the code.
   - Do NOT optimize the code.
   - Do NOT rename variables or functions.
   - Do NOT add new functionality.
   - Keep the original structure and content as unchanged as possible.

3. In both cases:

   - Return ONLY the resulting source code.
   - Do NOT use Markdown code fences.
   - Do NOT provide explanations.
   - Do NOT add any text before or after the code.
   - Make sure the resulting code is syntactically valid for the
     target language.

SOURCE CODE:

{request.code}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert programming language "
                    "converter."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    converted_code = response.choices[0].message.content

    return {
        "converted_code": converted_code
    }
