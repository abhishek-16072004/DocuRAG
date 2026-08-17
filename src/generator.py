from groq import Groq
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )

client = Groq(api_key=api_key)


def generate_answer(question, retrieved_chunks):

    # --------------------------------
    # Create context from chunks
    # --------------------------------

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"Source: Page {chunk['page_number']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)


    # --------------------------------
    # Create prompt
    # --------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using the retrieved document context.

Rules:

1. Carefully read ALL the provided context before answering.

2. If the context directly answers the question,
   answer directly.

3. You may summarize or combine information
   from multiple retrieved chunks.

4. Do not invent facts that are not supported
   by the context.

5. If the document provides implementation details
   rather than a formal definition, explain what
   can reasonably be understood from those details.

6. Only say:
   "I couldn't find this information in the document."
   when the retrieved context genuinely contains
   no useful information.

7. Keep the answer concise and clear.


DOCUMENT CONTEXT:
-----------------
{context}
-----------------


USER QUESTION:
{question}


ANSWER:
"""


    # --------------------------------
    # Call LLM
    # --------------------------------

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using "
                    "retrieved document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )


    # --------------------------------
    # Return answer
    # --------------------------------

    return response.choices[0].message.content