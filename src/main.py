from retriever import Retriever
from generator import generate_answer


retriever = Retriever()

question = input("\nAsk a question: ")

results = retriever.retrieve(
    question,
    k=3
)


# --------------------------------
# Check retrieval
# --------------------------------

if not results:

    print("\nI couldn't find relevant information in the document.")

else:

    print("\n===== RETRIEVED SOURCES =====")

    for result in results:

        print(
            f"Page {result['page_number']} "
            f"(Distance: {result['distance']:.3f})"
        )


    # --------------------------------
    # Generate answer
    # --------------------------------

    answer = generate_answer(
        question,
        results
    )


    print("\n===== ANSWER =====")
    print(answer)


    # --------------------------------
    # Sources
    # --------------------------------

    print("\n===== SOURCES =====")

    pages = sorted(
        set(
            result["page_number"]
            for result in results
        )
    )

    for page in pages:
        print(f"📄 Page {page}")