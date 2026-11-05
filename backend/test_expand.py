import rag

try:
    resp = rag.client.chat.completions.create(
        model=rag.MODEL,
        messages=[
            {"role": "system", "content": "Convert the question into 8-12 code keywords, separated by spaces. Output only the keywords."},
            {"role": "user", "content": "What endpoints does the API expose?"},
        ],
        temperature=0,
        max_tokens=60,
    )
    print("MODEL:", rag.MODEL)
    print("CONTENT:", repr(resp.choices[0].message.content))
    print("FINISH REASON:", resp.choices[0].finish_reason)
except Exception as e:
    print("ERROR:", type(e).__name__, e)