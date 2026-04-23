from g4f.client import Client

client = Client(
base_url="https://api.deepinfra.com/v1/openai",

)

response = client.chat.completions.create(
model="meta-llama/Meta-Llama-3.1-8B-Instruct",
messages=[{"role": "user", "content": "Say hello in one sentence."}],
stream=True,
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)