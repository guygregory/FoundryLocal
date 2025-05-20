# Foundry Local Code Samples
![Foundry Local](https://github.com/user-attachments/assets/03ced8c2-4325-4d52-aefb-2b66c13b1334)


## Overview

**Foundry Local** unlocks instant, on-device AI by allowing users to run generative AI models directly on their local machines, without relying on cloud-based inference. This approach offers enhanced privacy, lower latency, and greater control over AI workloads, empowering developers and organizations to build and experiment with AI locally.

Learn more about Foundry Local from the [official documentation](http://aka.ms/foundry-local-docs) and the [announcment blog from Microsoft Build](http://aka.ms/FoundryLocal).

---

## Quickstart Code Samples

This repository includes quickstart Python samples for using Foundry Local, based on the quickstart guide from Microsoft Learn.

### 1. Basic Quickstart

The `quickstart.py` sample demonstrates how to:

- Download and run a suitable model on your device using an alias.
- Interact with the local Foundry service using the OpenAI Python SDK.

```python
import openai
from foundry_local import FoundryLocalManager

# Use an alias to select the most suitable model
alias = "phi-3.5-mini"

# Start Foundry Local and load the model
manager = FoundryLocalManager(alias)

# Connect OpenAI client to the local Foundry endpoint
client = openai.OpenAI(
    base_url=manager.endpoint,
    api_key=manager.api_key  # Not required for local usage
)

# Generate a response from the local model
response = client.chat.completions.create(
    model=manager.get_model_info(alias).id,
    max_tokens=4096,
    messages=[{"role": "user", "content": "What is the golden ratio?"}]
)
print(response.choices[0].message.content)
```

### 2. Streaming Quickstart

The `quickstart-stream.py` sample shows how to stream responses from the model:

```python
import openai
from foundry_local import FoundryLocalManager

alias = "phi-3.5-mini"
manager = FoundryLocalManager(alias)

client = openai.OpenAI(
    base_url=manager.endpoint,
    api_key=manager.api_key
)

stream = client.chat.completions.create(
    model=manager.get_model_info(alias).id,
    max_tokens=4096,
    messages=[{"role": "user", "content": "What is the golden ratio?"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Gradio UI Example

The `fl-stream-gradio.py` sample provides a Gradio-powered web interface for chatting with Foundry Local models.

**Key Features:**
- Interactive chat UI using Gradio.
- Streams model outputs in real time.
- Maintains chat history for multi-turn conversations.

```python
import openai
from foundry_local import FoundryLocalManager
import gradio as gr

alias = "phi-3.5-mini"
manager = FoundryLocalManager(alias)
client = openai.OpenAI(
    base_url=manager.endpoint,
    api_key=manager.api_key
)
model_id = manager.get_model_info(alias).id

def generate_response(user_prompt: str, history: list[tuple[str, str]]):
    # [Function body elided for brevity. See fl-stream-gradio.py for details.]
    pass

with gr.Blocks() as demo:
    # [UI setup goes here]
    pass

demo.launch()
```

See [`fl-stream-gradio.py`](./fl-stream-gradio.py) for the full implementation.

---

## Resources

- [Official Foundry Local Documentation](http://aka.ms/foundry-local-docs)
- [Microsoft Foundry Local Announcement](http://aka.ms/FoundryLocal)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Gradio Documentation](https://www.gradio.app/)

---

## License

This repository is provided for sample and educational purposes.
