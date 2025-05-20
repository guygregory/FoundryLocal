import openai
from foundry_local import FoundryLocalManager
import gradio as gr

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------

alias = "phi-3.5-mini"

# Create a FoundryLocalManager instance. This will start the Foundry Local
# service if it is not already running and load the specified model.
manager = FoundryLocalManager(alias)

# Configure the client to use the local Foundry service
client = openai.OpenAI(
    base_url=manager.endpoint,
    api_key=manager.api_key  # API key is not required for local usage
)

# Grab the concrete model ID once so we don't have to look it up on every call
model_id = manager.get_model_info(alias).id

# -----------------------------------------------------------------------------
# Gradio front‑end
# -----------------------------------------------------------------------------

def generate_response(user_prompt: str):
    """Stream tokens from the local model to the Gradio output box.

    During streaming:
      • clear the user's prompt
      • disable Submit and Clear buttons
    When streaming completes, re-enable the buttons.
    """
    partial_answer = ""
    first_chunk = True

    # Immediately clear input and disable buttons
    yield (
        gr.update(value=""),                 # output_box (keep empty for now)
        gr.update(value=""),                 # user_input cleared
        gr.update(interactive=False),        # submit_btn disabled
        gr.update(interactive=False),        # clear_btn  disabled
    )

    # Create a streaming completion request
    stream = client.chat.completions.create(
        model=model_id,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_prompt}],
        stream=True,
    )

    # Stream tokens to the output box
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            content = delta.content.lstrip() if first_chunk else delta.content
            first_chunk = False
            partial_answer += content

            # Keep buttons disabled while streaming
            yield (
                partial_answer,
                gr.update(value=""),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

    # Streaming finished – re-enable buttons
    yield (
        partial_answer,
        gr.update(value=""),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )

# Helper to clear both boxes
def clear_fields():
    """Return empty values for Assistant output and user input."""
    return "", ""

with gr.Blocks(
    fill_height=True,
    fill_width=True,
    # Add CSS to make the textbox a flex child that can scroll vertically.
    css="""
    .scrollable-output { flex: 1 1 auto; min-height: 0; }
    .scrollable-output textarea { overflow-y: auto; }
    """
) as demo:
    # Large output box at the top (read-only)
    output_box = gr.Textbox(
        label="Assistant",
        lines=20,
        interactive=False,
        placeholder="The model's response will appear here, streamed live…",
        elem_classes="scrollable-output"  # class used in CSS above
    )

    # Small input box with a submit button below
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Your message", lines=1, scale=4)
            with gr.Row():
                submit_btn = gr.Button("Submit", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

    # Wire up both the button click and the ⏎ keypress in the textbox
    submit_btn.click(
        fn=generate_response,
        inputs=user_input,
        outputs=[output_box, user_input, submit_btn, clear_btn],  # updated outputs
    )
    user_input.submit(
        fn=generate_response,
        inputs=user_input,
        outputs=[output_box, user_input, submit_btn, clear_btn],  # updated outputs
    )

    # Clear button resets both boxes
    clear_btn.click(
        fn=clear_fields,
        inputs=None,
        outputs=[output_box, user_input],
    )

# -----------------------------------------------------------------------------
# Run the app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue()  # enable streaming
    demo.launch()
