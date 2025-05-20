import openai
from foundry_local import FoundryLocalManager
import gradio as gr

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------

# By using an alias, the most suitable model will be downloaded to your
# end‑user's device.
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

def generate_response(user_prompt: str, history: list[tuple[str, str]]):
    """Stream tokens from the local model to the Gradio output box.

    During streaming:
      • clear the user's prompt
      • disable Submit and Clear buttons
    When streaming completes, re-enable the buttons.
    """
    partial_answer, first_chunk = "", True

    # build a complete messages list with explicit speaker labels
    messages = []
    for user_turn, assistant_turn in history:
        messages.append({"role": "user", "content": f"User: {user_turn}"})
        if assistant_turn:                       # skip empty assistant slots
            messages.append({"role": "assistant", "content": f"Assistant: {assistant_turn}"})

    # add current prompt with label
    messages.append({"role": "user", "content": f"User: {user_prompt}"})

    # append user prompt with empty assistant reply slot
    history.append((user_prompt, ""))

    # immediately clear input & disable buttons
    yield (
        history,                      # Chatbot content
        gr.update(value=""),          # user_input
        gr.update(interactive=False), # submit_btn
        gr.update(interactive=False), # clear_btn
        history                       # state
    )

    # Create a streaming completion request with full history
    stream = client.chat.completions.create(
        model=model_id,
        max_tokens=4096,
        messages=messages,          # CHANGED: send entire conversation
        stream=True,
    )

    # Stream tokens to the output box
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            content = delta.content.lstrip() if first_chunk else delta.content
            first_chunk = False
            partial_answer += content
            history[-1] = (user_prompt, partial_answer)

            yield (
                history,
                gr.update(value=""),
                gr.update(interactive=False),
                gr.update(interactive=False),
                history
            )

    # finish streaming – re-enable buttons
    yield (
        history,
        gr.update(value=""),
        gr.update(interactive=True),
        gr.update(interactive=True),
        history
    )

# NEW: helper to clear both boxes
def clear_fields():
    """Clear both boxes and reset history."""
    return [], "", []   # Chatbot empty, user_input empty, history reset

with gr.Blocks(
    fill_height=True,
    fill_width=True,
    # Add CSS to make the textbox a flex child that can scroll vertically.
    css="""
    .scrollable-chatbot {
        flex: 1 1 auto;
        min-height: 0;
        overflow-y: auto;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
    }
    """,
    title=f"Foundry Local - {alias}"      # NEW: page title
) as demo:
    gr.Markdown(f"""<h2 style="text-align: center;">Foundry Local - {alias}</h2>""")
    chat_state = gr.State([])  # NEW

    # Chatbot replaces Textbox
    output_box = gr.Chatbot(
        label="Assistant",
        elem_classes="scrollable-chatbot"
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
        inputs=[user_input, chat_state],
        outputs=[output_box, user_input, submit_btn, clear_btn, chat_state],
    )
    user_input.submit(
        fn=generate_response,
        inputs=[user_input, chat_state],
        outputs=[output_box, user_input, submit_btn, clear_btn, chat_state],
    )

    # NEW: clear button resets both boxes
    clear_btn.click(
        fn=clear_fields,
        inputs=None,
        outputs=[output_box, user_input, chat_state],
    )

# -----------------------------------------------------------------------------
# Run the app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue()  # enable streaming
    demo.launch()
