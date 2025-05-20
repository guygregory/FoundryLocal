import openai
from foundry_local import FoundryLocalManager

# By using an alias, the most suitable model will be downloaded 
# to your end-user's device. 
alias = "phi-3.5-mini"

# Create a FoundryLocalManager instance. This will start the Foundry
# Local service if it is not already running and load the specified model.
manager = FoundryLocalManager(alias)
# The remaining code uses the OpenAI Python SDK to interact with the local model.
# Configure the client to use the local Foundry service
client = openai.OpenAI(
    base_url=manager.endpoint,
    api_key=manager.api_key  # API key is not required for local usage
)
# Set the model to use and generate a response
response = client.chat.completions.create(
    model=manager.get_model_info(alias).id,
    max_tokens=4096,
    messages=[{"role": "user", "content": "What is the golden ratio?"}]
)
print(response.choices[0].message.content)