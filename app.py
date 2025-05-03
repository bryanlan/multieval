import streamlit as st
import requests
import json
import concurrent.futures
import pyperclip
import os # Added for file path handling if needed later

# Attempt to import the API key
try:
    from keys import OPENROUTER_API_KEY
except ImportError:
    st.error("API key not found. Please create a 'keys.py' file with your OPENROUTER_API_KEY variable.")
    OPENROUTER_API_KEY = None # Set to None if import fails

# Predefined Model Lists (Update these with your desired models from OpenRouter)
INFERENCE_MODELS = [
    "microsoft/mai-ds-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-prover-v2",
    "deepseek/deepseek-v3-base:free",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-flash-preview",
    "x-ai/grok-3-mini-beta",
    "openai/gpt-4o-mini",
    "openai/o4-mini",
    "openai/o4-mini-high",
    "openai/o3",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "anthropic/claude-3-opus",
    "mistralai/mistral-large",
]

EVALUATION_MODELS = [
    "openai/o3",
    "openai/o4-mini",
    "google/gemini-2.5-pro-preview-03-25",
    "microsoft/mai-ds-r1:free",
]

# --- Initialize Session State ---
def initialize_session_state():
    """Initializes required keys in Streamlit's session state if they don't exist."""
    defaults = {
        'core_prompt': '',
        'context': '',
        'eval_template': """Evaluate the following response based on the core prompt and context.
Core Prompt: {core_prompt}
Context: {context}
Response(s):
{response_allmodels}

Evaluation Criteria: [Your criteria here - e.g., helpfulness, accuracy, tone]
Rating (1-5):
Justification:""", # Added basic template structure
        'selected_inference_llms': [],
        'selected_eval_llms': [],
        'temperature': 1.0,
        'max_tokens': 4096, # Increased default based on spec
        'inference_results': None,
        'evaluation_results': None,
        'generated_eval_prompt': None,
        # Internal state keys for file uploaders if needed later
        'prompt_set_loader_key': 0,
        'proj_state_loader_key': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Helper function for a single API call ---
def call_openrouter(model_name: str, user_message: str, temperature: float, max_tokens: int, api_key: str) -> tuple[str, str]:
    """
    Makes a single API call to the OpenRouter /chat/completions endpoint.

    Args:
        model_name: The identifier of the model to call (e.g., "openai/gpt-4o").
        user_message: The complete prompt/message to send to the model.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.
        api_key: The OpenRouter API key.

    Returns:
        A tuple containing (model_name, response_content_or_error_message).
    """
    if not api_key:
        return (model_name, "Error: OpenRouter API Key is missing.")

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        # Recommended OpenRouter headers
        "HTTP-Referer": "http://localhost:8501", # Adjust if deployed elsewhere
        "X-Title": "LLM Comparison Tool (Streamlit)", # App name for OpenRouter logs
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Add other OpenRouter parameters here if needed (e.g., top_p, transforms)
    }
    response_text = "" # Initialize for potential use in error reporting

    try:
        # Make the API call with a timeout
        response = requests.post(api_url, headers=headers, json=payload, timeout=180) # 3-minute timeout
        response_text = response.text # Store raw response text for debugging errors

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response
        result_data = response.json()

        # Extract the response content, checking the structure carefully
        if 'choices' in result_data and len(result_data['choices']) > 0 and 'message' in result_data['choices'][0] and 'content' in result_data['choices'][0]['message']:
            response_content = result_data['choices'][0]['message']['content']
            return (model_name, response_content.strip())
        else:
            # Handle cases where the response structure is unexpected but request was 'successful' (status 200)
            error_detail = result_data.get('error', {}).get('message', 'Unexpected response structure')
            return (model_name, f"Error: {error_detail} - Raw Response: {response_text}")

    except requests.exceptions.Timeout:
        # Handle request timeout specifically
        return (model_name, f"Error: API Request Timed Out ({model_name})")
    except requests.exceptions.RequestException as e:
        # Handle other potential request errors (network issues, invalid URL, etc.)
        # Try to extract a more specific error message from the response body if available
        error_detail = f"{e}"
        try:
            error_json = json.loads(response_text)
            if 'error' in error_json and 'message' in error_json['error']:
                 error_detail = error_json['error']['message']
        except (json.JSONDecodeError, TypeError): # Handle cases where response is not JSON or text is empty
             error_detail = response_text if response_text else f"{e}" # Use raw text or original exception
        return (model_name, f"Error: API Request Failed ({model_name}) - {error_detail}")
    except KeyError as e:
         # Handle unexpected structure after successful JSON parsing (e.g., missing 'choices')
        return (model_name, f"Error: Could not parse response structure ({model_name}, KeyError: {e}) - Raw Response: {response_text}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        import traceback
        tb_str = traceback.format_exc()
        return (model_name, f"Error: An unexpected error occurred ({model_name}) - {e}\\nTraceback:\\n{tb_str}")

# --- Helper function for generating the evaluation prompt ---
def generate_eval_prompt(core: str, context: str, template: str, results: list[tuple[str, str]]) -> str | None:
    """
    Generates the evaluation prompt by replacing placeholders in the provided template.

    Args:
        core: The core prompt text.
        context: The context text.
        template: The evaluation template string with placeholders.
        results: A list of tuples, where each tuple is (model_name, inference_response_or_error).

    Returns:
        The formatted evaluation prompt string, or None if generation is not possible.
    """
    if not template or not results:
        # Cannot generate if template or results are missing
        return None

    # Format all inference results into a single block, marking errors
    response_block = ""
    for model, response in results:
        is_error = response.strip().startswith("Error:")
        response_block += f"--- Response from: {model} {'(ERROR)' if is_error else ''} ---\n{response}\n\n"
    response_block = response_block.strip()

    # Combine inputs for different potential placeholders (Handle empty context)
    core_response_allmodels = f"Core Prompt:\n{core}\n\n{response_block}"
    core_context_part = f"Context:\n{context}\n\n" if context else ""
    core_context_response_allmodels = f"Core Prompt:\n{core}\n\n{core_context_part}{response_block}"

    # Perform replacements - Replace more specific placeholders first
    generated_prompt = template
    # Placeholders based on V3 Spec:
    # {core_context_response_allmodels}, {core_response_allmodels}, {response_allmodels}, {core_prompt}, {context}
    if "{core_context_response_allmodels}" in generated_prompt:
         generated_prompt = generated_prompt.replace("{core_context_response_allmodels}", core_context_response_allmodels)
    if "{core_response_allmodels}" in generated_prompt: # Might be redundant if first was used
         generated_prompt = generated_prompt.replace("{core_response_allmodels}", core_response_allmodels)
    if "{response_allmodels}" in generated_prompt:
         generated_prompt = generated_prompt.replace("{response_allmodels}", response_block)
    if "{core_prompt}" in generated_prompt:
         generated_prompt = generated_prompt.replace("{core_prompt}", core)
    if "{context}" in generated_prompt:
         generated_prompt = generated_prompt.replace("{context}", context if context else "[No Context Provided]") # Be explicit

    # Compatibility for simple {response} placeholder (if others aren't used)
    if "{response}" in generated_prompt and "{response_allmodels}" not in template and "{core_response_allmodels}" not in template and "{core_context_response_allmodels}" not in template:
        generated_prompt = generated_prompt.replace("{response}", response_block)

    return generated_prompt.strip()

# --- Helper function to prepare Prompt Set JSON ---
def prepare_prompt_set_json() -> str:
    """Gathers prompt data from session state and returns an indented JSON string."""
    prompt_set_data = {
        "core_prompt": st.session_state.get('core_prompt', ''),
        "context": st.session_state.get('context', ''),
        "eval_template": st.session_state.get('eval_template', '')
    }
    return json.dumps(prompt_set_data, indent=2)

# --- Helper function to prepare Project State JSON ---
def prepare_project_state_json() -> str:
    """Gathers the entire relevant session state and returns an indented JSON string."""
    keys_to_save = [
        'core_prompt', 'context', 'eval_template',
        'selected_inference_llms', 'selected_eval_llms',
        'temperature', 'max_tokens',
        'inference_results', 'evaluation_results', 'generated_eval_prompt'
    ]
    project_state_data = {}
    for key in keys_to_save:
        project_state_data[key] = st.session_state.get(key) # Safely get current value

    return json.dumps(project_state_data, indent=2)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="LLM Comparison Tool",
    layout="wide"
)

st.title("LLM Comparison & Evaluation Tool")

# --- Main Tabs ---
config_tab, output_tab, eval_tab = st.tabs(['Configuration', 'Output', 'Evaluation'])

with config_tab:
    st.header("Configuration")
    # st.write("Configure prompts, models, and parameters here.") # Remove placeholder

    # --- Project State Management ---
    st.subheader("Project State")
    uploaded_state_file = st.file_uploader(
        "Load Project State (.json)",
        type="json",
        key=f"proj_state_loader_{st.session_state.get('proj_state_loader_key', 0)}" # Use key trick
    )
    # --- Load Project State Logic ---
    if uploaded_state_file is not None:
        try:
            loaded_state = json.loads(uploaded_state_file.getvalue().decode("utf-8"))

            # Define keys expected in the state file (mirroring prepare function)
            expected_keys = [
                'core_prompt', 'context', 'eval_template',
                'selected_inference_llms', 'selected_eval_llms',
                'temperature', 'max_tokens',
                'inference_results', 'evaluation_results', 'generated_eval_prompt'
            ]

            # Update session state key by key, using .get() on loaded data
            for key in expected_keys:
                if key in loaded_state:
                    st.session_state[key] = loaded_state[key]
                # else: # Optional: Warn if a key is missing in the loaded file, keeping current value
                #     st.warning(f"Key '{key}' not found in loaded state file, keeping current value.")

            st.toast("Project State loaded successfully!", icon="✅")
            # Increment key to reset file uploader & trigger rerun
            st.session_state.proj_state_loader_key = st.session_state.get('proj_state_loader_key', 0) + 1
            st.rerun()

        except json.JSONDecodeError:
            st.error("Error: Invalid JSON format in the uploaded state file.")
        except Exception as e:
            st.error(f"An unexpected error occurred while loading the project state: {e}")

    # --- Save Project State ---
    st.download_button(
        "Save Project State",
        data=prepare_project_state_json(), # Call helper function
        file_name="project_state.json",
        mime="application/json",
        key="proj_state_saver",
        # disabled=False, # Enable the button
        help="Save the current prompts, selections, parameters, and results to a JSON file."
    )
    st.divider() # Visual separator

    # --- Prompt Set Management ---
    st.subheader("Prompts")
    st.session_state.core_prompt = st.text_area(
        "Core Prompt",
        value=st.session_state.core_prompt, # Read from state
        height=100
    )
    st.session_state.context = st.text_area(
        "Context",
        value=st.session_state.context, # Read from state
        height=200
    )
    st.session_state.eval_template = st.text_area(
        "Evaluation Prompt Template",
        value=st.session_state.eval_template, # Read from state
        height=200,
        help="Use placeholders: {core_prompt}, {context}, {response_allmodels}, {core_response_allmodels}, {core_context_response_allmodels}"
    )

    # --- Load Prompt Set ---
    uploaded_prompt_file = st.file_uploader(
        "Load Prompt Set (.json)",
        type="json",
        key=f"prompt_set_loader_{st.session_state.get('prompt_set_loader_key', 0)}" # Use key trick
    )
    if uploaded_prompt_file is not None:
        try:
            # Read and parse the uploaded JSON file
            prompt_data = json.loads(uploaded_prompt_file.getvalue().decode("utf-8"))

            # Validate and update session state
            required_keys = ["core_prompt", "context", "eval_template"]
            if all(key in prompt_data for key in required_keys):
                st.session_state.core_prompt = prompt_data["core_prompt"]
                st.session_state.context = prompt_data["context"]
                st.session_state.eval_template = prompt_data["eval_template"]
                st.toast("Prompt Set loaded successfully!", icon="✅")
                # Increment key to reset file uploader
                st.session_state.prompt_set_loader_key = st.session_state.get('prompt_set_loader_key', 0) + 1
                st.rerun()
            else:
                st.error(f"Error: Uploaded JSON is missing one or more required keys: {required_keys}")

        except json.JSONDecodeError:
            st.error("Error: Invalid JSON format in the uploaded file.")
        except Exception as e:
            st.error(f"An unexpected error occurred while loading the prompt set: {e}")
        # File uploader widget handles clearing the display name on rerun after key change

    # --- Save Prompt Set ---
    st.download_button(
        "Save Prompt Set",
        data=prepare_prompt_set_json(), # Call helper function here
        file_name="prompt_set.json",
        mime="application/json",
        key="prompt_set_saver",
        # disabled=False, # Enable the button
        help="Save only the Core Prompt, Context, and Evaluation Template to a JSON file."
    )
    st.divider() # Visual separator

    # --- LLM Selection ---
    st.subheader("LLM Selection")
    
    # Callback to handle inference LLM selections
    def on_inference_change():
        # Get selections from the widget's session state value
        st.session_state.selected_inference_llms = st.session_state.ms_inference
    
    # Callback to handle eval LLM selections
    def on_eval_change():
        # Get selections from the widget's session state value
        st.session_state.selected_eval_llms = st.session_state.ms_eval

    # Multiselect for inference LLMs
    st.multiselect(
        "Select Inference LLMs",
        options=INFERENCE_MODELS,
        default=st.session_state.get('selected_inference_llms', []),
        key="ms_inference",
        on_change=on_inference_change,
        # If we don't reuse default, setting it to empty list when none selected
        help="Select one or more LLMs to use for inference."
    )
    
    # Show current selections
    if st.session_state.get('selected_inference_llms'):
        models = st.session_state.selected_inference_llms
        st.caption(f"**Selected inference LLMs:** {len(models)} model(s)")
    
    st.divider()
    
    # Multiselect for evaluation LLMs
    st.multiselect(
        "Select Evaluation LLMs",
        options=EVALUATION_MODELS,
        default=st.session_state.get('selected_eval_llms', []),
        key="ms_eval",
        on_change=on_eval_change,
        help="Select one or more LLMs to use for evaluation."
    )
    
    # Show current selections
    if st.session_state.get('selected_eval_llms'):
        models = st.session_state.selected_eval_llms
        st.caption(f"**Selected evaluation LLMs:** {len(models)} model(s)")
    
    st.divider()

    # --- API Parameters ---
    st.subheader("API Parameters")
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=2.0,
        value=st.session_state.temperature, # Read from state
        step=0.1
    )
    st.session_state.max_tokens = st.number_input(
        "Max Output Tokens",
        min_value=50, # Set a more reasonable minimum
        value=st.session_state.max_tokens, # Read from state
        step=100
    )

with output_tab:
    st.header("Output")
    # st.write("Run inference and view model responses here.") # Remove placeholder

    run_inference_button = st.button("Run Inference", key="run_inference")

    if run_inference_button:
        # --- Input Validation ---
        if not OPENROUTER_API_KEY:
            st.error("OpenRouter API key is missing. Please configure it in keys.py.")
        elif not st.session_state.selected_inference_llms:
            st.warning("Please select at least one inference LLM in the Configuration tab.")
        elif not st.session_state.core_prompt:
             st.warning("Please provide a Core Prompt in the Configuration tab.")
        else:
            st.session_state.inference_results = [] # Clear previous results
            st.session_state.evaluation_results = None # Clear eval results
            st.session_state.generated_eval_prompt = None # Clear generated prompt

            selected_models = st.session_state.selected_inference_llms
            results = []
            futures = []

            # Combine core prompt and context ONCE
            user_message = f"Core Prompt:\n{st.session_state.core_prompt}"
            if st.session_state.context:
                 user_message += f"\n\nContext:\n{st.session_state.context}"

            # --- Run Inference in Parallel ---
            with st.spinner(f"Running inference on {len(selected_models)} model(s)..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit tasks
                    for model_name in selected_models:
                        futures.append(executor.submit(
                            call_openrouter,
                            model_name=model_name,
                            user_message=user_message,
                            temperature=st.session_state.temperature,
                            max_tokens=st.session_state.max_tokens,
                            api_key=OPENROUTER_API_KEY
                        ))

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            # Optional: Update progress more granularly if needed
                            # st.write(f"Received result for: {result[0]}")
                        except Exception as exc:
                            # This catches errors during the future's execution itself,
                            # though call_openrouter should handle most API errors internally.
                            st.error(f'An error occurred getting result: {exc}')
                            # Find which model failed if possible (might be tricky here)
                            # results.append(("Unknown Model", f"Error during future execution: {exc}"))


            # --- Store Results ---
            # Sort results alphabetically by model name for consistent display
            results.sort(key=lambda x: x[0])
            st.session_state.inference_results = results
            st.success(f"Inference complete for {len(results)} model(s).")

            # --- Generate Evaluation Prompt ---
            if results: # Only generate if we got some results (even errors count)
                generated_prompt = generate_eval_prompt(
                    core=st.session_state.core_prompt,
                    context=st.session_state.context,
                    template=st.session_state.eval_template,
                    results=st.session_state.inference_results
                )
                if generated_prompt:
                    st.session_state.generated_eval_prompt = generated_prompt
                    st.info("Evaluation prompt generated based on results.")
                else:
                    st.warning("Could not generate evaluation prompt (missing template or results?).")


            # Rerun to update the display sections immediately
            st.rerun()


    # --- Display Inference Results (outside the button click logic) ---
    st.subheader("Inference Results")
    if 'inference_results' in st.session_state and st.session_state.inference_results:
        # Display results using expanders
        for model, response in st.session_state.inference_results:
            with st.expander(f"Response from: {model}", expanded=True):
                st.markdown(response) # Display response using markdown
                # Add copy button for individual response
                if st.button(f"Copy Response##{model}", key=f"copy_{model}"): # Use unique key per button
                     pyperclip.copy(response)
                     st.toast(f"Copied response from {model}!")

    elif not run_inference_button: # Show info only if button wasn't just pressed
        st.info("Configure settings and click 'Run Inference' to see results.")

with eval_tab:
    st.header("Evaluation")
    st.write("Generate evaluation prompt and view evaluation results here.")
    # We will add components here later

    # --- Recalculate Evaluation Prompt on Tab Render (if results exist) ---
    # This ensures the prompt reflects the current template even if changed after inference
    if st.session_state.get('inference_results'):
        current_generated_prompt = generate_eval_prompt(
            core=st.session_state.core_prompt,
            context=st.session_state.context,
            template=st.session_state.eval_template,
            results=st.session_state.inference_results
        )
        # Update session state only if generation was successful
        if current_generated_prompt is not None:
             st.session_state.generated_eval_prompt = current_generated_prompt
        # Avoid showing warnings here on every render, rely on post-inference warning

    # --- Display Generated Evaluation Prompt ---
    st.subheader("Generated Evaluation Prompt")
    # Read the potentially updated value from session state
    eval_prompt_value = st.session_state.get('generated_eval_prompt', '')
    st.text_area(
        "Evaluation Prompt (Auto-Generated)", value=eval_prompt_value, height=300,
        disabled=True, key='eval_prompt_display',
        help="This prompt is generated based on the template and inference results."
    )

    col1, col2, _ = st.columns([1, 1, 3]) # Create columns for buttons
    with col1:
        copy_eval_button = st.button("Copy Eval Prompt", key='copy_eval', disabled=not eval_prompt_value)
    with col2:
        run_eval_button = st.button(
            "Run Evaluation",
            key='run_eval',
            disabled=(not st.session_state.get('inference_results') or not st.session_state.selected_eval_llms or not eval_prompt_value),
            help="Requires inference results, selected evaluation LLMs, and a generated prompt."
        )

    # --- Logic for Copy Button ---
    if copy_eval_button:
        if eval_prompt_value:
            pyperclip.copy(eval_prompt_value)
            st.toast("Evaluation prompt copied to clipboard!")
        else:
            st.warning("No evaluation prompt generated yet.")


    # --- Evaluation Execution Logic ---
    if run_eval_button:
        # Double-check conditions (though button is disabled, good practice)
        if not st.session_state.selected_eval_llms:
            st.warning("Please select at least one evaluation LLM.")
        elif not st.session_state.generated_eval_prompt:
            st.warning("No evaluation prompt has been generated. Run inference first.")
        elif not OPENROUTER_API_KEY:
             st.error("OpenRouter API key is missing. Please configure it in keys.py.")
        else:
            st.session_state.evaluation_results = [] # Clear previous eval results
            selected_eval_models = st.session_state.selected_eval_llms
            eval_prompt = st.session_state.generated_eval_prompt
            eval_results = []
            eval_futures = []

            # --- Run Evaluation in Parallel ---
            with st.spinner(f"Running evaluation on {len(selected_eval_models)} model(s)..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit tasks
                    for model_name in selected_eval_models:
                        eval_futures.append(executor.submit(
                            call_openrouter, # Reuse the same helper function
                            model_name=model_name,
                            user_message=eval_prompt, # Use the generated prompt
                            temperature=st.session_state.temperature, # Use same temp/tokens for eval
                            max_tokens=st.session_state.max_tokens,
                            api_key=OPENROUTER_API_KEY
                        ))

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(eval_futures):
                        try:
                            result = future.result()
                            eval_results.append(result)
                        except Exception as exc:
                            st.error(f'An error occurred getting evaluation result: {exc}')
                            # results.append(("Unknown Eval Model", f"Error: {exc}"))

            # --- Store Evaluation Results ---
            eval_results.sort(key=lambda x: x[0]) # Sort by model name
            st.session_state.evaluation_results = eval_results
            st.success(f"Evaluation complete for {len(eval_results)} model(s).")
            st.rerun() # Update display


    st.divider()
    st.subheader("Evaluation Results")
    # --- Display Evaluation Results ---
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        # Display results using expanders
        for model, response in st.session_state.evaluation_results:
            with st.expander(f"Evaluation from: {model}", expanded=True):
                st.markdown(response) # Display eval response using markdown
                # Optionally add copy button for eval responses too
                if st.button(f"Copy Eval##{model}", key=f"copy_eval_{model}"):
                     pyperclip.copy(response)
                     st.toast(f"Copied evaluation from {model}!")
    elif not run_eval_button: # Show only if button wasn't just pressed
        st.info("Run evaluation using the generated prompt and selected evaluation LLMs to see results.")

# --- End of App --- 