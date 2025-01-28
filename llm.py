#llm.py

import google.generativeai as genai
import logging
import config

# Configure the Gemini API key
genai.configure(api_key=config.gemini_key)

def analyze_market_with_external_data(prompt):
    """
    Analyze market data using Google's Gemini Pro 1.5 LLM with structured prompts.
    """
    try:
        # Define the Gemini model with a system instruction
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction="You are an expert forex trading analyst. Provide actionable insights and concise explanations."
        )
        
        # Configure generation settings
        generation_config = genai.GenerationConfig(
            max_output_tokens=10000,  # Limit the response length
            temperature=0.7  # Balance creativity with relevance
        )
        
        # Generate content using the prompt and the specified configuration
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Return the generated text content
        return response.text
    
    except Exception as e:
        logging.error(f"Error in Gemini Pro analysis: {e}")
        return "Error in analysis."