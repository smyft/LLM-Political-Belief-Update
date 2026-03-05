"""
Script to convert political proposals to corresponding actions using LLM API.

This script:
1. Loads policy proposals from policy_options.json
2. Loads the prompt template from proposal2action.txt
3. Calls the LLM API for each proposal to generate corresponding actions
4. Saves the results to a JSON file

Usage:
    python proposal2action.py              # Process all proposals
    python proposal2action.py --debug     # Process only first 3 proposals for testing
    python proposal2action.py --debug 5   # Process first 5 proposals for testing
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

# Add the parent directory to sys.path so src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.unified_llm_interface import APIInterface


def load_json_file(file_path):
    """Load JSON file and return parsed content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data, file_path):
    """Save data to JSON file with proper formatting."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_prompt_template(file_path):
    """Load the prompt template from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_llm_response(response_content):
    """
    Parse the LLM response to extract JSON object with the new format.
    
    New format:
    {
      "political_proposal": "The original input proposal text",
      "actions": [
        {
          "action_type": "Personal Commitment",
          "action_description": "A concise description..."
        },
        ...
      ]
    }
    
    Handles various formats that the LLM might return.
    """
    # Try to find JSON object in the response
    # First, try direct JSON parsing
    try:
        # Remove markdown code blocks if present
        cleaned_content = response_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
        
        cleaned_content = cleaned_content.strip()
        parsed = json.loads(cleaned_content)
        
        # Validate the expected structure
        if isinstance(parsed, dict) and "political_proposal" in parsed and "actions" in parsed:
            return parsed
        # If it's a list, try to find the object within
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "political_proposal" in item and "actions" in item:
                    return item
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object pattern with keys
    try:
        # Find the first { and last } to extract JSON object
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = response_content[start_idx:end_idx+1]
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "political_proposal" in parsed and "actions" in parsed:
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    # If all parsing fails, return None
    return None


def convert_proposal_to_action(policy_proposal, prompt_template, model, api_interface, debug=False):
    """
    Call LLM to convert a policy proposal to corresponding actions.
    
    Args:
        policy_proposal: The policy proposal text
        prompt_template: The prompt template with {POLICY_PROPOSAL} placeholder
        model: The LLM model to use
        debug: Whether to print debug information
        
    Returns:
        Dictionary containing the proposal and generated actions, or None if failed
    """
    # Prepare the user message by replacing placeholder
    user_message = prompt_template.replace("{POLICY_PROPOSAL}", policy_proposal)
    
    # Create dialogue history
    dialogue_history = [{"role": "user", "content": user_message}]
    
    if debug:
        print(f"    Calling LLM with model: {model}")
    
    response = api_interface.chat(
        dialogue_history,
        temperature=0.7,  # Use some creativity
        seed=42,
        max_tokens=2000,
        show_progress=False
    )[0]

    if "error" in response:
        print(f"  Failed to get response for proposal: {policy_proposal[:50]}...")
        return None
    
    response_content = response.get("generated_text", "")
    
    if debug:
        print(f"    Raw response (first 500 chars): {response_content[:500]}...")
    
    # Parse the JSON response
    parsed_result = parse_llm_response(response_content)
    
    if parsed_result is None:
        print(f"  Warning: Could not parse JSON response for proposal: {policy_proposal[:50]}...")
        print(f"  Raw response: {response_content[:200]}...")
        # Store raw response as fallback
        return {
            "political_proposal": policy_proposal,
            "actions": [],
            "raw_response": response_content,
            "parse_error": True
        }
    
    # Add the original proposal to ensure consistency
    parsed_result["political_proposal"] = policy_proposal
    
    if debug:
        print(f"    Successfully parsed {len(parsed_result.get('actions', []))} actions")
    
    return parsed_result


def process_all_proposals(policy_options, prompt_template, model, output_file, api_interface, batch_delay=1, debug=False, max_items=None):
    """
    Process all policy proposals and convert them to actions.
    
    Args:
        policy_options: Dictionary of policy categories and proposals
        prompt_template: The prompt template
        model: The LLM model to use
        output_file: Path to save the results
        batch_delay: Delay between API calls (seconds)
        debug: Whether to print debug information
        max_items: Maximum number of proposals to process (None = all)
    """
    results = {}
    total_proposals = sum(len(proposals) for proposals in policy_options.values())
    
    if max_items is not None:
        total_proposals = min(total_proposals, max_items)
    
    processed = 0
    
    print(f"Processing {total_proposals} policy proposals...")
    if max_items is not None:
        print(f"(Debug mode: limiting to {max_items} proposals)")
    print(f"Using model: {model}")
    print(f"Results will be saved to: {output_file}")
    print("-" * 50)
    
    for category, proposals in policy_options.items():
        print(f"\nProcessing category: {category} ({len(proposals)} proposals)")
        results[category] = []
        
        for i, proposal in enumerate(proposals):
            # Check if we've reached the max_items limit
            if max_items is not None and processed >= max_items:
                print(f"\nReached max_items limit ({max_items}). Stopping.")
                break
            
            processed += 1
            print(f"  [{processed}/{total_proposals}] Processing proposal {i+1}/{len(proposals)}...")
            
            result = convert_proposal_to_action(
                policy_proposal=proposal,
                prompt_template=prompt_template,
                model=model,
                api_interface=api_interface,
                debug=debug
            )
            
            if result:
                num_actions = len(result.get('actions', []))
                results[category].append(result)
                print(f"    Successfully converted to {num_actions} actions")
            else:
                results[category].append({
                    "political_proposal": proposal,
                    "actions": [],
                    "error": "Failed to get response from LLM"
                })
            
            # Save intermediate results
            save_json_file(results, output_file)
            
            # Add delay between API calls to avoid rate limiting
            if processed < total_proposals:
                time.sleep(batch_delay)
        
        # Check if we've reached the max_items limit after each category
        if max_items is not None and processed >= max_items:
            break
    
    print("\n" + "=" * 50)
    print(f"Processing complete! Results saved to: {output_file}")
    
    return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert political proposals to corresponding actions using LLM API")
    parser.add_argument("--debug", nargs='?', const=3, type=int, 
                        help="Run in debug mode. Optionally specify number of proposals to process (default: 3)")
    parser.add_argument("--model", type=str, default="google/gemini-3-pro-preview",
                        help="LLM model to use (default: google/gemini-3-pro-preview)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    # Determine debug mode and max_items
    debug = args.debug is not None
    max_items = args.debug if debug else None
    
    if debug:
        print(f"DEBUG MODE ENABLED")
        if max_items:
            print(f"Will process up to {max_items} proposals")
        print("-" * 50)


    # Configuration
    DATA_DIR = Path("./data")
    MODEL = "google/gemini-3-pro-preview"
    OUTPUT_FILE = DATA_DIR / "proposal_actions.json"
    BATCH_DELAY = args.delay # Delay between API calls
    
    # File paths
    policy_options_file = DATA_DIR / "policy_options.json"
    prompt_template_file = DATA_DIR / "proposal2action.txt"
    
    # Check if files exist
    if not policy_options_file.exists():
        print(f"Error: Policy options file not found: {policy_options_file}")
        sys.exit(1)
    
    if not prompt_template_file.exists():
        print(f"Error: Prompt template file not found: {prompt_template_file}")
        sys.exit(1)
    
    # Load data
    print("Loading policy options...")
    policy_options = load_json_file(policy_options_file)
    
    print("Loading prompt template...")
    prompt_template = load_prompt_template(prompt_template_file)
    
    # Show some statistics
    total_proposals = sum(len(proposals) for proposals in policy_options.values())
    print(f"Found {len(policy_options)} categories with {total_proposals} total proposals")
    
    if debug:
        print(f"\n[DEBUG] First 3 categories: {list(policy_options.keys())[:3]}")
        for cat in list(policy_options.keys())[:3]:
            print(f"  {cat}: {policy_options[cat][:2]}...")

    # Initialize API interface from src/models
    api_interface = APIInterface(model_name=MODEL)
    
    # Process proposals
    results = process_all_proposals(
        policy_options=policy_options,
        prompt_template=prompt_template,
        model=MODEL,
        output_file=OUTPUT_FILE,
        api_interface=api_interface,
        batch_delay=BATCH_DELAY,
        debug=debug,
        max_items=max_items
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for category, proposals in results.items():
        successful = sum(1 for p in proposals if "error" not in p and "parse_error" not in p)
        print(f"{category}: {successful}/{len(proposals)} proposals processed successfully")
    
    if debug:
        print("\n[DEBUG] Sample output (first category, first proposal):")
        if results:
            first_cat = list(results.keys())[0]
            if results[first_cat]:
                print(json.dumps(results[first_cat][0], indent=2, ensure_ascii=False)[:1000])


if __name__ == "__main__":
    main()
