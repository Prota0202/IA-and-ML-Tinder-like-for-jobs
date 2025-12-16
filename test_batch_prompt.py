#!/usr/bin/env python3
import sys, json
sys.path.insert(0, '.')

from CV import call_mistral_generic

# Test simple batch prompt
profile = {'localisation': {'ville': 'Bruxelles'}, 'experience_years': 5}
offers_sample = [
    {'titreoffre': 'Python Developer', 'localisation': 'Bruxelles'},
    {'titreoffre': 'Data Scientist', 'localisation': 'Liège'},
]

profile_line = f"Candidat: {profile['localisation']['ville']}, {profile['experience_years']}y exp."
offers_block = 'Offres:\n' + '\n'.join([f"{i+1}. {o['titreoffre']}" for i, o in enumerate(offers_sample)])

prompt = (
    f"{profile_line}\n{offers_block}\n\n"
    "Rate each offer 0-1 (0=bad, 1=excellent).\n"
    "RESPOND WITH ONLY: [0.85, 0.42, ...]\n"
    "NO TEXT, NO MARKDOWN. JUST NUMBERS IN BRACKETS."
)

print('=== PROMPT ===')
print(prompt)
print('\n=== CALLING MISTRAL ===')
response = call_mistral_generic(prompt)
print('=== RESPONSE ===')
print(repr(response))

# Try to extract JSON
response_text = str(response).strip()
response_text = response_text.replace("```json", "").replace("```", "").strip()
start_idx = response_text.find("[")
end_idx = response_text.rfind("]")
if start_idx != -1 and end_idx != -1:
    json_str = response_text[start_idx:end_idx+1]
    print(f"\n=== JSON FOUND ===\n{json_str}")
    try:
        scores = json.loads(json_str)
        print(f"Parsed: {scores}")
    except Exception as e:
        print(f"JSON parse error: {e}")
else:
    print("\n=== NO JSON FOUND ===")

