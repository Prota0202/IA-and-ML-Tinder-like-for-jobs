from zamzar import ZamzarClient
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZAMZAR_API_KEY")
if not api_key:
    raise ValueError("Pas de clé API trouvé")

zamzar = ZamzarClient(api_key)

input_pdf = "Labos_sécurités___PI2C_championship_vulnerability.pdf"      # PDF d'entrée
target_format = "txt"             # Format de sortie
output_folder = "."               # Dossier de sortie (ici: le même dossier)
output_filename = "labosecu.txt"

job = zamzar.convert(input_pdf, target_format)
job.store(output_folder).delete_all_files()

print(f"Conversion terminée. Le fichier texte est dans {output_folder}/{output_filename}")