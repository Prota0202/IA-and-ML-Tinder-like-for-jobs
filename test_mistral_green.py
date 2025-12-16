# test_mistral_green.py
import time
from green_metrics import tracker

# Simulation d'appel Mistral
mock_mistral_response = {
    "usage": {"prompt_tokens": 500, "completion_tokens": 150} # On simule un gros prompt
}

print("=== Démarrage mesure ===")
tracker.start()

# On simule un travail du PC pendant 2 secondes pour que CodeCarbon ait le temps de mesurer
time.sleep(2) 
# On ajoute des faux tokens pour voir si le Scope 3 s'ajoute bien
tracker.add_mistral_tokens(mock_mistral_response)

tracker.stop()
metrics = tracker.get_report()

print("\n=== RAPPORT FINAL ===")
print(f"💻 Scope 2 (Local BEL):       {metrics['scope2_local']:.8f} kgCO2")
print(f"⚡ Scope 2 (Cloud Élec FR):  {metrics['scope2_cloud_elec']:.8f} kgCO2")
print(f"🏭 Scope 3 (Embodied):       {metrics['scope3_cloud_embodied']:.8f} kgCO2")
print(f"☁️ Scope Cloud Total:       {metrics['scope3_cloud_total']:.8f} kgCO2")
print(f"🌍 TOTAL:                    {metrics['total_co2']:.8f} kgCO2")
print(f"🚗 Équivalent:               {metrics['equiv_km']:.2f} km en voiture")