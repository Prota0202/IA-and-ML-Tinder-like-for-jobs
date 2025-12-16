from codecarbon import OfflineEmissionsTracker # <--- CHANGEMENT ICI (Offline)
import logging

# On réduit le bruit dans la console
logging.getLogger("codecarbon").setLevel(logging.WARNING)

class GreenTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GreenTracker, cls).__new__(cls)
            cls._instance.init_tracker()
        return cls._instance

    def init_tracker(self):
        # --- SCOPE 2 : L'électricité de ton PC ---
        # On utilise OfflineEmissionsTracker pour forcer la Belgique sans erreur
        try:
            self.tracker = OfflineEmissionsTracker(
                project_name="JobMatcher_App",
                country_iso_code="BEL", # Force le mix énergétique belge (~0.150 kg/kWh)
                output_file="emissions.csv",
                on_csv_write="append",
                save_to_file=True,
                log_level="warning"
            )
        except Exception as e:
            print(f"⚠️ Erreur CodeCarbon, passage en mode défaut: {e}")
            # Fallback au cas où, mais Offline devrait marcher
            from codecarbon import EmissionsTracker
            self.tracker = EmissionsTracker(output_dir=".")

        # --- SCOPE 3 : Les Tokens (Cloud) ---
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_emissions_local = 0.0

    def start(self):
        self.tracker.start()

    def stop(self):
        # Arrête le compteur local
        self.last_emissions_local = self.tracker.stop()
        return self.last_emissions_local

    def add_mistral_tokens(self, response):
        """Récupère automatiquement les tokens Mistral (dict avec clé 'usage')"""
        try:
            if isinstance(response, dict) and 'usage' in response:
                usage = response['usage']
                in_tok = usage.get('prompt_tokens', 0)
                out_tok = usage.get('completion_tokens', 0)
                self.total_input_tokens += in_tok
                self.total_output_tokens += out_tok
                print(f"[GreenTracker] Mistral tokens: {in_tok} in, {out_tok} out")
        except Exception as e:
            print(f"⚠️ Impossible de compter les tokens Mistral: {e}")

    def get_report(self):
        """Calcule le TOTAL (Local + Cloud)"""
        
        # 1. Calcul Scope 3 (Cloud / Tokens)
        # Formule académique : 0.0004 kWh pour 1000 tokens
        kwh_per_1000_tokens = 0.0004
        carbon_intensity_cloud = 0.475 # kgCO2/kWh (Moyenne mondiale)
        
        total_tokens = self.total_input_tokens + self.total_output_tokens
        energy_llm_kwh = (total_tokens / 1000) * kwh_per_1000_tokens
        emissions_llm_kg = energy_llm_kwh * carbon_intensity_cloud
        
        # 2. Total
        total_co2_kg = self.last_emissions_local + emissions_llm_kg
        
        return {
            "scope2_local": self.last_emissions_local,
            "scope3_cloud": emissions_llm_kg,
            "total_tokens": total_tokens,
            "total_tokens_input": self.total_input_tokens,
            "total_tokens_output": self.total_output_tokens,
            "total_co2": total_co2_kg,
            "equiv_km": total_co2_kg / 0.12
        }

    def reset(self):
        """Réinitialise les compteurs pour une nouvelle session"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_emissions_local = 0.0
        print("[GreenTracker] Compteurs réinitialisés")

tracker = GreenTracker()