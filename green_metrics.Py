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
        
        total_tokens = self.total_input_tokens + self.total_output_tokens
        
        # Nouveau calcul réaliste
        # Scope 2 (Électricité Cloud - Mistral France)
        kwh_per_1000_tokens = 0.0004
        carbon_intensity_france = 0.0595  # kgCO2/kWh (Moyenne France car Mistral hébergé en France)
        co2_elec = (total_tokens / 1000) * kwh_per_1000_tokens * carbon_intensity_france

        # Scope 3 (Matériel/Embodied Carbon)
        # Large2 utilise 1,14g pour 400 tokens -> 0,00285g par token 
        # Large2 est 5 fois plus puissant que small -> on divise par 5
        embodied_carbon_per_1000_tokens = ((1.14 / 5) * 1000 / 400) / 1000  # conversion en kgCO2
        co2_embodied = (total_tokens / 1000) * embodied_carbon_per_1000_tokens
        
        print(f"[GreenTracker] Co2 élec cloud: {co2_elec:.6f} kg, Co2 embodied: {co2_embodied:.6f} kg")
        
        # Total Cloud (élec + embodied)
        co2_cloud_total = co2_elec + co2_embodied
        
        # Total final (Local PC + Cloud)
        total_co2_kg = self.last_emissions_local + co2_cloud_total
        
        return {
            "scope2_local": self.last_emissions_local,
            "scope2_cloud_elec": co2_elec,
            "scope3_cloud_embodied": co2_embodied,
            "scope3_cloud_total": co2_cloud_total,
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