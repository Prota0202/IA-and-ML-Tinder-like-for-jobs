import logging
from codecarbon import EmissionsTracker

logging.getLogger("codecarbon").setLevel(logging.WARNING)

class GreenTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.tracker = None
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.total_emissions_kg = 0.0
        self._initialized = True

    def init_tracker(self):
        """Initialise le tracker CodeCarbon (sans paramètres obsolètes)"""
        try:
            # Ne passe pas de paramètres qui n'existent plus
            self.tracker = EmissionsTracker(measure_power_secs=1)
            self.tracker.start()
        except Exception as e:
            print(f"[GreenTracker] Erreur init CodeCarbon: {e}")
            self.tracker = None

    def start(self):
        """Démarre le tracking"""
        if self.tracker is None:
            self.init_tracker()
        if self.tracker:
            try:
                self.tracker.start()
            except Exception:
                pass

    def stop(self):
        """Arrête et log les émissions"""
        if self.tracker:
            try:
                emissions = self.tracker.stop()
                if emissions and emissions > 0:
                    self.total_emissions_kg += emissions / 1000  # g -> kg
                    print(f"[GreenTracker] Émissions: {emissions:.2f}g CO2")
            except Exception as e:
                print(f"[GreenTracker] Erreur stop: {e}")

    def add_mistral_tokens(self, input_tokens: int, output_tokens: int):
        """Log les tokens consommés par un appel Mistral"""
        self.total_tokens_input += input_tokens
        self.total_tokens_output += output_tokens
        print(f"[GreenTracker] Mistral tokens: {input_tokens} in, {output_tokens} out (total: {self.total_tokens_input + self.total_tokens_output})")

    def get_report(self) -> dict:
        """Retourne un rapport des émissions et tokens"""
        return {
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_tokens": self.total_tokens_input + self.total_tokens_output,
            "total_emissions_kg": round(self.total_emissions_kg, 6),
        }

tracker = GreenTracker()