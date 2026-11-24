"""
Pricing d'options par Monte Carlo et formule analytique de Black-Scholes.

Trois blocs :
1. Processus stochastiques (mouvement brownien géométrique, diffusion à sauts de Merton)
2. Produits dérivés et pricers (Call européen, pricer Monte Carlo, pricer analytique BS)
3. Exemple d'utilisation et tests
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.stats as si


# =============================================================================
# 1. PROCESSUS STOCHASTIQUES
# =============================================================================

class StochasticProcess(ABC):
    """Interface commune à tous les processus de diffusion."""

    @abstractmethod
    def simulate(self, T: float, N_steps: int, N_paths: int) -> np.ndarray:
        """Simule N_paths trajectoires sur [0, T] avec N_steps pas.

        Retourne un tableau de forme (N_paths, N_steps + 1), la première
        colonne étant la valeur initiale S0.
        """
        pass


class GeometricBrownianMotion(StochasticProcess):
    """Mouvement brownien géométrique : dS = mu*S dt + sigma*S dW."""

    def __init__(self, S0: float, mu: float, sigma: float):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T: float, N_steps: int, N_paths: int) -> np.ndarray:
        dt = T / N_steps
        dW = np.random.normal(0, np.sqrt(dt), (N_paths, N_steps))
        W = np.cumsum(dW, axis=1)
        time_grid = np.linspace(dt, T, N_steps)
        drift_term = (self.mu - 0.5 * self.sigma**2) * time_grid
        S = self.S0 * np.exp(drift_term + self.sigma * W)
        return np.hstack((np.full((N_paths, 1), self.S0), S))


class MertonJumpDiffusion(StochasticProcess):
    """Diffusion à sauts de Merton : GBM + sauts log-normaux d'intensité lambd."""

    def __init__(self, S0: float, mu: float, sigma: float,
                 lambd: float, jump_mu: float, jump_sigma: float):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lambd = lambd
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma

    def simulate(self, T: float, N_steps: int, N_paths: int) -> np.ndarray:
        dt = T / N_steps
        time_grid = np.linspace(dt, T, N_steps)

        # Partie diffusive
        dW = np.random.normal(0, np.sqrt(dt), (N_paths, N_steps))
        W = np.cumsum(dW, axis=1)

        # Partie sauts : nombre de sauts par pas (Poisson) x taille (normale)
        N_jumps = np.random.poisson(self.lambd * dt, (N_paths, N_steps))
        jump_sizes = np.random.normal(self.jump_mu, self.jump_sigma,
                                      (N_paths, N_steps))
        J = np.cumsum(N_jumps * jump_sizes, axis=1)

        # Compensateur pour garder la martingale sous la mesure risque-neutre
        k = np.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1
        compensator = self.lambd * k * time_grid

        drift_term = (self.mu - 0.5 * self.sigma**2) * time_grid - compensator
        S = self.S0 * np.exp(drift_term + self.sigma * W + J)
        return np.hstack((np.full((N_paths, 1), self.S0), S))


# =============================================================================
# 2. PRODUITS DÉRIVÉS ET PRICING
# =============================================================================

class Option(ABC):
    """Produit dérivé défini par un strike et une maturité."""

    def __init__(self, strike: float, maturity: float):
        self.strike = strike
        self.maturity = maturity

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Payoff à maturité pour chaque trajectoire simulée."""
        pass


class EuropeanCall(Option):
    """Call européen : max(S_T - K, 0)."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        return np.maximum(S_T - self.strike, 0.0)


class Pricer(ABC):
    """Interface commune aux méthodes de valorisation."""

    @abstractmethod
    def price(self, option: Option) -> float:
        pass


class MonteCarloPricer(Pricer):
    """Valorisation par espérance actualisée du payoff sur trajectoires simulées."""

    def __init__(self, process: StochasticProcess, r: float,
                 N_steps: int = 252, N_paths: int = 100000):
        self.process = process
        self.r = r
        self.N_steps = N_steps
        self.N_paths = N_paths

    def price(self, option: Option) -> float:
        paths = self.process.simulate(option.maturity, self.N_steps, self.N_paths)
        payoffs = option.payoff(paths)
        discount_factor = np.exp(-self.r * option.maturity)
        return discount_factor * np.mean(payoffs)


class AnalyticalBlackScholesPricer(Pricer):
    """Formule fermée de Black-Scholes (call européen uniquement)."""

    def __init__(self, S0: float, r: float, sigma: float):
        self.S0 = S0
        self.r = r
        self.sigma = sigma

    def price(self, option: Option) -> float:
        if isinstance(option, EuropeanCall):
            d1 = (np.log(self.S0 / option.strike) +
                  (self.r + 0.5 * self.sigma**2) * option.maturity) / \
                 (self.sigma * np.sqrt(option.maturity))
            d2 = d1 - self.sigma * np.sqrt(option.maturity)
            call_price = (self.S0 * si.norm.cdf(d1, 0.0, 1.0) -
                          option.strike * np.exp(-self.r * option.maturity) *
                          si.norm.cdf(d2, 0.0, 1.0))
            return call_price
        raise NotImplementedError("Formule non implémentée pour ce produit.")


# =============================================================================
# 3. UTILISATION ET TESTS
# =============================================================================

if __name__ == "__main__":
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    call_option = EuropeanCall(strike=K, maturity=T)

    # --- Black-Scholes : analytique vs Monte Carlo ---
    gbm = GeometricBrownianMotion(S0=S0, mu=r, sigma=sigma)
    mc_pricer = MonteCarloPricer(process=gbm, r=r, N_steps=252, N_paths=100000)
    bs_pricer = AnalyticalBlackScholesPricer(S0=S0, r=r, sigma=sigma)

    price_mc = mc_pricer.price(call_option)
    price_bs = bs_pricer.price(call_option)

    print("--- Modèle de Black-Scholes ---")
    print(f"Prix Call (Analytique) : {price_bs:.4f}")
    print(f"Prix Call (Monte Carlo): {price_mc:.4f}")

    # --- Modèle à sauts de Merton ---
    mjd = MertonJumpDiffusion(S0=S0, mu=r, sigma=sigma,
                              lambd=1.0, jump_mu=-0.1, jump_sigma=0.1)
    mc_pricer_jump = MonteCarloPricer(process=mjd, r=r, N_steps=252, N_paths=100000)
    price_jump = mc_pricer_jump.price(call_option)

    print("\n--- Modèle à Sauts (Merton) ---")
    print(f"Prix Call (Monte Carlo): {price_jump:.4f}")
    print("Noter l'impact de l'asymétrie induite par les sauts sur le prix.")
