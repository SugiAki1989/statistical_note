import math
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Rating:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __repr__(self):
        return f"Rating(mu={self.mean:.3f}, sigma={self.std:.3f})"


class GameInfo:
    def __init__(self, beta, dynamics_factor, draw_probability):
        self.beta = beta
        self.dynamics_factor = dynamics_factor
        self.draw_probability = draw_probability


class PairwiseComparison(Enum):
    WIN = 1
    DRAW = 0
    LOSE = -1


def square(x):
    return x * x


def normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def v_t(delta, margin, c):
    t = (delta - margin) / c
    return normal_pdf(t) / normal_cdf(t)


def w_t(delta, margin, c):
    x = (delta - margin) / c
    v = v_t(delta, margin, c)
    return v * (v + x)


# def v_within_margin(delta, margin, c):
#     x1 = (-margin - delta) / c
#     x2 = (margin - delta) / c
#     return (normal_pdf(x1) - normal_pdf(x2)) / (normal_cdf(x2) - normal_cdf(x1))


# def w_within_margin(delta, margin, c):
#     x1 = (-margin - delta) / c
#     x2 = (margin - delta) / c
#     v = v_within_margin(delta, margin, c)
#     return v * v + (
#         (x2 * normal_pdf(x2) - x1 * normal_pdf(x1))
#         / (normal_cdf(x2) - normal_cdf(x1))
#     )


def calculate_new_rating(game, self_rating, opp_rating, comparison):
    logger.info("==============================")
    logger.info("=== TrueSkill Update Start ===")
    logger.info("==============================")

    # ===== メッセージ① prior =====
    logger.info("[1] prior------------------------------------------------------------------")
    logger.info(f"self = {self_rating}")
    logger.info(f"opp  = {opp_rating}")
    logger.info(f"comparison = {comparison}")
    logger.info("")

    # ===== メッセージ② skill → performance =====
    logger.info("[2] performance dist------------------------------------------------------------------")
    logger.info(f"N(μ1 = {self_rating.mean:.2f}, σ1^2 + β^2 = {square(self_rating.std) + square(game.beta):.3f})")
    logger.info(f"N(μ2 = {opp_rating.mean:.2f}, σ2^2 + β^2 = {square(opp_rating.std) + square(game.beta):.3f})")
    logger.info("")

    # ===== メッセージ③ performance difference =====
    if comparison == PairwiseComparison.LOSE:
        mean_delta = opp_rating.mean - self_rating.mean
    else:
        mean_delta = self_rating.mean - opp_rating.mean
    m = self_rating.mean - opp_rating.mean
    c = math.sqrt(
        square(self_rating.std)
        + square(opp_rating.std)
        + 2 * square(game.beta)
    )
    logger.info("[3] performance difference------------------------------------------------------------------")
    logger.info(f"μ1-μ2 = {m:.3f}")
    logger.info(f"√(σ1^2 + σ2^2 + 2β^2)= {c:.3f}")
    logger.info(f"σ1^2 + σ2^2 + 2β^2 = {square(c):.3f}")
    logger.info(f"N(μ1-μ2 = {mean_delta:.3f}, σ1^2 + σ2^2 + 2β^2 = {square(c):.3f})")
    logger.info("")

    # ===== メッセージ④ truncation =====

    draw_margin = 0.0  # 簡略化（DrawProbability 未使用）
    # if comparison == PairwiseComparison.DRAW:
    #     v = v_within_margin(mean_delta, draw_margin, c)
    #     w = w_within_margin(mean_delta, draw_margin, c)
    #     rank_mult = 1
    # else:
    v = v_t(mean_delta, draw_margin, c)
    w = w_t(mean_delta, draw_margin, c)
    rank_mult = comparison.value
    m_star_d = mean_delta + c * v
    v_star_d = square(c) * (1 - w)
    pi_prior = 1/square(c)
    pi_post = 1/v_star_d
    tau_prior = m / square(c)
    tau_post = m_star_d / v_star_d
    pi_msg_tmp = pi_post - pi_prior
    tau_msg_tmp = tau_post - tau_prior
    pi_msg = tau_msg_tmp / pi_msg_tmp
    tau_msg = 1 / pi_msg_tmp

    logger.info("[4] truncation correction------------------------------------------------------------------")
    logger.info(f"v(t) = {v:.3f}, w(t) = {w:.3f}")
    logger.info(f"m^*_d = {m_star_d:.3f}, v^*_d = {v_star_d:.3f}")
    logger.info(f"π_prior = {pi_prior:.3f}, π_post = {pi_post:.3f}")
    logger.info(f"τ_prior = {tau_prior:.3f}, τ_post = {tau_post:.3f}")
    logger.info(f"m_msg_tmp = {pi_msg_tmp:.3f}, τ_msg_tmp = {tau_msg_tmp:.3f}")
    logger.info(f"π_msg = {pi_msg:.3f}, τ_msg = {tau_msg:.3f}")
    logger.info(f"N(π_msg = {pi_msg:.3f}, τ_msg = {tau_msg:.3f})")
    logger.info("")

    # ===== メッセージ⑤ performance update =====
    if comparison == PairwiseComparison.WIN:
        m_p = pi_msg + opp_rating.mean
        v_p = tau_msg + square(opp_rating.std) + square(game.beta)
    else:
        m_p = self_rating.mean - pi_msg
        v_p = tau_msg + square(self_rating.std) + square(game.beta)

    logger.info("[5] performance update------------------------------------------------------------------")
    logger.info(f"μ_p = {m_p:.3f}")
    logger.info(f"σ_p = {v_p:.3f}")
    logger.info(f"N_tmp(μ_p = {m_p:.3f}, σ_p = {v_p:.3f})")
    logger.info(f"N(μ_p = {m_p:.3f}, σ_p = {v_p+square(game.beta):.3f})")
    logger.info("")

    # ===== メッセージ⑥ skill posterior update =====
    m_likelihood = m_p
    v_likelihood = v_p + square(game.beta)
    v2_prime = ((1/square(self_rating.std)) + (1/v_likelihood))**-1
    v_prime = math.sqrt(v2_prime)
    m_prime = v2_prime * ((self_rating.mean/self_rating.std**2) + (m_likelihood / v_likelihood))

    logger.info("[6] skill posterior update------------------------------------------------------------------")
    logger.info(f"N(μ_likelihood = {m_likelihood:.3f}, σ_likelihood = {v_likelihood:.3f})")

    logger.info(f"v2_prime = {v2_prime:.3f}")
    logger.info(f"v_prime = {v_prime:.3f}")
    logger.info(f"m_prime = {m_prime:.3f}")
    logger.info(f"N(μ_updated = {m_prime:.3f}, σ_updated = {v_prime:.3f})")
    logger.info("============================")
    logger.info("=== TrueSkill Update End ===")
    logger.info("============================")

    return Rating(m_prime, v_prime)
