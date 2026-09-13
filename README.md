# Algorithmic Trading in Financial Markets Using Deep Reinforcement Learning and LSTM

This repository contains the official source code and implementation of my M.Sc. thesis project, which achieved a **Perfect Score of 20/20**. The project introduces **SDQN** (Sequential Double DQN), an advanced framework that integrates Long Short-Term Memory (LSTM) networks with **Double Deep Q-Networks (DDQN)** to optimize automated trading strategies.

## 🚀 Project Overview
Predicting financial time-series data is highly challenging due to non-stationarity and noise. This project leverages **LSTM** layers to extract robust temporal features from historical market indicators (fetched via `pandas-datareader`). These sequential representations are then fed into the proposed **SDQN** agent. By employing a **Double DQN** architecture implemented in **PyTorch**, the model effectively mitigates the overestimation bias common in standard DQN, leading to highly stable and profitable trading decisions (Buy, Sell, Hold).

## 🛠️ Dependencies
The project is built using Python 3.7.4. All required libraries are listed in the `requirements.txt` file:
* **Deep Learning Framework:** PyTorch 1.5.0 & TensorBoard
* **Environment:** OpenAI Gym
* **Data Processing:** Pandas, NumPy, SciPy, Scikit-Learn
* **Data Visualization:** Seaborn, Matplotlib
* **Financial Data & Utilities:** Pandas-datareader, Statsmodels, Requests, TQDM, Tabulate

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

## 📊 Core Architecture (SDQN)
1. **Temporal Feature Extraction:** LSTM networks process sequential financial data to capture long-term dependencies.
2. **Double Q-Learning:** Separates action selection from action evaluation using PyTorch target networks to handle market volatility.
3. **Advanced Visualization:** High-quality analytical plots stored automatically in the `Figures` directory.

## 💻 Usage
Simulating (training and testing) a chosen supported algorithmic trading strategy on a chosen supported stock is performed by running the following command:

```bash
python main.py -strategy STRATEGY -stock STOCK
```

* **STRATEGY:** The name of the trading strategy (e.g., `SDQN`, or default `TDQN`).
* **STOCK:** The ticker name of the stock (by default `Apple`).

The performance of this algorithmic trading policy will be automatically displayed in the terminal, and some graphs will be generated and stored in the folder named `Figures`.

## 📈 Results & Visualizations

Below are the comprehensive performance charts and indicators generated during the training and backtesting phases, stored automatically in the `Figures` directory:

### 1. Trading Execution & Capital Growth
The upper chart displays the historical price asset with executed Long (green triangles) and Short (red triangles) signals. The lower chart illustrates the exponential growth of the portfolio's total capital over the trading horizon, validating the robustness of the strategy.
<p align="center">
  <img src="Figures/price_capital_backtest.png" alt="Price and Capital Backtest" width="650">
</p>

### 2. Action-Value (Q-Values) Dynamics
This chart illustrates the tracking of Q-values for Short and Long actions over time, showcasing how the agent evaluates and updates its action preferences based on market states.
<p align="center">
  <img src="Figures/q_values.png" alt="Q-Values Tracking" width="550">
</p>

### 3. Training Optimization & Reward Convergence
* **Total Rewards:** The agent successfully shifts from negative random exploration to positive reward accumulation, demonstrating excellent learning convergence.
* **Sharpe Ratio Progression:** Monitoring the Sharpe Ratio behavior across training and testing episodes to ensure the policy optimizes risk-adjusted returns.

<p align="center">
  <img src="Figures/total_reward_convergence.png" alt="Total Reward Convergence" width="450" style="display: inline-block; margin-right: 10px;">
  <img src="Figures/sharpe_ratio_episodes.png" alt="Sharpe Ratio per Episode" width="450" style="display: inline-block;">
</p>

### 4. Quantitative Performance Metrics (Train vs. Test)
The tables below present a highly detailed statistical breakdown of the trading policy. Notably, the agent achieved a remarkable **Sharpe Ratio of 2.251** and a **Profitability rate of 71.43%** during the out-of-sample Testing phase, significantly proving its generalization power.

<p align="center">
  <img src="Figures/train_performance_table.jpg" alt="Train Performance Metrics" width="380" style="display: inline-block; margin-right: 20px;">
  <img src="Figures/test_performance_table.jpg" alt="Test Performance Metrics" width="380" style="display: inline-block;">
</p>


## 📬 Contact
* **Developer:** Mohammadreza Ebadi
* **Email:** mohammadreza.ebadi@gmail.com

