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
Below are the key performance charts generated during the backtesting phase and saved in the `Figures` folder:

### 1. Cumulative Returns (Equity Curve)
This chart illustrates the portfolio value growth of the SDQN agent compared to standard baselines over the trading horizon.
<p align="center">
  <img src="Figures/equity_curve.png" alt="Equity Curve" width="600">
</p>

### 2. SDQN Training Reward Convergence
This plot displays the total reward convergence per episode, demonstrating the training stability achieved by the Double DQN architecture.
<p align="center">
  <img src="Figures/training_reward.png" alt="Training Rewards" width="600">
</p>

## 📬 Contact
* **Developer:** Mohammadreza Ebadi
* **Email:** mohammadreza.ebadi@gmail.com

