import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import ta  # pip install ta

class MultiStockEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_dict,
        tickers,
        window_size=30,
        initial_cash=1_000_000,
        fee_ratio=0.001,
        slippage=0.005
    ):
        super().__init__()
        self.df_dict = df_dict  # dict of {ticker: df}
        self.tickers = tickers
        self.num_stocks = len(tickers)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.fee_ratio = fee_ratio
        self.slippage = slippage

        # 自动识别所有因子特征（去除非数值型和索引列）
        sample_df = next(iter(self.df_dict.values()))
        self.feature_cols = [c for c in sample_df.columns if c not in ["datetime", "instrument"] and np.issubdtype(sample_df[c].dtype, np.number)]
        self.feature_dims = len(self.feature_cols)

        # 填充缺失值
        for t in tickers:
            df = self.df_dict[t].copy()
            df.fillna(0, inplace=True)
            self.df_dict[t] = df

        self.max_steps = min([len(df) for df in self.df_dict.values()]) - window_size - 1

        # 连续动作空间：每支股票目标持仓比例
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_stocks,), dtype=np.float32
        )

        # Observation空间：每支股票的 window_size * feature_dim + cash_ratio + 当前持仓比例
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_stocks * self.feature_dims + 1 + self.num_stocks),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机选择一个合法的起始时间点
        rng = np.random.default_rng(seed)
        self.current_step = rng.integers(self.window_size, self.max_steps)
        self.cash = self.initial_cash
        self.stocks_held = np.zeros(self.num_stocks, dtype=np.float32)
        self.total_asset = self.cash
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        action = action / (np.sum(action) + 1e-8)  # 归一化总仓位为1

        prices = self._get_current_prices()
        slipped_prices = prices * (1 + self.slippage * np.random.uniform(-1, 1, size=self.num_stocks))

        stock_values = self.stocks_held * prices
        self.total_asset = self.cash + np.sum(stock_values)

        target_values = self.total_asset * action
        delta_values = target_values - stock_values

        # 买入/卖出操作
        for i in range(self.num_stocks):
            price = slipped_prices[i]
            delta = delta_values[i]
            if delta > 0:  # 买入
                max_shares = int((self.cash - self.fee_ratio * delta) / price)
                shares_to_buy = min(int(delta / price), max_shares)
                cost = shares_to_buy * price * (1 + self.fee_ratio)
                if cost <= self.cash:
                    self.stocks_held[i] += shares_to_buy
                    self.cash -= cost
            elif delta < 0:  # 卖出
                shares_to_sell = min(int(-delta / price), int(self.stocks_held[i]))
                proceeds = shares_to_sell * price * (1 - self.fee_ratio)
                self.stocks_held[i] -= shares_to_sell
                self.cash += proceeds

        self.current_step += 1

        new_prices = self._get_current_prices()
        new_total_asset = self.cash + np.sum(self.stocks_held * new_prices)
        reward = np.log(new_total_asset / self.total_asset)
        self.total_asset = new_total_asset

        terminated = self.current_step >= self.max_steps
        if self.total_asset < self.initial_cash * 0.5:
            terminated = True
        self.done = terminated

        return self._get_observation(), reward, terminated, False, {}

    def _get_current_prices(self):
        return np.array([
            self.df_dict[t].loc[self.current_step, "$adjclose"] for t in self.tickers
        ])

    def _get_observation(self):
        import concurrent.futures
        def get_features(t):
            df = self.df_dict[t]
            slice = df.iloc[self.current_step - self.window_size:self.current_step]
            return slice[self.feature_cols].values

        with concurrent.futures.ThreadPoolExecutor() as executor:
            features_list = list(executor.map(get_features, self.tickers))
        features = np.hstack(features_list)

        price_now = self._get_current_prices()
        stock_val = price_now * self.stocks_held
        total_val = self.cash + np.sum(stock_val)
        allocation = stock_val / (total_val + 1e-8)
        cash_ratio = np.array([[self.cash / (total_val + 1e-8)] for _ in range(self.window_size)])
        alloc_matrix = np.tile(allocation, (self.window_size, 1))

        obs = np.concatenate([features, cash_ratio, alloc_matrix], axis=1)
        return obs.astype(np.float32)

    def render(self):
        print(
            f"Step: {self.current_step}, Asset: {self.total_asset:.2f}, "
            f"Cash: {self.cash:.2f}, Holdings: {self.stocks_held}"
        )
