import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.stats as stats

# 设置Tushare token (需要在Tushare官网注册获取)
ts.set_token('token')
pro = ts.pro_api()

# 设置matplotlib字体为SimHei，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AlphaFactors:
    def __init__(self, stock_list, start_date, end_date, freq='D'):
        """
        初始化Alpha因子计算类

        参数:
            stock_list: 股票列表，如 ['000001.SZ', '600000.SH']
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            freq: 数据频率，'D'表示日线，'W'表示周线
        """
        self.stock_list = stock_list
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.data = {}  # 存储获取的原始数据
        self.factors = {}  # 存储计算好的因子数据
        self.evaluation = {}  # 存储评估指标

        # 因子类型映射
        self.factor_types = {
            'open_close_ratio': {'type': '价格', 'frequency': '高频', 'direction': '多空'},
            'rsi': {'type': '技术', 'frequency': '中频', 'direction': '多空'},
            'volatility': {'type': '风险', 'frequency': '中频', 'direction': '多空'},
            'volume_change': {'type': '量能', 'frequency': '中频', 'direction': '多空'},
            'macd': {'type': '技术', 'frequency': '中频', 'direction': '多空'}
        }

        # 因子来源映射
        self.factor_sources = {
            'open_close_ratio': '开盘价与收盘价的比率，反映日内价格变动',
            'rsi': '相对强弱指标，衡量价格变动速度和幅度',
            'volatility': '价格波动率，衡量价格波动程度',
            'volume_change': '交易量变化率，反映市场参与度变化',
            'macd': '指数平滑异同移动平均线，用于判断股票买卖信号'
        }

    def get_stock_data(self):
        """获取股票交易数据"""
        for stock in self.stock_list:
            try:
                # 获取日线行情数据
                df_daily = pro.daily(ts_code=stock, start_date=self.start_date, end_date=self.end_date)
                df_daily = df_daily.sort_values('trade_date')

                # 获取行业数据
                df_industry = pro.stock_basic(exchange='', list_status='L',
                                              fields='ts_code,industry')
                industry = df_industry[df_industry['ts_code'] == stock]['industry'].values[
                    0] if not df_industry.empty else '未知行业'

                # 检查关键列是否存在
                required_daily_columns = ['trade_date', 'open', 'high', 'low', 'close', 'vol']
                missing_daily = [col for col in required_daily_columns if col not in df_daily.columns]
                if missing_daily:
                    print(f"警告: {stock} 的日线数据缺少列: {', '.join(missing_daily)}")

                # 存储数据
                self.data[stock] = {
                    'daily': df_daily,
                    'industry': industry
                }

                print(f"成功获取 {stock} ({industry}) 的数据")
            except Exception as e:
                print(f"获取 {stock} 的数据时出错: {e}")

    def calculate_open_close_ratio(self):
        """计算开盘价/收盘价比率因子"""
        for stock in self.stock_list:
            if stock in self.data:
                df = self.data[stock]['daily'].copy()

                # 检查必要的列
                if 'open' not in df.columns or 'close' not in df.columns:
                    print(f"{stock} 数据中缺少'open'或'close'列")
                    continue

                df['open_close_ratio'] = df['open'] / df['close']

                df['trade_date'] = pd.to_datetime(df['trade_date'])
                self.factors.setdefault(stock, {})['open_close_ratio'] = df[['trade_date', 'open_close_ratio']]

    def calculate_rsi(self, period=14):
        """计算相对强弱指标(RSI)因子"""
        for stock in self.stock_list:
            if stock in self.data:
                df = self.data[stock]['daily'].copy()

                # 检查必要的列
                if 'close' not in df.columns:
                    print(f"{stock} 数据中缺少'close'列")
                    continue

                # 计算价格变动
                df['change'] = df['close'] - df['close'].shift(1)

                # 计算上涨和下跌幅度
                df['gain'] = np.where(df['change'] > 0, df['change'], 0)
                df['loss'] = np.where(df['change'] < 0, -df['change'], 0)

                # 计算平均上涨和下跌幅度
                df['avg_gain'] = df['gain'].rolling(window=period).mean()
                df['avg_loss'] = df['loss'].rolling(window=period).mean()

                # 避免除零错误
                df['avg_loss'] = df['avg_loss'].replace(0, np.nan)

                # 计算相对强度(RS)和相对强弱指标(RSI)
                df['rs'] = df['avg_gain'] / df['avg_loss']
                df['rsi'] = 100 - (100 / (1 + df['rs']))

                # 处理缺失值
                df['rsi'] = df['rsi'].fillna(50)

                df['trade_date'] = pd.to_datetime(df['trade_date'])
                self.factors.setdefault(stock, {})['rsi'] = df[['trade_date', 'rsi']]

    def calculate_volatility(self, period=20):
        """计算价格波动率因子"""
        for stock in self.stock_list:
            if stock in self.data:
                df = self.data[stock]['daily'].copy()

                # 检查必要的列
                if 'close' not in df.columns:
                    print(f"{stock} 数据中缺少'close'列")
                    continue

                # 计算对数收益率
                df['log_return'] = np.log(df['close'] / df['close'].shift(1))

                # 计算滚动波动率
                df['volatility'] = df['log_return'].rolling(window=period).std() * np.sqrt(252)

                # 处理缺失值
                df['volatility'] = df['volatility'].fillna(0)

                df['trade_date'] = pd.to_datetime(df['trade_date'])
                self.factors.setdefault(stock, {})['volatility'] = df[['trade_date', 'volatility']]

    def calculate_volume_change(self, period=5):
        """计算交易量变化率因子"""
        for stock in self.stock_list:
            if stock in self.data:
                df = self.data[stock]['daily'].copy()

                # 检查必要的列
                if 'vol' not in df.columns:
                    print(f"{stock} 数据中缺少'vol'列")
                    continue

                # 计算交易量变化率
                df['volume_change'] = df['vol'].pct_change(period)

                # 处理缺失值
                df['volume_change'] = df['volume_change'].fillna(0)

                df['trade_date'] = pd.to_datetime(df['trade_date'])
                self.factors.setdefault(stock, {})['volume_change'] = df[['trade_date', 'volume_change']]

    def calculate_macd(self, short_window=12, long_window=26, signal_window=9):
        """计算MACD因子"""
        for stock in self.stock_list:
            if stock in self.data:
                df = self.data[stock]['daily'].copy()

                # 检查必要的列
                if 'close' not in df.columns:
                    print(f"{stock} 数据中缺少'close'列")
                    continue

                short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
                long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
                df['macd'] = short_ema - long_ema
                df['signal_line'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
                df['macd_histogram'] = df['macd'] - df['signal_line']

                df['trade_date'] = pd.to_datetime(df['trade_date'])
                self.factors.setdefault(stock, {})['macd'] = df[['trade_date', 'macd', 'signal_line', 'macd_histogram']]

    def merge_factors(self):
        """合并所有计算好的因子"""
        merged_factors = {}

        for stock in self.stock_list:
            if stock in self.factors and self.factors[stock]:
                # 获取第一个因子的数据框作为基础
                first_factor = next(iter(self.factors[stock].values()))
                merged = first_factor.copy()

                # 确保日期列是datetime类型
                merged['trade_date'] = pd.to_datetime(merged['trade_date'])

                # 合并其他因子
                for factor_name, factor_df in self.factors[stock].items():
                    if factor_name != list(self.factors[stock].keys())[0]:
                        # 确保要合并的数据框的日期列是datetime类型
                        factor_df['trade_date'] = pd.to_datetime(factor_df['trade_date'])
                        merged = pd.merge(merged, factor_df, on='trade_date', how='outer')

                # 按日期排序
                merged = merged.sort_values('trade_date')

                # 填充缺失值（使用前向填充）
                merged = merged.ffill()  # 修复FutureWarning

                merged_factors[stock] = merged

        return merged_factors

    def plot_factors(self, stock):
        """可视化指定股票的所有因子"""
        if stock in self.factors and self.factors[stock]:
            factors = self.factors[stock]

            # 创建一个图形
            fig, axes = plt.subplots(len(factors), 1, figsize=(12, 4 * len(factors)))
            fig.suptitle(f"{stock} ({self.data[stock]['industry']}) 的Alpha因子可视化", fontsize=16)

            # 如果只有一个因子，axes不是数组，需要转换为数组
            if len(factors) == 1:
                axes = [axes]

            # 绘制每个因子
            for i, (factor_name, factor_df) in enumerate(factors.items()):
                ax = axes[i]
                ax.plot(pd.to_datetime(factor_df['trade_date']), factor_df[factor_name])
                ax.set_title(
                    f"{factor_name} ({self.factor_types[factor_name]['type']}, {self.factor_types[factor_name]['frequency']})")
                ax.grid(True)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            try:
                plt.show()
            except Exception as e:
                print(f"显示图表时出错: {e}")
                print("尝试保存图表到文件...")
                try:
                    fig.savefig(f"{stock}_factors.png")
                    print(f"图表已保存为 {stock}_factors.png")
                except Exception as save_e:
                    print(f"保存图表失败: {save_e}")
        else:
            print(f"没有找到 {stock} 的因子数据")

    def calculate_returns(self, stock, factor_name, holding_period=20, quantile=0.2):
        """
        基于因子值计算投资组合收益

        参数:
            stock: 股票代码
            factor_name: 因子名称
            holding_period: 持有期，单位为交易日
            quantile: 分位数，用于划分多空组合
        """
        if stock in self.factors and factor_name in self.factors[stock]:
            # 获取因子数据和价格数据
            factor_df = self.factors[stock][factor_name].copy()

            # 检查价格数据是否存在close列
            if stock not in self.data or 'daily' not in self.data[stock] or 'close' not in self.data[stock][
                'daily'].columns:
                print(f"错误: {stock} 的价格数据中缺少'close'列，无法计算{factor_name}的收益")
                return None

            price_df = self.data[stock]['daily'].copy()

            # 确保两个数据框的日期列都是datetime类型
            factor_df['trade_date'] = pd.to_datetime(factor_df['trade_date'])
            price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])

            # 确保两个数据框都按日期排序
            factor_df = factor_df.sort_values('trade_date')
            price_df = price_df.sort_values('trade_date')

            # 合并因子和价格数据
            merged_df = pd.merge(factor_df, price_df[['trade_date', 'close']], on='trade_date')

            # 检查合并后的数据是否包含因子列和close列
            if factor_name not in merged_df.columns:
                print(f"错误: 合并后的数据中缺少{factor_name}列，无法计算收益")
                return None

            if 'close' not in merged_df.columns:
                print(f"错误: 合并后的数据中缺少'close'列，无法计算{factor_name}的收益")
                return None

            # 计算未来收益率
            merged_df[f'return_{holding_period}d'] = merged_df['close'].pct_change(holding_period).shift(
                -holding_period)

            # 基于因子值生成交易信号
            if factor_name == 'open_close_ratio':
                # 开盘/收盘比率策略，比率大于1看多，小于1看空
                merged_df['signal'] = np.where(merged_df['open_close_ratio'] > 1, 1,
                                               np.where(merged_df['open_close_ratio'] < 1, -1, 0))
            elif factor_name == 'rsi':
                # RSI策略，超买超卖信号
                merged_df['signal'] = np.where(merged_df['rsi'] < 30, 1,
                                               np.where(merged_df['rsi'] > 70, -1, 0))
            elif factor_name == 'volatility':
                # 波动率策略，高波动看空，低波动看多
                merged_df['vol_ma'] = merged_df['volatility'].rolling(window=20).mean()
                merged_df['signal'] = np.where(merged_df['volatility'] < merged_df['vol_ma'], 1,
                                               np.where(merged_df['volatility'] > merged_df['vol_ma'], -1, 0))
            elif factor_name == 'volume_change':
                # 交易量变化策略，大幅增加看多，大幅减少看空
                merged_df['signal'] = np.where(merged_df['volume_change'] > 0.5, 1,
                                               np.where(merged_df['volume_change'] < -0.5, -1, 0))
            elif factor_name == 'macd':
                # MACD策略，MACD线在信号线之上看多，之下看空
                merged_df['signal'] = np.where(merged_df['macd'] > merged_df['signal_line'], 1, -1)

            # 计算策略收益
            merged_df['strategy_return'] = merged_df['signal'] * merged_df[f'return_{holding_period}d']

            # 计算多头组合和空头组合的收益
            long_returns = merged_df[merged_df['signal'] == 1][f'return_{holding_period}d']
            short_returns = -merged_df[merged_df['signal'] == -1][f'return_{holding_period}d']  # 空头收益取负

            # 计算多空组合的平均收益
            avg_long_return = long_returns.mean()
            avg_short_return = short_returns.mean()
            avg_spread_return = avg_long_return - avg_short_return

            return {
                'long_return': avg_long_return,
                'short_return': avg_short_return,
                'spread_return': avg_spread_return,
                'returns_data': merged_df
            }
        else:
            print(f"没有找到 {stock} 的 {factor_name} 因子数据")
            return None

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        """
        计算夏普比率

        参数:
            returns: 收益率序列
            risk_free_rate: 无风险利率，默认使用3%的年化利率
        """
        # 将年化无风险利率转换为日收益率
        daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

        # 计算超额收益
        excess_returns = returns - daily_risk_free_rate

        # 计算夏普比率 (年化)
        if excess_returns.std() == 0:
            return np.nan

        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self, returns):
        """
        计算最大回撤

        参数:
            returns: 收益率序列
        """
        if returns.empty:
            return np.nan

        # 计算累积收益
        cumulative_returns = (1 + returns).cumprod()

        # 计算滚动最大收益
        rolling_max = cumulative_returns.cummax()

        # 计算回撤
        drawdown = (cumulative_returns / rolling_max) - 1

        # 计算最大回撤
        max_drawdown = drawdown.min()

        return max_drawdown

    def calculate_sortino_ratio(self, returns, risk_free_rate=0.03):
        """
        计算索提诺比率

        参数:
            returns: 收益率序列
            risk_free_rate: 无风险利率，默认使用3%的年化利率
        """
        # 将年化无风险利率转换为日收益率
        daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

        # 计算超额收益
        excess_returns = returns - daily_risk_free_rate

        # 计算下行标准差 (只考虑负收益的标准差)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return np.nan

        downside_std = downside_returns.std()

        # 计算索提诺比率 (年化)
        if downside_std == 0:
            return np.nan

        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
        return sortino_ratio

    def calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        """
        计算信息比率

        参数:
            portfolio_returns: 投资组合收益率序列
            benchmark_returns: 基准收益率序列
        """
        # 计算超额收益
        active_returns = portfolio_returns - benchmark_returns

        # 计算信息比率
        if active_returns.std() == 0:
            return np.nan

        information_ratio = np.sqrt(252) * active_returns.mean() / active_returns.std()
        return information_ratio

    def evaluate_factor(self, stock, factor_name, holding_period=20, risk_free_rate=0.03):
        """
        评估因子的有效性和绩效

        参数:
            stock: 股票代码
            factor_name: 因子名称
            holding_period: 持有期，单位为交易日
            risk_free_rate: 无风险利率，默认使用3%的年化利率
        """
        # 检查因子数据是否存在
        if stock not in self.factors or factor_name not in self.factors[stock]:
            print(f"没有找到 {stock} 的 {factor_name} 因子数据，跳过评估")
            return None

        # 计算基于因子的投资组合收益
        returns_data = self.calculate_returns(stock, factor_name, holding_period)

        if returns_data and returns_data['returns_data'] is not None:
            # 获取多头、空头和多空组合的收益
            long_returns = returns_data['returns_data'][returns_data['returns_data']['signal'] == 1][
                f'return_{holding_period}d'].dropna()
            short_returns = -returns_data['returns_data'][returns_data['returns_data']['signal'] == -1][
                f'return_{holding_period}d'].dropna()  # 空头收益取负
            spread_returns = long_returns - short_returns

            # 获取基准收益 (使用股票自身的平均收益作为基准)
            benchmark_returns = returns_data['returns_data'][f'return_{holding_period}d'].dropna().mean()

            # 计算评估指标
            evaluation = {
                'factor_name': factor_name,
                'factor_type': self.factor_types[factor_name]['type'],
                'factor_frequency': self.factor_types[factor_name]['frequency'],
                'factor_direction': self.factor_types[factor_name]['direction'],
                'factor_source': self.factor_sources[factor_name],
                'long_return': returns_data['long_return'],
                'short_return': returns_data['short_return'],
                'spread_return': returns_data['spread_return'],
                'long_sharpe': self.calculate_sharpe_ratio(long_returns, risk_free_rate),
                'short_sharpe': self.calculate_sharpe_ratio(short_returns, risk_free_rate),
                'spread_sharpe': self.calculate_sharpe_ratio(spread_returns, risk_free_rate),
                'long_max_drawdown': self.calculate_max_drawdown(long_returns),
                'short_max_drawdown': self.calculate_max_drawdown(short_returns),
                'spread_max_drawdown': self.calculate_max_drawdown(spread_returns),
                'long_sortino': self.calculate_sortino_ratio(long_returns, risk_free_rate),
                'short_sortino': self.calculate_sortino_ratio(short_returns, risk_free_rate),
                'spread_sortino': self.calculate_sortino_ratio(spread_returns, risk_free_rate),
                'information_ratio': self.calculate_information_ratio(
                    returns_data['returns_data']['strategy_return'].dropna(),
                    benchmark_returns
                )
            }

            # 计算因子IC和IR
            factor_series = returns_data['returns_data'][factor_name].dropna()
            return_series = returns_data['returns_data'][f'return_{holding_period}d'].dropna()
            factor_series, return_series = factor_series.align(return_series, join='inner')
            ic = stats.spearmanr(factor_series, return_series)[0]

            evaluation['ic'] = ic
            evaluation['ir'] = ic * np.sqrt(252 / holding_period)

            self.evaluation.setdefault(stock, {})[factor_name] = evaluation

            return evaluation
        else:
            print(f"无法计算 {stock} 的 {factor_name} 因子评估指标")
            return None
    def plot_factor_evaluation(self, stock, factor_name):
        """
        可视化因子评估结果

        参数:
            stock: 股票代码
            factor_name: 因子名称
        """
        if stock in self.evaluation and factor_name in self.evaluation[stock]:
            eval_data = self.evaluation[stock][factor_name]
            returns_data = self.calculate_returns(stock, factor_name)['returns_data']

            # 创建一个2x2的图形
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"{stock} {factor_name} 因子评估", fontsize=16)

            # 1. 因子值与价格对比
            ax1 = axes[0, 0]
            ax1.plot(pd.to_datetime(returns_data['trade_date']), returns_data[factor_name], label=factor_name)
            ax1.set_title(f"{factor_name} 因子值")
            ax1.grid(True)
            ax1.legend()

            ax1_twin = ax1.twinx()
            ax1_twin.plot(pd.to_datetime(returns_data['trade_date']), returns_data['close'], 'r-', alpha=0.3,
                          label='收盘价')
            ax1_twin.legend(loc='upper right')

            # 2. 策略收益
            ax2 = axes[0, 1]
            cumulative_returns = (1 + returns_data['strategy_return'].dropna()).cumprod() - 1
            ax2.plot(cumulative_returns.index, cumulative_returns, label='策略累积收益')
            ax2.set_title('策略累积收益')
            ax2.grid(True)
            ax2.legend()

            # 3. 多头空头收益分布
            ax3 = axes[1, 0]
            long_returns = returns_data[returns_data['signal'] == 1][f'return_{20}d'].dropna()
            short_returns = -returns_data[returns_data['signal'] == -1][f'return_{20}d'].dropna()

            if not long_returns.empty and not short_returns.empty:
                ax3.hist(long_returns, bins=20, alpha=0.5, label='多头收益')
                ax3.hist(short_returns, bins=20, alpha=0.5, label='空头收益')
                ax3.set_title('多头与空头收益分布')
                ax3.grid(True)
                ax3.legend()

            # 4. 因子评估指标
            ax4 = axes[1, 1]
            metrics = {
                '年化夏普比率': eval_data['spread_sharpe'],
                '最大回撤': eval_data['spread_max_drawdown'],
                '索提诺比率': eval_data['spread_sortino'],
                '信息比率': eval_data['information_ratio'],
                'IC值': eval_data['ic'],
                'IR值': eval_data['ir']
            }

            y_pos = np.arange(len(metrics))
            ax4.barh(y_pos, list(metrics.values()), align='center')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(list(metrics.keys()))
            ax4.set_title('因子评估指标')
            ax4.grid(True)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            try:
                plt.show()
            except Exception as e:
                print(f"显示图表时出错: {e}")
                print("尝试保存图表到文件...")
                try:
                    fig.savefig(f"{stock}_{factor_name}_evaluation.png")
                    print(f"图表已保存为 {stock}_{factor_name}_evaluation.png")
                except Exception as save_e:
                    print(f"保存图表失败: {save_e}")
        else:
            print(f"没有找到 {stock} 的 {factor_name} 因子评估数据")

    def analyze_factor_correlation(self, stock):
        """
        分析因子之间的相关性

        参数:
            stock: 股票代码
        """
        if stock in self.factors:
            # 合并所有因子
            factors_df = self.merge_factors()[stock]

            # 计算相关系数矩阵
            corr_matrix = factors_df.drop('trade_date', axis=1).corr()

            # 打印相关系数矩阵
            print(f"\n{stock} 的因子相关性矩阵:")
            print(corr_matrix.to_string())

            # 绘制热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.4f')
            plt.title(f"{stock} 因子相关性热力图")
            plt.tight_layout()
            try:
                plt.show()
            except Exception as e:
                print(f"显示图表时出错: {e}")
                print("尝试保存图表到文件...")
                try:
                    plt.savefig(f"{stock}_factor_correlation.png")
                    print(f"图表已保存为 {stock}_factor_correlation.png")
                except Exception as save_e:
                    print(f"保存图表失败: {save_e}")
            finally:
                plt.close()  # 关闭图表以释放资源

            return corr_matrix
        else:
            print(f"没有找到 {stock} 的因子数据")
            return None

    def evaluate_all_factors(self, holding_period=20, risk_free_rate=0.03):
        """
        评估所有股票的所有因子
        """
        for stock in self.stock_list:
            print(f"\n评估 {stock} 的因子...")
            for factor_name in self.factor_types.keys():
                if stock in self.factors and factor_name in self.factors[stock]:
                    eval_result = self.evaluate_factor(stock, factor_name, holding_period, risk_free_rate)
                    if eval_result:
                        print(f"  {factor_name}:")
                        print(f"    多头平均收益: {eval_result['long_return']:.4f}")
                        print(f"    空头平均收益: {eval_result['short_return']:.4f}")
                        print(f"    多空收益差: {eval_result['spread_return']:.4f}")
                        print(f"    夏普比率: {eval_result['spread_sharpe']:.4f}")
                        print(f"    最大回撤: {eval_result['spread_max_drawdown']:.4f}")
                        print(f"    IC值: {eval_result['ic']:.4f}")
                        print(f"    IR值: {eval_result['ir']:.4f}")
                else:
                    print(f"  跳过 {stock} 的 {factor_name} 因子评估，数据不存在")

    def calculate_average_sharpe(self):
        """
        计算所有因子的平均夏普比率
        """
        all_sharpe_ratios = {}

        for stock in self.stock_list:
            print(f"\n{stock}:")
            stock_sharpe = {}
            has_valid_data = False

            for factor_name in self.factor_types.keys():
                if stock in self.evaluation and factor_name in self.evaluation[stock]:
                    sharpe = self.evaluation[stock][factor_name]['spread_sharpe']
                    stock_sharpe[factor_name] = sharpe
                    print(f"  {factor_name}: {sharpe:.4f}")
                    if not np.isnan(sharpe):
                        has_valid_data = True
                else:
                    print(f"  {factor_name}: nan")

            if has_valid_data:
                valid_sharpes = [s for s in stock_sharpe.values() if not np.isnan(s)]
                avg_sharpe = np.mean(valid_sharpes)
                all_sharpe_ratios[stock] = avg_sharpe
                print(f"  平均夏普比率: {avg_sharpe:.4f}")
            else:
                print(f"  无法计算平均夏普比率")
                all_sharpe_ratios[stock] = np.nan

        return all_sharpe_ratios


# 使用示例
if __name__ == "__main__":
    # 定义要分析的股票列表
    stock_list = ['000001.SZ', '600000.SH', '000858.SZ']

    # 定义分析的时间范围
    start_date = '20230101'
    end_date = '20231231'

    # 创建Alpha因子计算器实例
    alpha_calculator = AlphaFactors(stock_list, start_date, end_date)

    # 获取股票数据
    alpha_calculator.get_stock_data()

    # 计算因子
    alpha_calculator.calculate_open_close_ratio()
    alpha_calculator.calculate_rsi()
    alpha_calculator.calculate_volatility()
    alpha_calculator.calculate_volume_change()
    alpha_calculator.calculate_macd()

    # 分析因子相关性
    for stock in stock_list:
        alpha_calculator.analyze_factor_correlation(stock)

    # 评估所有因子
    alpha_calculator.evaluate_all_factors()

    # 计算平均夏普比率
    alpha_calculator.calculate_average_sharpe()

    # 可视化因子评估结果
    for stock in stock_list:
        for factor_name in alpha_calculator.factor_types.keys():
            alpha_calculator.plot_factor_evaluation(stock, factor_name)