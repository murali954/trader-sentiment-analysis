import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Loading and preprocessing data...")

try:
    trader_df = pd.read_csv('data/historical_data.csv')
    sentiment_df = pd.read_csv('data/fear_greed_index.csv')
    print(f"Loaded {len(trader_df)} trading records and {len(sentiment_df)} sentiment records")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure the files exist in the data/ directory")
    exit()

trader_df['date'] = pd.to_datetime(trader_df['Timestamp IST'], dayfirst=True).dt.date
sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

merged_df = pd.merge(trader_df, sentiment_df[['date', 'classification']], on='date', how='left')

print(f"\nData Quality Check:")
print(f"Total records after merge: {len(merged_df)}")
print(f"Records with sentiment data: {merged_df['classification'].notna().sum()}")
print(f"Records missing sentiment: {merged_df['classification'].isna().sum()}")

merged_df = merged_df.dropna(subset=['classification'])
print(f"Records after removing missing sentiment: {len(merged_df)}")

print(f"\nSentiment distribution:")
print(merged_df['classification'].value_counts())

print(f"\nBasic statistics:")
print(merged_df[['Execution Price', 'Size Tokens', 'Size USD', 'Closed PnL']].describe())

merged_df['volume'] = merged_df['Size USD']  # Using existing USD size
merged_df['profitable'] = merged_df['Closed PnL'] > 0
merged_df['return_pct'] = (merged_df['Closed PnL'] / merged_df['Size USD']) * 100

fig = plt.figure(figsize=(20, 15))

plt.subplot(2, 3, 1)
sns.boxplot(data=merged_df, x='classification', y='Closed PnL')
plt.title('PnL Distribution by Market Sentiment', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Closed PnL (USD)')

plt.subplot(2, 3, 2)
volumes = merged_df.groupby('classification')['volume'].sum()
volumes.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Total Volume by Market Sentiment', fontsize=14, fontweight='bold')
plt.ylabel('Volume (USD)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
profitability = merged_df.groupby('classification')['profitable'].mean() * 100
profitability.plot(kind='bar', color='lightgreen', alpha=0.8)
plt.title('Win Rate by Market Sentiment', fontsize=14, fontweight='bold')
plt.ylabel('Win Rate (%)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 4)
avg_pnl = merged_df.groupby('classification')['Closed PnL'].mean()
colors = ['red' if x < 0 else 'green' for x in avg_pnl.values]
avg_pnl.plot(kind='bar', color=colors, alpha=0.7)
plt.title('Average PnL by Market Sentiment', fontsize=14, fontweight='bold')
plt.ylabel('Average PnL (USD)')
plt.xticks(rotation=45)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.subplot(2, 3, 5)
trade_counts = merged_df['classification'].value_counts()
trade_counts.plot(kind='bar', color='orange', alpha=0.8)
plt.title('Number of Trades by Market Sentiment', fontsize=14, fontweight='bold')
plt.ylabel('Number of Trades')
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
sns.violinplot(data=merged_df, x='classification', y='return_pct')
plt.title('Return % Distribution by Market Sentiment', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Return %')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("DETAILED STATISTICAL ANALYSIS")
print("="*80)

print("\n1. WIN RATE ANALYSIS:")
print("-" * 40)
profitability_stats = merged_df.groupby('classification')['profitable'].agg(['count', 'sum', 'mean'])
profitability_stats.columns = ['Total_Trades', 'Winning_Trades', 'Win_Rate']
profitability_stats['Win_Rate'] = profitability_stats['Win_Rate'] * 100
print(profitability_stats.round(2))

print("\n2. PnL STATISTICS BY SENTIMENT:")
print("-" * 40)
pnl_stats = merged_df.groupby('classification')['Closed PnL'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(2)
print(pnl_stats)

print("\n3. RISK-ADJUSTED METRICS:")
print("-" * 40)
risk_metrics = pd.DataFrame()
for sentiment in merged_df['classification'].unique():
    subset = merged_df[merged_df['classification'] == sentiment]['Closed PnL']
    risk_metrics.loc[sentiment, 'Mean_PnL'] = subset.mean()
    risk_metrics.loc[sentiment, 'Std_PnL'] = subset.std()
    risk_metrics.loc[sentiment, 'Sharpe_Ratio'] = subset.mean() / subset.std() if subset.std() != 0 else 0
    risk_metrics.loc[sentiment, 'Max_Drawdown'] = subset.min()
    risk_metrics.loc[sentiment, 'Best_Trade'] = subset.max()

print(risk_metrics.round(3))

print("\n4. VOLUME ANALYSIS:")
print("-" * 40)
volume_stats = merged_df.groupby('classification')['volume'].agg([
    'sum', 'mean', 'median', 'count'
]).round(2)
volume_stats.columns = ['Total_Volume', 'Avg_Volume', 'Median_Volume', 'Trade_Count']
print(volume_stats)

print("\n5. STATISTICAL SIGNIFICANCE TESTS:")
print("-" * 40)

sentiments = merged_df['classification'].unique()

print("Pairwise t-tests for PnL differences:")
for i, sent1 in enumerate(sentiments):
    for sent2 in sentiments[i+1:]:
        group1 = merged_df[merged_df['classification'] == sent1]['Closed PnL']
        group2 = merged_df[merged_df['classification'] == sent2]['Closed PnL']
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"{sent1} vs {sent2}: t-stat={t_stat:.3f}, p-value={p_value:.4f} {significance}")

print("\n6. CORRELATION ANALYSIS:")
print("-" * 40)
sentiment_dummies = pd.get_dummies(merged_df['classification'], prefix='sentiment')
correlation_df = pd.concat([
    merged_df[['Closed PnL', 'volume', 'return_pct']], 
    sentiment_dummies
], axis=1)

correlations = correlation_df.corr()['Closed PnL'].sort_values(ascending=False)
print("Correlations with PnL:")
for var, corr in correlations.items():
    if var != 'Closed PnL':
        print(f"{var}: {corr:.4f}")

print("\n7. PERFORMANCE RANKING:")
print("-" * 40)
performance_summary = pd.DataFrame()
for sentiment in sentiments:
    subset = merged_df[merged_df['classification'] == sentiment]
    performance_summary.loc[sentiment, 'Win_Rate_%'] = (subset['profitable'].mean() * 100)
    performance_summary.loc[sentiment, 'Avg_PnL'] = subset['Closed PnL'].mean()
    performance_summary.loc[sentiment, 'Total_PnL'] = subset['Closed PnL'].sum()
    performance_summary.loc[sentiment, 'Sharpe_Ratio'] = (subset['Closed PnL'].mean() / subset['Closed PnL'].std()) if subset['Closed PnL'].std() != 0 else 0
    performance_summary.loc[sentiment, 'Trade_Count'] = len(subset)

performance_summary = performance_summary.round(3)
print("\nPerformance Summary (ranked by Sharpe Ratio):")
print(performance_summary.sort_values('Sharpe_Ratio', ascending=False))

print("\n8. MONTHLY PERFORMANCE BREAKDOWN:")
print("-" * 40)
merged_df['month'] = pd.to_datetime(merged_df['date']).dt.month
monthly_performance = merged_df.groupby(['classification', 'month'])['Closed PnL'].agg(['count', 'mean', 'sum']).round(2)
print("Top 10 best month-sentiment combinations by average PnL:")
print(monthly_performance.sort_values('mean', ascending=False).head(10))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)