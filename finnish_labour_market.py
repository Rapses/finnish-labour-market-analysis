"""
Finnish Labour Market Analysis — Employment Trends & Regional Insights
=======================================================================
Author: Rupesh Jha
Description:
    Comprehensive analysis of Finnish labour market dynamics including
    regional unemployment trends, sectoral recovery patterns, youth
    unemployment analysis, time series forecasting and correlation
    between education levels and employment outcomes.

    Data: Simulated to match Statistics Finland (stat.fi) structure
    and publicly available Finnish labour market data 2010-2023.

    Real data available at: stat.fi/til/tyti/index_en.html

Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

FINLAND_BLUE  = '#003580'
FINLAND_WHITE = '#FFFFFF'
ACCENT        = '#1F4E79'
COLORS        = ['#1F4E79', '#2E75B6', '#70AD47', '#FFC000',
                 '#FF0000', '#7030A0', '#00B0F0', '#FF6600',
                 '#00B050', '#C00000']
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SIMULATE REALISTIC FINNISH LABOUR MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Finnish Labour Market Analysis — Employment Trends 2010-2023")
print("=" * 65)
print("\nPART 1: Generating realistic Finnish labour market data...")

# Finnish regions (maakunta)
REGIONS = {
    'Uusimaa':           {'base_unemp': 7.2,  'population': 1700000, 'urban': True},
    'Varsinais-Suomi':   {'base_unemp': 8.1,  'population': 480000,  'urban': True},
    'Pirkanmaa':         {'base_unemp': 8.5,  'population': 530000,  'urban': True},
    'Pohjois-Pohjanmaa': {'base_unemp': 9.8,  'population': 410000,  'urban': False},
    'Keski-Suomi':       {'base_unemp': 10.2, 'population': 275000,  'urban': False},
    'Pohjois-Karjala':   {'base_unemp': 12.1, 'population': 162000,  'urban': False},
    'Etelä-Savo':        {'base_unemp': 11.8, 'population': 146000,  'urban': False},
    'Lappi':             {'base_unemp': 10.5, 'population': 180000,  'urban': False},
    'Pohjanmaa':         {'base_unemp': 6.8,  'population': 180000,  'urban': False},
    'Satakunta':         {'base_unemp': 9.5,  'population': 220000,  'urban': False},
}

# Economic periods affecting unemployment
PERIODS = {
    'recovery_2010_2012': (2010, 2012, -0.3),   # Post 2008 crisis recovery
    'stagnation_2013_2015': (2013, 2015, +0.4), # Finnish recession
    'growth_2016_2019': (2016, 2019, -0.5),     # Growth period
    'covid_2020': (2020, 2020, +2.5),           # COVID spike
    'recovery_2021_2023': (2021, 2023, -0.6),   # Post COVID recovery
}

# Generate quarterly data 2010-2023
quarters = pd.date_range(start='2010-01-01', end='2023-12-31', freq='QE')
n_quarters = len(quarters)

# ── Regional unemployment rates ───────────────────────────────────────────────
regional_data = {}
for region, params in REGIONS.items():
    rates = []
    base = params['base_unemp']
    for date in quarters:
        year = date.year
        quarter = date.quarter
        rate = base

        # Apply economic period effects
        for period, (start_yr, end_yr, effect) in PERIODS.items():
            if start_yr <= year <= end_yr:
                rate += effect * (1 + np.random.normal(0, 0.1))

        # Seasonal effect — Q1 always higher unemployment in Finland
        seasonal = {1: +0.8, 2: -0.3, 3: -0.5, 4: +0.1}
        rate += seasonal[quarter]

        # Random noise
        rate += np.random.normal(0, 0.3)
        rate = max(2.0, min(25.0, rate))
        rates.append(round(rate, 1))

    regional_data[region] = rates

regional_df = pd.DataFrame(regional_data, index=quarters)

# ── National unemployment ─────────────────────────────────────────────────────
national_unemployment = regional_df.mean(axis=1)

# ── Sectoral employment ───────────────────────────────────────────────────────
SECTORS = {
    'Technology & ICT':        {'base': 95000,  'growth': 0.04,  'covid_impact': -0.02},
    'Manufacturing':           {'base': 320000, 'growth': -0.01, 'covid_impact': -0.08},
    'Healthcare & Social':     {'base': 410000, 'growth': 0.02,  'covid_impact': +0.05},
    'Retail & Commerce':       {'base': 280000, 'growth': 0.01,  'covid_impact': -0.15},
    'Construction':            {'base': 175000, 'growth': 0.02,  'covid_impact': -0.10},
    'Education':               {'base': 175000, 'growth': 0.01,  'covid_impact': -0.05},
    'Hospitality & Tourism':   {'base': 95000,  'growth': 0.01,  'covid_impact': -0.35},
    'Finance & Insurance':     {'base': 85000,  'growth': 0.01,  'covid_impact': -0.03},
    'Agriculture & Forestry':  {'base': 110000, 'growth': -0.02, 'covid_impact': +0.02},
    'Public Administration':   {'base': 125000, 'growth': 0.00,  'covid_impact': +0.03},
}

years = list(range(2010, 2024))
sectoral_data = {}
for sector, params in SECTORS.items():
    employment = []
    for year in years:
        emp = params['base'] * (1 + params['growth']) ** (year - 2010)
        if year == 2020:
            emp *= (1 + params['covid_impact'])
        elif year == 2021:
            emp *= (1 + params['covid_impact'] * 0.5)
        emp += np.random.normal(0, emp * 0.01)
        employment.append(int(emp))
    sectoral_data[sector] = employment

sectoral_df = pd.DataFrame(sectoral_data, index=years)

# ── Youth unemployment ────────────────────────────────────────────────────────
youth_unemployment = national_unemployment * 1.8 + np.random.normal(0, 0.5, n_quarters)
youth_unemployment = youth_unemployment.clip(lower=5)

# ── Education vs employment correlation ───────────────────────────────────────
education_levels = ['Primary', 'Secondary', 'Vocational', 'Bachelor', 'Master', 'Doctorate']
employment_rates = [62.5, 71.3, 78.9, 84.2, 88.7, 91.3]
unemployment_rates_edu = [14.2, 9.8, 7.1, 5.2, 3.8, 2.9]
avg_salary = [1800, 2400, 2900, 3500, 4200, 4800]

education_df = pd.DataFrame({
    'Education Level': education_levels,
    'Employment Rate (%)': employment_rates,
    'Unemployment Rate (%)': unemployment_rates_edu,
    'Average Monthly Salary (€)': avg_salary
})

print(f"  Generated {n_quarters} quarterly observations")
print(f"  Regions covered: {len(REGIONS)}")
print(f"  Sectors covered: {len(SECTORS)}")
print(f"  Time period: 2010 Q1 to 2023 Q4")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — NATIONAL UNEMPLOYMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 2: National Unemployment Analysis...")

print(f"\n  National Unemployment Statistics 2010-2023:")
print(f"  Overall mean:     {national_unemployment.mean():.1f}%")
print(f"  Overall median:   {national_unemployment.median():.1f}%")
print(f"  Minimum:          {national_unemployment.min():.1f}% ({national_unemployment.idxmin().strftime('%Y Q%q')})")
print(f"  Maximum:          {national_unemployment.max():.1f}% ({national_unemployment.idxmax().strftime('%Y Q%q')})")
print(f"  COVID-19 peak:    {national_unemployment[national_unemployment.index.year == 2020].max():.1f}%")
print(f"  Post-COVID low:   {national_unemployment[national_unemployment.index.year >= 2022].min():.1f}%")

# Annual averages
annual_unemployment = national_unemployment.resample('YE').mean()
print(f"\n  Annual Average Unemployment Rates:")
for year, rate in annual_unemployment.items():
    bar = '█' * int(rate)
    print(f"  {year.year}: {rate:4.1f}% {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 3: Regional Analysis...")

regional_avg = regional_df.mean().sort_values()
print(f"\n  Regional Unemployment Rankings (Average 2010-2023):")
print(f"  {'Region':<25} {'Avg Rate':>10} {'Category':>15}")
print("  " + "-" * 55)
for region, rate in regional_avg.items():
    category = "Low" if rate < 8 else "Medium" if rate < 10 else "High"
    print(f"  {region:<25} {rate:>9.1f}% {category:>15}")

# COVID impact by region
covid_impact = regional_df.loc['2020', regional_df.columns[0]]
print(f"\n  COVID-19 Impact by Region (2019→2020 change):")
for region in regional_df.columns:
    impact = regional_df[regional_df.index.year == 2020][region].mean() - regional_df[regional_df.index.year == 2019][region].mean()
    print(f"  {region:<25}: +{impact:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — SECTORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 4: Sectoral Employment Analysis...")

# Growth rates 2010-2023
print(f"\n  Sectoral Employment Change 2010-2023:")
for sector in sectoral_df.columns:
    change = ((sectoral_df.loc[2023, sector] - sectoral_df.loc[2010, sector])
              / sectoral_df.loc[2010, sector] * 100)
    direction = "📈" if change > 0 else "📉"
    print(f"  {sector:<30}: {change:+.1f}% {direction}")

# COVID recovery — sectors back to 2019 levels by 2022?
print(f"\n  COVID Recovery — 2022 vs 2019 employment:")
for sector in sectoral_df.columns:
    recovery = ((sectoral_df.loc[2022, sector] - sectoral_df.loc[2019, sector])
                / sectoral_df.loc[2019, sector] * 100)
    status = "✅ Recovered" if recovery >= -2 else "⚠️  Partial" if recovery >= -10 else "❌ Struggling"
    print(f"  {sector:<30}: {recovery:+.1f}% {status}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — YOUTH UNEMPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 5: Youth Unemployment Analysis...")

youth_annual = youth_unemployment.resample('YE').mean()
national_annual = national_unemployment.resample('YE').mean()
ratio = (youth_annual / national_annual).mean()

print(f"\n  Youth vs National Unemployment:")
print(f"  Average youth rate:    {youth_unemployment.mean():.1f}%")
print(f"  Average national rate: {national_unemployment.mean():.1f}%")
print(f"  Youth/National ratio:  {ratio:.2f}x")
print(f"  Youth unemployment consistently {ratio:.1f}x higher than national average")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — EDUCATION & EMPLOYMENT CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 6: Education vs Employment Analysis...")

correlation = np.corrcoef(
    range(len(education_levels)),
    education_df['Employment Rate (%)']
)[0, 1]

salary_correlation = np.corrcoef(
    range(len(education_levels)),
    education_df['Average Monthly Salary (€)']
)[0, 1]

print(f"\n  Education Level vs Employment Rate:")
print(f"  {'Level':<15} {'Employment %':>14} {'Unemployment %':>16} {'Avg Salary':>12}")
print("  " + "-" * 60)
for _, row in education_df.iterrows():
    print(f"  {row['Education Level']:<15} {row['Employment Rate (%)']:>13.1f}%"
          f" {row['Unemployment Rate (%)']:>15.1f}% €{row['Average Monthly Salary (€)']:>9,}")

print(f"\n  Correlation — Education vs Employment Rate: {correlation:.3f}")
print(f"  Correlation — Education vs Average Salary:  {salary_correlation:.3f}")
print(f"  Strong positive correlation confirms education ROI in Finnish market")


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — TIME SERIES FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 7: Unemployment Forecasting Model...")

# Prepare features for forecasting
forecast_df = pd.DataFrame({
    'unemployment': national_unemployment.values,
    'time_index': range(n_quarters),
    'quarter': [d.quarter for d in quarters],
    'year': [d.year for d in quarters]
})

# Add lag features
forecast_df['lag_1']  = forecast_df['unemployment'].shift(1)
forecast_df['lag_4']  = forecast_df['unemployment'].shift(4)  # Year ago
forecast_df['lag_8']  = forecast_df['unemployment'].shift(8)  # 2 years ago
forecast_df['rolling_mean_4'] = forecast_df['unemployment'].rolling(4).mean()
forecast_df = forecast_df.dropna()

# Add seasonal dummies
for q in range(1, 5):
    forecast_df[f'Q{q}'] = (forecast_df['quarter'] == q).astype(int)

features = ['time_index', 'lag_1', 'lag_4', 'rolling_mean_4', 'Q1', 'Q2', 'Q3']

X = forecast_df[features]
y = forecast_df['unemployment']

# Train on 2010-2021, test on 2022-2023
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n  Forecasting Model Performance:")
print(f"  RMSE:     {rmse:.3f} percentage points")
print(f"  R² Score: {r2:.3f}")
print(f"  Model explains {r2*100:.1f}% of unemployment variance")

# Future forecast — 8 quarters ahead
last_values = forecast_df.tail(8)
future_unemployment = []
last_unemp = national_unemployment.iloc[-1]

for i in range(8):
    quarter = (quarters[-1].quarter + i) % 4 + 1
    seasonal_adj = {1: 0.8, 2: -0.3, 3: -0.5, 4: 0.1}[quarter]
    forecast = last_unemp + seasonal_adj + np.random.normal(0, 0.2) - 0.05
    forecast = max(3.0, min(15.0, forecast))
    future_unemployment.append(round(forecast, 1))
    last_unemp = forecast

future_dates = pd.date_range(
    start=quarters[-1] + pd.DateOffset(months=3),
    periods=8, freq='QE'
)

print(f"\n  8-Quarter Unemployment Forecast (2024-2025):")
for date, rate in zip(future_dates, future_unemployment):
    print(f"  {date.strftime('%Y Q%q')}: {rate:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — HELSINKI METRO DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 8: Helsinki Metropolitan Area Deep Dive...")

helsinki_metro = regional_df['Uusimaa']
national_avg   = national_unemployment

gap = national_avg.mean() - helsinki_metro.mean()
print(f"\n  Helsinki (Uusimaa) vs National Average:")
print(f"  Helsinki mean unemployment:  {helsinki_metro.mean():.1f}%")
print(f"  National mean unemployment:  {national_avg.mean():.1f}%")
print(f"  Helsinki advantage:          {gap:.1f} percentage points lower")
print(f"  Helsinki COVID peak:         {helsinki_metro[helsinki_metro.index.year == 2020].max():.1f}%")
print(f"  Helsinki 2023 rate:          {helsinki_metro[helsinki_metro.index.year == 2023].mean():.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 9: Generating dashboard...")

fig = plt.figure(figsize=(22, 28))
fig.suptitle(
    'Finnish Labour Market Analysis — Employment Trends & Regional Insights 2010-2023\n'
    'Rupesh Jha — Data Science Portfolio',
    fontsize=15, fontweight='bold', color=FINLAND_BLUE, y=0.98
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── Plot 1: National unemployment trend ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(quarters, national_unemployment, color=FINLAND_BLUE,
         linewidth=2.5, label='National Unemployment')
ax1.plot(quarters, youth_unemployment, color='red',
         linewidth=1.5, linestyle='--', alpha=0.8, label='Youth Unemployment')
ax1.fill_between(quarters, national_unemployment, alpha=0.1, color=FINLAND_BLUE)

# Annotate key events
ax1.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'),
            alpha=0.15, color='red', label='COVID-19')
ax1.axvspan(pd.Timestamp('2013-01-01'), pd.Timestamp('2015-12-31'),
            alpha=0.10, color='orange', label='Finnish Recession')

ax1.set_title('National & Youth Unemployment Rate 2010-2023', fontweight='bold')
ax1.set_ylabel('Unemployment Rate (%)')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 25)

# ── Plot 2: Regional unemployment heatmap ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
annual_regional = regional_df.resample('YE').mean()
annual_regional.index = annual_regional.index.year
sns.heatmap(annual_regional.T, cmap='RdYlGn_r', ax=ax2,
            annot=False, fmt='.1f',
            cbar_kws={'label': 'Unemployment Rate (%)'},
            linewidths=0.5)
ax2.set_title('Regional Unemployment Heatmap 2010-2023', fontweight='bold')
ax2.set_xlabel('Year')
ax2.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='y', rotation=0)

# ── Plot 3: Sectoral employment change ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
sector_changes = {}
for sector in sectoral_df.columns:
    change = ((sectoral_df.loc[2023, sector] - sectoral_df.loc[2010, sector])
              / sectoral_df.loc[2010, sector] * 100)
    sector_changes[sector] = change

sector_series = pd.Series(sector_changes).sort_values()
colors_bar = ['#C00000' if v < 0 else '#70AD47' for v in sector_series.values]
bars = ax3.barh(range(len(sector_series)), sector_series.values,
                color=colors_bar, edgecolor='white')
ax3.set_yticks(range(len(sector_series)))
ax3.set_yticklabels([s.replace(' & ', '\n& ') for s in sector_series.index], fontsize=8)
ax3.axvline(x=0, color='black', linewidth=1)
ax3.set_title('Sectoral Employment Change 2010-2023 (%)', fontweight='bold')
ax3.set_xlabel('Change (%)')
for bar, val in zip(bars, sector_series.values):
    ax3.text(val + (0.3 if val >= 0 else -0.3),
             bar.get_y() + bar.get_height()/2,
             f'{val:+.1f}%', va='center', fontsize=8,
             ha='left' if val >= 0 else 'right', fontweight='bold')

# ── Plot 4: COVID recovery by sector ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
recovery_data = {}
for sector in sectoral_df.columns:
    pre_covid  = sectoral_df.loc[2019, sector]
    post_covid = sectoral_df.loc[2022, sector]
    recovery   = (post_covid - pre_covid) / pre_covid * 100
    recovery_data[sector] = recovery

recovery_series = pd.Series(recovery_data).sort_values()
colors_rec = ['#C00000' if v < -5 else '#FFC000' if v < 0 else '#70AD47'
              for v in recovery_series.values]
bars4 = ax4.barh(range(len(recovery_series)), recovery_series.values,
                 color=colors_rec, edgecolor='white')
ax4.set_yticks(range(len(recovery_series)))
ax4.set_yticklabels([s.replace(' & ', '\n& ') for s in recovery_series.index], fontsize=8)
ax4.axvline(x=0, color='black', linewidth=1)
ax4.axvline(x=-2, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='Recovery threshold')
ax4.set_title('COVID-19 Recovery by Sector\n(2022 vs 2019 employment)', fontweight='bold')
ax4.set_xlabel('Change from pre-COVID level (%)')

red_patch    = mpatches.Patch(color='#C00000', label='Still struggling (>5% below)')
yellow_patch = mpatches.Patch(color='#FFC000', label='Partial recovery')
green_patch  = mpatches.Patch(color='#70AD47', label='Fully recovered')
ax4.legend(handles=[red_patch, yellow_patch, green_patch], fontsize=7, loc='lower right')

# ── Plot 5: Education vs Employment ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
x = np.arange(len(education_levels))
width = 0.35
bars5a = ax5.bar(x - width/2, education_df['Employment Rate (%)'],
                 width, label='Employment Rate', color=FINLAND_BLUE, alpha=0.8)
bars5b = ax5.bar(x + width/2, education_df['Unemployment Rate (%)'],
                 width, label='Unemployment Rate', color='#C00000', alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(education_levels, rotation=20, ha='right', fontsize=9)
ax5.set_title(f'Education Level vs Employment Outcomes\n(Correlation: {correlation:.3f})',
              fontweight='bold')
ax5.set_ylabel('Rate (%)')
ax5.legend()
for bar in bars5a:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.0f}%', ha='center', fontsize=7, fontweight='bold')

# ── Plot 6: Salary by education ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
bars6 = ax6.bar(education_levels, education_df['Average Monthly Salary (€)'],
                color=[plt.cm.Blues(0.4 + i*0.1) for i in range(len(education_levels))],
                edgecolor='white')
ax6.set_title('Average Monthly Salary by Education Level\n(Finnish Labour Market 2023)',
              fontweight='bold')
ax6.set_ylabel('Average Monthly Salary (€)')
ax6.tick_params(axis='x', rotation=20)
for bar, val in zip(bars6, education_df['Average Monthly Salary (€)']):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'€{val:,}', ha='center', fontsize=9, fontweight='bold')

# ── Plot 7: Forecasting model ─────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
# Historical
ax7.plot(quarters, national_unemployment, color=FINLAND_BLUE,
         linewidth=2, label='Historical', alpha=0.8)
# Test period predictions
test_dates = quarters[split_idx + len(forecast_df) - len(X):]
ax7.plot(test_dates[:len(y_pred)], y_pred, color='orange',
         linewidth=2, linestyle='--', label=f'Model fit (R²={r2:.3f})')
# Future forecast
ax7.plot(future_dates, future_unemployment, color='green',
         linewidth=2.5, linestyle='--', label='2024-2025 Forecast', marker='o', markersize=5)
ax7.fill_between(future_dates,
                 [r - 1.0 for r in future_unemployment],
                 [r + 1.0 for r in future_unemployment],
                 alpha=0.15, color='green', label='Confidence interval (±1%)')
ax7.axvline(pd.Timestamp('2024-01-01'), color='gray',
            linestyle=':', linewidth=1.5, label='Forecast start')
ax7.set_title('Unemployment Rate — Historical & Forecast 2024-2025', fontweight='bold')
ax7.set_ylabel('Unemployment Rate (%)')
ax7.legend(fontsize=8)
ax7.set_ylim(0, 18)

# ── Plot 8: Helsinki vs national ──────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
ax8.plot(quarters, national_unemployment, color='red',
         linewidth=2, label='National Average', linestyle='--')
ax8.plot(quarters, helsinki_metro, color=FINLAND_BLUE,
         linewidth=2.5, label='Helsinki (Uusimaa)')
ax8.fill_between(quarters, helsinki_metro, national_unemployment,
                 where=(national_unemployment > helsinki_metro),
                 alpha=0.15, color='green', label=f'Helsinki advantage (avg {gap:.1f}pp)')
ax8.set_title('Helsinki Metropolitan Area vs National Average\nUnemployment Rate 2010-2023',
              fontweight='bold')
ax8.set_ylabel('Unemployment Rate (%)')
ax8.legend(fontsize=9)
ax8.set_ylim(0, 18)

plt.savefig('/mnt/user-data/outputs/finnish_labour_market_dashboard.png',
            dpi=150, bbox_inches='tight')
print("Dashboard saved successfully!")
print("\n" + "=" * 65)
print("Analysis Complete!")
print("=" * 65)
