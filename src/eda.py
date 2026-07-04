import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
STATIC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'static'))

def run_eda():
    # Ensure output directory exists
    os.makedirs(STATIC_DIR, exist_ok=True)
    print(f"Ensuring output directory exists at: {STATIC_DIR}")
    
    # Load dataset
    csv_path = os.path.join(DATA_DIR, 'raw_transactions_ulb.csv')
    print(f"Loading raw transaction data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Expected dataset not found at {csv_path}. Please run data_ingestion.py first."
        )
        
    df = pd.read_csv(csv_path)
    
    # Global styling
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })

    # =========================================================================
    # PLOT 1: Class Imbalance Count Plot
    # =========================================================================
    print("Generating Plot 1: Class Imbalance Count Plot...")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    sns.countplot(
        x='Class', 
        data=df, 
        hue='Class',
        palette={0: '#1f77b4', 1: '#d62728'}, 
        legend=False,
        ax=ax1
    )
    
    ax1.set_title("Distribution of Transaction Classes (Log Scale)", pad=15)
    ax1.set_xlabel("Class (0: Legitimate, 1: Fraudulent)", labelpad=10)
    ax1.set_ylabel("Count (Logarithmic)", labelpad=10)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'])
    ax1.set_yscale('log')
    
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(
            f"{int(height):,}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center', 
            va='center', 
            xytext=(0, 8), 
            textcoords='offset points',
            weight='bold', 
            fontsize=12
        )
        
    plt.tight_layout()
    plot1_path = os.path.join(STATIC_DIR, 'class_imbalance.png')
    fig1.savefig(plot1_path, dpi=150)
    plt.close(fig1)
    print(f"Saved: {plot1_path}")

    # =========================================================================
    # PLOT 2: Amount Distribution (KDE)
    # =========================================================================
    print("Generating Plot 2: Transaction Amount Distribution (KDE)...")
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    
    subset_df = df[df['Amount'] <= 250]
    
    sns.kdeplot(
        data=subset_df[subset_df['Class'] == 0], 
        x='Amount', 
        fill=True, 
        color='#1f77b4', 
        label='Legitimate (Class 0)', 
        alpha=0.4, 
        linewidth=2,
        ax=ax2
    )
    
    sns.kdeplot(
        data=subset_df[subset_df['Class'] == 1], 
        x='Amount', 
        fill=True, 
        color='#d62728', 
        label='Fraudulent (Class 1)', 
        alpha=0.4, 
        linewidth=2,
        ax=ax2
    )
    
    ax2.set_title("Transaction Amount Distribution Density (Under $250)", pad=15)
    ax2.set_xlabel("Transaction Amount ($)", labelpad=10)
    ax2.set_ylabel("Probability Density", labelpad=10)
    ax2.set_xlim(0, 250)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plot2_path = os.path.join(STATIC_DIR, 'amount_distribution.png')
    fig2.savefig(plot2_path, dpi=150)
    plt.close(fig2)
    print(f"Saved: {plot2_path}")

    # =========================================================================
    # PLOT 3: Feature Correlation Heatmap
    # =========================================================================
    print("Generating Plot 3: Feature Correlation Heatmap...")
    fig3, ax3 = plt.subplots(figsize=(6, 10))
    
    corr_matrix = df.corr()
    class_corr = corr_matrix[['Class']].drop('Class').sort_values(by='Class', ascending=False)
    
    sns.heatmap(
        class_corr, 
        annot=True, 
        fmt=".3f", 
        cmap="coolwarm", 
        vmin=-0.4, 
        vmax=0.4, 
        cbar_kws={'label': 'Correlation Coefficient'},
        linewidths=0.5,
        ax=ax3
    )
    
    ax3.set_title("Correlation of Features with Target Class", pad=15)
    ax3.set_xlabel("")
    ax3.set_ylabel("Features", labelpad=10)
    
    plt.tight_layout()
    plot3_path = os.path.join(STATIC_DIR, 'correlation_heatmap.png')
    fig3.savefig(plot3_path, dpi=150)
    plt.close(fig3)
    print(f"Saved: {plot3_path}")

    # =========================================================================
    # PLOT 4: Diurnal Cycle of Transactions (Hidden Story 1)
    # =========================================================================
    # EDUCATIONAL NOTE ON TIME FEATURE DIURNAL CYCLE:
    # - The 'Time' feature represents seconds elapsed since the first transaction.
    # - There are 86,400 seconds in a day. We can convert 'Time' to a 24-hour scale:
    #   `Hour = (Time % 86400) / 3600`
    # - Real human transactions show a diurnal rhythm (low between 1:00 AM - 5:00 AM, 
    #   high during peak business hours).
    # - Fraudulent activity behaves differently: automated scripts or attackers 
    #   often execute fraud during sleeping hours when human vigilance is low, or they 
    #   operate globally across time zones, resulting in a flatter or night-skewed cycle.
    # - Comparing these curves exposes a highly informative pattern for classifiers.
    
    print("Generating Plot 4 (Hidden Story): Diurnal Cycle of Transactions...")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Extract hour of day (0 to 24)
    df['Hour'] = (df['Time'] % 86400) / 3600
    
    # Plot densities comparing Class 0 vs Class 1
    sns.kdeplot(
        data=df[df['Class'] == 0],
        x='Hour',
        fill=True,
        color='#1f77b4',
        label='Legitimate (Class 0)',
        alpha=0.3,
        linewidth=2.5,
        ax=ax4
    )
    sns.kdeplot(
        data=df[df['Class'] == 1],
        x='Hour',
        fill=True,
        color='#d62728',
        label='Fraudulent (Class 1)',
        alpha=0.3,
        linewidth=2.5,
        ax=ax4
    )
    
    ax4.set_title("Transaction Density Over Time of Day (24-Hour Cycle)", pad=15)
    ax4.set_xlabel("Hour of the Day (0.0 = Midnight, 12.0 = Noon)", labelpad=10)
    ax4.set_ylabel("Probability Density", labelpad=10)
    ax4.set_xlim(0, 24)
    ax4.set_xticks(range(0, 25, 2))
    ax4.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plot4_path = os.path.join(STATIC_DIR, 'time_diurnal_distribution.png')
    fig4.savefig(plot4_path, dpi=150)
    plt.close(fig4)
    print(f"Saved: {plot4_path}")

    # =========================================================================
    # PLOT 5: PCA Feature Space Separability (Hidden Story 2)
    # =========================================================================
    # EDUCATIONAL NOTE ON OVERPLOTTING & SEPARABILITY:
    # - V14 and V17 are the two features that have the strongest negative correlation 
    #   with our target Class (meaning lower values are associated with fraud).
    # - If we make a scatter plot of all 284,807 transactions, the huge pile of 
    #   legitimate transactions (blue) will cover all the fraud points (red) due 
    #   to "overplotting".
    # - To handle this:
    #   1. We downsample the Class 0 points to a representative sample (e.g., 5,000 points).
    #   2. We plot all 492 Class 1 (fraud) points on top of the sample.
    #   3. We adjust transparency (`alpha`) so density is readable.
    # - This clearly reveals that fraudulent transactions cluster in a specific 
    #   subregion of V14 vs V17, demonstrating why they are separable.
    
    print("Generating Plot 5 (Hidden Story): PCA Separability Scatter Plot...")
    fig5, ax5 = plt.subplots(figsize=(9, 7))
    
    # Downsample class 0 for clean visualization without overplotting
    df_class_0_sample = df[df['Class'] == 0].sample(n=5000, random_state=42)
    df_class_1 = df[df['Class'] == 1]
    
    # Plot legitimate samples
    ax5.scatter(
        df_class_0_sample['V14'], 
        df_class_0_sample['V17'], 
        c='#1f77b4', 
        label='Legitimate (Sampled)', 
        alpha=0.3, 
        s=15, 
        edgecolors='none'
    )
    
    # Plot all fraud points on top
    ax5.scatter(
        df_class_1['V14'], 
        df_class_1['V17'], 
        c='#d62728', 
        label='Fraudulent (All)', 
        alpha=0.7, 
        s=30, 
        edgecolors='black', 
        linewidths=0.5
    )
    
    ax5.set_title("PCA Feature Space Separability: V14 vs V17", pad=15)
    ax5.set_xlabel("V14 (Anonymized Principal Component)", labelpad=10)
    ax5.set_ylabel("V17 (Anonymized Principal Component)", labelpad=10)
    ax5.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plot5_path = os.path.join(STATIC_DIR, 'pca_separability_scatter.png')
    fig5.savefig(plot5_path, dpi=150)
    plt.close(fig5)
    print(f"Saved: {plot5_path}")

    print("All EDA Visualizations including hidden stories complete!")

if __name__ == "__main__":
    run_eda()
