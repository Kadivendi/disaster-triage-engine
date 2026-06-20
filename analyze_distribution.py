import pandas as pd
import sys

def main():
    try:
        df = pd.read_csv("data/historical_events.csv")
    except FileNotFoundError:
        print("Dataset not found. Run training pipeline first to generate data.")
        sys.exit(1)
        
    print("Dataset loaded successfully.")
    print(f"Total events: {len(df)}")
    
    print("\nClass Distribution:")
    dist = df['severity'].value_counts(normalize=True) * 100
    for severity, pct in dist.items():
        print(f"  - {severity}: {pct:.2f}%")
        
    print("\nFeature Summary:")
    print(df.describe().round(2).T[['mean', 'std', 'min', 'max']])

if __name__ == "__main__":
    main()
