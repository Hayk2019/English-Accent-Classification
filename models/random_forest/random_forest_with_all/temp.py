import pandas as pd

# Load your CSV
df = pd.read_csv("result.csv")  # Replace with actual filename

# Separate rows where accent is 'India'
india_rows = df[df['accent'] == 'India']

# Set how many you want to keep (e.g., 20% of them)
fraction_to_keep = 0.65  # Keep 20% of Indian accent row

# Sample the desired fraction randomly
reduced_india = india_rows.sample(frac=fraction_to_keep, random_state=42)

# Get all other accents unchanged
other_rows = df[df['accent'] != 'India']

# Combine reduced India rows with others
final_df = pd.concat([reduced_india, other_rows])

# Optional: Shuffle the rows again if you like
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new CSV
final_df.to_csv("reduced_india_accent.csv", index=False)

print("Reduced 'India' accent rows and saved to 'reduced_india_accent.csv'")

