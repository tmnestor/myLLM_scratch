"""
Dataset generation for text classification examples.

This module generates and saves classification datasets with examples from
four categories: Business, Sports, Science, and Technology.
"""

import pandas as pd
import numpy as np
import os
import sys

def generate_text_classification_datasets():
    """
    Generate classification datasets with examples from four categories:
    Business, Sports, Science, and Technology.
    
    Creates:
    - train.csv: Training dataset
    - val.csv: Validation dataset 
    - test.csv: Test dataset
    """
    
    # Base examples for each category
    base_texts = [
        "Stock markets plunge on fears of global recession",
        "Government announces new economic stimulus package",
        "Tech company launches revolutionary AI assistant",
        "Latest smartphone sales exceed expectations",
        "The football team won the championship last night",
        "Tennis player reaches semifinals after tough match",
        "Olympic committee announces host city for next games",
        "Basketball player signs record-breaking contract",
        "Space agency successfully launches new satellite",
        "Scientists discover new species in Amazon rainforest",
        "Research team makes breakthrough in quantum computing",
        "New medical treatment shows promising results in trials",
        "Global treaty signed to reduce carbon emissions",
        "Hurricane causes major damage in coastal cities",
        "President addresses nation on foreign policy",
        "Election results spark widespread protests"
    ]
    
    base_labels = [1, 1, 3, 3, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1]  # Corresponding labels
    
    # Expand dataset by creating variations
    texts = []
    labels = []
    
    # Business category variations (label 1)
    business_variations = [
        "Stock market reaches all-time high amid economic optimism",
        "Investors concerned about inflation as markets fluctuate",
        "Central bank announces interest rate decision",
        "Major corporation reports quarterly earnings above expectations",
        "Oil prices surge following supply disruptions",
        "Cryptocurrency values plummet in market correction",
        "New regulations impact financial sector operations",
        "Trade negotiations continue between major economies",
        "Companies announce merger to create industry giant",
        "Startup secures record funding from venture capitalists",
        "Economic indicators suggest continued growth",
        "Consumer spending increases during holiday season",
        "Labor market shows signs of improvement with new jobs",
        "Real estate market trends shifting in urban areas",
        "Banking sector implements new customer security measures",
        "Supply chain disruptions affect global manufacturing"
    ]
    
    # Sports category variations (label 0)
    sports_variations = [
        "Team celebrates victory in season finale",
        "Player sets new record in championship game",
        "Coach announces retirement after successful season",
        "Underdog team upsets favorite in playoff match",
        "Athlete overcomes injury to return to competition",
        "International tournament draws record viewership",
        "Team trades star player in surprising move",
        "New stadium construction announced for local team",
        "Athletes prepare for upcoming championship event",
        "Sports league announces rule changes for next season",
        "Team signs rookie player to long-term contract",
        "Fans return to stadiums as restrictions ease",
        "Rivalry match ends in dramatic finish",
        "Player wins award for outstanding performance",
        "Team faces sanctions for rule violations",
        "Marathon runner achieves personal best time"
    ]
    
    # Science/Tech category variations (label 2)
    science_variations = [
        "Astronomers observe distant galaxy for the first time",
        "Researchers develop new method for clean energy generation",
        "Study reveals impacts of climate change on ecosystems",
        "New species of deep-sea creatures discovered by expedition",
        "Genetic research provides insights into rare diseases",
        "Physics experiment confirms theoretical predictions",
        "Environmental monitoring shows pollution reduction",
        "Archaeological dig uncovers ancient civilization artifacts",
        "Scientists develop new method for carbon capture",
        "Marine biologists track migration patterns of ocean species",
        "Study documents effects of habitat restoration efforts",
        "Research team publishes findings on pandemic origins",
        "Geology survey identifies mineral deposits in remote region",
        "Conservation efforts save endangered species from extinction",
        "Weather patterns analyzed for climate prediction models",
        "Breakthrough in renewable energy storage announced"
    ]
    
    # Technology category variations (label 3)
    tech_variations = [
        "New smartphone features cutting-edge camera technology",
        "Social media platform introduces content safety features",
        "Tech company unveils latest virtual reality headset",
        "Software update addresses security vulnerabilities",
        "Streaming service expands content library for subscribers",
        "Electric vehicle manufacturer increases production capacity",
        "New app helps users manage personal finances",
        "Smart home devices gain popularity among consumers",
        "Gaming company releases highly anticipated title",
        "Cloud computing service expands global data centers",
        "Wearable technology monitors health metrics in real-time",
        "Robotics company demonstrates new automation capabilities",
        "Digital payment methods see increased adoption rates",
        "Operating system update improves device performance",
        "Cybersecurity experts warn of emerging threats",
        "E-commerce platform expands delivery options"
    ]
    
    # Add base examples
    for text, label in zip(base_texts, base_labels):
        texts.append(text)
        labels.append(label)
    
    # Add variations for each category
    for text in business_variations:
        texts.append(text)
        labels.append(1)
    
    for text in sports_variations:
        texts.append(text)
        labels.append(0)
    
    for text in science_variations:
        texts.append(text)
        labels.append(2)
    
    for text in tech_variations:
        texts.append(text)
        labels.append(3)
    
    # Repeat variations to reach approximately 10x the original size
    # We have 16 originals + 64 variations = 80 examples
    # To get 10x original (160), we need to add 80 more
    additional_needed = 16 * 10 - len(texts)
    
    if additional_needed > 0:
        # Add modified versions of existing examples by prepending phrases
        modifiers = [
            "Breaking news: ", "Update: ", "Just in: ", "New report: ",
            "Latest: ", "Analysts say ", "Sources confirm ", "Today's headlines: "
        ]
        
        current_size = len(texts)
        for i in range(additional_needed):
            # Select a random example and modifier
            idx = i % current_size
            modifier = modifiers[i % len(modifiers)]
            texts.append(modifier + texts[idx])
            labels.append(labels[idx])
    
    # Create DataFrame with category labels
    category_names = ["Sports", "Business", "Science", "Technology"]
    category_labels = [category_names[label] for label in labels]
    
    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "category": category_labels
    })
    
    # Split into train/val/test sets (70/15/15 split)
    np.random.seed(42)  # For reproducibility
    train_df, val_test_df = train_test_split(df, test_size=0.3)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5)
    
    # Ensure data directory exists
    # First try project root data directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save to CSV files
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
    print(f"Created datasets:")
    print(f"  - Training: {len(train_df)} examples")
    print(f"  - Validation: {len(val_df)} examples")
    print(f"  - Test: {len(test_df)} examples")
    print(f"Saved to {data_dir}/")


def train_test_split(df, test_size=0.2):
    """
    Split a dataframe into train and test sets, ensuring balanced class distribution.
    
    Args:
        df (DataFrame): Input dataframe
        test_size (float): Proportion of data to use for test set
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_indices = []
    test_indices = []
    
    # Group by label and split each group
    for label, group in df.groupby("label"):
        indices = group.index.tolist()
        np.random.shuffle(indices)
        
        # Calculate split point
        split = int(len(indices) * (1 - test_size))
        
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    
    return df.loc[train_indices], df.loc[test_indices]


if __name__ == "__main__":
    generate_text_classification_datasets()