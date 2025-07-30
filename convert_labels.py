import pandas as pd
import os
import shutil

def convert_to_binary_classification():
    """Convert 3-class labels to 2-class (Buy/Sell only)"""
    
    # Read the enhanced labels
    labels_df = pd.read_csv('data/enhanced_labels.csv')
    
    print(f"Original dataset size: {len(labels_df)}")
    print("Original label distribution:")
    print(labels_df['label'].value_counts().sort_index())
    
    # Filter out HOLD class (label 2) and keep only BUY (1) and SELL (0)
    binary_df = labels_df[labels_df['label'] != 2].copy()
    
    print(f"\nFiltered dataset size: {len(binary_df)}")
    print("Filtered label distribution:")
    print(binary_df['label'].value_counts().sort_index())
    
    # Save the binary classification dataset
    binary_df.to_csv('data/binary_labels.csv', index=False)
    
    # Copy corresponding images to a new directory
    binary_chart_dir = 'data/binary_charts'
    os.makedirs(binary_chart_dir, exist_ok=True)
    
    copied_count = 0
    for _, row in binary_df.iterrows():
        src_path = os.path.join('data/enhanced_charts', row['filename'])
        dst_path = os.path.join(binary_chart_dir, row['filename'])
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
    
    print(f"\nCopied {copied_count} images to {binary_chart_dir}")
    print(f"Binary labels saved to data/binary_labels.csv")
    
    # Print final statistics
    print(f"\nFinal dataset statistics:")
    print(f"Total samples: {len(binary_df)}")
    print(f"SELL samples (label 0): {len(binary_df[binary_df['label'] == 0])}")
    print(f"BUY samples (label 1): {len(binary_df[binary_df['label'] == 1])}")
    
    return binary_df

if __name__ == "__main__":
    convert_to_binary_classification()
