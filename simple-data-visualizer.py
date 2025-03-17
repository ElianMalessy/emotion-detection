import os
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image
import torch
from torch.utils.data import DataLoader
from models import CNN, EmotionsDataset  # Make sure this import works with your project structure
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter

def analyze_emotion_dataset(csv_path, img_dir, model_path=None, output_dir="emotion_analysis"):
    """
    Analyze the emotion dataset and visualize various aspects
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file with image data
    img_dir : str
        Path to the directory containing images
    model_path : str or None
        Path to the trained model file (optional)
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {csv_path}...")
    data = pl.read_csv(csv_path)
    
    # Determine column names
    columns = data.columns
    
    # Assuming format is similar to: user.id, image, emotion
    image_col = columns[1] if len(columns) > 2 else columns[0]  # Second column should be image filename
    emotion_col = columns[2] if len(columns) > 2 else columns[1]  # Third column should be emotion
    
    # Check if user.id column exists and drop it if needed
    if "user.id" in data.columns:
        data = data.drop("user.id")
        # After dropping, we need to make sure we use the correct indices
        image_col = data.columns[0]  # First column should now be image filename
        emotion_col = data.columns[1]  # Second column should now be emotion
    
    # Ensure emotion column has lowercase values
    data = data.with_columns(
        pl.col(emotion_col).str.to_lowercase().alias("emotion")
    )
    emotion_col = "emotion"  # We've created a new standardized column
    
    # Get emotion classes
    emotions = data[emotion_col].unique().sort().to_list()
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
    
    print(f"Found {len(emotions)} emotion classes: {emotions}")
    print(f"Total number of images: {len(data)}")
    
    # 1. Analyze and visualize class distribution
    print("\n1. Analyzing class distribution...")
    emotion_counts = data.group_by(emotion_col).agg(pl.len().alias("count"))
    total = emotion_counts["count"].sum()
    emotion_counts = emotion_counts.with_columns(
        (pl.col("count") / total * 100).alias("percentage")
    )
    
    # Sort by count for better visualization
    emotion_counts = emotion_counts.sort("count", descending=True)
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.bar(emotion_counts[emotion_col], emotion_counts["count"])
    plt.title("Distribution of Emotions in Dataset")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for i, count in enumerate(emotion_counts["count"]):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "emotion_distribution.png"))
    plt.close()
    
    print("Class distribution statistics:")
    for row in emotion_counts.iter_rows(named=True):
        print(f"  {row[emotion_col]}: {row['count']} images ({row['percentage']:.2f}%)")
    
    # 2. Analyze image properties (from a sample)
    print("\n2. Analyzing image properties...")
    sample_size = min(500, len(data))
    sample_indices = np.random.choice(len(data), sample_size, replace=False)
    
    dimensions = []
    grayscale_count = 0
    color_count = 0
    
    for idx in sample_indices:
        # Convert to Python int to avoid numpy int32 issues
        idx = int(idx)
        # Get the filename using .item() to extract the value
        img_filename = data[idx, image_col]
        img_path = os.path.join(img_dir, img_filename)
        try:
            img = Image.open(img_path)
            dimensions.append(img.size)
            if img.mode == 'L':
                grayscale_count += 1
            else:
                color_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate stats on dimensions
    widths = [d[0] for d in dimensions]
    heights = [d[1] for d in dimensions]
    
    print(f"Dimension Statistics (from {sample_size} samples):")
    print(f"  Width - Mean: {np.mean(widths):.2f}, Std: {np.std(widths):.2f}, Min: {min(widths)}, Max: {max(widths)}")
    print(f"  Height - Mean: {np.mean(heights):.2f}, Std: {np.std(heights):.2f}, Min: {min(heights)}, Max: {max(heights)}")
    print(f"  Grayscale images: {grayscale_count} ({grayscale_count/len(dimensions)*100:.2f}%)")
    print(f"  Color images: {color_count} ({color_count/len(dimensions)*100:.2f}%)")
    
    # Plot dimension distribution
    plt.figure(figsize=(10, 5))
    plt.hist2d(widths, heights, bins=20, cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('Distribution of Image Dimensions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_dimensions.png"))
    plt.close()
    
    # 3. Visualize sample images for each emotion
    print("\n3. Visualizing sample images for each emotion...")
    
    # Determine the number of emotions and samples per emotion
    num_emotions = len(emotions)
    samples_per_emotion = min(5, min([len(data.filter(pl.col(emotion_col) == emotion)) for emotion in emotions]))
    
    # Create grid of sample images
    if samples_per_emotion > 0:
        fig_height = 3 * num_emotions
        plt.figure(figsize=(15, fig_height))
        
        for i, emotion in enumerate(emotions):
            emotion_data = data.filter(pl.col(emotion_col) == emotion)
            
            # Get random samples
            if len(emotion_data) > samples_per_emotion:
                sample_idxs = np.random.choice(len(emotion_data), samples_per_emotion, replace=False)
            else:
                sample_idxs = range(len(emotion_data))
            
            for j, s_idx in enumerate(sample_idxs):
                # Convert to Python int to avoid numpy int32 issues
                s_idx = int(s_idx)
                # Get the filename
                img_filename = emotion_data[s_idx, image_col]
                img_path = os.path.join(img_dir, img_filename)
                try:
                    img = Image.open(img_path)
                    plt.subplot(num_emotions, samples_per_emotion, i*samples_per_emotion + j + 1)
                    plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
                    if j == 0:
                        plt.title(f"{emotion}")
                    plt.axis('off')
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "emotion_samples.png"))
        plt.close()
    
    # 4. Analyze train-test split
    print("\n4. Analyzing train-test split...")
    
    # Get labels for stratification
    labels = data[emotion_col].to_list()
    
    # Perform stratified split
    train_indices, val_indices = train_test_split(
        range(len(data)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    
    # Count emotions in training and validation sets
    train_emotions = [labels[i] for i in train_indices]
    val_emotions = [labels[i] for i in val_indices]
    
    train_counts = Counter(train_emotions)
    val_counts = Counter(val_emotions)
    
    # Plot split comparison
    plt.figure(figsize=(12, 6))
    
    # Set up for grouped bar chart
    bar_width = 0.35
    index = np.arange(len(emotions))
    
    # Calculate percentages
    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    
    train_percents = [train_counts.get(emotion, 0) / train_total * 100 for emotion in emotions]
    val_percents = [val_counts.get(emotion, 0) / val_total * 100 for emotion in emotions]
    
    # Plot percentages
    plt.bar(index - bar_width/2, train_percents, bar_width, label='Training')
    plt.bar(index + bar_width/2, val_percents, bar_width, label='Validation')
    
    plt.xlabel('Emotion')
    plt.ylabel('Percentage')
    plt.title('Emotion Distribution in Train/Val Sets')
    plt.xticks(index, emotions, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_split.png"))
    plt.close()
    
    print(f"Training set: {len(train_indices)} images ({len(train_indices)/(len(train_indices)+len(val_indices))*100:.1f}%)")
    print(f"Validation set: {len(val_indices)} images ({len(val_indices)/(len(train_indices)+len(val_indices))*100:.1f}%)")
    
    # 5. Per-User Analysis (if user.id is available in the original CSV)
    print("\n5. Performing per-user analysis...")
    
    try:
        # Load the data with user.id
        user_data = pl.read_csv(csv_path)
        
        if "user.id" in user_data.columns:
            # Get column indices
            user_id_col = "user.id"
            user_img_col = user_data.columns[1]  # Second column should be image filename
            user_emotion_col = user_data.columns[2]  # Third column should be emotion
            
            # Ensure emotion column is lowercase
            user_data = user_data.with_columns(
                pl.col(user_emotion_col).str.to_lowercase().alias("emotion")
            )
            user_emotion_col = "emotion"
            
            # Get count of unique users
            unique_users = user_data[user_id_col].n_unique()
            
            # Count emotions per user
            user_emotion_counts = user_data.group_by([user_id_col, user_emotion_col]).agg(pl.len().alias("count"))
            
            # Get emotion diversity per user
            user_diversity = user_emotion_counts.group_by(user_id_col).agg(
                pl.n_unique(user_emotion_col).alias("unique_emotions")
            )
            
            # Get total images per user
            user_totals = user_emotion_counts.group_by(user_id_col).agg(
                pl.sum("count").alias("total_images")
            )
            
            # Plot user statistics
            plt.figure(figsize=(12, 6))
            
            # Distribution of unique emotions per user
            plt.hist(user_diversity["unique_emotions"], bins=range(1, len(emotions) + 2))
            plt.xlabel('Number of Unique Emotions')
            plt.ylabel('Number of Users')
            plt.title('Distribution of Emotion Diversity per User')
            plt.xticks(range(1, len(emotions) + 1))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "user_emotion_diversity.png"))
            plt.close()
            
            # Distribution of total images per user
            plt.figure(figsize=(12, 6))
            plt.hist(user_totals["total_images"], bins=20)
            plt.xlabel('Number of Images')
            plt.ylabel('Number of Users')
            plt.title('Distribution of Image Count per User')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "user_image_count.png"))
            plt.close()
            
            print(f"Total Users: {unique_users}")
            print(f"Emotions per User: mean={user_diversity['unique_emotions'].mean():.2f}")
            print(f"Images per User: mean={user_totals['total_images'].mean():.2f}")
        else:
            print("No user.id column found in the CSV. Skipping per-user analysis.")
    except Exception as e:
        print(f"Error during per-user analysis: {e}")
        print("Skipping per-user analysis.")
    
    # 6. Model Performance (if model is available)
    if model_path and os.path.exists(model_path):
        print("\n6. Analyzing model performance...")
        
        try:
            # Initialize device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load the model
            model = CNN().to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # Create a validation dataset and dataloader
            val_dataset = EmotionsDataset(data, emotion_to_idx, img_dir, train=False)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            # Evaluate model
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            # Create confusion matrix (raw counts)
            cm = np.zeros((len(emotions), len(emotions)), dtype=int)
            for true_label, pred_label in zip(all_labels, all_preds):
                cm[true_label, pred_label] += 1
            
            # Create normalized confusion matrix (row-wise percentages)
            # Each row will sum to 100%, showing how predictions are distributed for each true class
            cm_percentage = np.zeros((len(emotions), len(emotions)))
            for i in range(len(emotions)):
                if np.sum(cm[i, :]) > 0:  # Avoid division by zero
                    cm_percentage[i, :] = cm[i, :] / np.sum(cm[i, :]) * 100
            
            # Plot raw count confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=emotions, yticklabels=emotions)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix (Raw Counts)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix_raw.png"))
            plt.close()
            
            # Plot percentage confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', 
                        xticklabels=emotions, yticklabels=emotions, vmin=0, vmax=100)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix (Percentages by Row)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix_percentage.png"))
            plt.close()
            
            # Calculate accuracy per class
            class_correct = np.zeros(len(emotions))
            class_total = np.zeros(len(emotions))
            
            for i in range(len(all_labels)):
                label = all_labels[i]
                class_total[label] += 1
                if all_preds[i] == label:
                    class_correct[label] += 1
            
            class_acc = class_correct / class_total
            
            # Plot per-class accuracy
            plt.figure(figsize=(12, 6))
            plt.bar(emotions, class_acc)
            plt.xlabel('Emotion')
            plt.ylabel('Accuracy')
            plt.title('Accuracy by Emotion Class')
            plt.xticks(rotation=45)
            
            # Add accuracy labels
            for i, acc in enumerate(class_acc):
                plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "class_accuracy.png"))
            plt.close()
            
            # Overall accuracy
            overall_acc = np.sum(class_correct) / np.sum(class_total)
            print(f"Overall Accuracy: {overall_acc:.4f}")
            
            print("Per-class Accuracy:")
            for i, emotion in enumerate(emotions):
                print(f"  {emotion}: {class_acc[i]:.4f}")
            
        except Exception as e:
            print(f"Error during model performance analysis: {e}")
            print("Skipping model performance analysis.")
    else:
        print("No model file provided or found. Skipping model performance analysis.")
    
    # Generate an HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Detection Dataset Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            .image-container {{ margin: 20px 0; text-align: center; }}
            .image-container img {{ max-width: 100%; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .caption {{ font-style: italic; color: #7f8c8d; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>Emotion Detection Dataset Analysis</h1>
        
        <h2>1. Class Distribution</h2>
        <div class="image-container">
            <img src="emotion_distribution.png" alt="Emotion Distribution">
            <div class="caption">Distribution of emotion classes in the dataset</div>
        </div>
        
        <h2>2. Image Properties</h2>
        <div class="image-container">
            <img src="image_dimensions.png" alt="Image Dimensions">
            <div class="caption">Distribution of image dimensions in the dataset</div>
        </div>
        
        <h2>3. Sample Images</h2>
        <div class="image-container">
            <img src="emotion_samples.png" alt="Emotion Samples">
            <div class="caption">Sample images for each emotion category</div>
        </div>
        
        <h2>4. Train-Test Split</h2>
        <div class="image-container">
            <img src="train_val_split.png" alt="Train-Test Split">
            <div class="caption">Comparison of emotion distribution in training and validation sets</div>
        </div>
        
        <h2>5. Per-User Analysis</h2>
        <div class="image-container">
            <img src="user_emotion_diversity.png" alt="User Emotion Diversity">
            <div class="caption">Distribution of emotion diversity per user</div>
        </div>
        <div class="image-container">
            <img src="user_image_count.png" alt="User Image Count">
            <div class="caption">Distribution of image count per user</div>
        </div>
        
        {f'''
        <h2>6. Model Performance</h2>
        <div class="image-container">
            <img src="confusion_matrix_raw.png" alt="Confusion Matrix (Raw)">
            <div class="caption">Confusion matrix showing raw counts of predictions</div>
        </div>
        <div class="image-container">
            <img src="confusion_matrix_percentage.png" alt="Confusion Matrix (Percentage)">
            <div class="caption">Confusion matrix showing percentages by row (rows sum to 100%)</div>
        </div>
        <div class="image-container">
            <img src="class_accuracy.png" alt="Class Accuracy">
            <div class="caption">Accuracy by emotion class</div>
        </div>
        ''' if model_path and os.path.exists(model_path) else ''}
        
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write(html_content)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Open {os.path.join(output_dir, 'report.html')} to view the report.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze emotion detection dataset")
    parser.add_argument("--csv", type=str, default="data/legend.csv", help="Path to the CSV file")
    parser.add_argument("--img_dir", type=str, default="images", help="Path to the image directory")
    parser.add_argument("--model", type=str, default=None, help="Path to the trained model file")
    parser.add_argument("--output", type=str, default="emotion_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    analyze_emotion_dataset(args.csv, args.img_dir, args.model, args.output)