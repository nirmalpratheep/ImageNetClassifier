import os
from collections import defaultdict

def verify_dataset_structure(data_dir):
    """
    Verify that the dataset has the correct structure for training
    Args:
        data_dir: Path to ILSVRC subset directory
    Returns:
        bool: True if structure is valid, False otherwise
    """
    # Required directory structure
    required_dirs = [
        os.path.join('Data', 'CLS-LOC', 'train'),
        os.path.join('Data', 'CLS-LOC', 'val')
    ]
    
    # Check main directories exist
    print("\nChecking directory structure...")
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing directory: {full_path}")
            return False
        print(f"✓ Found directory: {full_path}")
    
    # Get train and val directories
    train_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'train')
    val_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'val')
    
    # Check class directories
    print("\nChecking class directories...")
    train_classes = set(os.listdir(train_dir))
    val_classes = set(os.listdir(val_dir))
    
    print(f"Found {len(train_classes)} classes in training set")
    print(f"Found {len(val_classes)} classes in validation set")
    
    # Check class consistency
    missing_in_val = train_classes - val_classes
    missing_in_train = val_classes - train_classes
    
    if missing_in_val:
        print(f"\n❌ Classes in train but missing in val: {missing_in_val}")
        return False
    if missing_in_train:
        print(f"\n❌ Classes in val but missing in train: {missing_in_train}")
        return False
    print("\n✓ All classes present in both train and val sets")
    
    # Check image files
    print("\nChecking image files...")
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int)
    }
    
    # Count training files
    for class_id in train_classes:
        train_class_dir = os.path.join(train_dir, class_id)
        images = [f for f in os.listdir(train_class_dir) if f.endswith('.JPEG')]
        stats['train'][class_id] = len(images)
        if len(images) == 0:
            print(f"❌ No images found in train/{class_id}")
            return False
    
    # Count validation files
    for class_id in val_classes:
        val_class_dir = os.path.join(val_dir, class_id)
        images = [f for f in os.listdir(val_class_dir) if f.endswith('.JPEG')]
        stats['val'][class_id] = len(images)
        if len(images) == 0:
            print(f"❌ No images found in val/{class_id}")
            return False
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Total classes: {len(train_classes)}")
    
    print("\nTraining set:")
    print(f"Total images: {sum(stats['train'].values())}")
    print(f"Average images per class: {sum(stats['train'].values()) / len(train_classes):.1f}")
    print(f"Min images in a class: {min(stats['train'].values())}")
    print(f"Max images in a class: {max(stats['train'].values())}")
    
    print("\nValidation set:")
    print(f"Total images: {sum(stats['val'].values())}")
    print(f"Average images per class: {sum(stats['val'].values()) / len(val_classes):.1f}")
    print(f"Min images in a class: {min(stats['val'].values())}")
    print(f"Max images in a class: {max(stats['val'].values())}")
    
    # Check dataset stats file
    stats_file = os.path.join(data_dir, 'dataset_stats.txt')
    if not os.path.exists(stats_file):
        print(f"\n❌ Missing dataset_stats.txt file")
        print("Please run calculate_stats.py first")
        return False
    
    # Verify stats file format
    try:
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            if len(lines) != 2:
                print(f"\n❌ Invalid dataset_stats.txt format")
                return False
            # Try to evaluate the lines as Python lists
            mean = eval(lines[0].strip())
            std = eval(lines[1].strip())
            if not (isinstance(mean, list) and isinstance(std, list) and
                   len(mean) == 3 and len(std) == 3):
                print(f"\n❌ Invalid mean/std format in dataset_stats.txt")
                return False
    except:
        print(f"\n❌ Error reading dataset_stats.txt")
        return False
    
    print("\n✓ Dataset structure verification completed successfully!")
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify ImageNet subset structure')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to ILSVRC subset directory')
    
    args = parser.parse_args()
    
    if verify_dataset_structure(args.data_dir):
        print("\n✓ Dataset is ready for training!")
    else:
        print("\n❌ Dataset verification failed!")
        print("Please fix the issues above before training.") 