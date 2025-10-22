import os
import shutil
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict

def get_validation_files_by_class(val_anno_dir):
    """
    Get validation files for each class using annotation files
    Args:
        val_anno_dir: Path to validation annotation directory
    Returns:
        dict: Mapping of class_id to list of validation filenames
    """
    class_to_files = defaultdict(list)
    
    print("Reading validation annotations...")
    for anno_file in tqdm(os.listdir(val_anno_dir)):
        if not anno_file.endswith('.xml'):
            continue
        
        # Parse annotation file
        tree = ET.parse(os.path.join(val_anno_dir, anno_file))
        root = tree.getroot()
        
        # Get image filename
        filename = root.find('filename').text + '.JPEG'
        
        # Get class(es) in this image
        for obj in root.findall('object'):
            class_id = obj.find('name').text
            class_to_files[class_id].append(filename)
    
    return class_to_files

def create_subset_structure(src_root, dst_root, subset_percentage=0.1):
    """
    Create a subset of ImageNet dataset maintaining the original structure
    Args:
        src_root: Path to original ILSVRC directory
        dst_root: Path where subset will be created
        subset_percentage: Percentage of data to include (default: 0.1 for 10%)
    """
    # Create main directories
    for main_dir in ['Annotations', 'Data', 'ImageSets']:
        os.makedirs(os.path.join(dst_root, main_dir, 'CLS-LOC'), exist_ok=True)
    
    # Get list of all classes from training directory
    train_dir = os.path.join(src_root, 'Data', 'CLS-LOC', 'train')
    all_classes = os.listdir(train_dir)
    
    # Randomly select classes
    num_classes = int(len(all_classes) * subset_percentage)
    selected_classes = set(random.sample(all_classes, num_classes))
    
    print(f"Selected {len(selected_classes)} classes out of {len(all_classes)}")
    
    # Save selected classes
    with open(os.path.join(dst_root, 'selected_classes.txt'), 'w') as f:
        for class_id in selected_classes:
            f.write(f"{class_id}\n")
    
    # Process training data
    print("\nProcessing training data...")
    for class_id in tqdm(selected_classes):
        # Copy training images
        src_class_path = os.path.join(train_dir, class_id)
        dst_class_path = os.path.join(dst_root, 'Data', 'CLS-LOC', 'train', class_id)
        if os.path.exists(src_class_path):
            shutil.copytree(src_class_path, dst_class_path, dirs_exist_ok=True)
        
        # Copy training annotations
        src_anno_path = os.path.join(src_root, 'Annotations', 'CLS-LOC', 'train', class_id)
        dst_anno_path = os.path.join(dst_root, 'Annotations', 'CLS-LOC', 'train', class_id)
        if os.path.exists(src_anno_path):
            shutil.copytree(src_anno_path, dst_anno_path, dirs_exist_ok=True)
    
    # Process validation data
    print("\nProcessing validation data...")
    val_anno_src = os.path.join(src_root, 'Annotations', 'CLS-LOC', 'val')
    val_anno_dst = os.path.join(dst_root, 'Annotations', 'CLS-LOC', 'val')
    val_data_src = os.path.join(src_root, 'Data', 'CLS-LOC', 'val')
    val_data_dst = os.path.join(dst_root, 'Data', 'CLS-LOC', 'val')
    
    # Create validation directories
    os.makedirs(val_anno_dst, exist_ok=True)
    os.makedirs(val_data_dst, exist_ok=True)
    
    # Get validation files for each class
    class_to_files = get_validation_files_by_class(val_anno_src)
    
    # Copy validation files for selected classes
    print("\nCopying validation files...")
    copied_count = defaultdict(int)
    for class_id in selected_classes:
        val_files = class_to_files.get(class_id, [])
        for filename in val_files:
            # Copy image
            src_img = os.path.join(val_data_src, filename)
            dst_img = os.path.join(val_data_dst, filename)
            shutil.copy2(src_img, dst_img)
            
            # Copy annotation
            anno_file = filename.replace('.JPEG', '.xml')
            src_anno = os.path.join(val_anno_src, anno_file)
            dst_anno = os.path.join(val_anno_dst, anno_file)
            shutil.copy2(src_anno, dst_anno)
            
            copied_count[class_id] += 1
    
    # Print validation statistics
    print("\nValidation files copied per class:")
    for class_id in selected_classes:
        count = copied_count[class_id]
        print(f"Class {class_id}: {count} files")
    
    # Organize validation files into class folders
    print("\nOrganizing validation files into class folders...")
    organize_validation_files(val_data_dst, val_anno_dst)
    
    # Process ImageSets
    print("\nProcessing ImageSets...")
    imagesets_src = os.path.join(src_root, 'ImageSets', 'CLS-LOC')
    imagesets_dst = os.path.join(dst_root, 'ImageSets', 'CLS-LOC')
    
    # Copy and filter ImageSets files
    for file in os.listdir(imagesets_src):
        with open(os.path.join(imagesets_src, file), 'r') as f:
            lines = f.readlines()
        
        filtered_lines = [line for line in lines 
                         if any(class_id in line for class_id in selected_classes)]
        
        with open(os.path.join(imagesets_dst, file), 'w') as f:
            f.writelines(filtered_lines)
    
    print(f"\nSubset created in: {dst_root}")
    print(f"Number of classes: {len(selected_classes)}")

def organize_validation_files(val_dir, anno_dir):
    """
    Organize validation files into class folders using annotations
    """
    # Create temporary directory
    temp_dir = val_dir + "_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Move all files to temporary directory
    for filename in os.listdir(val_dir):
        if filename.endswith('.JPEG'):
            shutil.move(os.path.join(val_dir, filename), 
                       os.path.join(temp_dir, filename))
    
    # Create class directories and move files
    print("Moving files to class directories...")
    for anno_file in tqdm(os.listdir(anno_dir)):
        if not anno_file.endswith('.xml'):
            continue
        
        # Parse annotation file
        tree = ET.parse(os.path.join(anno_dir, anno_file))
        root = tree.getroot()
        
        # Get image filename and class
        filename = root.find('filename').text + '.JPEG'
        class_id = root.find('object').find('name').text
        
        # Create class directory
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        
        # Move file to class directory
        src_path = os.path.join(temp_dir, filename)
        dst_path = os.path.join(class_dir, filename)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
    
    # Clean up
    shutil.rmtree(temp_dir)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to original ILSVRC directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path where subset will be created')
    parser.add_argument('--percentage', type=float, default=0.1,
                      help='Percentage of classes to include (default: 0.1)')
    
    args = parser.parse_args()
    
    create_subset_structure(args.data_dir, args.output_dir, args.percentage)