import os
import glob
import argparse

def adjust_labels(input_dir, output_dir, min_padding=0.2, max_padding=0.5, padding_percentage=0.5):
    """
    Adjusts labels by adding padding based on duration.
    
    padding = clamp(duration * padding_percentage, min_padding, max_padding)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    search_pattern = os.path.join(input_dir, '*.txt')
    label_files = glob.glob(search_pattern)
    
    print(f"Found {len(label_files)} label files in {input_dir}")
    
    for file_path in label_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                    
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2]
                
                duration = end - start
                
                # Calculate padding
                calculated_padding = duration * padding_percentage
                padding = max(min_padding, min(max_padding, calculated_padding))
                
                # Apply padding
                new_start = max(0.0, start - padding)
                new_end = end + padding
                
                # Write new line
                f_out.write(f"{new_start:.6f}\t{new_end:.6f}\t{label}\n")
                
        print(f"Processed: {filename} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Adjust label padding')
    parser.add_argument('--input_dir', type=str, default='data/01_raw/labels', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='data/01_raw/labels_padded', help='Output directory')
    parser.add_argument('--min_padding', type=float, default=0.2, help='Minimum padding in seconds')
    parser.add_argument('--max_padding', type=float, default=0.5, help='Maximum padding in seconds')
    parser.add_argument('--padding_percentage', type=float, default=0.5, help='Padding percentage of duration')
    
    args = parser.parse_args()
    
    adjust_labels(args.input_dir, args.output_dir, args.min_padding, args.max_padding, args.padding_percentage)

if __name__ == "__main__":
    main()
