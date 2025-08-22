#!/usr/bin/env python3
"""Debug script to find the line 862 issue."""

def check_gpu_neurons_file():
    """Check the gpu_neurons.py file for syntax issues."""
    try:
        with open('core/gpu_neurons.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"File has {len(lines)} lines")
        
        # Check around line 862
        for i in range(860, min(870, len(lines))):
            line_num = i + 1
            line_content = lines[i].rstrip()
            print(f"Line {line_num}: {repr(line_content)}")
        
        # Look for unterminated triple quotes
        in_triple_quote = False
        triple_quote_start = None
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Count triple quotes in this line
            triple_quotes = line.count('"""')
            
            if triple_quotes % 2 == 1:  # Odd number of triple quotes
                if not in_triple_quote:
                    in_triple_quote = True
                    triple_quote_start = line_num
                    print(f"Triple quote started at line {line_num}")
                else:
                    in_triple_quote = False
                    print(f"Triple quote ended at line {line_num} (started at {triple_quote_start})")
                    triple_quote_start = None
        
        if in_triple_quote:
            print(f"ERROR: Unterminated triple quote starting at line {triple_quote_start}")
        else:
            print("No unterminated triple quotes found")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_gpu_neurons_file()