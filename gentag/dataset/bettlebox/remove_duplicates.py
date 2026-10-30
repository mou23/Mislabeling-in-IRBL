#!/usr/bin/env python3
"""
Script to remove duplicate bug entries from XML files in the bettlebox dataset.
Duplicates are identified by bug_id.
"""

import sys
from collections import OrderedDict

# Try to use lxml for better XML handling, fall back to ElementTree
try:
    from lxml import etree as ET
    USING_LXML = True
    print("Using lxml for XML parsing (more robust)")
except ImportError:
    import xml.etree.ElementTree as ET
    USING_LXML = False
    print("Using xml.etree.ElementTree for XML parsing")


def parse_xml_file(filepath):
    """Parse XML file and return the tree and root."""
    try:
        if USING_LXML:
            # lxml can handle more malformed XML with recover mode
            parser = ET.XMLParser(recover=True, encoding='utf-8', huge_tree=True)
            tree = ET.parse(filepath, parser=parser)
            root = tree.getroot()
        else:
            # Standard ElementTree
            tree = ET.parse(filepath)
            root = tree.getroot()
        return tree, root
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        print(f"Trying alternative parsing method...")
        try:
            # Try reading the file with explicit encoding and character replacement
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Replace invalid XML characters
            import re
            # Remove control characters except tab, newline, carriage return
            content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)

            if USING_LXML:
                parser = ET.XMLParser(recover=True, encoding='utf-8', huge_tree=True)
                root = ET.fromstring(content.encode('utf-8'), parser=parser)
            else:
                root = ET.fromstring(content)
            return None, root
        except Exception as e2:
            print(f"Alternative parsing also failed: {e2}")
            sys.exit(1)


def remove_duplicates(root):
    """
    Remove duplicate bug entries based on bug_id.
    Returns a new root with unique bugs and statistics.
    """
    seen_bug_ids = OrderedDict()
    duplicates = []
    
    # Iterate through all bug elements
    for bug in root.findall('bug'):
        bug_id_elem = bug.find('bug_id')
        if bug_id_elem is not None and bug_id_elem.text:
            bug_id = bug_id_elem.text.strip()
            
            if bug_id in seen_bug_ids:
                # This is a duplicate
                duplicates.append(bug_id)
            else:
                # First occurrence, keep it
                seen_bug_ids[bug_id] = bug
    
    # Create new root with unique bugs
    new_root = ET.Element('root')
    for bug in seen_bug_ids.values():
        new_root.append(bug)
    
    return new_root, len(seen_bug_ids), len(duplicates)


def write_xml_file(root, filepath):
    """Write the XML tree to a file with proper formatting."""
    try:
        if USING_LXML:
            # lxml has better formatting options
            tree = ET.ElementTree(root)
            tree.write(filepath, encoding='utf-8', xml_declaration=True,
                      pretty_print=True)
        else:
            # Standard ElementTree
            tree = ET.ElementTree(root)
            # indent is available in Python 3.9+
            try:
                ET.indent(tree, space="  ")
            except AttributeError:
                pass  # indent not available in older Python versions
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
        print(f"Written to: {filepath}")
    except Exception as e:
        print(f"Error writing {filepath}: {e}")
        sys.exit(1)


def main():
    """Main function to process both XML files."""
    files_to_process = [
        ('test.xml', 'test_no_duplicates.xml'),
        ('train.xml', 'train_no_duplicates.xml')
    ]
    
    print("=" * 60)
    print("Removing duplicates from bettlebox XML files")
    print("=" * 60)
    
    for input_file, output_file in files_to_process:
        print(f"\nProcessing: {input_file}")
        print("-" * 60)
        
        # Parse the input file
        tree, root = parse_xml_file(input_file)
        original_count = len(root.findall('bug'))
        print(f"Original bug count: {original_count}")
        
        # Remove duplicates
        new_root, unique_count, duplicate_count = remove_duplicates(root)
        print(f"Unique bugs: {unique_count}")
        print(f"Duplicates removed: {duplicate_count}")
        
        # Write the output file
        write_xml_file(new_root, output_file)
        
        # Verify
        verify_tree, verify_root = parse_xml_file(output_file)
        final_count = len(verify_root.findall('bug'))
        print(f"Verified final count: {final_count}")
        
        if final_count == unique_count:
            print(f"✓ Success: {output_file} created with {final_count} unique bugs")
        else:
            print(f"✗ Warning: Count mismatch in {output_file}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

