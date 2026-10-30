# Duplicate Removal Summary

## Overview
This document summarizes the duplicate removal process for the bettlebox XML dataset files.

## Script
- **File**: `remove_duplicates.py`
- **Purpose**: Remove duplicate bug entries from XML files based on `bug_id`
- **Method**: Uses lxml (with fallback to xml.etree.ElementTree) for robust XML parsing

## Results

### test.xml
- **Original bug count**: 10,445
- **Unique bugs**: 8,955
- **Duplicates removed**: 1,490
- **Output file**: `test_no_duplicates.xml`

### train.xml
- **Original bug count**: 2,989
- **Unique bugs**: 2,989
- **Duplicates removed**: 0
- **Output file**: `train_no_duplicates.xml`

## Files Generated
1. `test_no_duplicates.xml` - Test dataset with duplicates removed
2. `train_no_duplicates.xml` - Train dataset with duplicates removed

## How to Use

Run the script from the bettlebox directory:
```bash
python remove_duplicates.py
```

## Technical Details
- Duplicates are identified by the `<bug_id>` field
- The first occurrence of each bug_id is kept
- XML structure and formatting are preserved
- The script handles malformed XML gracefully using lxml's recovery mode
- Output files maintain UTF-8 encoding with proper XML declaration

## Notes
- The train.xml file had no duplicates
- The test.xml file had 1,490 duplicate entries (14.3% of total)
- All duplicates have been successfully removed while preserving data integrity

