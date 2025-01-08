import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import csv
import json
from typing import List
from models.models import ReviewItem

def process_dataset(input_json: str, output_json: str):
    reviews = []
    
    with open(input_json, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        # Assuming the input JSON is a list of review objects
        for review_data in data:
            review = ReviewItem(
                reviewId=str(review_data['id']),
                text=review_data['comment'],
                type=review_data['label']
            )
            reviews.append(review.model_dump())
    
    output_data = {
        "reviews": reviews
    }
    
    with open(output_json, 'w', encoding='utf-8') as jsonfile:
        json.dump(output_data, jsonfile, indent=2)
        
    # Print debug traces
    total_reviews = len(output_data["reviews"])
    print(f"\nTotal number of reviews: {total_reviews}")
    
    # Count reviews per type
    type_counts = {}
    for review in output_data["reviews"]:
        review_type = review["type"]
        type_counts[review_type] = type_counts.get(review_type, 0) + 1
        
    print("\nReviews per type:")
    for review_type, count in sorted(type_counts.items()):
        print(f"{review_type}: {count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_dataset.py <input_json> <output_json>")
        sys.exit(1)
        
    input_json = sys.argv[1]
    output_json = sys.argv[2]
    process_dataset(input_json, output_json)
