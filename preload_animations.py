import os
import json
import logging
from s3_storage import S3Storage
from animation_cache import AnimationCache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common animation prompts that are likely to be requested
COMMON_ANIMATIONS = [
    {
        "prompt": "Show the Pythagorean theorem with a visual proof",
        "title": "Pythagorean Theorem Visual Proof",
        "job_id": "pythagorean_theorem",
        "s3_key": "preloaded/pythagorean_theorem.mp4"
    },
    {
        "prompt": "Demonstrate the quadratic formula with animation",
        "title": "Quadratic Formula Visualization",
        "job_id": "quadratic_formula",
        "s3_key": "preloaded/quadratic_formula.mp4"
    },
    {
        "prompt": "Show how derivatives work with a visual explanation",
        "title": "Derivative Concept Visualization",
        "job_id": "derivatives",
        "s3_key": "preloaded/derivatives.mp4"
    },
    {
        "prompt": "Explain the concept of integration with area under a curve",
        "title": "Integration as Area Under Curve",
        "job_id": "integration",
        "s3_key": "preloaded/integration.mp4"
    },
    {
        "prompt": "Visualize the unit circle and trigonometric functions",
        "title": "Unit Circle and Trigonometry",
        "job_id": "unit_circle",
        "s3_key": "preloaded/unit_circle.mp4"
    },
    {
        "prompt": "Show how matrix multiplication works visually",
        "title": "Matrix Multiplication Visualization",
        "job_id": "matrix_multiplication",
        "s3_key": "preloaded/matrix_multiplication.mp4"
    },
    {
        "prompt": "Demonstrate the binomial theorem with animation",
        "title": "Binomial Theorem Visualization",
        "job_id": "binomial_theorem",
        "s3_key": "preloaded/binomial_theorem.mp4"
    },
    {
        "prompt": "Explain the concept of limits in calculus",
        "title": "Limits in Calculus",
        "job_id": "limits",
        "s3_key": "preloaded/limits.mp4"
    },
    {
        "prompt": "Visualize the Fibonacci sequence and golden ratio",
        "title": "Fibonacci and Golden Ratio",
        "job_id": "fibonacci",
        "s3_key": "preloaded/fibonacci.mp4"
    },
    {
        "prompt": "Show how probability distributions work",
        "title": "Probability Distributions",
        "job_id": "probability",
        "s3_key": "preloaded/probability.mp4"
    }
]

def main():
    # Initialize S3 storage
    s3_storage = S3Storage()
    
    if not s3_storage.is_enabled:
        logger.error("S3 storage is not configured. Cannot preload animations.")
        return
    
    # Initialize animation cache
    animation_cache = AnimationCache(s3_storage)
    
    # Preload animations
    animation_cache.preload_common_animations(COMMON_ANIMATIONS)
    
    logger.info(f"Preloaded {len(COMMON_ANIMATIONS)} common animations to cache")
    
    # Save the list of preloaded animations for reference
    with open("preloaded_animations.json", "w") as f:
        json.dump(COMMON_ANIMATIONS, f, indent=2)
    
    logger.info("Saved preloaded animations list to preloaded_animations.json")

if __name__ == "__main__":
    main()
