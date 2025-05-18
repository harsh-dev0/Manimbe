import requests
import json
import time
import sys

# Base URL for the API
BASE_URL = "http://localhost:8000"  # Change this if your server is running on a different port

def test_api_credit_exhausted():
    """Test if the API credit exhausted URL is returned correctly"""
    print("Testing API credit exhausted scenario...")
    
    # Send a request that will trigger the API credit exhausted scenario
    # This is a complex prompt that should trigger the API credit exhausted fallback
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": "FORCE_API_CREDIT_EXHAUSTED_TEST"}
    )
    
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(response.text)
        return False
    
    # Get the job ID from the response
    data = response.json()
    job_id = data.get("id")
    
    if not job_id:
        print("Error: No job ID in response")
        return False
    
    print(f"Job ID: {job_id}")
    
    # Poll the job status until it's completed or failed
    max_attempts = 30
    for attempt in range(max_attempts):
        status_response = requests.get(f"{BASE_URL}/status/{job_id}")
        
        if status_response.status_code != 200:
            print(f"Error checking status: {status_response.status_code}")
            return False
        
        status_data = status_response.json()
        status = status_data.get("status")
        
        print(f"Status: {status}")
        
        if status in ["completed", "failed"]:
            break
            
        time.sleep(1)
    
    # Check if the video URL is the expected one for API credit exhausted
    expected_url = "https://manim-ai-videos.s3.us-east-1.amazonaws.com/videos/0251cdf9-76dc-4484-85d7-85dd8252ea89.mp4"
    actual_url = status_data.get("video_url")
    
    print(f"Expected URL: {expected_url}")
    print(f"Actual URL: {actual_url}")
    
    return expected_url == actual_url

def test_error_scenario():
    """Test if the error scenario URL is returned correctly"""
    print("\nTesting error scenario...")
    
    # Send a request that will trigger an error during video creation
    # This is a prompt that should cause an error in the Manim code execution
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": "FORCE_ERROR_SCENARIO_TEST"}
    )
    
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(response.text)
        return False
    
    # Get the job ID from the response
    data = response.json()
    job_id = data.get("id")
    
    if not job_id:
        print("Error: No job ID in response")
        return False
    
    print(f"Job ID: {job_id}")
    
    # Poll the job status until it's completed or failed
    max_attempts = 30
    for attempt in range(max_attempts):
        status_response = requests.get(f"{BASE_URL}/status/{job_id}")
        
        if status_response.status_code != 200:
            print(f"Error checking status: {status_response.status_code}")
            return False
        
        status_data = status_response.json()
        status = status_data.get("status")
        
        print(f"Status: {status}")
        
        if status in ["completed", "failed"]:
            break
            
        time.sleep(1)
    
    # Check if the video URL is the expected one for error scenario
    expected_url = "https://manim-ai-videos.s3.us-east-1.amazonaws.com/videos/ec808a73-d0e8-4573-9057-5c1580fa1e11.mp4"
    actual_url = status_data.get("video_url")
    
    print(f"Expected URL: {expected_url}")
    print(f"Actual URL: {actual_url}")
    
    return expected_url == actual_url

def main():
    """Run all tests"""
    api_credit_test_passed = test_api_credit_exhausted()
    error_test_passed = test_error_scenario()
    
    print("\nTest Results:")
    print(f"API Credit Exhausted Test: {'PASSED' if api_credit_test_passed else 'FAILED'}")
    print(f"Error Scenario Test: {'PASSED' if error_test_passed else 'FAILED'}")
    
    if api_credit_test_passed and error_test_passed:
        print("\nAll tests passed! URLs are being returned correctly.")
        return 0
    else:
        print("\nSome tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
