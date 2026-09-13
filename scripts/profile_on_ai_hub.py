import os
import json
import qai_hub as hub

def profile_on_ai_hub():
    """
    Uploads the ONNX model to Qualcomm AI Hub, compiles it,
    and runs a profiling job on a Snapdragon X Elite physical device.
    """
    model_path = os.path.join("models", "card_detector.onnx")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running scripts/train_card_detector.py")
        return

    print("Uploading model to Qualcomm AI Hub...")
    model = hub.upload_model(model_path)
    print(f"Model uploaded successfully. Model ID: {model.model_id}")

    # Specify the target device for profiling
    target_device_name = "Snapdragon X Elite CRD"
    print(f"Configuring device: {target_device_name}...")
    device = hub.Device(target_device_name)
    
    print("Submitting profiling job to AI Hub...")
    # A profile job typically implies a compile step for ONNX on QNN
    profile_job = hub.submit_profile_job(
        model=model,
        device=device,
    )
    
    print(f"Job submitted! You can view the dashboard here:")
    print(f"URL: {profile_job.url}")
    print("\nWaiting for the job to complete on cloud hardware... this may take a few minutes.")
    
    profile_job.wait()
    print("Job completed!")
    
    # Download the profile metrics
    print("Downloading profiling results...")
    profile_data = profile_job.download_profile()
    
    # Write the results out to a JSON file
    output_json = "profiling_results.json"
    
    # We dump the raw output or format it cleanly
    with open(output_json, "w") as f:
        json.dump(profile_data, f, indent=4)
        
    print(f"Saved complete profiling metrics to {output_json}")
    
    # Output quick summary to terminal if possible
    try:
        # Based on qai-hub dict structure
        inference_time = profile_data['execution_summary']['estimated_inference_time']
        print(f"Estimated Inference Time: {inference_time} microseconds")
    except KeyError:
        print("Results downloaded. Check the JSON for detailed latency and memory usage.")

if __name__ == "__main__":
    # Ensure script is run from project root
    if not os.path.exists("models"):
        print("Please run this script from the project root (e.g., `python scripts/profile_on_ai_hub.py`)")
    else:
        try:
            profile_on_ai_hub()
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\nEnsure you have configured your AI Hub API token!")
            print("Run: qai-hub configure --api_token <YOUR_TOKEN>")
