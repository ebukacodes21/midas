import os
import cv2
import boto3
import exifread
import torch
import tempfile
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_from_s3(s3, bucket, prefix, download_dir):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".jpg") or key.endswith(".jpeg"):
                local_path = Path(download_dir) / Path(key).name
                s3.download_file(bucket, key, str(local_path))

def upload_to_s3(s3, bucket, local_path, s3_path):
    s3.upload_file(str(local_path), bucket, s3_path)

def estimate_depth(image_path):
    model_type = "DPT_Large"
    model = torch.hub.load("intel-isl/MiDaS", model_type).to(DEVICE).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    img = cv2.imread(image_path)
    img_input = transform(img).to(DEVICE)
    with torch.no_grad():
        prediction = model(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 65535, cv2.NORM_MINMAX).astype('uint16')
    depth_path = str(image_path).replace(".jpg", "_depth.png")
    cv2.imwrite(depth_path, depth_map_normalized)
    return depth_path

def extract_gps(img_path):
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f)
    gps_lat = tags.get('GPS GPSLatitude')
    gps_lat_ref = tags.get('GPS GPSLatitudeRef')
    gps_lon = tags.get('GPS GPSLongitude')
    gps_lon_ref = tags.get('GPS GPSLongitudeRef')

    if not all([gps_lat, gps_lat_ref, gps_lon, gps_lon_ref]):
        return None, None

    def dms_to_dd(dms):
        d, m, s = [float(x.num) / float(x.den) for x in dms.values]
        return d + m / 60.0 + s / 3600.0

    lat = dms_to_dd(gps_lat)
    if gps_lat_ref.values[0] != 'N':
        lat = -lat
    lon = dms_to_dd(gps_lon)
    if gps_lon_ref.values[0] != 'E':
        lon = -lon
    return lat, lon

def main():
    user_id = os.environ.get("USER_ID")
    job_id = os.environ.get("JOB_ID")
    bucket = os.environ.get("S3_BUCKET")

    if not all([user_id, job_id, bucket]):
        print("❌ Missing required environment variables.")
        return

    input_prefix = f"uploads/{user_id}/{job_id}/"
    output_prefix = f"processed/{user_id}/{job_id}/"

    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Download images
        download_from_s3(s3, bucket, input_prefix, temp_dir)

        # Step 2: Process each image
        for img_file in Path(temp_dir).glob("*.jpg"):
            print(f"🔍 Processing {img_file.name}")
            depth_path = estimate_depth(str(img_file))
            lat, lon = extract_gps(str(img_file))

            # Upload depth map
            s3_output_path = output_prefix + Path(depth_path).name
            upload_to_s3(s3, bucket, depth_path, s3_output_path)

            # Save GPS metadata as .txt
            gps_file = str(img_file).replace(".jpg", "_gps.txt")
            with open(gps_file, "w") as f:
                if lat and lon:
                    f.write(f"{lat},{lon}")
                else:
                    f.write("No GPS found")
            upload_to_s3(s3, bucket, gps_file, output_prefix + Path(gps_file).name)

        print("✅ All files processed.")

        # Optional: Trigger Meshroom job
        # trigger_meshroom_job(user_id, job_id, bucket)

# def trigger_meshroom_job(user_id, job_id, bucket):
#     batch = boto3.client("batch")
#     response = batch.submit_job(
#         jobName=f"meshroom-{job_id}",
#         jobQueue=os.environ["MESHROOM_JOB_QUEUE"],
#         jobDefinition=os.environ["MESHROOM_JOB_DEF"],
#         containerOverrides={
#             "environment": [
#                 {"name": "USER_ID", "value": user_id},
#                 {"name": "JOB_ID", "value": job_id},
#                 {"name": "S3_BUCKET", "value": bucket}
#             ]
#         }
#     )
#     print(f"📤 Meshroom job submitted: {response['jobId']}")

if __name__ == "__main__":
    main()
