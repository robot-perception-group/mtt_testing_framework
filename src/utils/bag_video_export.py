#!/usr/bin/env python3
import os
import sys
import cv2
import rosbag
from cv_bridge import CvBridge
from tqdm import tqdm  # Import tqdm for progress bars

if len(sys.argv) < 4:
    print("Usage: bag_video_export.py <bagfile> <image_topic> <outdir>")
    sys.exit(1)

bagfile, topic, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.exists(outdir):
    os.makedirs(outdir)

bag = rosbag.Bag(bagfile, "r")
bridge = CvBridge()

# Count total messages for progress bar
total_msgs = bag.get_message_count(topic)

# First pass: read all frames, collect (timestamp, image-path)
print(f"Extracting frames from {topic}...")
frame_list = []
for idx, (topic, msg, t) in enumerate(tqdm(bag.read_messages(topics=[topic]), total=total_msgs, desc="Extracting frames")):
    cv_img = bridge.compressed_imgmsg_to_cv2(msg)  # or imgmsg_to_cv2 for raw
    fname = os.path.join(outdir, f"frame{idx:06d}.png")
    cv2.imwrite(fname, cv_img)
    frame_list.append((t.to_sec(), fname))

bag.close()

# Second pass: build the ffmpeg 'concat' file
print("Creating ffmpeg concat file...")
concat_path = os.path.join(outdir, "frames.txt")
with open(concat_path, "w") as f:
    for i in tqdm(range(len(frame_list)), desc="Writing concat file"):
        ts, fname = frame_list[i]
        f.write(f"file '{os.path.basename(fname)}'\n")
        # compute duration until next frame (or repeat last)
        if i < len(frame_list) - 1:
            duration = frame_list[i+1][0] - ts
        else:
            duration = 1.0/30.0   # fallback to ~30 Hz
        f.write(f"duration {duration:.6f}\n")

# Finally: call ffmpeg (requires ffmpeg be on your $PATH)
print("Starting ffmpeg encoding process...")
import subprocess
output_file = os.path.join(outdir, "output.mp4")

subprocess.call([
    "ffmpeg", "-f", "concat", "-safe", "0",
    "-i", concat_path,
    "-vf", "format=yuv420p",
    "-c:v", "libx264",
    output_file
])

print(f"Video successfully exported to {output_file}")
