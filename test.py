import pyrealsense2 as rs

pipeline = rs.pipeline()
config   = rs.config()
config.enable_device("814412071258")   # D455 serial
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

profile = pipeline.start(config)
intr    = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

sensor_width_mm      = 12.7             # D455 1/2" sensor
focal_length_mm      = (intr.fx / intr.width) * sensor_width_mm
horizontal_aperture  = sensor_width_mm

print(f"focal_length:        {focal_length_mm:.4f}")
print(f"horizontal_aperture: {horizontal_aperture}")
pipeline.stop()