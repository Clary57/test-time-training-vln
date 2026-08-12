import habitat_sim
import habitat_sim.utils.common as utils
from habitat_sim.sensor import CameraSensorSpec
import matplotlib.pyplot as plt
import numpy as np

# 1. Simulator config
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "/nvme-ssd1/zwy/data/scene_dataset/hm3d/00001-UVdNNRcVyV1/UVdNNRcVyV1.basis.glb"

# 2. Camera sensor config (NEW API)
color_sensor_spec = CameraSensorSpec()
color_sensor_spec.uuid = "color_sensor"
color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
color_sensor_spec.resolution = [540, 1000]   # [H, W]
color_sensor_spec.position = [-9.0, 16.0, 8.0]   # (x, y, z)
color_sensor_spec.orientation = [-np.pi / 3, 0.0, 0.0]  # pitch, yaw, roll
color_sensor_spec.hfov = 90.0
color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

# 3. Agent config
agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [color_sensor_spec]

# 4. Create simulator
sim = habitat_sim.Simulator(
    habitat_sim.Configuration(sim_cfg, [agent_cfg])
)

# Scene info (still valid)
scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
print("Scene bounding box:", scene_bb)
print("Scene center:", scene_bb.center())

# 5. Get observation
obs = sim.get_sensor_observations()
img = obs["color_sensor"]  # (H, W, 4) uint8 RGBA

# 6. Ensure RGBA + transparent background
if img.shape[-1] == 3:
    rgba = np.concatenate(
        [img, 255 * np.ones_like(img[..., :1])], axis=-1
    )
else:
    rgba = img.copy()

# Make black background transparent
mask = np.all(rgba[..., :3] == 0, axis=-1)
rgba[mask, 3] = 0

# 7. Save image
plt.imsave("long_term.png", rgba)

sim.close()
