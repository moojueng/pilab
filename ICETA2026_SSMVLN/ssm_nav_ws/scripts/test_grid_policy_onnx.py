#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/grid_ssm_policy.onnx")

x = np.zeros((1, 32), dtype=np.float32)
x[0, 0:25] = 0
x[0, 25] = 4
x[0, 26] = 0
x[0, 27] = 4
x[0, 28] = 10
x[0, 29] = 20
x[0, 30] = 5
x[0, 31] = 2

outputs = session.run(None, {"input": x})
action_logits, reward_values = outputs

print("action_logits:", action_logits)
print("reward_values:", reward_values)
print("pred_action:", int(np.argmax(action_logits[0])))
print("best_reward_action:", int(np.argmax(reward_values[0])))
