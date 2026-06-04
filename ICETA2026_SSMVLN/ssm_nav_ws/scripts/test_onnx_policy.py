import argparse
import numpy as np
import onnxruntime as ort


ACTION_NAMES = ["up", "down", "left", "right"]


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ssm_policy_action.onnx")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("input:", input_name, session.get_inputs()[0].shape)
    print("output:", output_name, session.get_outputs()[0].shape)

    # Feature order:
    # current_x, current_y, goal_x, goal_y,
    # delta_goal_x, delta_goal_y, path_remaining,
    # vision_mean, vision_dark, vision_edge,
    # free_up, free_down, free_left, free_right

    sample = np.array([
        [
            0.0,   # current_x
            0.0,   # current_y
            1.0,   # goal_x
            1.0,   # goal_y
            1.0,   # delta_goal_x
            1.0,   # delta_goal_y
            0.5,   # path_remaining
            0.5,   # vision_mean
            0.25,  # vision_dark
            0.5,   # vision_edge
            0.0,   # free_up
            1.0,   # free_down
            0.0,   # free_left
            1.0,   # free_right
        ]
    ], dtype=np.float32)

    logits = session.run([output_name], {input_name: sample})[0]
    probs = softmax(logits)

    action_id = int(np.argmax(probs, axis=1)[0])

    print("logits:", logits[0])
    print("probs:", probs[0])
    print("pred_action:", action_id, ACTION_NAMES[action_id])


if __name__ == "__main__":
    main()
