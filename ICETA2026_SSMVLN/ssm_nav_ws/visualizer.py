import os
import rclpy
from rclpy.node import Node

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s_nav_msgs.msg import VisionGraph


class NavVisualizer(Node):
    def __init__(self):
        super().__init__("nav_visualizer")

        self.subscription = self.create_subscription(
            VisionGraph,
            "/planned_path",
            self.listener_callback,
            10
        )

        self.output_path = "/home/mj/my_research/ssm_nav_ws/nav_graph_path.png"
        self.saved = False

        self.get_logger().info("Waiting for /planned_path...")

    def listener_callback(self, msg):
        if self.saved:
            return

        nodes = {node.node_id: node for node in msg.nodes}

        plt.figure(figsize=(8, 6))

        for node in msg.nodes:
            x1 = node.position.x
            y1 = node.position.y

            for neighbor_id in node.neighbors:
                if neighbor_id not in nodes:
                    continue

                neighbor = nodes[neighbor_id]
                x2 = neighbor.position.x
                y2 = neighbor.position.y

                plt.plot([x1, x2], [y1, y2], color="lightgray", linewidth=1)

        xs = [node.position.x for node in msg.nodes]
        ys = [node.position.y for node in msg.nodes]

        plt.scatter(xs, ys, c="steelblue", s=80, edgecolors="black", zorder=3)

        for node in msg.nodes:
            plt.text(
                node.position.x + 0.05,
                node.position.y + 0.05,
                str(node.node_id),
                fontsize=8
            )

        path_ids = list(msg.path_node_ids)

        if len(path_ids) >= 2:
            path_x = []
            path_y = []

            for node_id in path_ids:
                if node_id not in nodes:
                    continue

                path_x.append(nodes[node_id].position.x)
                path_y.append(nodes[node_id].position.y)

            plt.plot(path_x, path_y, color="red", linewidth=3, zorder=4)
            plt.scatter(path_x, path_y, c="red", s=100, zorder=5)

        plt.title("SSM-Nav 2D Graph and Planned Path")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        plt.savefig(self.output_path, dpi=200)
        plt.close()

        self.get_logger().info(f"Saved visualization: {self.output_path}")
        self.saved = True
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = NavVisualizer()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
