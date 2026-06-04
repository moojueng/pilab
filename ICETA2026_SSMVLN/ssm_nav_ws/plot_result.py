import csv
import matplotlib.pyplot as plt

def plot_graph():
    try:
        ids, xs, ys = [], [], []
        with open('/home/mj/my_research/ssm_nav_ws/graph_dump.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(int(row['id']))
                xs.append(float(row['x']))
                ys.append(float(row['y']))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(xs, ys, c='green', s=100, edgecolors='black', label='Generated Nodes')
        
        for i, txt in enumerate(ids):
            plt.annotate(f"N{txt}", (xs[i], ys[i]), xytext=(5, 5), textcoords='offset points')
            
        plt.title('SSM-Nav-Onboard: Vision-Graph Node Map')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.gca().invert_yaxis() # Matrix coordinate to Plot coordinate
        plt.grid(True)
        plt.legend()
        
        plt.savefig('/home/mj/my_research/ssm_nav_ws/vision_graph_result.png')
        print("Success: vision_graph_result.png generated.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_graph()
