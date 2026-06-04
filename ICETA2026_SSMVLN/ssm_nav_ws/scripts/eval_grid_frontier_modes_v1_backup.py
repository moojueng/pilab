#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ACTIONS = [(-1,0),(0,1),(1,0),(0,-1),(0,0)]

class SsmBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        residual = x
        gate = torch.sigmoid(self.gate_proj(x))
        x = F.silu(self.in_proj(x)) * gate
        x = self.out_proj(x)
        return self.norm(x + residual)

class GridPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, layers=3, actions=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([SsmBlock(hidden_dim) for _ in range(layers)])
        self.action_head = nn.Linear(hidden_dim, actions)
        self.reward_head = nn.Linear(hidden_dim, actions)
    def forward(self, x):
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        return self.action_head(x), self.reward_head(x)

@dataclass(frozen=True, order=True)
class Pos:
    r: int
    c: int

@dataclass
class Metrics:
    mode: str
    success: int = 0
    steps: int = 0
    collisions: int = 0
    revisits: int = 0
    revisit_ratio: float = 0.0
    unique_visited: int = 0
    observed_nodes: int = 0
    observed_edges: int = 0
    target_seen_step: int = -1
    fallback_count: int = 0
    frontier_switches: int = 0

class GridEvaluator:
    def __init__(self, grid, model, mode='utility_commit', vision_radius=2, max_steps=200, start=Pos(14,0)):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.observed = [[-1 for _ in range(self.cols)] for _ in range(self.rows)]
        self.visits = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.robot = start
        self.model = model
        self.mode = mode
        self.vision_radius = vision_radius
        self.max_steps = max_steps
        self.traj = []
        self.visited = set()
        self.fallback_count = 0
        self.frontier_switches = 0
        self.current_subgoal = None
        self.commit_left = 0
        self.node_set = set()
        self.observed_edges = 0

    def in_bounds(self,p): return 0 <= p.r < self.rows and 0 <= p.c < self.cols
    def hidden_obstacle(self,p): return (not self.in_bounds(p)) or self.grid[p.r][p.c] == 1
    def free_observed(self,p): return self.in_bounds(p) and self.observed[p.r][p.c] in (0,2)
    def neighbors4(self,p): return [Pos(p.r-1,p.c),Pos(p.r,p.c+1),Pos(p.r+1,p.c),Pos(p.r,p.c-1)]
    def observe(self):
        R=self.vision_radius
        for dr in range(-R,R+1):
            for dc in range(-R,R+1):
                p=Pos(self.robot.r+dr,self.robot.c+dc)
                if self.in_bounds(p): self.observed[p.r][p.c]=self.grid[p.r][p.c]
    def target_visible(self):
        R=self.vision_radius
        for dr in range(-R,R+1):
            for dc in range(-R,R+1):
                p=Pos(self.robot.r+dr,self.robot.c+dc)
                if self.in_bounds(p) and self.observed[p.r][p.c]==2: return True
        return False
    def build_graph(self):
        self.node_set=set()
        self.observed_edges=0
        for r in range(self.rows):
            for c in range(self.cols):
                p=Pos(r,c)
                if self.free_observed(p): self.node_set.add(p)
        for p in self.node_set:
            for n in self.neighbors4(p):
                if n in self.node_set: self.observed_edges += 1
    def is_frontier(self,p):
        return self.free_observed(p) and any(self.in_bounds(n) and self.observed[n.r][n.c] == -1 for n in self.neighbors4(p))
    def build_feature(self, prev_action):
        x=[]
        for dr in range(-2,3):
            for dc in range(-2,3):
                p=Pos(self.robot.r+dr,self.robot.c+dc)
                x.append(float(self.observed[p.r][p.c]) if self.in_bounds(p) else -1.0)
        frontier_count=0
        for p in self.node_set:
            for n in self.neighbors4(p):
                if self.in_bounds(n) and self.observed[n.r][n.c] == -1: frontier_count += 1
        x += [float(self.robot.r),float(self.robot.c),float(prev_action),float(len(self.node_set)),float(self.observed_edges),float(frontier_count),float(len(self.visited))]
        return torch.tensor([x], dtype=torch.float32)
    def policy_action(self, prev_action):
        with torch.no_grad():
            logits, rewards = self.model(self.build_feature(prev_action))
            scores = rewards[0] + 0.2 * logits[0]
            return int(torch.argmax(scores).item())
    def action_next(self,a):
        dr,dc=ACTIONS[a]
        return Pos(self.robot.r+dr,self.robot.c+dc)
    def action_valid(self,a):
        return 0 <= a < 4 and self.free_observed(self.action_next(a)) and not self.hidden_obstacle(self.action_next(a))
    def action_from_to(self,a,b):
        d=(b.r-a.r,b.c-a.c)
        return {(-1,0):0,(0,1):1,(1,0):2,(0,-1):3}.get(d,4)
    def visit_count_action(self,a):
        if not (0 <= a < 4): return 999
        p=self.action_next(a)
        return self.visits[p.r][p.c] if self.in_bounds(p) else 999
    def action_loops(self,a): return self.visit_count_action(a) >= 6
    def bfs_parent(self, goal=None):
        q=deque([self.robot]); parent={self.robot:None}; dist={self.robot:0}
        found=None
        while q:
            cur=q.popleft()
            if goal is not None and cur==goal:
                found=cur; break
            if goal is None and cur!=self.robot and self.is_frontier(cur):
                found=cur; break
            for n in self.neighbors4(cur):
                if self.free_observed(n) and n not in parent:
                    parent[n]=cur; dist[n]=dist[cur]+1; q.append(n)
        return found,parent,dist
    def first_action_to(self, goal):
        found,parent,_=self.bfs_parent(goal)
        if found is None: return 4
        cur=goal
        while parent.get(cur) is not None and parent[cur] != self.robot:
            cur=parent[cur]
        return self.action_from_to(self.robot, cur)
    def nearest_frontier_action(self):
        found,parent,_=self.bfs_parent(None)
        if found is not None:
            cur=found
            while parent.get(cur) is not None and parent[cur] != self.robot:
                cur=parent[cur]
            return self.action_from_to(self.robot, cur)
        for a in range(4):
            if self.action_valid(a) and self.visit_count_action(a)==0: return a
        for a in range(4):
            if self.action_valid(a): return a
        return 4
    def collect_frontiers(self): return [p for p in self.node_set if p!=self.robot and self.is_frontier(p)]
    def unknown_count(self,p,radius):
        cnt=0
        for dr in range(-radius,radius+1):
            for dc in range(-radius,radius+1):
                q=Pos(p.r+dr,p.c+dc)
                if self.in_bounds(q) and self.observed[q.r][q.c] == -1: cnt += 1
        return cnt
    def local_visit_penalty(self,p,radius=1):
        s=0
        for dr in range(-radius,radius+1):
            for dc in range(-radius,radius+1):
                q=Pos(p.r+dr,p.c+dc)
                if self.in_bounds(q): s += min(self.visits[q.r][q.c],5)
        return float(s)
    def free_degree(self,p): return sum(1 for n in self.neighbors4(p) if self.free_observed(n))
    def direction_cost(self,p,prev_action):
        if prev_action not in range(4): return 0.0
        dr,dc=p.r-self.robot.r,p.c-self.robot.c
        best = (0 if dr<0 else 2) if abs(dr)>abs(dc) else (1 if dc>0 else 3 if dc<0 else prev_action)
        return 0.0 if best==prev_action else 1.0
    def distance_to(self,p):
        _,_,dist=self.bfs_parent(None)
        return dist.get(p,10**6)
    def frontier_score(self,p,prev_action,dist=None):
        
        if dist is not None:
            travel = float(dist.get(p, 10**6))
        else:
            travel = float(self.distance_to(p))
        if travel >= 10**6: return -1e9
        info=float(self.unknown_count(p,self.vision_radius+1))
        prior=float(self.unknown_count(p,3))
        revisit=self.local_visit_penalty(p,1)
        dead=1.0 if self.free_degree(p)<=1 else 0.0
        turn=self.direction_cost(p,prev_action)
        semantic_proxy=info/10.0
        return 2.5*semantic_proxy + 1.5*info + 1.2*prior - 1.0*travel - 1.3*revisit - 0.8*dead - 0.3*turn
    def best_frontier(self,prev_action):
        fs=self.collect_frontiers()
        if not fs: return None
        _,_,dist=self.bfs_parent(None)
        return max(fs, key=lambda p:self.frontier_score(p,prev_action,dist))
    def utility_action(self,prev_action):
        if self.mode == 'utility_commit' and self.current_subgoal is not None and self.commit_left > 0 and self.free_observed(self.current_subgoal):
            a=self.first_action_to(self.current_subgoal)
            if a in range(4):
                self.commit_left -= 1
                return a
        best=self.best_frontier(prev_action)
        if best is not None:
            if best != self.current_subgoal: self.frontier_switches += 1
            self.current_subgoal=best
            self.commit_left = 4 if self.mode == 'utility_commit' else 0
            a=self.first_action_to(best)
            if a in range(4): return a
        return self.nearest_frontier_action()
    def choose_action(self, policy_action, prev_action):
        if self.mode == 'policy_only':
            if self.action_valid(policy_action) and not self.action_loops(policy_action): return policy_action
            self.fallback_count += 1
            return self.nearest_frontier_action()
        if self.mode == 'nearest':
            if self.action_valid(policy_action) and self.visit_count_action(policy_action)==0: return policy_action
            self.fallback_count += 1
            return self.nearest_frontier_action()
        if self.action_valid(policy_action) and not self.action_loops(policy_action) and self.visit_count_action(policy_action)==0:
            return policy_action
        self.fallback_count += 1
        return self.utility_action(prev_action)
    def move(self,p):
        if self.hidden_obstacle(p): return False
        self.robot=p; return True
    def run(self):
        m=Metrics(mode=self.mode)
        if self.hidden_obstacle(self.robot):
            m.collisions=1; return m
        prev=4
        for step in range(self.max_steps):
            self.observe(); self.build_graph()
            self.traj.append(self.robot); self.visits[self.robot.r][self.robot.c]+=1
            if self.target_visible():
                m.success=1; m.steps=step; m.target_seen_step=step; break
            pa=self.policy_action(prev)
            a=self.choose_action(pa,prev)
            if a==4:
                m.steps=step; break
            nxt=self.action_next(a)
            if not self.move(nxt):
                m.collisions += 1
                a=self.utility_action(prev) if self.mode!='nearest' else self.nearest_frontier_action()
                self.fallback_count += 1
                if not self.move(self.action_next(a)):
                    m.steps=step; break
            if self.robot in self.visited: m.revisits += 1
            self.visited.add(self.robot)
            prev=a
        else:
            m.steps=self.max_steps
        m.observed_nodes=len(self.node_set); m.observed_edges=self.observed_edges; m.unique_visited=len(self.visited)
        m.fallback_count=self.fallback_count; m.frontier_switches=self.frontier_switches
        m.revisit_ratio=(m.revisits/m.steps) if m.steps>0 else 0.0
        return m
    def save_outputs(self,out_dir):
        out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
        with open(out/'trajectory.csv','w',newline='') as f:
            w=csv.writer(f); w.writerow(['step','r','c']); w.writerows((i,p.r,p.c) for i,p in enumerate(self.traj))
        with open(out/'observed_map.csv','w',newline='') as f: csv.writer(f).writerows(self.observed)
        scale=24
        img=Image.new('RGB',(self.cols*scale,self.rows*scale),'white')
        d=ImageDraw.Draw(img)
        for r in range(self.rows):
            for c in range(self.cols):
                v=self.observed[r][c]
                color=(170,170,170) if v==-1 else (0,0,0) if v==1 else (255,0,0) if v==2 else (255,255,255)
                d.rectangle([c*scale,r*scale,(c+1)*scale-1,(r+1)*scale-1],fill=color,outline=(120,120,120))
        for p in self.visited:
            d.rectangle([p.c*scale,p.r*scale,(p.c+1)*scale-1,(p.r+1)*scale-1],fill=(110,180,255),outline=(120,120,120))
        for p in self.traj:
            d.rectangle([p.c*scale,p.r*scale,(p.c+1)*scale-1,(p.r+1)*scale-1],fill=(255,220,0),outline=(120,120,120))
        if self.traj:
            p=self.traj[0]; d.rectangle([p.c*scale,p.r*scale,(p.c+1)*scale-1,(p.r+1)*scale-1],fill=(180,0,255))
        p=self.robot; d.rectangle([p.c*scale,p.r*scale,(p.c+1)*scale-1,(p.r+1)*scale-1],fill=(0,255,0))
        img.save(out/'path.png')

def load_grid(path):
    rows=[]
    with open(path) as f:
        for line in f:
            if line.strip(): rows.append([int(x) for x in line.strip().split(',')])
    return rows

def load_model(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=GridPolicyNet(ckpt['input_dim'],ckpt.get('hidden_dim',128),ckpt.get('layers',3),ckpt.get('actions',5))
    model.load_state_dict(ckpt['model_state_dict']); model.eval(); return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--maps', nargs='+', required=True)
    ap.add_argument('--model', default='models/grid_ssm_policy.pt')
    ap.add_argument('--modes', nargs='+', default=['nearest','utility'])
    ap.add_argument('--out', default='results/grid_sim/frontier_modes')
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--vision-radius', type=int, default=2)
    ap.add_argument('--start-r', type=int, default=14)
    ap.add_argument('--start-c', type=int, default=0)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    model=load_model(args.model)
    rows=[]
    for map_path in args.maps:
        name=Path(map_path).stem
        grid=load_grid(map_path)
        for mode in args.modes:
            ev=GridEvaluator(grid,model,mode,args.vision_radius,args.max_steps,Pos(args.start_r,args.start_c))
            m=ev.run()
            case_out=out/mode/name
            ev.save_outputs(case_out)
            with open(case_out/'metrics.csv','w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(m.__dict__.keys()))
                w.writeheader(); w.writerow(m.__dict__)
            row={'map':name,**m.__dict__}
            rows.append(row)
            print(row)
    fields=list(rows[0].keys()) if rows else []
    with open(out/'summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    # per-mode aggregate
    aggs=[]
    for mode in args.modes:
        subset=[r for r in rows if r['mode']==mode]
        if not subset: continue
        n=len(subset); succ=sum(int(r['success']) for r in subset)
        successful=[r for r in subset if int(r['success'])==1]
        aggs.append({
            'mode':mode,
            'maps':n,
            'success_rate':succ/n,
            'avg_steps_all':sum(float(r['steps']) for r in subset)/n,
            'avg_steps_success':sum(float(r['steps']) for r in successful)/max(len(successful),1),
            'avg_revisit_ratio':sum(float(r['revisit_ratio']) for r in subset)/n,
            'avg_fallback_count':sum(float(r['fallback_count']) for r in subset)/n,
            'avg_frontier_switches':sum(float(r['frontier_switches']) for r in subset)/n,
        })
    with open(out/'aggregate.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggs[0].keys())); w.writeheader(); w.writerows(aggs)
    print('\nAGGREGATE')
    for a in aggs: print(a)

if __name__=='__main__': main()
