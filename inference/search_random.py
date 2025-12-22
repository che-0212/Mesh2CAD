import sys
import os
import random
import math
from pathlib import Path
import numpy as np


from search import Search


class SearchRandom(Search):

    def __init__(self, env, log_dir=None):
        super().__init__(env, log_dir)
        self.log_probs = False

    def search(self, agent, budget, score_function=None, screenshot=False):
        super().search(agent, budget, score_function, screenshot)
        # the length of rollout is the same as the number of planar faces as a maximum
        rollout_length = 0
        for node in self.target_graph["nodes"]:
            if node["surface_type"] == "PlaneSurfaceType":
                rollout_length += 1
        if rollout_length < 2:
            # There exist some designs with no planar faces that we can't handle
            # We need at least 2 faces
            raise Exception("Not enough valid planar faces in target")
        elif rollout_length > 2:
            rollout_length = math.ceil(rollout_length / 2)

        rollout_attempt = 0
        used_budget = 0
        max_score = 0
        max_scores = []

        while used_budget < budget:
            # We begin each rollout an empty graph
            cur_graph = self.env.get_empty_graph()
            for i in range(rollout_length):
                actions, action_probabilities = agent.get_actions_probabilities(cur_graph, self.target_graph)
                # Filter for clearly bad actions
                action_probabilities = self.filter_bad_actions(cur_graph, actions, action_probabilities)
                action = np.random.choice(actions, 1, p=action_probabilities)[0]
                new_graph, cur_iou = self.env.extrude(action["start_face"], action["end_face"], action["operation"])
                take_screenshot = screenshot
                # debug
                #import pdb; pdb.set_trace() 
                
                start_face_data = [node for node in self.target_graph["nodes"] if node["id"] == action["start_face"]]
                end_face_data = [node for node in self.target_graph["nodes"] if node["id"] == action["end_face"]]

                def _format_vectors(data_list):
                    formatted = []
                    for node in data_list:
                        # 深拷贝节点数据避免修改原始数据
                        node_copy = node.copy()
                        # 处理 points：每3个值为一组
                        points = node_copy.get("points", [])
                        node_copy["points"] = [points[i:i+3] for i in range(0, len(points), 3)]
                        # 处理 normals：每3个值为一组
                        normals = node_copy.get("normals", [])
                        node_copy["normals"] = [normals[i:i+3] for i in range(0, len(normals), 3)]
                        formatted.append(node_copy)
                    return formatted

                # 将数据格式化
                formatted_start = _format_vectors(start_face_data)
                formatted_end = _format_vectors(end_face_data)

                if cur_iou is not None:
                    max_score = max(max_score, cur_iou)
                else:
                    # We only want to take screenshots when something changes
                    take_screenshot = False
                if new_graph is not None:
                    cur_graph = new_graph
                
                log_data = {
                    "rollout_attempt": rollout_attempt,
                    "rollout_step": i,
                    "rollout_length": rollout_length,
                    "used_budget": used_budget,
                    "budget": budget,
                    "start_face": action["start_face"],
                    "end_face": action["end_face"],
                    "operation": action["operation"],
                    "current_iou": cur_iou,
                    "max_iou": max_score,
                    # ==== 新增字段 ====
                    "valid_start_face_data": formatted_start,
                    "valid_end_face_data": formatted_end
                }
                if self.log_probs:
                    probs = np.sort(action_probabilities).tolist()
                    log_data["probabilities"] = probs

                self.log.log(log_data, take_screenshot)
                max_scores.append(max_score)
                # Stop early if we find a solution
                if math.isclose(max_score, 1, abs_tol=0.00001):
                    return max_scores
                used_budget += 1
                # Stop if the rollout hits the budget
                if used_budget >= budget:
                    break
            print(f"[{used_budget}/{budget}] Score: {max_score}")
            # Revert to the target and remove all reconstruction
            self.env.revert_to_target()
            rollout_attempt += 1
        return max_scores
