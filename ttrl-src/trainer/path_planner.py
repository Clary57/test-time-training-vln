import numpy as np
from numpy import ndarray
from typing import List, Tuple


class DiscretePathPlanner:
    """Runs a path planning algorithm that greedily minimizes the Euclidean
    distance to a goal location as given in polar coordinates. The path plan
    is created assuming discrete actions MOVE_FORWARD, TURN_LEFT, and
    TURN_RIGHT. The plan further assumes infinite space devoid of obstacles.
    """

    RAD_15DEG = np.deg2rad(15.0)

    def __init__(
        self,
        forward_distance: float = 0.25,
        turn_angle: float = RAD_15DEG,
        goal_radius: float = 0.13,
        step_limit: int = 200,
    ) -> None:
        """
        Args:
            forward_distance: forward step size in meters
            turn_angle: radians of Left and Right actions
            goal_radius: planner stops when within this radius in meters
            step_limit: maximum number of steps in a path plan.
        """
        assert np.isclose((np.pi * 2) % turn_angle, 0.0)
        self._forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.num_turns_in_circle = int((np.pi * 2) / self.turn_angle)
        self.goal_radius = goal_radius
        self.step_limit = step_limit
    
    def plan_trajectory(self, trajectory, predicted_arrive = None)-> List[int]:
        predicted_arrive = np.array(predicted_arrive)
        start_position = np.array([0.0, 0.0])
        current_position = np.array([0.0, 0.0])
        current_heading = 0.0


        step_distance = np.linalg.norm(np.array(trajectory[-1][:2]) - current_position)
        heading_diff = (np.arctan2(trajectory[-1][-2], trajectory[-1][-1]) - current_heading + 3*np.pi) % (2*np.pi) - np.pi
        is_physically_stopping = (
            step_distance < 0.1 and 
            abs(heading_diff) < np.pi / 24
        )
        if is_physically_stopping:
            return [0]

        path_plan = []
        for idx, (relative_pose, stop_flag) in enumerate(zip(trajectory, predicted_arrive)):
            tmp_path = []
            goal_position = np.array(relative_pose[:2]) # x, y
            ideal_heading = np.arctan2(relative_pose[-2], relative_pose[-1]) # sin cos (-pi,pi)

            step_distance = np.linalg.norm(goal_position - current_position)
            heading_diff = (ideal_heading - current_heading + 3*np.pi) % (2*np.pi) - np.pi
            is_physically_stopping = (
                step_distance < 0.05 and 
                abs(heading_diff) < np.pi / 36
            )

            is_physically_stopping = True
            if idx < 1 and np.all(predicted_arrive[idx:] > 0.8) and is_physically_stopping:
                path_plan.append(0)
                break

            def distance_to_goal(pos: ndarray) -> float:
                return np.linalg.norm(pos - goal_position)

            while round(distance_to_goal(current_position), 3) > self.goal_radius:
                # generate all possible next forward positions
                # pick the candidate that minimizes distance to goal
                next_position = min(
                    self.generate_candidate_positions(
                        current_position, current_heading
                    ),
                    key=lambda p: distance_to_goal(p[0]),
                )

                # make an action plan for it and add to total action plan
                current_position, current_heading, num_turns = next_position

                if num_turns > self.num_turns_in_circle // 2:
                    right_turns = self.num_turns_in_circle - num_turns
                    tmp_path.extend([3] * right_turns)
                else:
                    tmp_path.extend([2] * num_turns)

                tmp_path.append(1)

                assert len(tmp_path) < self.step_limit, "reached step limit"

            # heading should face away from the starting point
            # ideal_heading = self.heading_to(start_position, goal_position)
            # ideal_heading = np.arctan2(relative_pose[1,0], relative_pose[0,0])
            # ideal_heading = relative_pose[-1]
            

            diff = (ideal_heading - current_heading + 3*np.pi) % (2*np.pi) - np.pi

            while abs(diff) > self.turn_angle / 2:
                if diff > 0:
                    current_heading += self.turn_angle
                    tmp_path.append(2)  # left
                else:
                    current_heading -= self.turn_angle
                    tmp_path.append(3)  # right

                diff = (ideal_heading - current_heading + 3*np.pi) % (2*np.pi) - np.pi
            path_plan.extend(tmp_path)
        if len(path_plan) == 0:
            path_plan = [0]
        return path_plan

    def plan_single_point(self, relative_pose, predicted_arrive = None) -> List[int]:
        """(r, theta) is a relative waypoint in polar coordinates where r is
        in meters and theta is in radians.
        relative_pose: x, y
        """
        start_position = np.array([0.0, 0.0])
        current_position = np.array([0.0, 0.0])
        current_heading = 0.0
        goal_position = relative_pose[:2] # x, y
        path_plan = []
        # if predicted_arrive > 0.7:
        #     return [0]

        def distance_to_goal(pos: ndarray) -> float:
            return np.linalg.norm(pos - goal_position)

        while round(distance_to_goal(current_position), 3) > self.goal_radius:
            # generate all possible next forward positions
            # pick the candidate that minimizes distance to goal
            next_position = min(
                self.generate_candidate_positions(
                    current_position, current_heading
                ),
                key=lambda p: distance_to_goal(p[0]),
            )

            # make an action plan for it and add to total action plan
            current_position, current_heading, num_turns = next_position

            if num_turns > self.num_turns_in_circle // 2:
                right_turns = self.num_turns_in_circle - num_turns
                path_plan.extend([3] * right_turns)
            else:
                path_plan.extend([2] * num_turns)

            path_plan.append(1)

            assert len(path_plan) < self.step_limit, "reached step limit"

        # heading should face away from the starting point
        # ideal_heading = self.heading_to(start_position, goal_position)
        # ideal_heading = np.arctan2(relative_pose[1,0], relative_pose[0,0])
        ideal_heading = relative_pose[-1]

        diff = (ideal_heading - current_heading + np.pi) % (2*np.pi) - np.pi

        while abs(diff) > self.turn_angle / 2:
            if diff > 0:
                current_heading += self.turn_angle
                path_plan.append(2)  # left
            else:
                current_heading -= self.turn_angle
                path_plan.append(3)  # right

            diff = (ideal_heading - current_heading + np.pi) % (2*np.pi) - np.pi

        if len(path_plan) == 0:
            path_plan = [0]
        return path_plan

    def generate_candidate_positions(
        self, position: ndarray, heading: float
    ) -> List[Tuple[ndarray, float, int]]:
        """
        returns:
            [(position, heading, num_left_turns), ...]
        """
        candidates = []
        angle = heading
        for i in range(round((np.pi * 2) / self.turn_angle)):
            pos_delta = [np.cos(angle)*self._forward_distance, np.sin(angle)*self._forward_distance]
            candidates.append((position + pos_delta, angle, i))
            angle = (angle + self.turn_angle) % (np.pi * 2)

        return candidates

    @staticmethod
    def heading_to(position_from: ndarray, position_to: ndarray) -> float:
        # different than utils.compute_heading_to
        delta_x = position_to[0] - position_from[0]
        delta_y = position_to[-1] - position_from[-1]
        delta_x = 0 if delta_x < 1e-6 else delta_x
        delta_y = 0 if delta_y < 1e-6 else delta_y
        xz_angle = np.arctan2(delta_y, delta_x)
        return (xz_angle + 2*np.pi) % (2 * np.pi)

    @staticmethod
    def normalize_angle(theta) -> ndarray:
        return (theta + 2*np.pi) % (2 * np.pi)
    
    @staticmethod
    def should_stop(trajectory, pos_thresh=0.05, ang_thresh=0.1):
        if np.linalg.norm(trajectory[0][:2]) < 0.12:
            return True
        return False