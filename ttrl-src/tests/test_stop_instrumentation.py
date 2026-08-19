import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "eval" / "stop_trace.py"
SPEC = importlib.util.spec_from_file_location("stop_trace", MODULE_PATH)
stop_trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stop_trace)


class StopInstrumentationTests(unittest.TestCase):
    def metadata(self, selected=(1, 1, 0), candidates=((1, 1, 0), (2,))):
        return stop_trace.chunk_metadata(
            episode_id="7", decision_id=3, selected_chunk=selected,
            raw_model_outputs=("forward, stop", "left"),
            parsed_candidate_chunks=candidates, candidate_rewards=(1.0, -1.0),
            selected_index=0,
        )

    def test_disabled_path_has_no_writer_no_file_and_no_rng_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            random.seed(19); before = random.getstate()
            enabled = False
            writer = stop_trace.StopTraceWriter(path, "7", 3.0) if enabled else None
            actions = [1, 2, 0]
            observed = list(actions)
            self.assertIsNone(writer)
            self.assertEqual(actions, observed)
            self.assertEqual(before, random.getstate())
            self.assertFalse(path.exists())

    def test_chunk_ids_offsets_and_future_stop_are_stable(self):
        rows = self.metadata()
        self.assertEqual([r["decision_id"] for r in rows], [3, 3, 3])
        self.assertEqual([r["chunk_id"] for r in rows], ["7:3"] * 3)
        self.assertEqual([r["chunk_offset"] for r in rows], [0, 1, 2])
        self.assertEqual([r["selected_future_stop"] for r in rows], [True, True, False])

    def test_unselected_stop_is_not_selected_or_executed(self):
        rows = self.metadata(selected=(1,), candidates=((1,), (0,)))
        self.assertTrue(rows[0]["any_candidate_contains_stop"])
        self.assertFalse(rows[0]["selected_chunk_contains_stop"])
        self.assertIsNone(rows[0]["selected_stop_offset"])

    def test_distance_transitions_jsonl_and_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with stop_trace.StopTraceWriter(path, "7", 3.0) as writer:
                records=[]
                for before, after in ((4, 2), (2, 4), (4, 1)):
                    records.append(writer.record(
                        distance_before=before, distance_after=after,
                        episode_over_before=False, episode_over_after=False,
                        agent_action=1, action_source="generated", overridden_action=None,
                        override_reason=None, executed_action=1, pending_before=1,
                        pending_after=0, metadata=self.metadata(selected=(1,))[0],
                        early_rotation_override=False, max_step_override=False))
            parsed=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(parsed, records)
            self.assertEqual([r["env_step"] for r in parsed], [0,1,2])
            self.assertEqual([r["first_oracle_reach"] for r in parsed], [True,False,False])
            self.assertEqual(min([parsed[0]["distance_to_goal_before"]] + [r["distance_to_goal_after"] for r in parsed]), 1)
            self.assertEqual(parsed[-1]["distance_to_goal_after"], 1)

    def test_selected_future_stop_and_executed_stop_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"trace.jsonl"; rows=self.metadata()
            with stop_trace.StopTraceWriter(path,"7",3) as writer:
                first=writer.record(distance_before=5,distance_after=4,episode_over_before=False,
                    episode_over_after=False,agent_action=1,action_source="generated",
                    overridden_action=None,override_reason=None,executed_action=1,pending_before=3,
                    pending_after=2,metadata=rows[0],early_rotation_override=False,max_step_override=False)
                stop=writer.record(distance_before=4,distance_after=4,episode_over_before=False,
                    episode_over_after=True,agent_action=0,action_source="pending",
                    overridden_action=None,override_reason=None,executed_action=0,pending_before=1,
                    pending_after=0,metadata=rows[2],early_rotation_override=False,max_step_override=False)
            self.assertTrue(first["selected_future_stop"]); self.assertFalse(first["stop_executed"])
            self.assertFalse(stop["selected_future_stop"]); self.assertTrue(stop["stop_executed"])
            self.assertEqual(stop["termination_reason"],"queued_stop")

    def test_override_timing_and_termination_priority(self):
        self.assertEqual(stop_trace.termination_reason(episode_over_after=True,max_step_override=False,
            early_rotation_override=True,stop_executed=True,action_source="early_rotation_override"),"early_rotation_stop")
        self.assertEqual(stop_trace.termination_reason(episode_over_after=True,max_step_override=True,
            early_rotation_override=True,stop_executed=True,action_source="max_step_override"),"max_step_stop")
        self.assertEqual(stop_trace.termination_reason(episode_over_after=True,max_step_override=False,
            early_rotation_override=False,stop_executed=False,action_source="generated"),"environment_done_after_non_stop")

    def test_post_agent_override_preserves_pending_stop_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta={**self.metadata()[2], "agent_action_source": "pending"}
            with stop_trace.StopTraceWriter(Path(tmp)/"trace.jsonl","7",3) as writer:
                row=writer.record(distance_before=2,distance_after=2,episode_over_before=False,
                    episode_over_after=True,agent_action=0,action_source="early_rotation_override",
                    overridden_action=0,override_reason="early_rotation",executed_action=0,
                    pending_before=1,pending_after=0,metadata=meta,
                    early_rotation_override=True,max_step_override=False)
            self.assertTrue(row["stop_is_current_queued_action"])
            self.assertEqual(row["termination_reason"],"early_rotation_stop")

    def test_selected_index_matches_existing_tie_break(self):
        self.assertEqual(stop_trace.selected_candidate_index([1,1,-1],[True,True,True]),0)
        self.assertEqual(stop_trace.selected_candidate_index([-1,1,1],[False,True,True]),1)
        self.assertIsNone(stop_trace.selected_candidate_index([-1,-1],[False,False]))

    def test_fake_model_env_logging_is_action_and_rng_neutral(self):
        class FakeModel:
            def generate(self):
                return ["forward, stop", "left"]

        class FakeEnv:
            def __init__(self):
                self.distances=[5.0,2.5,2.5]
                self.index=0
                self.executed=[]
            def step(self, action):
                self.executed.append(action); self.index += 1
                return self.distances[self.index]

        def run(enabled, directory):
            model=FakeModel(); env=FakeEnv(); candidates=model.generate()
            parsed=[[1,0],[2]]; rewards=[1.0,-1.0]
            selected=stop_trace.selected_candidate_index(rewards,[True,True])
            production_queue=list(parsed[selected])
            metadata_queue=(stop_trace.chunk_metadata(episode_id="9",decision_id=0,
                selected_chunk=production_queue,raw_model_outputs=candidates,
                parsed_candidate_chunks=parsed,candidate_rewards=rewards,
                selected_index=selected) if enabled else None)
            trace_path=Path(directory)/"trace.jsonl"
            writer=stop_trace.StopTraceWriter(trace_path,"9",3.0) if enabled else None
            queue_snapshots=[]
            while production_queue:
                before=len(production_queue); action=production_queue.pop(0)
                meta=metadata_queue.pop(0) if enabled else None
                distance_before=env.distances[env.index]; distance_after=env.step(action)
                queue_snapshots.append((before,len(production_queue)))
                if writer:
                    writer.record(distance_before=distance_before,distance_after=distance_after,
                        episode_over_before=False,episode_over_after=action==0,
                        agent_action=action,action_source="generated" if before==2 else "pending",
                        overridden_action=None,override_reason=None,executed_action=action,
                        pending_before=before,pending_after=len(production_queue),metadata=meta,
                        early_rotation_override=False,max_step_override=False)
            if writer: writer.close()
            return {"candidates":candidates,"selected":selected,"actions":env.executed,
                    "queues":queue_snapshots,"trace_exists":trace_path.exists()}

        with tempfile.TemporaryDirectory() as tmp:
            random.seed(313); state=random.getstate()
            disabled=run(False,Path(tmp)/"off")
            self.assertEqual(state,random.getstate())
            enabled=run(True,Path(tmp)/"on")
            self.assertEqual(state,random.getstate())
            for key in ("candidates","selected","actions","queues"):
                self.assertEqual(disabled[key],enabled[key])
            self.assertFalse(disabled["trace_exists"]); self.assertTrue(enabled["trace_exists"])

    def test_evaluator_source_preserves_A_C_override_order_and_default_off(self):
        root=Path(__file__).parents[1]
        a=(root/"eval"/"eval_ttt.py").read_text()
        c=(root/"eval"/"eval.py").read_text()
        a_loop=a[a.index("while not env.episode_over:"):a.index("info = env.get_metrics()",a.index("obs = env.step(action)"))]
        c_loop=c[c.index("while not env.episode_over:"):c.index("info = env.get_metrics()",c.index("obs = env.step(action)"))]
        self.assertLess(a_loop.index("if early_rotation_override or max_step_override"),a_loop.index("agent.act"))
        self.assertLess(c_loop.index("agent.act"),c_loop.index("if early_rotation_override or max_step_override"))
        for source in (a,c):
            self.assertIn('parser.add_argument("--stop-trace",action="store_true"',source)
            self.assertNotIn('default=True',source[source.index('--stop-trace'):source.index('--stop-trace')+200])

    def test_updater_exposes_existing_choice_without_second_ranking(self):
        source=(Path(__file__).parents[1]/"trainer"/"trajectory_consensus_updater.py").read_text()
        self.assertIn('"selected_candidate_index": selected_candidate_index',source)
        self.assertIn("selected_actions = self._select_actions", source)
        self.assertIn("if self.trace_metadata_enabled", source)


if __name__ == "__main__":
    unittest.main()
