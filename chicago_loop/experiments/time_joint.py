"""
Timing of the exact joint station assignment on random station states (8 stations, 2 chargers each).

  python -m chicago_loop.experiments.time_joint
"""
import random, time
from chicago_loop.env.fleet import StationState, VehicleState, VehicleType
import chicago_loop.models.optimizer as O
cls = [c for c in vars(O).values() if isinstance(c, type) and hasattr(c, "_joint_assign")][0]
opt = cls.__new__(cls); opt.beta = 1.0; opt.charge_wait_weight = 3.0; opt.joint_stats = dict(steps=0, differs=0, fallback=0)
rng = random.Random(1)
S = [f"s{k}" for k in range(8)]
for n in (2, 4, 6, 8, 10, 12):
    ts = []
    for rep in range(3 if n >= 10 else 10):
        s_map = {s: StationState(s, s, num_piles=2, in_service=[rng.uniform(0, 40) for _ in range(2)],
                                 committed=[(rng.uniform(0, 5), 40.0) for _ in range(rng.randint(0, 3))]) for s in S}
        must = [f"v{i}" for i in range(n)]
        v_map = {v: VehicleState(v, list(VehicleType)[0], "n", soc=rng.uniform(0.1, 0.2)) for v in must}
        tts = {(v, s): rng.uniform(0.5, 5) for v in must for s in S}
        t0 = time.perf_counter(); opt._joint_assign(must, S, s_map, tts, v_map); ts.append(time.perf_counter() - t0)
    print(n, f"{sum(ts)/len(ts):.4f} s")
