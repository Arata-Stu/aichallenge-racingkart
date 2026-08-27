import csv

from virtual_scan_rl.track import TrackProgress, completed_lap_count


def test_projection_does_not_jump_to_distant_segment_index(tmp_path):
    path = tmp_path / "track.csv"
    points = [(0, 0), (1, 0), (2, 0), (2, 0.2), (1, 0.2), (0, 0.2)]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["x", "y"])
        writer.writeheader()
        for x, y in points:
            writer.writerow({"x": x, "y": y})
    tracker = TrackProgress(str(path), search_back=0, search_forward=1, max_step_m=2.0)
    tracker.reset(0.1, 0.0, 0.0)
    delta, _, _ = tracker.update(0.9, 0.19)
    assert tracker.segment_index in (0, 1)
    assert 0.0 <= delta <= 2.0


def test_explicitly_closed_raceline_does_not_create_zero_length_segment(tmp_path):
    path = tmp_path / "closed_track.csv"
    points = [(0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["x", "y"])
        writer.writeheader()
        for x, y in points:
            writer.writerow({"x": x, "y": y})

    tracker = TrackProgress(str(path))

    assert tracker.points.tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert tracker.total_length == 4.0


def test_lap_completion_arms_first_awsim_transition_and_accepts_geometry():
    track_length = 354.18

    assert completed_lap_count(14.47, track_length, awsim_lap_transitions=1) == 0
    assert completed_lap_count(14.47, track_length, awsim_lap_transitions=2) == 1
    assert completed_lap_count(track_length, track_length, awsim_lap_transitions=0) == 1
