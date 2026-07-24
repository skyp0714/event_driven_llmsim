import json
import re
from collections.abc import Mapping
from .logger import get_logger


_BACKGROUND_TRANSFER_COMPLETE = re.compile(
    r"Background transfer complete\t([^\t\n]+)\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t(\d+)")
_HBF_BACKGROUND_COMPLETE = re.compile(
    r"HBF background complete\t([^\t\n]+)\t(\d+)\t(\d+)\t(\d+)")
_CONTROL_EVENT = re.compile(r"Control event\t([^\t\n]+)\t(\d+)")
_CONTROL_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_BACKGROUND_FABRIC_CAPABILITY = "Analytical control capability\tcold-fabric-v1"
_ENDPOINT_PARK_CAPABILITY = "Analytical control capability\tendpoint-park-v1"
_HBF_BACKGROUND_CAPABILITY = (
    "Analytical control capability\thbf-background-v1")
_POST_ENDPOINT_BARRIER_CAPABILITY = (
    "Analytical control capability\tpost-endpoint-barrier-v1")
_HBF_COMPONENT = re.compile(r"^[A-Za-z0-9_.:-]+$")


class ExactControlSchedule:
    """Track one-shot absolute-time callbacks issued to analytical ASTRA."""

    def __init__(self):
        self._sequence = 0
        self._pending_by_id = {}
        self._pending_by_time = {}
        self._completed = {}

    def arm(self, absolute_time_ns, current_time_ns):
        timestamp = Controller._validate_nonnegative_int(
            absolute_time_ns, "absolute_time_ns")
        current = Controller._validate_nonnegative_int(
            current_time_ns, "current_time_ns")
        if timestamp <= current:
            return None
        existing = self._pending_by_time.get(timestamp)
        if existing is not None:
            return None
        event_id = f"python-ready.{self._sequence}"
        self._sequence += 1
        self._pending_by_id[event_id] = timestamp
        self._pending_by_time[timestamp] = event_id
        return Controller.control_at_command(event_id, timestamp)

    def complete(self, event_id, absolute_time_ns):
        timestamp = Controller._validate_nonnegative_int(
            absolute_time_ns, "absolute_time_ns")
        if event_id in self._completed:
            if self._completed[event_id] != timestamp:
                raise RuntimeError(
                    "Duplicate control callback changed its timestamp: "
                    f"event={event_id}")
            return False
        try:
            expected = self._pending_by_id.pop(event_id)
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown analytical control callback {event_id!r}") from exc
        if expected != timestamp:
            raise RuntimeError(
                "Analytical control callback timestamp changed: "
                f"event={event_id}, expected={expected}, observed={timestamp}")
        self._pending_by_time.pop(timestamp)
        self._completed[event_id] = timestamp
        return True

    def has_pending(self):
        return bool(self._pending_by_id)

    def next_pending_time(self):
        return min(self._pending_by_time, default=None)


class SameTimeControlBarrier:
    """Commit Python effects after every ASTRA endpoint at one timestamp.

    The congestion-aware backend delivers this callback only after the event
    queue and every endpoint report at the timestamp have been drained.
    Python can therefore defer tied GPU and HBF completions until this
    callback without advancing analytical time.
    """

    def __init__(self):
        self._sequence = 0
        self._pending_id = None
        self._pending_time = None
        self._completed = {}

    def arm(self, current_time_ns):
        timestamp = Controller._validate_nonnegative_int(
            current_time_ns, "current_time_ns")
        if self._pending_id is not None:
            if self._pending_time != timestamp:
                raise RuntimeError(
                    "A same-time barrier is still pending at another "
                    f"timestamp: pending={self._pending_time}, "
                    f"requested={timestamp}")
            return None
        event_id = f"python-tie.{self._sequence}"
        self._sequence += 1
        self._pending_id = event_id
        self._pending_time = timestamp
        return Controller.control_after_endpoints_command(
            event_id, timestamp)

    def owns(self, event_id):
        return event_id == self._pending_id or event_id in self._completed

    def complete(self, event_id, current_time_ns):
        timestamp = Controller._validate_nonnegative_int(
            current_time_ns, "current_time_ns")
        if event_id in self._completed:
            if self._completed[event_id] != timestamp:
                raise RuntimeError(
                    "Duplicate same-time barrier changed its timestamp")
            return False
        if event_id != self._pending_id:
            raise RuntimeError(
                f"Unknown same-time barrier callback {event_id!r}")
        if timestamp != self._pending_time:
            raise RuntimeError(
                "Same-time barrier timestamp changed: "
                f"expected={self._pending_time}, observed={timestamp}")
        self._completed[event_id] = timestamp
        self._pending_id = None
        self._pending_time = None
        return True

    def has_pending(self):
        return self._pending_id is not None

    def pending_time(self):
        return self._pending_time


class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        self._auxiliary_command_provider = None
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            if line == "":
                self._raise_backend_eof(p, "Waiting")
            # For debugging
            # print(line, end='')
            out.append(line)
            p.stdout.flush()
        return out

    def check_end(self, p):
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            line = p.stdout.readline()
            if line == "":
                self._raise_backend_eof(p, "final completion")
            out.append(line)
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    @staticmethod
    def _raise_backend_eof(p, expected):
        return_code = p.poll()
        stderr = (
            p.stderr.read()
            if return_code is not None and p.stderr is not None
            else ""
        )
        detail = stderr.strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        suffix = f"; stderr: {detail}" if detail else ""
        raise RuntimeError(
            "ASTRA-Sim closed stdout before "
            f"{expected} (returncode={return_code}){suffix}")

    def set_auxiliary_command_provider(self, provider):
        if provider is not None and not callable(provider):
            raise TypeError("auxiliary command provider must be callable")
        self._auxiliary_command_provider = provider

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        commands = []
        if input != "exit" and self._auxiliary_command_provider is not None:
            commands = list(self._auxiliary_command_provider(input) or ())
        for command in commands:
            if (not isinstance(command, str) or not command
                    or "\n" in command or "\r" in command):
                raise ValueError(
                    "auxiliary commands must be non-empty single-line strings")
            self.logger.debug("ASTRA auxiliary command: %s", command)
            p.stdin.write(command+'\n')
        self.logger.debug("ASTRA endpoint command: %s", input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parse_output(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return

    @staticmethod
    def parse_protocol_event(output):
        """Parse a model callback or exact analytical control callback.

        ``read_wait`` returns all output through the next wait marker.  Cold
        HBM transfer completions do not own an ASTRA endpoint and therefore
        must be distinguished from the legacy ``sys[...]`` model report.
        """
        text = "".join(output) if isinstance(output, (list, tuple)) else output
        transfer = _BACKGROUND_TRANSFER_COMPLETE.search(text)
        if transfer:
            return {
                "type": "background_transfer_complete",
                "job_id": transfer.group(1),
                "arrival_ns": int(transfer.group(2)),
                "completion_ns": int(transfer.group(3)),
                "bytes_per_lane": int(transfer.group(4)),
                "lane_count": int(transfer.group(5)),
                "critical_lane_start_ns": int(transfer.group(6)),
            }
        hbf_background = _HBF_BACKGROUND_COMPLETE.search(text)
        if hbf_background:
            return {
                "type": "hbf_background_complete",
                "job_id": hbf_background.group(1),
                "arrival_ns": int(hbf_background.group(2)),
                "completion_ns": int(hbf_background.group(3)),
                "stage_count": int(hbf_background.group(4)),
            }
        control = _CONTROL_EVENT.search(text)
        if control:
            return {
                "type": "control_event",
                "event_id": control.group(1),
                "time_ns": int(control.group(2)),
            }

        pattern = (
            r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
            r"exposed communication (\d+) cycles\.")
        model = re.search(pattern, text)
        if model:
            return {
                "type": "model_complete",
                "sys": int(model.group(1)),
                "id": int(model.group(2)),
                "cycle": int(model.group(3)),
                "communication_cycle": int(model.group(4)),
            }
        return None

    @staticmethod
    def has_background_fabric_capability(output):
        text = "".join(output) if isinstance(output, (list, tuple)) else output
        return _BACKGROUND_FABRIC_CAPABILITY in text

    @staticmethod
    def has_endpoint_park_capability(output):
        text = "".join(output) if isinstance(output, (list, tuple)) else output
        return _ENDPOINT_PARK_CAPABILITY in text

    @staticmethod
    def has_hbf_background_capability(output):
        text = "".join(output) if isinstance(output, (list, tuple)) else output
        return _HBF_BACKGROUND_CAPABILITY in text

    @staticmethod
    def has_post_endpoint_barrier_capability(output):
        text = "".join(output) if isinstance(output, (list, tuple)) else output
        return _POST_ENDPOINT_BARRIER_CAPABILITY in text

    @staticmethod
    def control_at_command(event_id, absolute_time_ns):
        Controller._validate_control_identifier(event_id, "event_id")
        timestamp = Controller._validate_nonnegative_int(
            absolute_time_ns, "absolute_time_ns")
        return f"control-at\t{event_id}\t{timestamp}"

    @staticmethod
    def control_after_endpoints_command(event_id, absolute_time_ns):
        Controller._validate_control_identifier(event_id, "event_id")
        timestamp = Controller._validate_nonnegative_int(
            absolute_time_ns, "absolute_time_ns")
        return f"control-after-endpoints\t{event_id}\t{timestamp}"

    @staticmethod
    def background_transfer_command(
            job_id, arrival_ns, bytes_per_lane, lanes,
            chunk_bytes=64 * 1024 * 1024):
        """Encode a congestion-aware cold-HBM transfer command.

        ``bytes_per_lane`` is the per-TP-rank byte count, not the aggregate
        across lanes. Each ``lanes`` item is a ``(logical_src, logical_dst)``
        pair; ASTRA resolves endpoint aliases before contention accounting.
        """
        Controller._validate_control_identifier(job_id, "job_id")
        arrival = Controller._validate_nonnegative_int(
            arrival_ns, "arrival_ns")
        byte_count = Controller._validate_positive_int(
            bytes_per_lane, "bytes_per_lane")
        chunk = Controller._validate_positive_int(chunk_bytes, "chunk_bytes")
        if not lanes:
            raise ValueError("lanes must contain at least one rank pair")
        encoded_lanes = []
        seen = set()
        for lane in lanes:
            if not isinstance(lane, (tuple, list)) or len(lane) != 2:
                raise TypeError("each lane must be a (source, destination) pair")
            source = Controller._validate_nonnegative_int(lane[0], "source")
            destination = Controller._validate_nonnegative_int(
                lane[1], "destination")
            if source == destination:
                raise ValueError("transfer lane source and destination differ")
            pair = (source, destination)
            if pair in seen:
                raise ValueError(f"duplicate transfer lane {pair}")
            seen.add(pair)
            encoded_lanes.append(f"{source}>{destination}")
        return "\t".join((
            "background-transfer", str(job_id), str(arrival),
            str(byte_count), str(chunk), ",".join(encoded_lanes)))

    @staticmethod
    def hbf_background_command(job_id, arrival_ns, stages):
        """Encode an asynchronous HBF DAG with explicit route resources.

        No P/D rank or PCIe resource is inferred here. Each stage must carry
        the exact execution-instance resources produced by its HBF route, so
        a caller cannot accidentally obtain an eight-link P4+D4 pool from the
        protocol layer.
        """
        Controller._validate_control_identifier(job_id, "job_id")
        arrival = Controller._validate_nonnegative_int(
            arrival_ns, "arrival_ns")
        if isinstance(stages, (str, bytes)):
            raise TypeError("stages must be an iterable of stage descriptors")
        try:
            raw_stages = list(stages)
        except TypeError as exc:
            raise TypeError(
                "stages must be an iterable of stage descriptors") from exc
        if not raw_stages:
            raise ValueError("stages must contain at least one HBF stage")

        encoded_stages = []
        stage_ids = set()
        allowed = {
            "id", "runtime_ns", "tensor_bytes", "resources", "deps",
        }
        for raw_stage in raw_stages:
            if hasattr(raw_stage, "as_dict"):
                raw_stage = raw_stage.as_dict()
            if not isinstance(raw_stage, Mapping):
                raise TypeError("each HBF stage must be a mapping")
            unknown = set(raw_stage) - allowed
            missing = allowed - set(raw_stage)
            if unknown or missing:
                raise ValueError(
                    "HBF stage fields must be exactly "
                    f"{sorted(allowed)}; missing={sorted(missing)}, "
                    f"unknown={sorted(unknown)}")

            stage_id = Controller._validate_hbf_component(
                raw_stage["id"], "stage id")
            if stage_id in stage_ids:
                raise ValueError(f"duplicate HBF stage id {stage_id!r}")
            stage_ids.add(stage_id)
            runtime = Controller._validate_positive_int(
                raw_stage["runtime_ns"], "runtime_ns")
            tensor_bytes = Controller._validate_nonnegative_int(
                raw_stage["tensor_bytes"], "tensor_bytes")
            resources = Controller._validate_hbf_components(
                raw_stage["resources"], "resources", allow_empty=False)
            dependencies = Controller._validate_hbf_components(
                raw_stage["deps"], "deps", allow_empty=True)
            encoded_stages.append({
                "id": stage_id,
                "runtime_ns": runtime,
                "tensor_bytes": tensor_bytes,
                "resources": resources,
                "deps": dependencies,
            })

        remaining = {}
        dependents = {stage_id: [] for stage_id in stage_ids}
        for stage in encoded_stages:
            stage_id = stage["id"]
            for dependency in stage["deps"]:
                if dependency not in stage_ids:
                    raise ValueError(
                        f"HBF stage {stage_id!r} has unknown dependency "
                        f"{dependency!r}")
                if dependency == stage_id:
                    raise ValueError(
                        f"HBF stage {stage_id!r} depends on itself")
                dependents[dependency].append(stage_id)
            remaining[stage_id] = len(stage["deps"])
        ready = [
            stage_id for stage_id, count in remaining.items() if count == 0
        ]
        visited = 0
        while ready:
            stage_id = ready.pop()
            visited += 1
            for dependent in dependents[stage_id]:
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    ready.append(dependent)
        if visited != len(encoded_stages):
            raise ValueError("HBF background descriptor contains a cycle")

        descriptor = json.dumps(
            {"v": 1, "stages": encoded_stages},
            separators=(",", ":"),
            sort_keys=True,
        )
        return "\t".join((
            "hbf-background", str(job_id), str(arrival), descriptor))

    @staticmethod
    def _validate_control_identifier(value, field):
        if not isinstance(value, str) or not _CONTROL_IDENTIFIER.fullmatch(value):
            raise ValueError(
                f"{field} must contain only letters, digits, '.', '_' or '-'")

    @staticmethod
    def _validate_hbf_component(value, field):
        if not isinstance(value, str) or not _HBF_COMPONENT.fullmatch(value):
            raise ValueError(
                f"{field} must contain only letters, digits, '.', '_', "
                "'-' or ':'")
        return value

    @staticmethod
    def _validate_hbf_components(values, field, *, allow_empty):
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{field} must be a sequence of identifiers")
        try:
            raw_values = list(values)
        except TypeError as exc:
            raise TypeError(
                f"{field} must be a sequence of identifiers") from exc
        if not allow_empty and not raw_values:
            raise ValueError(f"{field} must not be empty")
        result = []
        seen = set()
        for value in raw_values:
            component = Controller._validate_hbf_component(value, field)
            if component in seen:
                raise ValueError(f"duplicate {field} entry {component!r}")
            seen.add(component)
            result.append(component)
        return result

    @staticmethod
    def _validate_nonnegative_int(value, field):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value

    @staticmethod
    def _validate_positive_int(value, field):
        result = Controller._validate_nonnegative_int(value, field)
        if result == 0:
            raise ValueError(f"{field} must be positive")
        return result
