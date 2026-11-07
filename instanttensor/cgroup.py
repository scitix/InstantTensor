"""Read visible cgroup v2 resources and count sharing within a process group.

Keys identify live kernel objects, not persistent historical identities. The boot
UUID must identify the real host; CGROUP_HOST_BOOT_ID can override the proc value.
Hidden ancestors cannot be queried. Resource readings are not an atomic snapshot.
"""

import contextlib
from dataclasses import dataclass
import os
import posixpath
import re
import uuid


@dataclass(frozen=True)
class CgroupKey:
    host_boot_id: str
    dev_major: int
    dev_minor: int
    inode: int


@dataclass(frozen=True)
class CgroupResources:
    key: CgroupKey
    path: str
    # None means this level supplies no finite constraint (unlimited or absent).
    avail_cpus: int | None
    avail_mem: int | None


@dataclass(frozen=True)
class CgroupSnapshot:
    # Current cgroup first, then each visible ancestor, including the mount root.
    entries: tuple[CgroupResources, ...]
    ancestors_complete: bool


@dataclass(frozen=True)
class CgroupUsage:
    process_counts: dict[CgroupKey, int]
    # Indexed by process-group rank; False means hidden ancestry may be missing.
    ancestors_complete_by_rank: tuple[bool, ...]


def _read(path):
    with open(path, encoding="utf-8", errors="surrogateescape") as stream:
        return stream.read()


def _components(path):
    parts = path.split("/")
    if not path.startswith("/") or any(p in (".", "..") for p in parts):
        raise ValueError("Unresolvable namespace-relative path: " + repr(path))
    return tuple(p for p in parts if p)


def _v2_path(text):
    for line in text.splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            _components(path)
            return path
    raise ValueError("No cgroup v2 membership found")


def _mounts(text):
    result = []
    for line in text.splitlines():
        left, right = line.split(" - ", 1)
        if right.split()[0] != "cgroup2":
            continue
        fields = left.split()
        root, point = (re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), field)
                       for field in fields[3:5])
        try:
            root_parts = _components(root)
        except ValueError:
            continue
        result.append({"id": int(fields[0]), "device": fields[2],
                       "root_parts": root_parts, "point": point})
    return sorted(result, key=lambda mount: len(mount["root_parts"]))


def _local_path(mount, path):
    root, parts = mount["root_parts"], _components(path)
    if root != parts[:len(root)]:
        return None
    return posixpath.join(mount["point"], *parts[len(root):])


def _checked_open(path, mount, stack):
    fd = os.open(path, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
    stack.callback(os.close, fd)
    info = dict(line.split(":", 1) for line in _read(f"/proc/self/fdinfo/{fd}").splitlines())
    st = os.fstat(fd)
    device = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    if int(info["mnt_id"]) != mount["id"] or device != mount["device"]:
        raise ValueError("Mount changed or another mount covers " + path)
    if st.st_nlink == 0:
        raise ValueError("Cgroup was removed during observation")
    return fd, st


def _read_at(fd, name):
    with os.fdopen(os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd),
                   encoding="ascii") as stream:
        return stream.read().strip()


def _optional_read_at(fd, name):
    try:
        return _read_at(fd, name)
    except FileNotFoundError:
        return None


def _parse_cpuset(text):
    cpus = set()
    for item in text.split(",") if text else ():
        bounds = item.split("-")
        start, end = int(bounds[0]), int(bounds[-1])
        if len(bounds) > 2 or start < 0 or end < start:
            raise ValueError("Invalid cpuset: " + text)
        cpus.update(range(start, end + 1))
    return tuple(sorted(cpus))


def _resources(fd, key, path):
    cpu_limits = []
    cpu_max = _optional_read_at(fd, "cpu.max")
    if cpu_max is not None:
        quota_text, period_text = cpu_max.split()
        if quota_text != "max":
            quota, period = int(quota_text), int(period_text)
            cpu_limits.append((quota + period - 1) // period)
    cpuset = _optional_read_at(fd, "cpuset.cpus.effective")
    if cpuset is not None:
        cpu_limits.append(len(_parse_cpuset(cpuset)))
    avail_mem = None
    limit = _optional_read_at(fd, "memory.max")
    if limit is not None and limit != "max":
        current = int(_read_at(fd, "memory.current"))
        stats = dict(line.split() for line in _read_at(fd, "memory.stat").splitlines())
        avail_mem = max(int(limit) - current + int(stats["inactive_file"]), 0)
    return CgroupResources(key, path, min(cpu_limits, default=None), avail_mem)


def query_resources(*, host_boot_id: str | None = None) -> CgroupSnapshot:
    """Query this process's visible v2 ancestry, without allocating any budget.

    Raises OSError/ValueError when v2 cannot be resolved or resources cannot be
    read. Missing controller files are represented explicitly, not as zero usage.
    The completeness flag uses the Linux kernfs root-inode convention (inode 1).
    """
    boot_id = get_host_boot_id(host_boot_id)
    membership = _v2_path(_read("/proc/self/cgroup"))
    errors = []
    for mount in _mounts(_read("/proc/self/mountinfo")):
        if _local_path(mount, membership) is None:
            continue
        try:
            with contextlib.ExitStack() as stack:
                entries, paths = [], []
                current = membership
                while True:
                    path = _local_path(mount, current)
                    fd, st = _checked_open(path, mount, stack)
                    key = CgroupKey(boot_id, os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
                    entries.append(_resources(fd, key, current))
                    paths.append(path)
                    if _components(current) == mount["root_parts"]:
                        break
                    current = posixpath.dirname(current)
                if _v2_path(_read("/proc/self/cgroup")) != membership:
                    raise ValueError("Cgroup membership changed during observation")
                for entry, path in zip(entries, paths):
                    _, st = _checked_open(path, mount, stack)
                    if entry.key != CgroupKey(boot_id, os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
                        raise ValueError("Cgroup changed during observation")
                return CgroupSnapshot(tuple(entries), entries[-1].key.inode == 1)
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(error))
    raise ValueError("Cannot resolve cgroup v2: " +
                     ("; ".join(errors) or "no compatible, accessible cgroup2 mount"))


def _encode_key(key):
    # Use 32-bit words so UUIDs and unsigned inode numbers fit signed int64
    # tensors exactly. Python hash() is process-dependent and is not used.
    host = uuid.UUID(key.host_boot_id).int
    return [(host >> shift) & 0xffffffff for shift in (96, 64, 32, 0)] + [
        key.dev_major, key.dev_minor, key.inode >> 32, key.inode & 0xffffffff,
    ]


def _decode_key(words):
    host = 0
    for word in words[:4]:
        host = (host << 32) | word
    return CgroupKey(str(uuid.UUID(int=host)), words[4], words[5],
                     (words[6] << 32) | words[7])


def get_host_boot_id(override=None):
    return str(uuid.UUID(override or os.getenv("CGROUP_HOST_BOOT_ID") or
                         _read("/proc/sys/kernel/random/boot_id").strip()))


def _gather_records(values, process_group, device):
    """Exchange variable-length int64 records without pickling Python objects."""
    if process_group is None:
        return [values]

    import torch
    import torch.distributed as dist

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(process_group) == "nccl" else torch.device("cpu")
    size = torch.tensor([len(values)], dtype=torch.int64, device=device)
    dist.all_reduce(size, op=dist.ReduceOp.MAX, group=process_group)
    rows = torch.empty((dist.get_world_size(process_group), size.item()),
                       dtype=torch.int64, device=device)
    record = rows[dist.get_rank(process_group)]
    record.zero_()
    record[:len(values)] = torch.tensor(values, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(rows.view(-1), record, group=process_group)
    return rows.tolist()


def count_process_group_users(
    snapshot: CgroupSnapshot, process_group=None, *, device=None,
) -> CgroupUsage:
    """Count group ranks sharing every reported cgroup, including all ancestors.

    Every group rank must call this function with a successfully queried snapshot.
    None means local-only (no collectives). With a group, all_reduce(MAX)
    negotiates ancestry length, then all_gather_into_tensor exchanges padded keys.
    Counts cover visible ancestry only; no claim is made about hidden ancestors
    or processes outside this process group. The snapshot is not modified.
    """
    keys = list(dict.fromkeys(entry.key for entry in snapshot.entries))
    if process_group is None:
        return CgroupUsage({key: 1 for key in keys}, (snapshot.ancestors_complete,))

    values = [len(keys), int(snapshot.ancestors_complete)]
    for key in keys:
        values.extend(_encode_key(key))
    counts, complete = {}, []
    for row in _gather_records(values, process_group, device):
        complete.append(bool(row[1]))
        for index in range(row[0]):
            key = _decode_key(row[2 + 8 * index:2 + 8 * (index + 1)])
            counts[key] = counts.get(key, 0) + 1
    return CgroupUsage(counts, tuple(complete))
