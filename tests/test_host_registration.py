import subprocess


def test_segmented_host_registration_fallback(tmp_path):
    executable = tmp_path / "test_host_registration"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            "csrc",
            "tests/cpp/test_host_registration.cpp",
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
