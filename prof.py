import subprocess
import time


def run_program(command):
    start_time = time.time()  # Record the start time
    try:
        # Run the program and capture its output
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        output = result.stdout
        return_code = result.returncode
    except subprocess.CalledProcessError as e:
        # If the program fails, capture the return code and stderr
        output = e.stderr
        return_code = e.returncode
    finally:
        end_time = time.time()  # Record the end time

    execution_time = end_time - start_time
    return output, return_code, execution_time


if __name__ == "__main__":
    testcases = [
        "q50",
        "q100",
        "q200",
        "q400",
        "q1000",
        "n50",
        "n100",
        "n200",
        "n400",
        "n1000",
        "nn50",
        "nn100",
        "nn200",
        "nn400",
        "nn1000",
        "c50",
        "c100",
        "c200",
        "c400",
        "c1000",
    ]
    seqdir = "seqout/"
    paradir = "paraout/"
    testdir = "testcases/"
    for testcase in testcases:
        command = [
            "srun",
            "-p",
            "mi2104x",
            "-t",
            "00:10:00",
            "./para",
            f"{testdir}{testcase}.in",
            f"{paradir}{testcase}.out",
        ]
        _, _, execution_time_para = run_program(command)
        command = [
            "srun",
            "-p",
            "mi2104x",
            "-t",
            "00:10:00",
            "./seq",
            f"{testdir}{testcase}.in",
            f"{seqdir}{testcase}.out",
        ]
        _, _, execution_time_seq = run_program(command)
        command = [
            "python",
            "validate.py",
            f"{seqdir}{testcase}.out",
            f"{paradir}{testcase}.out",
        ]
        output, _, _ = run_program(command)
        output = output.split()[0]
        print(f"\nTestcase {testcase}:")
        print(f"    Parallel version execution time: {execution_time_para:.4f} seconds")
        print(f"    Sequential version execution time: {execution_time_seq:.4f} seconds")
        if output == "Accepted":
            print(f"    \033[92mAccepted\033[0m")
        else:
            print(f"    \033[93mWrong Answer\033[0m")
