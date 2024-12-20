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
    # Replace 'your_program' and arguments with the program you want to run
    testcases = [
        "normal1000",
        "normal100",
        "normal50",
        "quad100",
        "quad50",
    ]
    tempdir = "profiling/"
    testdir = "testcases/"
    for testcase in testcases:
        command = ["sbatch", "--version"]
        output, return_code, execution_time = run_program(command)
        output = output.split()[0]
        if output == "Accepted":
            print("\033[1;32;40mAccepted\033[0m 1;32;40m")
        print(f"Output:\n{output}")
        print(f"Return Code: {return_code}")
        print(f"Execution Time: {execution_time:.4f} seconds")
