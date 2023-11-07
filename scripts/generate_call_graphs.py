__author__ = "Saumik"
__date__ = "10/31/2023"

""" 

python .\generate_call_graph.py <python script> <path to png file to be generated>

"""

import subprocess
import os
import re
import sys

def run_cProfile(script_path, output_file):
    cmd = [sys.executable, "-m", "cProfile", "-o", output_file, script_path]
    subprocess.run(cmd)

def run_gprof2dot(input_file, temp_dot_file):
    cmd = f"gprof2dot -f pstats {input_file} > {temp_dot_file}"
    subprocess.run(cmd, shell=True)

def filter_dot_file(temp_dot_file, filtered_dot_file):
    nodes_to_remove = set()
    filtered_lines = []

    with open(temp_dot_file, "r") as infile:
        for line in infile:
            if "frozen importlib" in line:
                # Extract the node identifier (usually a number)
                match = re.search(r'(\d+)\s+\[', line)
                if match:
                    nodes_to_remove.add(match.group(1))
            else:
                filtered_lines.append(line)

    with open(filtered_dot_file, "w") as outfile:
        for line in filtered_lines:
            # Remove any edges connected to the nodes we want to remove
            if any(node in line for node in nodes_to_remove):
                continue
            outfile.write(line)

def run_dot(filtered_dot_file, output_png_file):
    cmd = f"dot -Tpng -o {output_png_file} {filtered_dot_file}"
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":

    script_path = sys.argv[1]
    profile_output = ".\\profile_data.pstats"
    temp_dot_file = ".\\temp.dot"
    filtered_dot_file = ".\\filtered.dot"
    output_png_file = sys.argv[2]

    # Step 0: Run cProfile on the script
    run_cProfile(script_path, profile_output)

    # Step 1: Run gprof2dot
    run_gprof2dot(profile_output, temp_dot_file)

    # Step 2: Filter the DOT file
    filter_dot_file(temp_dot_file, filtered_dot_file)

    # Step 3: Run dot to generate PNG
    run_dot(filtered_dot_file, output_png_file)

    # Remove temporary files if needed
    os.remove(temp_dot_file)
    os.remove(profile_output)
    os.remove(filtered_dot_file)

