import argparse


def calculate_success_rate(file_name, threshold):
    print(f'using success threshold of {threshold}')
    with open(file_name, 'r') as f:
        contents = f.read()
    lines = contents.split('\n')
    traj_idx = 0
    successes = 0
    total = 0
    traj_found = True
    while traj_found:
        traj_found = False
        data = None
        for line in lines:
            if line.startswith(f'{traj_idx}: '):
                traj_found = True
                data = line
        if traj_found:
            # Extracting numbers...
            data = data.replace(':', ',').split(',')
            final_dist = float(data[2])
            with open('scores.txt', 'a') as s:
                s.write(str(final_dist) + '\n')
            if final_dist <= threshold:
                successes += 1
            total += 1
        traj_idx += 1
    print(f'{1.0*successes/total*100}% success rate, of {total} trials.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str, help='path to outputted benchmark results file')
    parser.add_argument('threshold', type=float, help='success threshold')
    args = parser.parse_args()
    calculate_success_rate(args.results_file, args.threshold)
