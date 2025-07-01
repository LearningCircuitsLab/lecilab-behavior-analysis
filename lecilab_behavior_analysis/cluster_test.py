# run the following shell command using python: rsync -azv hvergara@mini:/home/hvergara/test.deleteme .
import subprocess

def main():
    ssh_command = (
        "ssh hvergara@mini 'ls /archive/training_village/'"
    )
    result = subprocess.run(
        ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    print(result.stdout.decode())


if __name__ == "__main__":
    main()