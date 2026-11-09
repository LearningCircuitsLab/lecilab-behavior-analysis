"""Paths of the data on each computer and on the cluster, and copying it with ssh / rsync."""
from typing import List
import socket
import subprocess




#IDIBAPS_TV_PROJECTS = "/archive/training_village/"
IDIBAPS_TV_PROJECTS = "/storage/training_village/"


def get_outpath():
    hostname = socket.gethostname()
    paths = {
        "lorena-ThinkPad-E550": "/home/emma/Desktop/EloiJacomet/data",
        "tectum": "/mnt/c/Users/HMARTINEZ/LeCiLab/data/behavioral_data",
        "tudou": "/home/kudongdong/data/LeciLab/behavioral_data",
        "setup2": "/home/kudongdong/Documents/data/LeciLab/behavioral_data",
        "minibaps": "/storage/training_village",
        "minibaps2": "/storage/training_village",
    }
    return paths.get(hostname, "default/path")


def get_idibaps_cluster_credentials():
    hostname = socket.gethostname()
    if hostname == "tectum":
        return {
            "username": "hvergara",
            "host": "mini2",
            # "port": 443,
        }
    elif hostname == "tudou":
        return {
            "username": "kudongdong",
            "host": "mini",
        }
    elif hostname == "setup2":
        return {
            "username": "kudongdong",
            "host": "mini",
        }
    else:
        raise ValueError("Unknown host")


def get_server_projects() -> List[str]:
    credentials = get_idibaps_cluster_credentials()    
    return get_folders_from_server(credentials, IDIBAPS_TV_PROJECTS)


def get_folders_from_server(credentials: dict, path: str) -> List[str]:
    # Create a single SSH connection to the remote server
    ssh_command = (
        f"ssh {credentials['username']}@{credentials['host']} "
        f"'ls {path}'"
    )
    result = subprocess.run(
        ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Decode the output and split it into a list of folder names
    if result.returncode == 0:
        folders = result.stdout.decode("utf-8").strip().split("\n")
    else:
        print(f"Error: {result.stderr.decode('utf-8')}")
        folders = []
    # Filter out any unwanted folders (e.g., .git)
    folders = [folder for folder in folders if not folder.startswith(".")]

    return folders


def get_animals_in_project(project_name: str) -> List[str]:
    credentials = get_idibaps_cluster_credentials()
    folders = get_folders_from_server(
        credentials,
        f"{IDIBAPS_TV_PROJECTS}{project_name}/sessions/",
    )

    # filter out any unwanted folders (e.g., .csv)
    folders = [folder for folder in folders if not folder.endswith(".csv")]

    return folders


def get_files_for_animal(project_name: str, animal_name: str) -> List[str]:
    credentials = get_idibaps_cluster_credentials()
    files = get_folders_from_server(
        credentials,
        f"{IDIBAPS_TV_PROJECTS}{project_name}/sessions/{animal_name}/",
    )

    return files


def rsync_cluster_data(
    project_name: str,
    file_path: str,
    credentials: dict,
    local_path: str,
    exclude: List[str] = None,
) -> bool:
    """
    This method syncs the session data from the server to the local machine.
    """
    remote_path = f"{credentials['username']}@{credentials['host']}:{IDIBAPS_TV_PROJECTS}{project_name}/{file_path}"
    rsync_command = "rsync -avz"
    if exclude:
        for pattern in exclude:
            rsync_command += f" --exclude='{pattern}'"
    rsync_command += f" {remote_path} {local_path}"
    result = subprocess.run(rsync_command, shell=True)
    if result.returncode != 0:
        print(f"Error syncing data for {file_path}: {result.stderr.decode('utf-8')}")
    return result.returncode == 0


def rsync_specific_file(
    file_path: str,
    credentials: dict,
    local_path: str,
) -> bool:
    """
    This method syncs the session data from the server to the local machine.
    """
    remote_path = f"{credentials['username']}@{credentials['host']}:{file_path}"
    rsync_command = f"rsync -avz {remote_path} {local_path}"
    result = subprocess.run(rsync_command, shell=True)
    if result.returncode != 0:
        print(f"Error syncing data for {file_path}: {result.stderr.decode('utf-8')}")
    return result.returncode == 0
