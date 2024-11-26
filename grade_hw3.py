from argparse import ArgumentParser
from pathlib import Path
import zipfile
import sys
import subprocess
import logging
import time


def unzip(zip_path: Path, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)

def run_one_submission(zip_path: Path, extract_dir: Path, log_dir: Path, gpu_id: int = 0):
    print(f"Running submission {zip_path}")

    # Set up logging for this submission
    student_name = zip_path.stem.split("_")[0]
    extract_path = extract_dir / student_name
    log_path = log_dir / f"{student_name}.log"
    
    # (Re)set logging for each submission
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='w'
    )
    logger = logging.getLogger(student_name)

    # extract zip file
    logger.info(f"Extracting {zip_path} to {extract_path}")
    try:
        unzip(zip_path, extract_path)
    except Exception as e:
        logger.error(f"Error extracting {zip_path}")
        logger.exception(e)
        logger.error('FAILED - unzip')
        return

    # find project directory
    logger.info(f"Searching for `run.py` in {extract_path}")
    project_dir = None
    if (extract_path / "run.py").exists():
        project_dir = extract_path
    else:
        logger.warning(f"No `run.py` found in {extract_path}, searching for it in subdirectories")
        for subdir in extract_path.iterdir():
            if (subdir / "run.py").exists():
                project_dir = subdir
                break
    if project_dir:
        logger.info(f"Found `run.py` in {project_dir}")
        logger.info(f'Project directory set to {project_dir}')
    else:
        logger.error(f"No `run.py` found in {extract_path}")
        logger.error('FAILED - find project dir')
        return

    # build virtual environment
    logger.info(f"Building virtual environment in {project_dir}")
    if (project_dir / "requirements.txt").exists():  # install from requirements.txt
        requirements_path = project_dir / "requirements.txt"

        logger.info(f"Installing dependencies from {requirements_path}")
        venv_result = subprocess.run(
            ["uv", "venv", "--python", "3.11"],
            cwd=project_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        logger.info(f"Create virtual environment output:\n{venv_result.stdout}")
        if venv_result.returncode != 0:
            logger.error('FAILED - create venv')
            return

        install_result = subprocess.run(
            "source .venv/bin/activate && uv pip install pip && uv pip install -r requirements.txt",
            cwd=project_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            executable="/bin/bash"
        )
        logger.info(f"Install dependencies output:\n{install_result.stdout}")
        if install_result.returncode != 0:
            logger.error('FAILED - install dependencies from requirements.txt')
            return

    elif (project_dir / "pyproject.toml").exists():  # install from pyproject.toml
        pyproject_path = project_dir / "pyproject.toml"

        logger.info(f"Installing dependencies from {pyproject_path}")
        sync_result = subprocess.run(
            ["uv", "sync", "--python", "3.11"],
            cwd=project_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        logger.info(f"Install dependencies output:\n{sync_result.stdout}")
        if sync_result.returncode != 0:
            logger.error('FAILED - install dependencies from pyproject.toml')
            return
    else:
        logger.error(f"No requirements.txt or pyproject.toml found in {project_dir}")
        logger.error('FAILED - build venv')
        return
    logger.info('SUCCESS - build venv')

    # run the submission
    logger.info(f"Start running {project_dir}")
    logger.info(f"Command: `.venv/bin/python run.py submission.csv`")
    start_time = time.perf_counter()
    run_result = subprocess.run(
        [".venv/bin/python", "run.py", 'submission.csv'],
        cwd=project_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )
    end_time = time.perf_counter()
    logger.info(f"================== Run output ==================\n{run_result.stdout}")
    logger.info(f"================================================")
    if run_result.returncode != 0:
        logger.error('FAILED - run')
        return
    run_time = (end_time - start_time) / 60
    logger.info(f"Submission {student_name} run finished")
    logger.info(f"Runtime: {run_time:.2f} minutes")
    logger.info('SUCCESS - run')

    # check submission.csv
    if (project_dir / "submission.csv").exists():
        logger.info(f"Found submission.csv in submission {student_name}")
        logger.info('SUCCESS - submission.csv')
    else:
        logger.error(f"No submission.csv found in submission {student_name}")
        logger.error('FAILED - submission.csv')


def main(args):
    submissions = Path(args.submissions)
    extract_dir = Path(args.extract)
    log_dir = Path(args.log)

    assert submissions.exists(), f"Error: {submissions} does not exist"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        print(f"Created {extract_dir}")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        print(f"Created {log_dir}")

    if submissions.is_file():  # single submission zip file
        if not submissions.suffix.lower() == '.zip':
            print(f"Error: {submissions} is not a zip file")
            sys.exit(1)
        run_one_submission(submissions, extract_dir, log_dir, gpu_id=args.gpu)

    elif submissions.is_dir():  # multiple submission zip files
        for submission_zip in submissions.glob("*.zip"):
            run_one_submission(submission_zip, extract_dir, log_dir, gpu_id=args.gpu)

    else:  # invalid path
        print(f"Error: {submissions} is not a valid directory or zip file")
        sys.exit(1)

if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("submissions", type=str, help="Path to the submissions directory or a single submission zip file")
    parse.add_argument("--extract", type=str, default="extracted_submissions", help="Path to the extracted directory")
    parse.add_argument("--log", type=str, default="logs", help="Path to the log directory")
    parse.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parse.parse_args()
    main(args)
