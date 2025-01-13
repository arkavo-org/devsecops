import os
import sys
import shutil
import subprocess


here = os.path.abspath(os.path.dirname(__file__))

def writeViteEnv(env, output_file=os.path.join(here, "webapp", ".env")):
    print("Writing environment file for web app")
    # Open the file for writing
    with open(output_file, "w") as f:
        for key, value in env.items():
            if not key.startswith("__") and isinstance(value, (str, int, float)):
                f.write(f"{key}={value}\n")

    print(f"Environment variables have been written to {output_file}")


def substitutions(currdir, env): 
    if os.path.isdir(currdir):
        try:
            for file in os.listdir(currdir):
                substitutions(os.path.join(currdir, file), env)
        except:
            print(f"Couldn't process {currdir}")
    else:
        if currdir.endswith(".template"):
            print("Applying substitutions to " + currdir)
            newFile = currdir.replace(".template","")
            with open(currdir, 'r') as f:
                templateText = f.read()
            for k, v in vars(env).items():
                templateText = templateText.replace("$"+k, str(v))
                newFile = newFile.replace("$"+k, str(v)) # also templetize the filename (!)
            print(f"Writing to {newFile}")
            with open(newFile, 'w+') as f:
                f.write(templateText)

        if currdir.endswith(".copy"):
            newFile = currdir.replace(".copy","")
            if not os.path.exists(newFile):
                print(f"Copying {currdir} to {newFile}")
                shutil.copy(currdir, newFile)

def initializeFiles():
    # Check if we are in a GitHub Actions environment
    in_github_actions = os.getenv("GITHUB_ACTIONS") == "true"
    print(in_github_actions)
    envFile = os.path.join(here, "env.py")
    envExampleFile = os.path.join(here, "env.example.py")
    if not os.path.isfile(envFile):
        shutil.copy(envExampleFile, envFile)
        print("env.py file did not exist and has been created. Please edit it to update the necessary values, then re-run this script.")
        
        # Exit only if not in GitHub Actions
        if not in_github_actions:
            sys.exit(1)
        else:
            print("Running in GitHub Actions, continuing without exiting.")

            import docker

def check_nvidia_gpu():
    print("NVIDIA GPU Detected on system")
    try:
        # Try nvidia-smi command
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    
def check_amd_gpu():
    print("AMD GPU Detected on system")
    try:
        # Try rocm-smi command
        subprocess.run(["rocm-smi"], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    