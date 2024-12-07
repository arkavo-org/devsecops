import os
import sys
import shutil

def substitutions(currdir, env): 
    if os.path.isdir(currdir):
        for file in os.listdir(currdir):
            substitutions(os.path.join(currdir, file), env)
    else:
        if currdir.endswith(".template"):
            print("Applying substitutions to " + currdir)
            with open(currdir, 'r') as f:
                templateText = f.read()
            for k, v in vars(env).items():
                templateText = templateText.replace("$"+k, str(v))
            with open(currdir.replace(".template",""), 'w+') as f:
                f.write(templateText)

def initializeFiles():
    # Check if we are in a GitHub Actions environment
    in_github_actions = os.getenv("GITHUB_ACTIONS") == "true"
    print(in_github_actions)
    
    if not os.path.isfile("env.py"):
        shutil.copy("env.example.py", "env.py")
        print("env.py file did not exist and has been created. Please edit it to update the necessary values, then re-run this script.")
        
        # Exit only if not in GitHub Actions
        if not in_github_actions:
            sys.exit(1)
        else:
            print("Running in GitHub Actions, continuing without exiting.")