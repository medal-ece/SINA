import os

def prepare_config_file():
    """
    Prepares a YAML configuration file with dynamically resolved paths.
    Combines static content with dynamically generated paths.
    """
    current_dir = os.getcwd()  # Get the current working directory
    project_path = os.path.dirname(current_dir)  # Get the parent directory (project root directory)
    
    # Dynamically generate the paths
    data_section = f"""# Data
path: {os.path.join(project_path, "Model training/data").replace("\\", "/")}
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
"""
    # Static sections of the config
    static_section = """\n# Keypoints
kpt_shape: [5, 3] # [number of keypoints, number of dim]

# Classes
names:
    0: Rectangle
"""

    # Combine both sections
    full_config = data_section + static_section

    # Save the combined config to a YAML file
    config_path = os.path.join(project_path, "Model training/config.yaml")
    with open(config_path, "w") as file:
        file.write(full_config)
    print(f"Config file prepared at {config_path}")
    return config_path, project_path

def prepare_paths(project_path):
    """
    Prepares and returns the source and destination paths for the training output.
    0: Rectangle
    """
    source_path = os.path.join(project_path, 'runs')
    destination_path = os.path.join(project_path, 'Current trained model')
    return source_path, destination_path