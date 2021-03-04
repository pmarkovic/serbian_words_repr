from azureml.core import Workspace, environment
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig


if __name__ == "__main__":
    ws = Workspace.from_config()

    # set up pytorch environment
    pytorch_env = Environment.from_conda_specification(
        name='serb2vec-env',
        file_path='./.azureml/environment.yml'
    )
    # Specify a GPU base image
    pytorch_env.docker.enabled = True
    pytorch_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'

    config = ScriptRunConfig(source_directory='./src',
                             script='train_model.py',
                             arguments=["--is_sg"],
                             compute_target='gpu-cluster',
                             environment=pytorch_env)


    experiment = Experiment(workspace=ws, name='sg300-experiment1')
    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)

    run.wait_for_completion(show_output=True)
    
    run.download_file(name='data/params.txt', output_file_path='./src/data/params.txt'), 

