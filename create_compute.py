from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Only letters, digits and dash are allowed
# Length: 2-16 characters
CLUSTER_NAME = "gpu-cluster"


ws = Workspace.from_config()

try:
    compute_target = ComputeTarget(workspace=ws, name=CLUSTER_NAME)
    print('Found existing compute target')
except ComputeTargetException as e:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           idle_seconds_before_scaledown=1200,
                                                           min_nodes=0,
                                                           max_nodes=4)

    compute_target = ComputeTarget.create(ws, CLUSTER_NAME, compute_config)

compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

