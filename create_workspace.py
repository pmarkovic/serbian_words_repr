from azureml.core import Workspace

WS_NAME = "serb2vec_ws"
SUBSCRIPTION_ID = "88da09b3-7d71-4965-a4e9-c59ddf1ffa8f"
RESOURCE_GROUP = "Serb2Vec"
LOCATION = "westeurope"

ws = Workspace.create(name=WS_NAME, 
                      subscription_id=SUBSCRIPTION_ID,
                      resource_group=RESOURCE_GROUP,
                      create_resource_group=True,
                      location=LOCATION)

ws.write_config(path=".azureml")