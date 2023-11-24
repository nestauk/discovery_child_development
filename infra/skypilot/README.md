Loosely based on [these docs](https://skypilot.readthedocs.io/en/latest/examples/gpu-jupyter.html#:~:text=SkyPilot%20makes%20the%20process%20of,managing%20provisioning%20and%20port%20forwarding.&text=View%20the%20supported%20GPUs%20with%20the%20sky%20show%2Dgpus%20command.&text=Enter%20the%20password%20or%20token,can%20create%20a%20new%20notebook.).

To run a jupyter notebook on skypilot:

- Navigate to this repo
- Activate the conda env and make sure requirements are installed
- Export secrets

```
export OPENAI_API_KEY= #<OpenAI API key>
```

- Then run

```
sky launch -c task --use-spot ./infra/skypilot/task.yaml
```

- Input "Y" when prompted to start the cluster.
- SSH into the cluster (means you can interact with the cluster through your local terminal):

```
ssh -L 8888:localhost:8888 task
```

- Navigate to the correct directory:

```
cd sky_workdir/discovery_child_development
```

- You may also need to deactivate a conda environment:

```
conda deactivate
```

- Activate the correct env with:

```
conda activate discovery_child_development
```

- Install and launch jupyter:

```
pip install jupyter
jupyter notebook
```
