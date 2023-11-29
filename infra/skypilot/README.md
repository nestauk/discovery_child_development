Loosely based on [these docs](https://skypilot.readthedocs.io/en/latest/examples/gpu-jupyter.html#:~:text=SkyPilot%20makes%20the%20process%20of,managing%20provisioning%20and%20port%20forwarding.&text=View%20the%20supported%20GPUs%20with%20the%20sky%20show%2Dgpus%20command.&text=Enter%20the%20password%20or%20token,can%20create%20a%20new%20notebook.).

To run a jupyter notebook on skypilot (you will need 2 terminals):

**In terminal 1**
1. Navigate to this repo
2. Activate the conda env and make sure you have a conda env with the requirements installed (run make install)
3. Pip install skypilot (causes problems if it's in `requirements.txt``, so you need to install it manually)

```
pip install skypilot
```

4. Then run

```
sky launch -c task --use-spot ./infra/skypilot/task.yaml
```

5. Input "Y" when prompted to start the cluster. The cluster should now launch and you will see that jupyter is running! Pay attention to the URLs provided here - you will need the `token` part

**In terminal 2**

6. In a **NEW** terminal, SSH into the cluster (means you can interact with the cluster through your local terminal):

```
ssh -L 8888:localhost:8888 task
```

7. Paste `localhost:8888` into a local browser. It may prompt you for a token - use one of the tokens from the URLs in step (5). You should now be able to run any jupyter notebooks from this repo on the cloud!

**IMPORTANT: When you are done...**

8. Close terminal 2 (or you can use the command `exit` - either is fine)
9. In terminal 1, `crtl+c` to exit jupyter
10. Now run the following command to kill the compute instance and enter `Y` when prompted:

```
sky down task
```
This terminates the cluster and stops it from costing us anything!

11. Run `sky status` to check you have no clusters still running. This should return something like "No existing clusters"
