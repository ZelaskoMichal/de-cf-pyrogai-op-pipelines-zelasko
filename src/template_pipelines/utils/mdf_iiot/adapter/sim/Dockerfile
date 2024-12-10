FROM acrci3415437dev01.azurecr.io/platform/simulator:beta

RUN apt-get update && apt-get install -y jq

COPY mk01iris.sim.json /configs/iris-demo.sim.json
COPY create_input_data.py /configs/create_input_data.py

# Use this to convert the hostname so we don't need to have two separate config files
RUN jq '.Grpc.HostName |= (sub("auto"; "mk01-iris-inspect-flowers"))' /configs/iris-demo.sim.json >> /configs/iris-demo.sim.json.new
RUN mv /configs/iris-demo.sim.json.new /configs/iris-demo.sim.json
