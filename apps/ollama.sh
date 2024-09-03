#!/usr/bin/env bash
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source  $(dirname $0)/functions

NAME="ollama"
IMAGE="ollama/ollama"
LLM_MODEL=$(config_lkp "LLM_MODEL" "llama2")

# This function is responsible for running creating a running the container
# and its dependencies.
_docker_run() {
	docker volume create $NAME > /dev/null
	docker run \
		--name $NAME \
        -d \
        --gpus=all \
        -v ollama:/root/.ollama \
        -p 11434:11434 \
		$DOCKER_NETWORK $IMAGE
}

# stop and remove the running container
_docker_stop() {
	docker stop -t "$STOP_TIMEOUT" "$NAME" > /dev/null
	docker rm -f "$NAME" > /dev/null
}

# print the project's metadata
_meta() {
	cat <<-EOM
		name: Ollama
		type: custom
		class: process
		icon_url: ollama.com/public/icon-32x32.png
		EOM
}


main "$1" "$NAME"
