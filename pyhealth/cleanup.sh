#!/bin/bash

podman rm -fa
podman rmi -fa
podman kill -a
podman system prune -a
