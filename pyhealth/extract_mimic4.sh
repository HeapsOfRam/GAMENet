#!/bin/bash

for f in *.gz; do
  STEM=$(basename "${f}" .gz)
  gunzip -c "${f}" > /home/ram/code/school/dl4h/final/submit/GAMENet/pyhealth/hiddendata/extracted/"${STEM}"
done
