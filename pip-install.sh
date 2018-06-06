#!/bin/sh
while read p; do
  pip install $p
done < requirements.pip
