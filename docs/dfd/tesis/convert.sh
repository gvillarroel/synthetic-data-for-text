#!/bin/bash
for  f in $(ls); do
    plantuml $f -nometadata -tsvg -o ../images/
done;