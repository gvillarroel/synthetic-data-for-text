#!/bin/bash
for  f in $(ls); do
    java -jar /root/.vscode-server/extensions/jebbs.plantuml-2.17.5/plantuml.jar $f -nometadata -tsvg -o ../images/
done;