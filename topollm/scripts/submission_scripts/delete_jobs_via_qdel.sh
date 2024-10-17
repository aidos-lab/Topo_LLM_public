#!/bin/bash

for i in {11972824..11972850}; do
    qdel $i
done

# One-liner for copy-pasting:
#
# for i in {12171420..12171476}; do qdel $i; done