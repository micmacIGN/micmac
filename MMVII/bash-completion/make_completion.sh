#!/bin/bash

VARNAME='_MMVII_COMPLETION_PYTHON'
REPLACE='${'${VARNAME}'})"'

sed -e'/< *'${REPLACE}'/a\EOT\n)"' mmvii-completion.template | sed -e'/< *'${REPLACE}'/rmmvii-completion.py' -e's/< *'${REPLACE}'.*/<<'"'"EOT"'"'/'  -e '/'${VARNAME}'/d' >  mmvii-completion

