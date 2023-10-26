#!/bin/bash

sed -e'/< *mmvii-completion.py)"/a\EOT\n)"' mmvii-completion.template | sed -e'/< *mmvii-completion.py)"/rmmvii-completion.py' -e's/< *mmvii-completion.py)".*/<<'"'"EOT"'"'/'   >  mmvii-completion

