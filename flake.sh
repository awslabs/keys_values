#!/bin/bash

flake8 . --count --select=E9,F63,F7,F82,F401 --show-source --statistics
