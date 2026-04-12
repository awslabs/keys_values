
#!/bin/bash

git ls-files | grep "\.py$" |  xargs python -m black
