
#!/bin/bash

git ls-files | grep "\.py$" |  xargs black
