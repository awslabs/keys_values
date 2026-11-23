# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# moved to examples/rl/grpo_rlvr.py; shim kept for queued jobs that still call this path
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "rl" / "grpo_rlvr.py"
    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])
