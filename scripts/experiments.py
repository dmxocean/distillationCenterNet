# -*- coding: utf-8 -*-
"""
Entry point for running a batch of experiments

Delegates to the experiments runner so the orchestration stays importable and testable inside the
package
"""

from runners.experiments import main

if __name__ == "__main__":
    main()
