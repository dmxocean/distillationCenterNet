# -*- coding: utf-8 -*-
"""
Entry point for baseline training

Delegates to the baseline runner so the orchestration stays importable and testable inside the
package
"""

from runners.baseline import main

if __name__ == "__main__":
    main()
