# -*- coding: utf-8 -*-
"""
Entry point for checkpoint evaluation

Delegates to the evaluate runner so the orchestration stays importable and testable inside the
package
"""

from runners.evaluate import main

if __name__ == "__main__":
    main()
