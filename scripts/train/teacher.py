# -*- coding: utf-8 -*-
"""
Entry point for teacher training

Delegates to the teacher runner so the orchestration stays importable and testable inside the
package
"""

from runners.teacher import main

if __name__ == "__main__":
    main()
