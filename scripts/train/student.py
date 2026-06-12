# -*- coding: utf-8 -*-
"""
Entry point for student training with knowledge distillation

Delegates to the student runner so the orchestration stays importable and testable inside the
package
"""

from runners.student import main

if __name__ == "__main__":
    main()
