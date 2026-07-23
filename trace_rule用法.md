```
Tracing rules and policies for TorchDynamo compilation decisions.

This module defines the rules that govern what code TorchDynamo should trace and compile
versus what should be executed eagerly. It contains functions and classes that determine:

- Which modules, functions, and objects should be skipped during tracing
- Which parts of the code should cause graph breaks
- How to handle different Python libraries and third-party packages
- Rules for determining when to inline functions vs calling them eagerly

Key components:
- Skip rules: Functions that return True if an object should be skipped during tracing
- Inlining rules: Policies for when to inline function calls during compilation
- Library-specific handling: Special cases for popular Python packages
- Performance heuristics: Rules that balance compilation overhead vs runtime benefits

These rules are critical for TorchDynamo's ability to automatically determine
compilation boundaries and optimize PyTorch programs effectively.
```
