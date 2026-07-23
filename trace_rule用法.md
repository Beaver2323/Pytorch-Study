```
This module contains classes and utilities for building variable trackers in Dynamo.
Variable trackers are used to convert Python values into symbolic representations
that can be traced and transformed during graph capture.

The key classes are:

- VariableBuilder: Handles source-tracked objects that need guards and proper
  reconstruction in the output graph. Used for inputs, module attributes, etc.

- SourcelessBuilder: Handles ephemeral objects created during tracing that don't
  need source tracking or guards. Used for temporary lists, intermediate values, etc.

Variable trackers enable Dynamo to track the flow of values through the program,
maintain guards for dynamic properties, and reconstruct values in the output graph.
The builders in this module handle converting Python values into appropriate
VariableTracker instances based on their type and usage context.
```
