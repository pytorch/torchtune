# Design Philosophy and Best Practices

TorchTune embodies PyTorch’s design philosophy
[[details](https://pytorch.org/docs/stable/community/design.html)], especially
"usability over performance". The code base is designed to be easy to read, use
(and re-use) and extend.

### Simplicity 

TorchTune code should be easy to read, use (and re-use) and extend. We expect
AND want users to read the internals of the codebase. Simple code is better than
complex tricks. While implementing a feature, keep in mind that not every user
will be a domain expert. For example, in most cases simply re-writing a class
(even if this duplicates code) might be a more desirable strategy than utilizing
complex module-swapping logic which only a subset of users will understand.


### Native PyTorch

Users shouldn’t need to learn N different frameworks to understand or contribute
to the core of TorchTune. They only need to understand PyTorch.  We should
provide integrations with other libraries and frameworks where these make sense.
But these integrations should not “pollute” the code base. Provide these through
wrapper functions around native implementations. This will also make it easier
to debug issues due to breakages in these external libraries.


### Correctness and Stability

PyTorch has very high user-trust. TorchTune should cultivate the same. 

- Components should have unit-tests to ensure numerical parity with reference
  implementations, and to catch breakages.
- Model implementations should have checkpoint-tests to ensure output parity
  with reference implementations, and to catch breakages.
- Training recipes should have integration tests to ensure performance parity
  with reference implementations on standard benchmarks, and to catch breakages.
- Clearly classify external APIs with “stable” or “experimental” tags to
  establish user expectation.


### Best Practices

1. **Modular Blocks instead of Monolithic Classes**. Stuffing all of the logic
   into a single class limits readability and makes it hard to reuse logic.
   Think about breaking the implementation into self-contained blocks which can
   be used independently from a given model. For example, attention mechanisms,
   embedding classes, transformer layers etc.

2. **Say no to Inheritance**. You really don’t need it AND it makes the code
   much harder to understand or refactor since the logic is spread across many
   files/classes. Where needed, consider using Protocols.

3. **Clean Interfaces**. There’s nothing more challenging than reading through
   functions/constructors with ~100 parameters. Think carefully about what needs
   to be exposed to the user and don’t hesitate to hard-code parameters until
   there is a need to make them configurable.

4. **Intrusive Configs**. Config objects should not intrude into the class
   implementation. Configs should interact with these classes through cleanly
   defined builder functions which convert the config into flat parameters
   needed to instantiate an object.

5. **Limit Generalization**. Attempting to generalize code before this is needed
   unnecessarily complicates implementations - you are anticipating use cases
   you don’t know a lot about. When you actually need to generalize a component,
   think about whether it’s worth it to complicate a given interface to stuff in
   more functionality. Don’t be afraid of code duplication if it makes things
   easier to read.

6. **Value Checks and Asserts**. Don’t check values in higher level modules -
   defer the checks to the modules where the values are actually used. This
   helps reduce the number of `raise` statements in code which generally
   hurts readability, but are critical for correctness.
