# Thoughts on the Test Setup for Torchtune

This is a proposal on how to organize tests inside torchtune. Currently there are a few issues, 
1. Tests are split into tests, integration tests and recipe tests with no common entrypoint. 
2. There is a lack of clear test annotations, for eg. which test requires network, which are slow, which should run when, etc.
3. Tests require assets but that is not clearly documented anywhere 

## Design 
1. All tests are by default unit tests unless marked otherwise. Unit tests should not need external assets or network. All such parts are mocked.
3. All integration tests ( aka everything non-unit ) should be marked appropriately ( eg. using @pytest.mark )
4. All tests reside in one top level directory tests/. Ideally the test directory structure matches the library structure 1:1 for ease of discoverability
5. All entry points are running test suites are standardized
  - torch test --> runs all unit tests
  - torch test --integration --> runs all integration tests 
  - torch test --all --> runs all tests  
5. All tests need to be self sufficient
(if certain test requires asset downlod or os environment varibables,
dev setup should cover those or failures of tests should be explicit with steps on how to setup properly)
