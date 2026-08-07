1. run model get ops: python test_llama.py &> ops.log    # generate_test_ops_auto.py will use ops.log
2. generate test code: python generate_test_ops_auto.py   # generate_test_ops_auto.py generate test_ops_auto.py using template test_ops_auto_template.py
3. run a generated test: pytest -v -s test_ops_auto.py::TestOpsAuto::test_auto_009_torch_arange