def test_imports():
    """
    Please don't import sklearn.feature_extraction to solve any of the problems in this assignment. 
    If you fail this test, we will give you a zero for this assignment.

    the 'a' in the file name is so this test is run first on a clean Python interpreter.
    """
    import sys
    import src
    # assert 'sklearn.feature_extraction' not in sys.modules.keys()
	# I commented out the above line per Mike D'Acry's recommendation in Campuswire post #2557
	# nltk internally imports sklearn.feature_extraction, which causes this test to fail.