By running pytest test_shared_strorage_connector.py, it run the test through the test_shared_storage_connector_hashes(); pytest -s to see full print records for a better understanding and investigation.

The generated KV folders and files will be stored in Path("storage_1/") during the test, which is located at where you execute this ut test file.

There are 11 designed input cases to test the feature. Excepted length (number of folders in the storage path), which is the number of unique hashes form, and the information about each input case, is explicit mentioned and printed in the script. There are 6 unique combinations in the 11 input cases, so there should be 6 unique hashes in the folder at the end. The code also checks whether the number of hashes is correct after each llm generate.


(The critical one, for 2 separate requests with identical text input, and image_1 and image_2 having same pixel size; they should create 2 different hashes in the storage path.)


(Above shows it correctly recognize the image order as well)

Note that the order in important to test it, therefore I did not make them separate test cases in pytest, but use a for loop to execute them one by one.
