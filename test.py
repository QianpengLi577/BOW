import os
os.system("python getfeature_train.py")
os.system("python getfeature_test.py")
# os.system("python getfeature_train_deep.py")
# os.system("python getfeature_test_deep.py")
os.system("python load_predict_SVM.py")
os.system("python load_predict_easymkl.py")
os.system("python load_predict_SVM_lowess.py")
os.system("python load_predict_SVM_lowess-1.py")
os.system("python load_predict_ksvd.py")
# os.system("python load_predict_SVM_deep.py")
# os.system("python load_predict_easymkl_deep.py")