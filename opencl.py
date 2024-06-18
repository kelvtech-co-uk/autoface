import cv2

# Check if OpenCL is available and enable it
print("OpenCL available:", cv2.ocl.haveOpenCL())
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    print("OpenCL enabled:", cv2.ocl.useOpenCL())
    #print(cv2.ocl.getPlatfomsInfo)
