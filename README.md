# FaceRecognition_attendance_system
Attendance System that uses face-recognition library. For each frame a feature vector (i.e 128 features) is generated to compare. 


Steps to Run the project:

1. Create a virtualenv
    python3 -m pip install --user virtualenv
    
    virtualenv venv
    source venv/bin/activate
    
2. Install packages
    pip install -r requirements.txt
    
3. Run the program
    python attandance.py
    
    
## Adding New Image
    To add a new image in database add that image in Images folder in the same directory as attandance.py
