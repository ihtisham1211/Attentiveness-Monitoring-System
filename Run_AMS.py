from LibandMethords import *

# Wrting in file
csvfile=open('AMS_Report.csv','w', newline='')#opening file
writer=csv.writer(csvfile)#loading file
fields = ['Time', 'State', 'Emotion', 'Yaw','Pitch','Roll']# creating fields in file
writer=csv.DictWriter(csvfile, fieldnames=fields) #setting fields
writer.writeheader() #writing fields in file


# code Timer
codeTimer = threadStopWatch() #code timer init
codeThread = threading.Thread(target = codeTimer.start()) #creating code timer thread
codeThread.start() #starting code timer

# parameters for loading data and images
emotion_model_path = 'training_output/fer2013.hdf5' #storing expression model path in var
emotion_labels = get_labels()#loading labels
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# init dlib's face detector (HOG-based)


# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model_dlib() #loading dlib model
emotion_classifier = load_model(emotion_model_path, compile=False) #loading expression model

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3] # 64 x 64 model shape input

# starting lists for calculating modes
emotion_window = []

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# starting video streaming
cv2.namedWindow('AMS')
video_capture = cv2.VideoCapture(0)
ID=1
Sample=0
EYE_AR_THRESH = 0.2
COUNTER = 0

#AMS timer
amsTimer = threadStopWatch()
amsThread = threading.Thread(target = amsTimer.start())

while True:
    a = 0
    landmarks = []# all landmarks will be stored here

    bgr_image = video_capture.read()[1] #getting video frame by frame in 480p
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) #changing the image into gray scale for better results
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    #for face_coordinates in faces:
    #Dlib
    faces, score, idx = detect_faces_dlib(face_detection,gray_image)# passsing face model and image to get face coordinates
    for face in faces:
        face_coordinates = make_face_coordinates_dlib(face)
        ID=ID+1
        Sample=Sample+1
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets) # applying dlib offset with our defined offsets to get only face in the image
        gray_face = gray_image[y1:y2, x1:x2]# only face in this image

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))#resize face to 64 x 64 to fit in model
        except:
            continue

        gray_face = preprocess_input(gray_face, True) # changing image into required form (samples, size1,size2,channels)
        gray_face = np.expand_dims(gray_face, 0) # Image has only 2 Dimension, But we require 4 dimension input to our model
        gray_face = np.expand_dims(gray_face, -1)
        
        emotion_prediction = emotion_classifier.predict(gray_face) #passing face to model
        emotion_probability = np.max(emotion_prediction) # seprate emotion with max probability
        emotion_label_arg = np.argmax(emotion_prediction) # get value of that emotion
        emotion_text = emotion_labels[emotion_label_arg] # get label with value
        emotion_mode = emotion_text # store text of emotion
        
        temp = '' # temp for shape
    #HEAD START
    # loop over the face detections
    for (i, rect) in enumerate(faces):
        shape = predictor(gray_image, rect) #Facial landmarks of the ROI and convert the 68 points into a NumPy array
        shape = face_utils.shape_to_np(shape)
        temp = shape
        #Eye start
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)# applying EAR
        rightEAR = eye_aspect_ratio(rightEye)# applying EAR

		# average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
    for (x, y) in temp:
    	a = a + 1
    	if a== 9 or a == 31 or a == 37 or a == 46 or a == 49 or a == 55: #chin  - Nose - left Eye - Right Eye  - Left mouth - right Mouth
    		cv2.circle(rgb_image, (x, y), 1, (0, 0, 255), -1)
    		landmarks.append(x)
    		landmarks.append(y)

    imgpts, modelpts, rotate_degree, nose = face_orientation(rgb_image, landmarks) #head pose function.

#----------------------------------------------------------------------------------------------
#---------------------------->Labeling of Attentive and Unattentive<---------------------------
#----------------------------------------------------------------------------------------------

    #Emotion labeling
    if emotion_text == 'angry' or emotion_text == 'fear' or emotion_text == 'sad':
        emotion_mode = 'Not Attentive'
    elif emotion_text == 'happy' or emotion_text == 'surprise' or emotion_text == 'neutral':
        emotion_mode = 'Attentive'

    
    #Headpose labeling
    result = 'dont know'
    head_mode = 'dont know'
    roll = int(rotate_degree[0])
    pitch = int(rotate_degree[1])
    yaw = int(rotate_degree[2])

#----------------------------------------------------------------------------------------------
#---------------------------->Checks of Attentive and Unattentive<---------------------------
#----------------------------------------------------------------------------------------------
    per = 0
    
    if roll<=30 and pitch <=30 and yaw<=25 and roll>=-30 and pitch >=-10 and yaw>=-25:
        head_mode = 'Attentive'
    else:
        head_mode = 'Not Attentive'
        
        #EYE
    if ear < EYE_AR_THRESH:
        COUNTER = COUNTER + 1
        if COUNTER > 10000:
            per = 0
    else:
        COUNTER = 0
            #head
        if head_mode == 'Attentive':
            per = per + 70
            #Emotion
        if emotion_mode == 'Attentive':
            per = per + 30
            

    if per >=50:
        result = 'Attentive ' + str(per) +'%'
        color = emotion_probability * np.asarray((0, 255, 0))#Green
        amsTimer.resume()
        writer.writerow({'Time': codeTimer.getInterval(), 'State':'Attentive',
                         'Emotion':emotion_text, 'Yaw':yaw,'Pitch':pitch,'Roll':roll})
    else:
        result = 'Not Attentive ' + str(per) +'%'
        color = emotion_probability * np.asarray((255, 0, 0))#red
        amsTimer.pause()
        writer.writerow({'Time': codeTimer.getInterval(), 'State':'Not Attentive',
                         'Emotion':emotion_text, 'Yaw':yaw,'Pitch':pitch,'Roll':roll})

    color = color.astype(int)
    color = color.tolist()
        
    cv2.putText(rgb_image, "ROLL:" + rotate_degree[0]+' Pitch: '+rotate_degree[1]+
                ' YAW: '+rotate_degree[2], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                thickness=1, lineType=1)
    
    cv2.putText(rgb_image,'Emotion: ' + emotion_text , (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                thickness=1, lineType=1)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, result, color, 0, -45, 1, 1)
    
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('AMS', bgr_image)#Show Final Output
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        codeThread.join()
        amsThread.join()
        writer.writerow({'Time':'', 'State':'',
                         'Emotion':'', 'Yaw':'',
                         'Pitch':'Total Attentive Time : ','Roll':amsTimer.getInterval()})
        csvfile.close()
        break

cv2.waitKey(0)