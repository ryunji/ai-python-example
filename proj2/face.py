# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# STEP 3
#img = ins_get_image('t1')
#img = cv2.imread('colabo.jpg', cv2.IMREAD_COLOR);
img = cv2.imread('h1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('h2.jpg', cv2.IMREAD_COLOR)
# STEP 4
#faces = app.get(img)
#print(faces)
faces1 = app.get(img)
faces2 = app.get(img2)



# STEP 5
# then print all-to-all face similarity
feats = []
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
#feats.append(faces[0].normed_embedding)
#feats.append(faces[0].normed_embedding)

#for face in faces:
#feats.append(face.normed_embedding)
#feats = np.array(feats, dtype=np.float32)


#sims = np.dot(feats[0], feats[1].T)
sims = np.dot(feat1, feat2                                                                                                                                                                                                                                                                                                                           .T)
print(sims)

#print(len(faces))
#print(faces[0].embedding)
#rimg = app.draw_on(img, faces)
#cv2.imwrite("./colabo_output.jpg", rimg)
