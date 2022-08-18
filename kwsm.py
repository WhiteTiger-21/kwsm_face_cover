import cv2
import numpy
import argparse

num = 0
parser = argparse.ArgumentParser()
parser.add_argument('--camera')
args = parser.parse_args()
if args.camera :
    num = int(args.camera)

# カメラキャプチャ
cap = cv2.VideoCapture(num)
#kwsm
kwsm = cv2.imread("kwsm.png", cv2.IMREAD_UNCHANGED);

# 分類器
# ここからダウンロード 
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

while True:
    # カメラから1フレームずつ取得
    ret, frame = cap.read()
    # フレームの反転
    frame = cv2.flip(frame, 1)


    # kwsmのもともとの縦横比を計算
    orig_height, orig_width = kwsm.shape[:2]
    aspect_ratio = orig_width/orig_height

    # 顔検出
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facerect = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(10, 10)
    )
    
    if len(facerect) > 1:
        #検出した顔の数だけ処理を行う
        for rect in facerect:
            # 顔サイズに合わせてkwsmをリサイズ
            icon = cv2.resize(kwsm,tuple([int(rect[2]*aspect_ratio*2), int(rect[3]*2)]))

            # 透過処理準備
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            icon = cv2.cvtColor(icon, cv2.COLOR_RGB2RGBA)

            # マスクの作成
            icon_mask = icon[:,:,3]
            _, binary = cv2.threshold(icon_mask, 10, 255, cv2.THRESH_BINARY)

            # カメラフレームとリサイズ済みkwsmのサイズを取得
            height, width = icon.shape[:2]
            frame_height, frame_width = frame.shape[:2]

            w1 = int(width*0.2)
            w2 = width - w1;

            h1 = int(height*0.1)
            h2 = height-h1
            
            # 合成時にはみ出さない場合だけ合成を行う
            if frame_height > rect[1]+height and frame_width > rect[0]+width and 0 < rect[0]-w1:
                # 合成する座標を指定
                roi = frame[rect[1]-h1:h2+rect[1], rect[0]-w1:w2+rect[0]]

                # カメラフレームのうち、顔座標に相当する部分をkwsmに置き換える
                # マスクを使い、笑い男アイコン背景の黒い部分を透過させる
                frame[rect[1]-h1:h2+rect[1], rect[0]-w1:w2+rect[0]] = numpy.where(numpy.expand_dims(binary == 255, -1),icon, roi)

    cv2.imshow('result', frame)

    # 何らかのキーが入力されると終了
    k = cv2.waitKey(1)
    if k != -1:
        break

cap.release()
cv2.destroyAllWindows()
