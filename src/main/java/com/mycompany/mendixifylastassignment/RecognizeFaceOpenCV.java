/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.mendixifylastassignment;

import org.bytedeco.javacpp.helper.opencv_objdetect;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import org.bytedeco.javacpp.opencv_highgui;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
/**
 *
 * @author mfi
 */
public class RecognizeFaceOpenCV {
    private static final String HAAR_CASCADE_FACE = "src\\main\\resources\\haarcascade_frontalface_default.xml";
    private final double scaleFactor = 4.0;
    private final int minNeighbours = 6;
    public static void main(String[] args) {
        IplImage image =  cvLoadImage("images\\test_image_1.jpg");
        RecognizeFaceOpenCV recognizedFaces;
        recognizedFaces = new RecognizeFaceOpenCV();
        CvSeq sign = recognizedFaces.findFace(image);
        recognizedFaces.drawSignRectangle(sign, image);

        opencv_highgui.cvShowImage("result", image);
        opencv_highgui.cvWaitKey(0);

    }

    public CvSeq findFace(IplImage image) {
        // load the cascade model for recognizing faces
        CvHaarClassifierCascade haarClassifier = new CvHaarClassifierCascade(cvLoad(HAAR_CASCADE_FACE));
        // convert the read image to a grayscale for the recognition, by first creating an empty image with correct size
        // and then converting the color to gray
        IplImage grayedImage;
        grayedImage = org.bytedeco.javacpp.opencv_core.cvCreateImage(
                new org.bytedeco.javacpp.opencv_core.CvSize(image.width(), image.height()), image.depth(), 1);
        cvCvtColor(image, grayedImage, CV_BGR2GRAY);
        opencv_core.CvMemStorage storage = opencv_core.CvMemStorage.create();
        CvSeq sign = opencv_objdetect.cvHaarDetectObjects(
                grayedImage, haarClassifier, storage, scaleFactor, minNeighbours, CV_HAAR_DO_CANNY_PRUNING);

        opencv_core.cvClearMemStorage(storage);
        return sign;
    }

    private void drawSignRectangle(CvSeq sign, IplImage image) {
        int numFaces = sign.total();
        for (int i = 0; i < numFaces; i++) {
            opencv_core.CvRect signCoordinate = new opencv_core.CvRect(opencv_core.cvGetSeqElem(sign, i));
            opencv_imgproc.cvRectangle(
                    image,
                    cvPoint(signCoordinate.x(), signCoordinate.y()),
                    cvPoint(signCoordinate.width() + signCoordinate.x(), signCoordinate.height() + signCoordinate.y()),
                    opencv_core.CvScalar.RED,
                    2,
                    CV_AA,
                    0);
        }
    }
}
