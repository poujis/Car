#include <iostream>
#include <stdio.h>
#include <string.h>
#include "car_lib.h"
#include <termios.h>
#include <unistd.h> 
#include <fcntl.h>
#include <sys/signal.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <pthread.h> 
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/gpu/device/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#define PI 3.1415926

using namespace std;
using namespace cv;

extern "C" {

 /**
   * @brief  To load image file to the buffer.
   * @param  file: pointer for load image file in local path
     outBuf: destination buffer pointer to load
     nw : width value of the destination buffer
     nh : height value of the destination buffer
   * @retval none
   */
 void OpenCV_load_file(char* file, unsigned char* outBuf, int nw, int nh)
 {
  Mat srcRGB;
  Mat dstRGB(nh, nw, CV_8UC3, outBuf);

  srcRGB = imread(file, CV_LOAD_IMAGE_COLOR); // rgb
  //cvtColor(srcRGB, srcRGB, CV_RGB2BGR);

  cv::resize(srcRGB, dstRGB, cv::Size(nw, nh), 0, 0, CV_INTER_LINEAR);
 }

 /**
   * @brief  To convert format from BGR to RGB.
   * @param  inBuf: buffer pointer of BGR image
     w: width value of the buffers
     h : height value of the buffers
     outBuf : buffer pointer of RGB image
   * @retval none
   */
 void OpenCV_Bgr2RgbConvert(unsigned char* inBuf, int w, int h, unsigned char* outBuf)
 {
  Mat srcRGB(h, w, CV_8UC3, inBuf);
  Mat dstRGB(h, w, CV_8UC3, outBuf);

  cvtColor(srcRGB, dstRGB, CV_BGR2RGB);
 }

 /**
   * @brief  Detect faces on loaded image and draw circles on the faces of the loaded image.
   * @param  file: pointer for load image file in local path
     outBuf: buffer pointer to draw circles on the detected faces
     nw : width value of the destination buffer
     nh : height value of the destination buffer
   * @retval none
   */
 void OpenCV_face_detection(char* file, unsigned char* outBuf, int nw, int nh)
 {
  Mat srcRGB = imread(file, CV_LOAD_IMAGE_COLOR);
  Mat dstRGB(nh, nw, CV_8UC3, outBuf);

  // Load Face cascade (.xml file)
  CascadeClassifier face_cascade;
  face_cascade.load("haarcascade_frontalface_alt.xml");

  // Detect faces
  std::vector<Rect> faces;
  face_cascade.detectMultiScale(srcRGB, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

  // Draw circles on the detected faces
  for (int i = 0; i < faces.size(); i++)
  {
   Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
   ellipse(srcRGB, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
  }

  cv::resize(srcRGB, dstRGB, cv::Size(nw, nh), 0, 0, CV_INTER_LINEAR);
 }

 /**
   * @brief  To bind two images on destination buffer.
   * @param  file1: file path of first image to bind
     file2: file path of second image to bind
     outBuf : destination buffer pointer to bind
     nw : width value of the destination buffer
     nh : height value of the destination buffer
   * @retval none
   */
 void OpenCV_binding_image(char* file1, char* file2, unsigned char* outBuf, int nw, int nh)
 {
  Mat srcRGB = imread(file1, CV_LOAD_IMAGE_COLOR);
  Mat srcRGB2 = imread(file2, CV_LOAD_IMAGE_COLOR);
  Mat dstRGB(nh, nw, CV_8UC3, outBuf);

  cv::resize(srcRGB2, srcRGB2, cv::Size(srcRGB2.cols / 1.5, srcRGB2.rows / 1.5));
  cv::Point location = cv::Point(280, 220);
  for (int y = std::max(location.y, 0); y < srcRGB.rows; ++y)
  {
   int fY = y - location.y;
   if (fY >= srcRGB2.rows)
    break;

   for (int x = std::max(location.x, 0); x < srcRGB.cols; ++x)
   {
    int fX = x - location.x;
    if (fX >= srcRGB2.cols)
     break;

    double opacity = ((double)srcRGB2.data[fY * srcRGB2.step + fX * srcRGB2.channels() + 3]) / 255.;
    for (int c = 0; opacity > 0 && c < srcRGB.channels(); ++c)
    {
     unsigned char overlayPx = srcRGB2.data[fY * srcRGB2.step + fX * srcRGB2.channels() + c];
     unsigned char srcPx = srcRGB.data[y * srcRGB.step + x * srcRGB.channels() + c];
     srcRGB.data[y * srcRGB.step + srcRGB.channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
    }
   }
  }

  cv::resize(srcRGB, dstRGB, cv::Size(nw, nh), 0, 0, CV_INTER_LINEAR);
 }

 /**
   * @brief  Apply canny edge algorithm and draw it on destination buffer.
   * @param  file: pointer for load image file in local path
     outBuf: destination buffer pointer to apply canny edge
     nw : width value of destination buffer
     nh : height value of destination buffer
   * @retval none
   */
 void OpenCV_canny_edge_image(char* file, unsigned char* outBuf, int nw, int nh)
 {
  Mat srcRGB = imread(file, CV_LOAD_IMAGE_COLOR);
  Mat srcGRAY;
  Mat dstRGB(nh, nw, CV_8UC3, outBuf);

  cvtColor(srcRGB, srcGRAY, CV_BGR2GRAY);
  // 케니 알고리즘 적용
  cv::Mat contours;
  cv::Canny(srcGRAY, // 그레이레벨 영상
   contours, // 결과 외곽선
   125,  // 낮은 경계값
   350);  // 높은 경계값

  // 넌제로 화소로 외곽선을 표현하므로 흑백 값을 반전
  //cv::Mat contoursInv; // 반전 영상
  //cv::threshold(contours, contoursInv, 128, 255, cv::THRESH_BINARY_INV);
  // 밝기 값이 128보다 작으면 255가 되도록 설정

  cvtColor(contours, contours, CV_GRAY2BGR);

  cv::resize(contours, dstRGB, cv::Size(nw, nh), 0, 0, CV_INTER_LINEAR);
 }

 /**
   * @brief  Detect the hough and draw hough on destination buffer.
   * @param  srcBuf: source pointer to hough transform
     iw: width value of source buffer
     ih : height value of source buffer
     outBuf : destination pointer to hough transform
     nw : width value of destination buffer
     nh : height value of destination buffer
   * @retval none
 */
 void OpenCV_object_detect(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh) {

  Mat dstRGB(nh, nw, CV_8UC3, outBuf);

 

  Mat srcRGB(ih, iw, CV_8UC3, srcBuf);

  Mat resRGB(ih, iw, CV_8UC3);

 

  int Greencnt = 0;

  printf("1");

  for (int j = 0; j <= nh; j++) {

   for (int i = 0; i <= nw; i++) {

 

    if (srcRGB.at<cv::Vec3b>(j, i)[0] <= 255 && srcRGB.at<cv::Vec3b>(j, i)[1] >= 0 && srcRGB.at<cv::Vec3b>(j, i)[2] <= 255) {

     Greencnt++;

    }

   }

  }

  if (Greencnt >= 10) {

   printf("detect");

  }

  printf("2");

  cv::resize(srcRGB, dstRGB, cv::Size(nw, nh), 0, 0, CV_INTER_LINEAR);

 }
 void OpenCV_hough_transform(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh)
{
  Scalar lineColor = cv::Scalar(255, 255, 255);
  Scalar c_B = cv::Scalar(255, 0, 0);
  Scalar c_G = cv::Scalar(0, 255, 0);
  Scalar c_R = cv::Scalar(0,0, 255);
  Scalar c_Black = cv::Scalar(0, 0, 0);

  Mat dstRGB(nh, nw, CV_8UC3, outBuf);

  Mat srcRGB(ih, iw, CV_8UC3, srcBuf);
  Mat resRGB(ih, iw, CV_8UC3);
  //cvtColor(srcRGB, srcRGB, CV_BGR2BGRA);

  // Ä³´Ï ¾Ë°í¸®Áò Àû¿ë
  cv::Mat contours;
  cv::Canny(srcRGB, contours, 150, 300);

  // ¼± °¨Áö À§ÇÑ ÇãÇÁ º¯È¯
  std::vector<cv::Vec2f> lines;
  cv::HoughLines(contours, lines, 1, PI / 180, // ´Ü°èº° Å©±â (1°ú ¥ð/180¿¡¼­ ´Ü°èº°·Î °¡´ÉÇÑ ¸ðµç °¢µµ·Î ¹ÝÁö¸§ÀÇ ¼±À» Ã£À½)
    50);  // ÅõÇ¥(vote) ÃÖ´ë °³¼ö

  // ¼± ±×¸®±â
  cv::Mat result(contours.rows, contours.cols, CV_8UC3, lineColor);
  //printf("Lines detected: %d\n", lines.size());

  // ¼± º¤ÅÍ¸¦ ¹Ýº¹ÇØ ¼± ±×¸®±â
  std::vector<cv::Vec2f>::const_iterator it = lines.begin();
  int left_line =0;
  int right_line =0;

  int pt1_x = 0, pt1_x1 = 0;
  int pt2_x = 0, pt2_x1 = 0;
  while (it != lines.end())
  {
    float rho = (*it)[0];   // Ã¹ ¹øÂ° ¿ä¼Ò´Â rho °Å¸®
    float theta = (*it)[1]; // µÎ ¹øÂ° ¿ä¼Ò´Â µ¨Å¸ °¢µµ

    if (theta < PI / 2. -PI / 13.  && theta > PI / 13.) // ¼öÁ÷ Çà
    {
      pt1_x += rho / cos(theta);
      pt1_x1 += (rho - result.rows * sin(theta)) / cos(theta);
      //cv::Point pt1(rho / cos(theta), 0); // Ã¹ Çà¿¡¼­ ÇØ´ç ¼±ÀÇ ±³Â÷Á¡   
      //cv::Point pt2((rho - result.rows * sin(theta)) / cos(theta), result.rows);
      // ¸¶Áö¸· Çà¿¡¼­ ÇØ´ç ¼±ÀÇ ±³Â÷Á¡
      //cv::line(srcRGB, pt1, pt2, c_B, 1); // ÇÏ¾á ¼±À¸·Î ±×¸®±â
      left_line ++;
    }
    else if (theta > PI / 2. + PI / 13. && theta < PI*1.0 - PI / 13.) // ¼öÁ÷ Çà
    {
      pt2_x += rho / cos(theta);
      pt2_x1 += (rho - result.rows * sin(theta)) / cos(theta);
      //cv::Point pt1(rho / cos(theta), 0); // Ã¹ Çà¿¡¼­ ÇØ´ç ¼±ÀÇ ±³Â÷Á¡   
      //cv::Point pt2((rho - result.rows * sin(theta)) / cos(theta), result.rows);
      // ¸¶Áö¸· Çà¿¡¼­ ÇØ´ç ¼±ÀÇ ±³Â÷Á¡
      //cv::line(srcRGB, pt1, pt2, c_R, 1); // ÇÏ¾á ¼±À¸·Î ±×¸®±â
      right_line ++;
    }
    else // ¼öÆò Çà
    {
      cv::Point pt5(0, rho / sin(theta)); // Ã¹ ¹øÂ° ¿­¿¡¼­ ÇØ´ç ¼±ÀÇ ±³Â÷Á¡  
      cv::Point pt6(result.cols, (rho - result.cols * cos(theta)) / sin(theta));
      // ¸¶Áö¸· ¿­¿¡¼­ ÇØ´ç ¼±ÀÇ ±³Â÷Á¡
      //cv::line(srcRGB, pt1, pt2, lineColor, 1); // ÇÏ¾á ¼±À¸·Î ±×¸®±â
    }
    //printf("line: rho=%f, theta=%f\n", rho, theta);
    ++it;
  }

  /// no count Exception Handling
  if(left_line != 0){
    cv::Point pt1(pt1_x/left_line,0);
    cv::Point pt2(pt1_x1/left_line, result.rows);
    cv::line(srcRGB, pt1, pt2, c_B, 1);
  }
  if(right_line != 0){
    cv::Point pt3(pt2_x/right_line,0);
    cv::Point pt4(pt2_x1/right_line, result.rows);
    cv::line(srcRGB, pt3, pt4, c_R, 1);
  }
  printf("l : %d // r : %d \n",left_line, right_line);
  
  cv::resize(srcRGB, dstRGB, cv::Size(nw, nh), 0, 0, CV_INTER_LINEAR);
}
 /**
   * @brief  Merge two source images of the same size into the output buffer.
   * @param  src1: pointer to parameter of rgb32 image buffer
     src2: pointer to parameter of bgr32 image buffer
     dst : pointer to parameter of rgb32 output buffer
     w : width of src and dst buffer
     h : height of src and dst buffer
   * @retval none
   */
 void OpenCV_merge_image(unsigned char* src1, unsigned char* src2, unsigned char* dst, int w, int h)
 {
  Mat src1AR32(h, w, CV_8UC4, src1);
  Mat src2AR32(h, w, CV_8UC4, src2);
  Mat dstAR32(h, w, CV_8UC4, dst);

  cvtColor(src2AR32, src2AR32, CV_BGRA2RGBA);

  for (int y = 0; y < h; ++y) {
   for (int x = 0; x < w; ++x) {
    double opacity = ((double)(src2AR32.data[y * src2AR32.step + x * src2AR32.channels() + 3])) / 255.;
    for (int c = 0; opacity > 0 && c < src1AR32.channels(); ++c) {
     unsigned char overlayPx = src2AR32.data[y * src2AR32.step + x * src2AR32.channels() + c];
     unsigned char srcPx = src1AR32.data[y * src1AR32.step + x * src1AR32.channels() + c];
     src1AR32.data[y * src1AR32.step + src1AR32.channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
    }
   }
  }

  memcpy(dst, src1AR32.data, w * h * 4);

 }

 void OpenCV_object_detect2(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh) {
  // 2. speed control ----------------------------------------------------------
  //printf("green stop\n");
   unsigned char status;
    short speed;
    unsigned char gain;
    int position, posInit, posDes, posRead;
    short angle;
    int channel;
    int data;
    char sensor;
    int i, j;
    int tol;
    char byte = 0x80;
    int greensum_y = 0;
    int greensum_x = 0;
    int greencnt = 0;
    int g_x = 0;
    int g_y = 0;
    int leftx=0;
    int rightx=0;
    int avg_x;
    bool loop_out = false;

    CarControlInit();
  //jobs to be done beforehand;
  // PositionControlOnOff_Write(UNCONTROL); // position controller must be OFF !!!

  // //control on/off
  // status = SpeedControlOnOff_Read();
  // //printf("SpeedControlOnOff_Read() = %d\n", status);
  // SpeedControlOnOff_Write(CONTROL);

  // //speed controller gain set
  // //P-gain
  // gain = SpeedPIDProportional_Read();        // default value = 10, range : 1~50
  // //printf("SpeedPIDProportional_Read() = %d \n", gain);
  // gain = 20;
  // SpeedPIDProportional_Write(gain);

  // //I-gain
  // gain = SpeedPIDIntegral_Read();        // default value = 10, range : 1~50
  // //printf("SpeedPIDIntegral_Read() = %d \n", gain);
  // gain = 20;
  // SpeedPIDIntegral_Write(gain);

  // //D-gain
  // gain = SpeedPIDDifferential_Read();        // default value = 10, range : 1~50
  // //printf("SpeedPIDDefferential_Read() = %d \n", gain);
  // gain = 20;
  // SpeedPIDDifferential_Write(gain);


  // //speed set   
  // speed = DesireSpeed_Read();
  // //printf("DesireSpeed_Read() = %d \n", speed);
  // speed = 50;
  // DesireSpeed_Write(speed);

  // sleep(2);  //run time

  // posDes = 10000;
  // position = posInit + posDes;
  // DesireEncoderCount_Write(position);

  // position = DesireEncoderCount_Read();
  // printf("DesireEncoderCount_Read() = %d\n", position);

  // tol = 50;    // tolerance
  // while (abs(posRead - position) > tol)
  // {
  //  posRead = EncoderCounter_Read();
  //  printf("EncoderCounter_Read() = %d\n", posRead);
  // }
  // sleep(1);

  Mat dstRGB(nh, nw, CV_8UC3, outBuf);
  Mat srcRGB(ih, iw, CV_8UC3, srcBuf);

  printf("ih : %d , iw : %d\n",ih,iw);
  PositionControlOnOff_Write(UNCONTROL); 
  SpeedControlOnOff_Write(CONTROL);
  speed = 30;
  DesireSpeed_Write(speed);
  for (j=10;j<ih;j++){
    for(i=10;i<iw-1;i++){
      if(srcRGB.at<cv::Vec3b>(j, i)[0] <= 30 && srcRGB.at<cv::Vec3b>(j, i)[1] <= 30 && srcRGB.at<cv::Vec3b>(j, i)[2] <= 30){
        leftx = i;
        printf("%d\n",i);
        loop_out = true;
        break;
      }
    }
    if(loop_out==true)
        break;
  }
  loop_out=false;
  for (j=ih-10;j>0;j--){
    for(i=iw-10;i>0;i--){
      if(srcRGB.at<cv::Vec3b>(j, i)[0] <= 30 && srcRGB.at<cv::Vec3b>(j, i)[1] <= 30 && srcRGB.at<cv::Vec3b>(j, i)[2] <= 30){
        rightx = i;
        loop_out=true;
        break;
      }
    }
    if(loop_out==true)
          break;
  }
  avg_x = (int)((leftx+rightx)/2);
  printf("leftx = %d\n",leftx);
  //printf("rightx= %d\n",rightx);
  //printf("avg_x : %d\n",avg_x);
  if(DistanceSensor(1)>=1200){
    DesireSpeed_Write(0);
  }
  else if(avg_x<140){
    angle=1700;
    SteeringServoControl_Write(angle);
  }
  else if(avg_x>=140 && avg_x<=180){
    angle=1500;
    SteeringServoControl_Write(angle);
  }
  else if(avg_x<=320){
    angle=1300;
    SteeringServoControl_Write(angle);
  }


  // for (int j = 0; j < ih; j++) {
  //  for (int i = 0; i < iw; i++) {
  //   if (srcRGB.at<cv::Vec3b>(j, i)[0] >= 220 && srcRGB.at<cv::Vec3b>(j, i)[1] >= 220 && srcRGB.at<cv::Vec3b>(j, i)[2] >= 220) {
  //    greencnt++;
  //    greensum_y += j;
  //    greensum_x += i;
  //    srcRGB.at<cv::Vec3b>(j, i)[0] = 255;
  //    srcRGB.at<cv::Vec3b>(j, i)[1] = 255;
  //    srcRGB.at<cv::Vec3b>(j, i)[2] = 255;

  //   }
  //   if(greencnt==150)
  //     break;
  //  }
  //  printf("%d \n",greencnt);

  //  // g_y = greensum_y / greencnt;
  //  // g_x = greensum_x / greencnt;


  // }
  //    if (greencnt >= 150) {
  //   //printf("detect green");

  //   //speed set   
  //   speed = DesireSpeed_Read();
  //   //printf("DesireSpeed_Read() = %d \n", speed);
  //   speed = 10;
  //   DesireSpeed_Write(speed); //run time

  //   speed = DesireSpeed_Read();
  //   //printf("DesireSpeed_Read() = %d \n", speed);

  //   speed = 0;
  //   DesireSpeed_Write(speed);
  //   greencnt=0;


  //  }

 
 cv::resize(srcRGB, dstRGB, cv::Size(nw, nh), 0, 0, CV_INTER_LINEAR);
}
}